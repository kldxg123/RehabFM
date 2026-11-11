# -*- coding: utf-8 -*-
"""数据管线：注册表解析、token 化（重采样 + z-score + 切窗）、合成数据生成。

对应设计文档第七章。真实数据接入时，只需为每个数据集实现一个
`iter_recording() -> dict[modality, np.ndarray(C, T)]` 适配器，
token 化与批处理全部复用本模块。
"""
import ast
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import MODALITIES, WINDOW_SEC, CONTEXT_WINDOWS

# 注册表中原始模态名 -> 统一模态名
MODALITY_MAP = {
    "EEG": "EEG", "EMG": "EMG", "ECG": "ECG", "EDA": "EDA",
    "IMU": "IMU", "Force": "FORCE", "Pressure": "FORCE",
    "Skeleton": "SKEL", "MoCap": "SKEL", "Kinematics": "SKEL",
    # RGB/Depth/Radar/fMRI/MRI：经姿态估计或 ROI 提取后再接入，见设计文档表 7
}


# 科研不可用数据集（rehabcorpus_registry_科研可用标注.xlsx "列1" 非空者）：
# 许可未经核验 / 非商业限制 / 定制限制性许可 —— 一律不得纳入训练
RESEARCH_EXCLUDED = {"D003", "D005", "D006", "D011", "D014", "D015", "D022"}


def load_registry(xlsx_path):
    """读取注册表，返回每个数据集统一后的模态列表、许可层级与科研可用标记。

    xlsx 为原始表或科研可用标注表均可：标注表含"列1"（非空 = 不可用），
    原始表则回退到内置 RESEARCH_EXCLUDED 清单（两者结论一致）。"""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    ann_col = "列1" if "列1" in df.columns else None
    rows = []
    for _, r in df.iterrows():
        mods = sorted({MODALITY_MAP[m] for m in ast.literal_eval(r["modalities"])
                       if m in MODALITY_MAP})
        redistrib = str(r["redistribution_allowed"]).upper() == "TRUE"
        if ann_col is not None:
            research_ok = pd.isna(r[ann_col])
        else:
            research_ok = r["dataset_id"] not in RESEARCH_EXCLUDED
        rows.append(dict(
            dataset_id=r["dataset_id"], short_name=r["short_name"],
            modalities=mods,
            license_tier="A" if redistrib else "B",
            research_ok=bool(research_ok),
            num_subjects=int(r["num_subjects"]),
        ))
    return rows


def zscore(x, eps=1e-6):
    """逐通道 z-score。x: (C, T)"""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def resample_linear(x, fs_in, fs_out):
    """线性插值重采样（轻量实现；正式管线建议 scipy.signal.resample_poly）。x: (C, T)"""
    if abs(fs_in - fs_out) < 1e-9:
        return x
    t_in = np.arange(x.shape[-1]) / fs_in
    t_out = np.arange(int(round(x.shape[-1] * fs_out / fs_in))) / fs_out
    return np.stack([np.interp(t_out, t_in, ch) for ch in x], axis=0).astype(np.float32)


def tokenize(recording):
    """一段记录 -> {modality: (W, C, Tw)} tokens + 每窗每通道统计特征（掩码重建目标）。

    recording: {modality: np.ndarray(C, T)}，采样率须为 config 中的 canonical_fs。
    返回 tokens: {modality: (W, C, Tw)}，stats: {modality: (W, C, 8)}
    统计特征 8 维 = [mean, std, min, max, energy, low/mid/high 频带能量比]，
    作为掩码信号建模的轻量重建目标（正式版可换 STFT 损失，见设计文档 5.1）。
    """
    tokens, stats = {}, {}
    for mod, x in recording.items():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # 防御：源数据 NaN/Inf 清零
        fs = MODALITIES[mod]["canonical_fs"]
        tw = int(WINDOW_SEC * fs)
        w_total = x.shape[-1] // tw
        if w_total == 0:
            continue
        x = zscore(x[:, : w_total * tw].astype(np.float32))
        xw = x.reshape(x.shape[0], w_total, tw).transpose(1, 0, 2)  # (W, C, Tw)
        tokens[mod] = xw
        # 轻量统计目标（8 维 = mean/std/min/max + 4 个频带能量；log1p 压缩量纲）
        s = [xw.mean(-1), xw.std(-1), xw.min(-1), xw.max(-1)]
        spec = np.abs(np.fft.rfft(xw, axis=-1))
        s += [np.log1p(b.mean(-1)) for b in np.array_split(spec, 4, axis=-1)]
        stats[mod] = np.stack(s, axis=-1).astype(np.float32)  # (W, C, 8)
    # 波形重建目标：通道均值 -> 平均池化到 32 点/窗（随样本变化，防表征坍缩）
    waves = {}
    for mod, xw in tokens.items():
        wv = xw.mean(axis=1)                                # (W, Tw) 通道均值
        W_, Tw_ = wv.shape
        pool = max(1, Tw_ // 32)
        wv = wv[:, : 32 * pool].reshape(W_, 32, pool).mean(-1)  # (W, 32)
        waves[mod] = wv.astype(np.float32)
    return tokens, stats, waves


def pad_collate(samples):
    """把变通道样本 pad 成批。返回 {modality: dict(x=(B,W,Cmax,Tw), cmask=(B,W,Cmax), stats=...)}"""
    mods = sorted({m for s in samples for m in s["tokens"]})
    batch = {}
    for mod in mods:
        owners = [i for i, s in enumerate(samples) if mod in s["tokens"]]
        cmax = max(s["tokens"][mod].shape[1] for s in (samples[i] for i in owners))
        W = samples[owners[0]]["tokens"][mod].shape[0]
        Tw = samples[owners[0]]["tokens"][mod].shape[2]
        B = len(owners)
        x = np.zeros((B, W, cmax, Tw), np.float32)
        st = np.zeros((B, W, cmax, 8), np.float32)
        wv = np.zeros((B, W, 32), np.float32)
        cmask = np.zeros((B, W, cmax), bool)
        wmask = np.zeros((B, W), bool)
        for bi, i in enumerate(owners):
            t = samples[i]["tokens"][mod]
            x[bi, :, : t.shape[1]] = t
            st[bi, :, : t.shape[1]] = samples[i]["stats"][mod]
            wv[bi] = samples[i]["waves"][mod]
            cmask[bi, :, : t.shape[1]] = True
            # Cached adapters use exactly-zero rows for context padding.
            # Capture validity BEFORE random masking so masked real windows remain prediction targets.
            wmask[bi] = np.any(t != 0, axis=(1, 2))
            if not wmask[bi].any():
                wmask[bi, 0] = True
        batch[mod] = dict(x=torch.from_numpy(x), stats=torch.from_numpy(st),
                          wave=torch.from_numpy(wv), cmask=torch.from_numpy(cmask),
                          wmask=torch.from_numpy(wmask),
                          dataset=torch.tensor([samples[i]["dataset"] for i in owners]),
                          index=torch.tensor(owners))
    return batch


# ---------------- 合成数据（真实数据挂载前验证管线用） ----------------

class SyntheticRehabDataset(Dataset):
    """模拟 6 个"数据集"、不同模态组合、含跨模态生理耦合与患者/对照类别差异。

    生理先验（合成）：
      - 共享节律 latent rho(t) 同时调制 EMG 包络与 IMU 加速度 → 跨模态可对比对齐；
      - y=1（患者）：EEG 8–12 Hz 能量下降 + EMG 包络减弱 → 下游可分类。
    """

    DATASETS = [  # (模态组合, 被试数) —— 模拟注册表中的不同模态组合
        (["EEG", "EMG", "IMU"], 60),
        (["EEG"], 40),
        (["IMU"], 50),
        (["EEG", "IMU"], 40),
        (["EMG", "IMU"], 30),
        (["IMU", "FORCE"], 30),
    ]
    N_CH = {"EEG": 8, "EMG": 4, "IMU": 6, "FORCE": 2, "ECG": 1, "EDA": 1, "SKEL": 50}

    def __init__(self, segments_per_subject=4, seed=0):
        rng = np.random.default_rng(seed)
        self.items = []  # (dataset_idx, subject_id, y, seg_seed)
        for di, (mods, n_sub) in enumerate(self.DATASETS):
            for s in range(n_sub):
                y = int(rng.random() < 0.5)
                for k in range(segments_per_subject):
                    self.items.append((di, s, y, rng.integers(0, 2 ** 31)))
        self.mods_of = {di: mods for di, (mods, _) in enumerate(self.DATASETS)}

    def __len__(self):
        return len(self.items)

    def _gen_modality(self, mod, rng, sources, y, W):
        """每通道 = 共享潜源的随机混合 + 小噪声（模拟真实生理多通道结构）。

        sources: dict(rho1=class 节律 (W*Tw_ref,), rho2=AR 噪声, rho3=步频节律)
        各模态取不同源子集 → 跨模态有真实共享信息，重建目标非退化。
        """
        fs = MODALITIES[mod]["canonical_fs"]
        Tw = int(WINDOW_SEC * fs)
        C = self.N_CH[mod]
        t = np.arange(W * Tw) / fs
        def up(src):  # 源重采样到本模态时间轴
            return np.interp(t, np.linspace(0, W * Tw / fs, len(src)), src).astype(np.float32)
        rho1 = up(sources["rho1"])                      # 类相关节律（10 vs 45 Hz）
        rho2 = up(sources["rho2"])                      # AR(1) 个体噪声
        rho3 = up(sources["rho3"])                      # 1.2 Hz 步态节律
        mixing = {
            "EEG":   [rho1, rho2],
            "EMG":   [rho2, rho3],
            "IMU":   [rho3, rho2],
            "FORCE": [rho3],
            "ECG":   [rho2], "EDA": [rho2], "SKEL": [rho3, rho2],
        }[mod]
        w = rng.uniform(0.5, 1.5, (C, len(mixing))).astype(np.float32)  # 随机混合矩阵
        x = sum(w[:, i:i+1] * s[None, :] for i, s in enumerate(mixing))
        x = x + 0.3 * rng.standard_normal((C, W * Tw)).astype(np.float32)  # 小噪声
        return x.astype(np.float32)

    def __getitem__(self, i):
        di, subj, y, seg_seed = self.items[i]
        rng = np.random.default_rng(seg_seed)
        W = CONTEXT_WINDOWS
        n = W * 256  # 源在 256 Hz 参考轴上生成
        t = np.arange(n) / 256
        f1 = 10.0 if y == 0 else 45.0                     # 类相关节律频率
        phase = rng.uniform(0, 6.28)
        rho1 = np.sin(2 * np.pi * f1 * t + phase)
        ar = np.zeros(n); e = rng.standard_normal(n)      # AR(1) 个体噪声
        for k in range(1, n):
            ar[k] = 0.98 * ar[k - 1] + e[k]
        rho2 = ar / (ar.std() + 1e-6)
        rho3 = np.abs(np.sin(2 * np.pi * 1.2 * t + rng.uniform(0, 6.28)))  # 步频
        sources = dict(rho1=rho1, rho2=rho2, rho3=rho3)
        recording = {m: self._gen_modality(m, rng, sources, y, W)
                     for m in self.mods_of[di]}
        tokens, stats, waves = tokenize(recording)
        return dict(tokens=tokens, stats=stats, waves=waves, y=y, dataset=di,
                    subject=di * 1000 + subj)  # 全局唯一被试 id，防跨数据集泄漏


def collate_fn(samples):
    batch = pad_collate(samples)
    batch["y"] = torch.tensor([s["y"] for s in samples])
    batch["dataset"] = torch.tensor([s["dataset"] for s in samples])
    batch["subject"] = torch.tensor([s["subject"] for s in samples])
    if all("win_y" in s for s in samples):
        batch["win_y"] = torch.from_numpy(np.stack([s["win_y"] for s in samples]))
    return batch
