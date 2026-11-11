# -*- coding: utf-8 -*-
"""阶段一：自监督预训练（对应设计文档第五章，图 3 / 表 4）。

三个损失：
  A. LOO-CL  跨模态对比对齐（InfoNCE，留一法）
  B. MASKED  掩码信号建模（重建被掩时间窗的统计特征）
  C. XPRED   跨模态互预测（可见模态融合表征 -> 目标模态统计特征）
外加模态丢弃 + 通道丢弃正则。

用法：
  python -m rehabfm.train_pretrain --epochs 3 --out ckpt/pretrain.pt
"""
import argparse
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import PRETRAIN as HP, MODEL as MODEL_CFG
MODEL_D = MODEL_CFG["d_model"]
from .data import SyntheticRehabDataset, collate_fn
from .model import RehabFM, MaskedStatsDecoder


def apply_window_mask(batch, ratio):
    """随机遮盖时间窗：数据清零 + 记录掩码位置（返回掩码 dict）。"""
    masks = {}
    for mod, pack in batch.items():
        if not isinstance(pack, dict) or "x" not in pack:
            continue
        B, W = pack["x"].shape[:2]
        m = torch.rand(B, W, device=pack["x"].device) < ratio
        m[:, 0] = False  # 至少保留首窗
        masks[mod] = m
        keep = (~m).view(B, W, 1, 1).float()
        pack["x"] = pack["x"] * keep
    return masks


class RoundRobinBatchSampler:
    """交替批次采样器：每个 batch 只含单一数据集，数据集按轮转次序切换。

    用于多数据集联合预训练时切断梯度互冲（负迁移）：异质数据集混在同一 batch
    里梯度方向互相拉扯；逐数据集成批 + 轮转，等效于多任务交替优化。
    配合 ConcatDataset 使用；过采样副本（同一 Dataset 多次 append）各自成为
    独立池，轮转时自然获得更多批次机会，等效过采样。
    """

    def __init__(self, cumulative_sizes, batch_size, seed=0):
        self.pools = []  # (start, end) 全局索引区间
        prev = 0
        for b in cumulative_sizes:
            self.pools.append((prev, int(b)))
            prev = int(b)
        self.bs = batch_size
        self.seed = seed
        self.n_batches = sum(max(1, -(-(e - s) // batch_size)) for s, e in self.pools)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        order = []  # 每池乱序后的全局索引队列
        for s, e in self.pools:
            idx = torch.randperm(e - s, generator=g) + s
            order.append(idx.tolist())
        pos = [0] * len(order)
        remaining = self.n_batches
        while remaining:
            for pi, pool in enumerate(order):
                if pos[pi] >= len(pool):
                    continue
                batch = pool[pos[pi]:pos[pi] + self.bs]
                pos[pi] += self.bs
                yield batch
                remaining -= 1
                if not remaining:
                    return

    def __len__(self):
        return self.n_batches


def masked_stats_loss(model, decoder, batch, masks):
    """B：重建被掩窗的 32 点波形（随样本变化的目标，常数编码器无法作弊）。"""
    loss, n = 0.0, 0
    for mod, m in masks.items():
        pack = batch[mod]
        tok = model.encode_modality(mod, pack["x"], pack["cmask"], pack.get("wmask"), pack.get("dataset"))[0]
        if model.geometry and "wmask" in pack:
            m = m & pack["wmask"]
        pred = decoder(tok)                                   # (B, W, 32)
        loss = loss + pred.sum() * 0.0  # differentiable zero for no masked windows
        if m.any():
            loss = loss + F.l1_loss(pred[m], pack["wave"][m])
            n += 1
    return loss / max(n, 1)


def loo_cl_loss(zs, owners, tau, n_batch):
    """A：留一法对比对齐（向量化实现，无逐对 GPU 同步）。

    任一单模态 z_m(i) <-> 该样本其余模态融合表征（均值）。
    先按样本聚合 F(i)=Σ_m z_m(i)、cnt(i)=模态数，则留一正样本
    p_m(i) = (F(i)-z_m(i)) / (cnt(i)-1)，全程张量运算。"""
    d = next(iter(zs.values())).shape[-1]
    dev = next(iter(zs.values())).device
    F_sum = torch.zeros(n_batch, d, device=dev)
    cnt = torch.zeros(n_batch, device=dev)
    for m, z in zs.items():
        F_sum.index_add_(0, owners[m], z)
        cnt.index_add_(0, owners[m], torch.ones(z.shape[0], device=dev))
    anchors, positives = [], []
    for m, z in zs.items():
        o = owners[m]
        valid = cnt[o] > 1
        if valid.any():
            a = z[valid]
            p = (F_sum[o[valid]] - a) / (cnt[o[valid]] - 1).clamp(min=1).unsqueeze(-1)
            anchors.append(a)
            positives.append(p)
    if sum(len(a) for a in anchors) < 2:
        return None
    a = F.normalize(torch.cat(anchors), dim=-1)
    p = F.normalize(torch.cat(positives), dim=-1)
    logits = a @ p.t() / tau
    labels = torch.arange(len(a), device=a.device)
    return F.cross_entropy(logits, labels)


def xpred_loss(model, zs, owners, predictors, n_batch):
    """C：可见模态融合表征 -> 目标模态窗级波形均值（向量化实现）。"""
    d = next(iter(zs.values())).shape[-1]
    dev = next(iter(zs.values())).device
    F_sum = torch.zeros(n_batch, d, device=dev)
    cnt = torch.zeros(n_batch, device=dev)
    for m, z in zs.items():
        F_sum.index_add_(0, owners[m], z)
        cnt.index_add_(0, owners[m], torch.ones(z.shape[0], device=dev))
    loss, n = 0.0, 0
    for tgt_mod, z in zs.items():
        o = owners[tgt_mod]
        valid = cnt[o] > 1
        if not valid.any():
            continue
        fused = (F_sum[o[valid]] - z[valid]) / (cnt[o[valid]] - 1).clamp(min=1).unsqueeze(-1)
        pred = predictors[tgt_mod](fused)
        tgt = model._stat_cache[tgt_mod][valid].mean(1)      # 目标模态平均波形
        loss = loss + F.l1_loss(pred, tgt)
        n += 1
    return loss / max(n, 1) if n else None


def supcon_loss(zs, owners, y, proj, n_batch, tau=0.1):
    """动作级监督对比（SupCon，batch 内）：同 y 的段互为正例，异 y 为负例。

    roundrobin 交替采样下每个 batch 恰好是单一数据集，标签空间一致，
    可直接比较 y；无标签数据集（grabmyo 等 y 全同）自动返回 None 跳过。
    作用机制：SSL 的掩码重建只约束窗级统计量，不约束"同类动作表征接近"——
    SupCon 补上这一环，针对动作识别类任务（pte/ultra/phytmo 等）。
    投影头 proj 仅预训练使用，不进入下游评估。"""
    if n_batch < 2 or y.unique().numel() < 2:
        return None
    d = next(iter(zs.values())).shape[-1]
    dev = next(iter(zs.values())).device
    F_sum = torch.zeros(n_batch, d, device=dev)
    cnt = torch.zeros(n_batch, device=dev)
    for m, z in zs.items():
        F_sum.index_add_(0, owners[m], z)
        cnt.index_add_(0, owners[m], torch.ones(z.shape[0], device=dev))
    zf = F_sum / cnt.clamp(min=1).unsqueeze(-1)          # (B, d) 样本级融合表征
    h = F.normalize(proj(zf).float(), dim=-1)            # (B, 128)，fp32 防 bf16 下 -inf 异常
    # 反坍缩铰链（关键）：SupCon 在"全部表征重合"处切向梯度恒为零（径向分量被
    # normalize 投影消除），是稳定不动点 —— 实测 supcon 损失 50 个 epoch 恒定
    # 3.3309 即投影头坍缩所致。z 上的 variance_loss 管不到投影头内部，
    # 必须对 h 本身加批内逐维 std 铰链（单位球面 128 维，gamma 取 0.05）。
    l_var_h = F.relu(0.05 - h.std(dim=0)).mean()
    sim = h @ h.T / tau
    eye = torch.eye(n_batch, dtype=torch.bool, device=dev)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    if int(pos.sum()) < 2:                                # 单类别/无正例批次跳过
        return None
    logits = sim.masked_fill(eye, float("-inf"))
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(logp.masked_fill(~pos, 0.0).sum(1) / pos.sum(1).clamp(min=1))
    return per_anchor[pos.any(1)].mean() + l_var_h


def variance_loss(zs, gamma=1.0):
    """VICReg 式方差正则：表征坍缩的稳定器。

    经验：InfoNCE 在"所有表征重合"处净梯度近似为零（拉向正样本与推离负样本
    方向相同、相互抵消），常数输出是稳定吸引子 —— 真实 FoG 与合成数据上均
    观测到 z 两两余弦=1.0 的完全坍缩。对每个模态 z 的批内逐维标准差做铰链
    约束（std >= gamma），直接打破该不动点。"""
    loss, n = 0.0, 0
    for m, z in zs.items():
        if z.shape[0] < 2:
            continue
        std = z.std(dim=0)
        loss = loss + F.relu(gamma - std).mean()
        n += 1
    return loss / max(n, 1) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--batch-size", type=int, default=HP["batch_size"])
    ap.add_argument("--lr", type=float, default=HP["lr"])
    ap.add_argument("--out", default="ckpt/pretrain.pt")
    ap.add_argument("--segments-per-subject", type=int, default=4)
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "gaitpdb", "gaitndd", "fog", "kneepad",
                             "inhouse", "multi", "mixed", "fullmix", "allmix"])
    ap.add_argument("--data-root", default=None, help="真实数据集目录（默认 ~/rehabfm/data/<名>）")
    ap.add_argument("--sampler", default="shuffle", choices=["shuffle", "roundrobin"],
                    help="roundrobin=交替批次（每 batch 单数据集轮转），抗多任务负迁移")
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 个 epoch 存一次中间 ckpt（评估过拟合轨迹），0=只存最终")
    args = ap.parse_args()

    torch.manual_seed(int(os.environ.get("REHABFM_SEED", "0")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "synthetic":
        ds = SyntheticRehabDataset(segments_per_subject=args.segments_per_subject)
        pred_mods = sorted({m for mods, _ in ds.DATASETS for m in mods})
    elif args.dataset == "fog":
        from .adapters_fog import FoGDataset
        ds = FoGDataset(os.path.expanduser(args.data_root or "~/rehabfm/data/09_FoG-Multimodal"))
        pred_mods = sorted(ds[0]["tokens"].keys())
    elif args.dataset == "kneepad":
        from .adapters_kneepad import KneEPADDataset
        ds = KneEPADDataset(os.path.expanduser(
            args.data_root or "~/rehabfm/data/03_KneE-PAD/extracted/dataset"))
        pred_mods = sorted(ds[0]["tokens"].keys())
    elif args.dataset == "inhouse":
        # 院内数据仅 19 例训练例进预训练；10 例留出集任何训练环节不得接触
        from .adapters_inhouse import InHouseRehabDataset
        ds = InHouseRehabDataset(os.path.expanduser(
            args.data_root or "~/rehabfm/data/90_InHouse/extracted"), split="train")
        pred_mods = sorted(ds[0]["tokens"].keys())
    elif args.dataset == "mixed":
        # 开源 + 院内 19 例混合预训练（用户方案）：院内过采样 REHABFM_INHOUSE_REPEAT 倍，
        # 防止 19 例在混合语料中被淹没（等权拼接下占比 <1% 等于没混）
        from torch.utils.data import ConcatDataset
        from .adapters_fog import FoGDataset
        from .adapters_gait import PhysioNetGaitDataset
        from .adapters_inhouse import InHouseRehabDataset
        base = os.path.expanduser(args.data_root or "~/rehabfm/data")
        rep = int(os.environ.get("REHABFM_INHOUSE_REPEAT", "6"))
        inhouse = InHouseRehabDataset(os.path.join(base, "90_InHouse/extracted"),
                                      split="train")
        parts = [
            PhysioNetGaitDataset(os.path.join(base, "12_GaitPDB"), dataset_idx=0),
            FoGDataset(os.path.join(base, "09_FoG-Multimodal")),          # idx 1
        ] + [inhouse] * rep                                               # idx 9
        ds = ConcatDataset(parts)
        pred_mods = sorted({m for p in {id(x): x for x in parts}.values()
                            for m in p[0]["tokens"]})
        print(f"[pretrain] mixed: open={len(parts[0]) + len(parts[1])} segs + "
              f"inhouse {len(inhouse)} segs x{rep} = {len(ds)}")
    elif args.dataset == "fullmix":
        # 全量混合预训练：全部科研可用外部数据 + 院内 19 例（过采样 REHABFM_INHOUSE_REPEAT 倍）
        # 外部：GaitPDB(步态EEG) + FoG(步态多模态) + U-Limb/MHH(上肢EMG) + GAITEX(IMU) + PHYTMO(IMU)
        # grabmyo/MultimodalGait 下载完成后可再并入
        from torch.utils.data import ConcatDataset
        from .adapters_fog import FoGDataset
        from .adapters_gait import PhysioNetGaitDataset
        from .adapters_inhouse import InHouseRehabDataset
        from .adapters_ulimb import ULimbMHHDataset
        from .adapters_gaitex import GaitexDataset
        from .adapters_phytmo import PhytmoDataset
        from .adapters_pte import PteUciDataset
        import glob
        base = os.path.expanduser(args.data_root or "~/rehabfm/data")
        rep = int(os.environ.get("REHABFM_INHOUSE_REPEAT", "2"))
        inhouse = InHouseRehabDataset(os.path.join(base, "90_InHouse/extracted"),
                                      split="train")
        # REHABFM_EXCLUDE 可排除指定数据源（逗号分隔：pte/grabmyo/multigait）
        exclude = set(os.environ.get("REHABFM_EXCLUDE", "").split(",")) - {""}
        opens = [
            PhysioNetGaitDataset(os.path.join(base, "12_GaitPDB"), dataset_idx=0),
            FoGDataset(os.path.join(base, "09_FoG-Multimodal")),            # idx 1
            ULimbMHHDataset(os.path.join(base, "02_U-Limb/extracted/MHH/Data/EMG")),  # idx 13
            GaitexDataset(os.path.join(base, "26_GAITEX/extracted/data")),  # idx 14
            PhytmoDataset(os.path.join(base, "24_PHYTMO/PHYTMO_extracted/inertial")),  # idx 10
        ]
        if "pte" not in exclude:
            opens.append(PteUciDataset(os.path.join(base, "25_PTE-UCI/extracted")))  # idx 15
        # 可选数据源：下载/缓存就绪才并入（目录存在且非空即尝试）
        grabmyo_dir = os.path.join(base, "29_GRABMyo")
        if "grabmyo" not in exclude and \
                sum(1 for _ in glob.glob(os.path.join(grabmyo_dir, "**", "*.hea"),
                                         recursive=True)) > 1000:
            from .adapters_grabmyo import GrabMyoDataset
            opens.append(GrabMyoDataset(grabmyo_dir))                       # idx 11
        else:
            print("[pretrain] fullmix: grabmyo 未就绪或被排除，跳过")
        mg_dir = os.path.join(base, "32_MultimodalGait/Dataset")
        if "multigait" not in exclude and \
                os.path.isdir(os.path.join(mg_dir, "EMG_Data")) and \
                len(os.listdir(os.path.join(mg_dir, "EMG_Data"))) > 100:
            from .adapters_multigait import MultimodalGaitDataset
            opens.append(MultimodalGaitDataset(mg_dir))                     # idx 12
        else:
            print("[pretrain] fullmix: MultimodalGait 未就绪或被排除，跳过")
        # 数据集批次权重（REHABFM_W="idx:倍数,..."）：轮转采样下每个副本独立成池，
        # 倍数>1 等效提高该数据集的批次频率——用于防止小体量关键任务数据被大语料稀释
        w = {}
        for kv in os.environ.get("REHABFM_W", "").split(","):
            if ":" in kv:
                k, v = kv.split(":")
                w[int(k)] = int(v)
        parts = []
        for p in opens:
            parts += [p] * w.get(p.dataset_idx, 1)
        parts += [inhouse] * rep                                          # idx 9
        ds = ConcatDataset(parts)
        pred_mods = sorted({m for p in {id(x): x for x in parts}.values()
                            for m in p[0]["tokens"]})
        print(f"[pretrain] fullmix: open={sum(len(p) for p in opens)} segs + "
              f"inhouse {len(inhouse)} segs x{rep} = {len(ds)}")
    elif args.dataset == "allmix":
        # 全量混合预训练：全部科研可用且适配器就绪的外部数据集（22 套）+ 院内 19 例
        # 每套通过各自 *_STRIDE / *_MAXREC / *_MAXSESS 环境变量限幅（目标 300-500 段/套），
        # REHABFM_W="idx:倍数,..." 批次加权沿用 v5 方案，院内 REHABFM_INHOUSE_REPEAT 倍过采样
        from torch.utils.data import ConcatDataset
        from .adapters_fog import FoGDataset
        from .adapters_gait import PhysioNetGaitDataset
        from .adapters_inhouse import InHouseRehabDataset
        from .adapters_ulimb import ULimbMHHDataset
        from .adapters_gaitex import GaitexDataset
        from .adapters_phytmo import PhytmoDataset
        from .adapters_pte import PteUciDataset
        from .adapters_grabmyo import GrabMyoDataset
        from .adapters_multigait import MultimodalGaitDataset
        from .adapters_gaitndd import GaitNDDDataset
        from .adapters_clingait import ClinGaitIMUDataset
        from .adapters_pdturning import PDTurningDataset
        from .adapters_gaitrec import GaitRecDataset
        from .adapters_moveport import MovePortDataset
        from .adapters_ulred import ULREDDataset
        from .adapters_hsstroke import HSStrokeDataset
        from .adapters_strokeeeg import StrokeEEGDataset
        from .adapters_llmi import LLMIStrokeDataset
        from .adapters_poststroke_ul import PostStrokeULDataset
        from .adapters_ultra import UltraMoCapDataset
        from .adapters_nordic import NordicWalkingDataset
        from .adapters_hipoa import HipOAGaitDataset
        from .adapters_mgposture import MultiGaitPostureDataset
        from .adapters_strokedepth import StrokeRehabIMUDataset
        base = os.path.expanduser(args.data_root or "~/rehabfm/data")
        rep = int(os.environ.get("REHABFM_INHOUSE_REPEAT", "2"))
        inhouse = InHouseRehabDataset(os.path.join(base, "90_InHouse/extracted"),
                                      split="train")
        opens = [
            PhysioNetGaitDataset(os.path.join(base, "12_GaitPDB"), dataset_idx=0),
            FoGDataset(os.path.join(base, "09_FoG-Multimodal")),            # 1
            ULimbMHHDataset(os.path.join(base, "02_U-Limb/extracted/MHH/Data/EMG")),  # 13
            GaitexDataset(os.path.join(base, "26_GAITEX/extracted/data")),  # 14
            PhytmoDataset(os.path.join(base, "24_PHYTMO/PHYTMO_extracted/inertial")),  # 10
            PteUciDataset(os.path.join(base, "25_PTE-UCI/extracted")),      # 15
            GrabMyoDataset(os.path.join(base, "29_GRABMyo")),               # 11
            MultimodalGaitDataset(os.path.join(base, "32_MultimodalGait/Dataset")),  # 12
            GaitNDDDataset(os.path.join(base, "13_GaitNDD")),               # 16
            GrabMyoDataset(os.path.join(base, "30_GRABMyoFlow/static_extension_WFDB"),
                           dataset_idx=17),
            ClinGaitIMUDataset(os.path.join(
                base, "04_ClinicalGait-IMU/extracted/dataset/dataset")),    # 18
            PDTurningDataset(os.path.join(base, "10_PD-Turning/extracted/IMU/IMU")),  # 19
            GaitRecDataset(os.path.join(base, "16_GaitRec")),               # 20
            MovePortDataset(os.path.join(base, "27_MovePort/extracted")),   # 21
            ULREDDataset(os.path.join(base, "23_UL-RED/extracted")),        # 22
            HSStrokeDataset(os.path.join(base, "20_HS-Stroke/extracted/sourcedata")),  # 23
            StrokeEEGDataset(os.path.join(base, "18_Stroke-EEG/extracted/edffile/edffile"),
                             participants=os.path.join(base, "18_Stroke-EEG/participants.tsv")),  # 24
            LLMIStrokeDataset(os.path.join(
                base, "19_LLMI-Stroke-EEG/extracted/sourcedata/sourcedata")),  # 25
            PostStrokeULDataset(os.path.join(base, "07_PostStroke-UL-KinEMG")),  # 26
            UltraMoCapDataset(os.path.join(base, "31_ULTRA-MoCap/extracted")),  # 27
            NordicWalkingDataset(os.path.join(base, "08_PD-NordicWalking/extracted")),  # 28
            HipOAGaitDataset(os.path.join(
                base, "17_HipOA-Gait/extracted/Patients C3D - Data/Patients")),  # 29
            MultiGaitPostureDataset(os.path.join(
                base, "28_MultiGaitPosture/extracted/processed_data")),     # 30
            StrokeRehabIMUDataset(os.path.join(
                base, "01_StrokeRehab-DepthIMU/files/imu_xlsx")),           # 31
        ]
        opens = [p for p in opens if len(p) > 0]   # 缓存未就绪的空集自动剔除
        # REHABFM_INCLUDE="0,1,13,..." 数据集精选白名单（按 dataset_idx），默认全选
        inc = os.environ.get("REHABFM_INCLUDE", "")
        if inc:
            keep = {int(x) for x in inc.split(",") if x.strip()}
            opens = [p for p in opens if p.dataset_idx in keep]
            print(f"[pretrain] allmix 精选: 保留 idx {sorted(keep)}")
        # 数据集批次权重（REHABFM_W="idx:倍数,..."）：轮转采样下每个副本独立成池
        w = {}
        for kv in os.environ.get("REHABFM_W", "").split(","):
            if ":" in kv:
                k, v = kv.split(":")
                w[int(k)] = int(v)
        parts = []
        for p in opens:
            parts += [p] * w.get(p.dataset_idx, 1)
        parts += [inhouse] * rep                                          # idx 9
        ds = ConcatDataset(parts)
        pred_mods = sorted({m for p in {id(x): x for x in parts}.values()
                            for m in p[0]["tokens"]})
        print(f"[pretrain] allmix: {len(opens)} 外部数据集 "
              f"open={sum(len(p) for p in opens)} segs + "
              f"inhouse {len(inhouse)} segs x{rep} = {len(ds)}")
        for p in opens:
            print(f"  idx={p.dataset_idx:>2d} {type(p).__name__:<24s} {len(p)} segs",
                  flush=True)
    elif args.dataset == "multi":
        # 多数据集联合预训练（对应设计文档 4.2 混合语料采样；当前为等权拼接）
        # 仅含科研可用数据集（rehabcorpus_registry_科研可用标注.xlsx）：
        # KneE-PAD(D003)/GSTRIDE(D014) 等许可未经核验，不得纳入训练
        from torch.utils.data import ConcatDataset
        from .adapters_fog import FoGDataset
        from .adapters_gait import PhysioNetGaitDataset
        base = os.path.expanduser(args.data_root or "~/rehabfm/data")
        parts = [
            PhysioNetGaitDataset(os.path.join(base, "12_GaitPDB"), dataset_idx=0),
            FoGDataset(os.path.join(base, "09_FoG-Multimodal")),          # idx 1
        ]
        ds = ConcatDataset(parts)
        pred_mods = sorted({m for p in parts for m in p[0]["tokens"]})
    else:
        from .adapters_gait import PhysioNetGaitDataset
        default_root = {"gaitpdb": "~/rehabfm/data/12_GaitPDB",
                        "gaitndd": "~/rehabfm/data/13_GaitNDD"}[args.dataset]
        ds = PhysioNetGaitDataset(os.path.expanduser(args.data_root or default_root),
                                  dataset_idx=0)
        pred_mods = sorted(ds[0]["tokens"].keys())
    if args.sampler == "roundrobin" and hasattr(ds, "cumulative_sizes"):
        sampler = RoundRobinBatchSampler(ds.cumulative_sizes, args.batch_size,
                                         seed=int(os.environ.get("REHABFM_SEED", "0")))
        dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
        print(f"[pretrain] sampler=roundrobin pools={len(sampler.pools)}")
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    model = RehabFM(n_datasets=int(os.environ.get('REHABFM_N_DATASETS', '16'))).to(device)
    decoder = MaskedStatsDecoder().to(device)
    d = MODEL_D
    predictors = nn.ModuleDict({m: nn.Linear(d, 32) for m in pred_mods}).to(device)
    # 动作级监督对比：REHABFM_SUPCON=λ（默认 0=关闭），投影头仅预训练使用
    supcon_lambda = float(os.environ.get("REHABFM_SUPCON", "0"))
    proj = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                         nn.Linear(128, 128)).to(device) if supcon_lambda > 0 else None
    print(f"[pretrain] device={device} supcon_lambda={supcon_lambda}")
    params = list(model.parameters()) + list(decoder.parameters()) + list(predictors.parameters())
    if proj is not None:
        params += list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=HP["weight_decay"],
                            betas=(0.9, 0.98))
    steps = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(HP["warmup_ratio"] * steps)))
        * 0.5 * (1 + math.cos(math.pi * s / max(1, steps))))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] backbone params: {n_params/1e6:.2f}M | segments: {len(ds)} | steps: {steps}")

    model.train()
    step = 0
    for ep in range(args.epochs):
        agg = dict(masked=0.0, loo_cl=0.0, xpred=0.0, var=0.0, supcon=0.0, total=0.0, nb=0, n_sc=0)
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else
                         {kk: vv.to(device) if torch.is_tensor(vv) else vv
                          for kk, vv in v.items()})
                     for k, v in batch.items()}
            model.train()
            masks = apply_window_mask(batch, HP["mask_window_ratio"])
            use_amp = device == "cuda" and os.environ.get("REHABFM_AMP", "1") == "1"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(batch, modality_drop_p=HP["modality_drop_p"],
                            channel_drop_p=HP["mask_channel_ratio"])
                model._stat_cache = {m: batch[m]["wave"] for m in out["z"]}  # (B, W, 32)
                n_batch = len(batch["y"])
                l_masked = masked_stats_loss(model, decoder, batch, masks)
                l_cl = loo_cl_loss(out["z"], out["owners"], HP["tau"], n_batch)
                l_xp = xpred_loss(model, out["z"], out["owners"], predictors, n_batch)
                l_var = variance_loss(out["z"])
                l_sc = supcon_loss(out["z"], out["owners"], batch["y"], proj,
                                   n_batch) if proj is not None else None
            loss = HP["loss_w"]["masked"] * l_masked
            if l_cl is not None:
                loss = loss + HP["loss_w"]["loo_cl"] * l_cl
            if l_xp is not None:
                loss = loss + HP["loss_w"]["xpred"] * l_xp
            if l_var is not None:
                loss = loss + HP["loss_w"]["var"] * l_var
            if l_sc is not None:
                loss = loss + supcon_lambda * l_sc
            opt.zero_grad()
            if not torch.isfinite(loss):
                # 数值防护：偶发异常批次（NaN/Inf）跳过，不污染权重
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(params, HP["grad_clip"])
            opt.step()
            sched.step()
            agg["masked"] += l_masked.item()
            agg["loo_cl"] += (l_cl.item() if l_cl is not None else 0.0)
            agg["xpred"] += (l_xp.item() if l_xp is not None else 0.0)
            agg["var"] += (l_var.item() if l_var is not None else 0.0)
            agg["supcon"] += (l_sc.item() if l_sc is not None else 0.0)
            agg["n_sc"] += (1 if l_sc is not None else 0)
            agg["total"] += loss.item()
            agg["nb"] += 1
            step += 1
        print(f"[pretrain] epoch {ep+1}/{args.epochs} "
              f"total={agg['total']/agg['nb']:.4f} masked={agg['masked']/agg['nb']:.4f} "
              f"loo_cl={agg['loo_cl']/agg['nb']:.4f} xpred={agg['xpred']/agg['nb']:.4f} "
              f"var={agg['var']/agg['nb']:.4f} supcon={agg['supcon']/max(agg['n_sc'],1):.4f}"
              f"(n={agg['n_sc']}/{agg['nb']})", flush=True)
        if args.save_every and (ep + 1) % args.save_every == 0:
            mid = args.out.replace(".pt", f"_e{ep+1}.pt")
            os.makedirs(os.path.dirname(mid) or ".", exist_ok=True)
            torch.save(dict(model=model.state_dict(), decoder=decoder.state_dict(),
                            predictors=predictors.state_dict()), mid)
            print(f"[pretrain] mid-save -> {mid}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(dict(model=model.state_dict(), decoder=decoder.state_dict(),
                    predictors=predictors.state_dict()), args.out)
    print(f"[pretrain] saved -> {args.out}")


if __name__ == "__main__":
    main()
