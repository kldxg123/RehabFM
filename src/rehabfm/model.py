# -*- coding: utf-8 -*-
"""模型：每模态输入茎 + 通道无关注意力池化 + 共享时序 Transformer。

对应设计文档第四章（图 2 / 表 3）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MODALITIES, SITE_LIST, MODEL, WINDOW_SEC


class ConvStem(nn.Module):
    """1D CNN 输入茎：单通道 5s 窗 -> d_model 向量。kernel 7/5/3/3, stride 2（表 3）。"""

    def __init__(self, d, native=False):
        super().__init__()
        self.native = native
        chs = [1, 32, 64, d, d]
        ks = [7, 5, 3, 3]
        strides = [4, 2, 2, 2]   # 首层 stride 4 降采样，控制激活显存
        blocks = []
        for i in range(4):
            blocks += [nn.Conv1d(chs[i], chs[i + 1], ks[i], stride=strides[i], padding=ks[i] // 2),
                       nn.GroupNorm(8, chs[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):            # x: (N, 1, Tw)
        h = self.net(x)              # (N, d, T')
        mean = h.mean(dim=-1)
        if self.native:
            patches = F.adaptive_avg_pool1d(h, 4).transpose(1, 2).flatten(1)
            return torch.cat([mean, patches], -1)
        return mean                 # 时间维全局池化 -> (N, d)


class SkeletonStem(nn.Module):
    """骨架/运动学输入茎：单关节坐标窗 -> d_model（MLP，正式版可换 GCN）。"""

    def __init__(self, d, in_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU())

    def forward(self, x):            # x: (N, 1, Tw)
        return self.net(x.squeeze(1).float())


class ChannelAgnosticPooling(nn.Module):
    """通道无关多头注意力池化（SleepFM 式）：变通道 -> 固定表示。

    对每窗，用可学习 query 对通道嵌入做 cross-attention，支持任意通道数/缺失通道。
    """

    def __init__(self, d, nhead, n_queries):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.out = nn.Linear(n_queries * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, ch_emb, cmask):
        # ch_emb: (B, W, C, d); cmask: (B, W, C) True=有效
        B, W, C, d = ch_emb.shape
        kv = ch_emb.reshape(B * W, C, d)
        q = self.queries.unsqueeze(0).expand(B * W, -1, -1)
        key_padding = ~cmask.reshape(B * W, C)
        # 防止全掩码行产生 NaN
        key_padding = key_padding & ~key_padding.all(dim=-1, keepdim=True)
        o, _ = self.attn(q, kv, kv, key_padding_mask=key_padding)   # (B*W, Q, d)
        return self.norm(self.out(o.reshape(B, W, -1)))             # (B, W, d)


class RehabFM(nn.Module):
    """RehabFM 骨干（Small 档）。

    forward(batch, modality_drop_p=0, channel_drop_p=0, train_mode=False) -> dict
      tokens:  {modality: (B, W, d)}       该模态的时序 token 序列
      z:       {modality: (B, d)}          模态级表征（注意力池化）
      owners:  {modality: LongTensor}      每个模态子批在原始 batch 中的样本下标
    """

    def __init__(self, n_datasets=8, geometry=False, native=False):
        super().__init__()
        self.geometry = geometry
        self.native = native
        if geometry:
            # Optional structural branch, seven scalars; old checkpoints are unchanged.
            self.channel_gain = nn.Parameter(torch.full((len(MODALITIES),), 0.1))
        d = MODEL["d_model"]
        self.stems = nn.ModuleDict({
            m: (ConvStem(d, native=native) if spec["stem"] == "conv"
                else SkeletonStem(d, int(WINDOW_SEC * spec["canonical_fs"])))
            for m, spec in MODALITIES.items()
        })
        if native:
            self.patch_proj = nn.Linear(4*d, d)
            self.spectral_proj = nn.Linear(16, d)
            self.native_gain = nn.Parameter(torch.full((2,), 0.1))
        self.chan_pool = ChannelAgnosticPooling(d, MODEL["nhead"], MODEL["pool_queries"])
        self.mod_emb = nn.Embedding(len(MODALITIES), MODEL["emb_cond"])
        self.site_emb = nn.Embedding(len(SITE_LIST), MODEL["emb_cond"])
        self.ds_emb = nn.Embedding(n_datasets, MODEL["emb_cond"])
        self.cond_proj = nn.Linear(3 * MODEL["emb_cond"], d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=MODEL["nhead"], dim_feedforward=MODEL["dim_ff"],
            dropout=MODEL["dropout"], batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, MODEL["num_layers"])
        self.final_norm = nn.LayerNorm(d)
        self.pos = nn.Parameter(torch.randn(1, 128, d) * 0.02)  # 简化正弦/RoPE：可学习位置
        self.z_norm = nn.LayerNorm(d)
        self.mod_index = {m: i for i, m in enumerate(MODALITIES)}
        self.site_index = {s: i for i, s in enumerate(SITE_LIST)}

    def _cond(self, mod, like, dataset_idx=None):
        B = like.shape[0]
        dev = like.device
        mi = self.mod_emb(torch.full((B,), self.mod_index[mod], device=dev))
        si = self.site_emb(torch.full((B,), self.site_index[MODALITIES[mod]["site"]], device=dev))
        ids = dataset_idx if self.geometry and dataset_idx is not None else like.new_zeros(B, dtype=torch.long)
        di = self.ds_emb(ids)
        policy = getattr(self, "unseen_conditioning", None)
        if self.geometry and policy is not None:
            known_ids = torch.tensor(policy["known_ids"], device=dev)
            known = torch.isin(ids,known_ids)
            if not known.all():
                rows = policy["modality_rows"].get(mod,policy["known_ids"])
                replacement = self.ds_emb.weight[rows].mean(0) if policy["mode"] == "mean" else torch.zeros_like(di[0])
                di = torch.where(known[:,None],di,replacement[None])
        return self.cond_proj(torch.cat([mi, si, di], dim=-1))  # (B, d)

    def encode_modality(self, mod, x, cmask, wmask=None, dataset_idx=None):
        """x: (B, W, C, Tw), cmask: (B, W, C) -> tokens (B, W, d), z (B, d)"""
        B, W, C, Tw = x.shape
        ch = x.reshape(B * W * C, 1, Tw)
        # 分块 + 梯度检查点：激活显存与 batch×窗×通道 解耦（A100 压测 OOM 的修复）
        outs = []
        for i in range(0, ch.shape[0], 16384):
            seg = ch[i:i + 16384]
            if self.training and seg.requires_grad:
                outs.append(torch.utils.checkpoint.checkpoint(
                    self.stems[mod], seg, use_reentrant=False))
            else:
                outs.append(self.stems[mod](seg))
        emb = torch.cat(outs, 0).reshape(B, W, C, -1)
        if self.native:
            d = MODEL["d_model"]
            if MODALITIES[mod]["stem"] == "conv":
                emb = emb[..., :d] + self.native_gain[0]*self.patch_proj(emb[..., d:])
            # Explicit relative band energy complements waveforms; it does not replace them.
            fs = MODALITIES[mod]["canonical_fs"]
            power = torch.fft.rfft(x.float(), dim=-1).abs().square()
            freq = torch.fft.rfftfreq(Tw, d=1.0/fs, device=x.device)
            edges = torch.tensor([.5,1,2,4,6,8,10,12,16,24,32,48,64,80,100], device=x.device)
            bins = torch.bucketize(freq, edges)
            band = power.new_zeros(B,W,C,16)
            band.scatter_add_(-1,bins.view(1,1,1,-1).expand_as(power),power)
            frac = band/power.sum(-1,keepdim=True).clamp_min(1e-8)
            spectral = torch.log1p(100*frac).to(emb.dtype)
            emb = emb + self.native_gain[1]*self.spectral_proj(spectral)
        if self.geometry:
            # Preserve stable CSV channel order through otherwise permutation-invariant pooling.
            # This is channel order, not an inferred anatomical coordinate.
            d = emb.shape[-1]
            positions = torch.arange(C, device=x.device, dtype=torch.float32)[:, None]
            frequencies = torch.exp(torch.arange(0, d, 2, device=x.device).float() * (-math.log(10000.0)/d))
            code = emb.new_zeros(C, d)
            code[:, 0::2] = torch.sin(positions*frequencies).to(emb.dtype)
            code[:, 1::2] = torch.cos(positions*frequencies).to(emb.dtype)
            emb = emb + self.channel_gain[self.mod_index[mod]] * code[None, None]
        tok = self.chan_pool(emb, cmask)                    # (B, W, d)
        # 条件（模态/部位/数据集）作为独立 [COND] token 前置，而非加在每个窗上 ——
        # 加性注入会让常量 cond 分量在窗均值池化后主导 z，导致跨样本表征趋同（坍缩）
        cond_tok = self._cond(mod, tok, dataset_idx).unsqueeze(1)        # (B, 1, d)
        seq = torch.cat([cond_tok, tok + self.pos[:, :W]], dim=1)
        if self.geometry and wmask is not None:
            padding = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), ~wmask], 1)
            enc = self.final_norm(self.encoder(seq, src_key_padding_mask=padding))
        else:
            enc = self.final_norm(self.encoder(seq))            # (B, W+1, d)
        tok = enc[:, 1:]
        if self.geometry and wmask is not None:
            tok = tok * wmask[:, :, None]
            z = self.z_norm(tok.sum(1)/wmask.sum(1).clamp_min(1)[:, None])
        else:
            z = self.z_norm(tok.mean(dim=1))                    # 信号窗均值池化 + LN
        return tok, z

    def forward(self, batch, modality_drop_p=0.0, channel_drop_p=0.0):
        tokens, zs, owners = {}, {}, {}
        for mod, pack in batch.items():
            if not isinstance(pack, dict) or "x" not in pack:
                continue
            x, cmask, idx = pack["x"], pack["cmask"], pack["index"]
            if self.training and channel_drop_p > 0:
                keep = torch.rand_like(cmask.float()) > channel_drop_p
                keep = keep | (~keep.any(-1, keepdim=True))  # 至少留 1 通道
                cmask = cmask & keep
            tok, z = self.encode_modality(mod, x, cmask, pack.get("wmask"), batch["dataset"][idx])
            tokens[mod], zs[mod], owners[mod] = tok, z, idx
        if self.training and modality_drop_p > 0 and len(zs) > 1:
            for mod in list(zs):
                if len(zs) > 1 and torch.rand(()) < modality_drop_p:
                    del tokens[mod], zs[mod], owners[mod]
        return dict(tokens=tokens, z=zs, owners=owners)


class MaskedStatsDecoder(nn.Module):
    """掩码建模的轻量解码器：token -> 该窗 32 点波形（通道均值，防坍缩目标）。"""

    def __init__(self):
        super().__init__()
        d = MODEL["d_model"]
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 32))

    def forward(self, tok):           # tok: (B, W, d) -> (B, W, 32)
        return self.net(tok)
