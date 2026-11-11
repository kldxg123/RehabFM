# -*- coding: utf-8 -*-
"""RehabFM 全局配置（Small 档，与设计文档表 3/表 4 对应）

可用环境变量覆盖关键规格（用于服务器 Base 档，不改代码）：
  REHABFM_D=256 REHABFM_NHEAD=8 REHABFM_LAYERS=6 REHABFM_CONTEXT=60
"""
import os

# ---------- 模态家族定义（对应设计文档 2.1 节 / 表 1） ----------
# 每个模态：标准采样率 canonical_fs、输入茎类型 stem、身体部位 site
MODALITIES = {
    "EEG":   {"canonical_fs": 256, "stem": "conv",     "site": "scalp"},
    "EMG":   {"canonical_fs": 256, "stem": "conv",     "site": "muscle"},
    "ECG":   {"canonical_fs": 128, "stem": "conv",     "site": "chest"},
    "EDA":   {"canonical_fs": 128, "stem": "conv",     "site": "skin"},
    "IMU":   {"canonical_fs": 100, "stem": "conv",     "site": "limb"},
    "FORCE": {"canonical_fs": 100, "stem": "conv",     "site": "foot"},
    "SKEL":  {"canonical_fs": 60,  "stem": "skeleton", "site": "whole_body"},
}
SITE_LIST = sorted({m["site"] for m in MODALITIES.values()})

# ---------- token 化（对应表 3/表 4，标注 ★ 为 SleepFM 设定） ----------
WINDOW_SEC = 5        # ★ 5 秒窗 = 1 个 token
CONTEXT_WINDOWS = int(os.environ.get("REHABFM_CONTEXT", 12))  # 正式: 60（5 min ★）

# ---------- 模型规格（Small 档；环境变量切 Base 档） ----------
MODEL = dict(
    d_model=int(os.environ.get("REHABFM_D", 64)),        # Base: 256（SleepFM 为 128）
    nhead=int(os.environ.get("REHABFM_NHEAD", 4)),       # Base: 8 ★
    num_layers=int(os.environ.get("REHABFM_LAYERS", 3)), # ★ Base: 6
    dim_ff=int(os.environ.get("REHABFM_D", 64)) * 4,     # FFN 4×
    dropout=0.3,       # ★
    pool_queries=8,    # ★ 通道无关池化的可学习 query 数
    emb_cond=16,       # 模态/部位/数据集嵌入维度
)

# ---------- 预训练超参（对应表 4，★ = SleepFM 设定） ----------
PRETRAIN = dict(
    lr=1e-3,           # ★
    weight_decay=0.05,
    batch_size=32,     # ★
    dropout=0.3,       # ★
    tau=0.07,          # InfoNCE 温度（CLIP 惯例）
    mask_window_ratio=0.3,   # 时间窗掩码比例（15–40% 区间）
    mask_channel_ratio=0.1,  # 通道丢弃
    modality_drop_p=0.25,    # 模态丢弃概率
    loss_w=dict(masked=1.0, loo_cl=1.0, xpred=0.5, var=1.0),  # A:B:C + 反坍缩方差正则
    epochs=3,                # ★ 全语料 1–3 epoch
    grad_clip=1.0,
    warmup_ratio=0.02,
)

# ---------- 下游微调超参（对应表 6） ----------
DOWNSTREAM = dict(
    probe_lr=1e-3, probe_weight_decay=1e-4, probe_epochs=50,
    lstm_hidden=64, lstm_layers=2, lstm_lr=1e-3, lstm_epochs=50,
    unfreeze_lr_backbone=1e-5, unfreeze_lr_head=1e-4,
)
