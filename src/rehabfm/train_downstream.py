# -*- coding: utf-8 -*-
"""阶段二：下游适配——冻结骨干 + 轻量头（对应设计文档第六章，表 5 / 表 6）。

演示三种策略在"患者 vs 对照"二分类上的差异：
  1. 线性探针（冻结 z）            —— 表 5 第 1 档
  2. LSTM 头（冻结 token 序列）    —— 表 5 第 2 档（SleepFM 式，约 0.1–1M 参数）
  3. 从零端到端基线（同架构不预训练）—— 表 8 必备基线，用于隔离预训练增益
切分严格按被试级（subject-level）。

用法：
  python -m rehabfm.train_downstream --ckpt ckpt/pretrain.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DOWNSTREAM as HP, MODEL
from .data import SyntheticRehabDataset, collate_fn
from .model import RehabFM


@torch.no_grad()
def extract_features(model, dl, device):
    """冻结骨干提特征：融合 z（各模态均值）+ 首模态 token 序列。
    若样本含 win_y（窗级标签，如 FoG），一并收集返回。"""
    model.eval()
    feats, toks, ys, subs, win_ys = [], [], [], [], []
    for batch in dl:
        batch = {k: (v.to(device) if torch.is_tensor(v) else
                     {kk: vv.to(device) if torch.is_tensor(vv) else vv
                      for kk, vv in v.items()})
                 for k, v in batch.items()}
        out = model(batch)
        B = len(batch["y"])
        z = torch.zeros(B, MODEL["d_model"], device=device)
        tok = torch.zeros(B, next(iter(out["tokens"].values())).shape[1],
                          MODEL["d_model"], device=device)
        for m in out["z"]:
            z[out["owners"][m]] += out["z"][m]
            tok[out["owners"][m]] += out["tokens"][m]
        n_mod = torch.zeros(B, device=device)
        for m in out["owners"]:
            n_mod[out["owners"][m]] += 1
        z = z / n_mod.clamp(min=1).unsqueeze(-1)
        tok = tok / n_mod.clamp(min=1).view(-1, 1, 1)
        feats.append(z.cpu())
        toks.append(tok.cpu())
        ys.append(batch["y"].cpu())
        subs.append(batch["subject"].cpu())
        if "win_y" in batch:
            win_ys.append(batch["win_y"].cpu())
    wy = torch.cat(win_ys) if win_ys else None
    return (torch.cat(feats), torch.cat(toks),
            torch.cat(ys), torch.cat(subs), wy)


class LSTMHead(nn.Module):
    """SleepFM 式下游头：两层双向 LSTM，hidden 64（表 6 ★）。"""

    def __init__(self, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(MODEL["d_model"], HP["lstm_hidden"], HP["lstm_layers"],
                            batch_first=True, bidirectional=True,
                            dropout=0.3 if HP["lstm_layers"] > 1 else 0.0)
        self.fc = nn.Linear(2 * HP["lstm_hidden"], n_classes)

    def forward(self, tok):
        h, _ = self.lstm(tok)
        return self.fc(h.mean(1))


def subject_split(subjects, ys, seed=0, train_ratio=0.8):
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects.numpy())
    rng.shuffle(uniq)
    ntr = int(len(uniq) * train_ratio)
    tr_sub, te_sub = set(uniq[:ntr].tolist()), set(uniq[ntr:].tolist())
    mask_tr = torch.tensor([s in tr_sub for s in subjects.tolist()])
    mask_te = torch.tensor([s in te_sub for s in subjects.tolist()])
    return mask_tr, mask_te


def fit_eval(name, make_head, Xtr, ytr, Xte, yte, epochs, lr, wd=0.0, sub_te=None):
    head = make_head()
    n_params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(Xtr), ytr)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(Xte).argmax(-1)
        acc = (pred == yte).float().mean().item()
    msg = f"  {name:<28s} params={n_params/1e3:7.1f}K  test_acc={acc:.3f}"
    if sub_te is not None:
        # 患者级多数投票：同一患者的所有段预测取多数类，再算患者级准确率
        subs = sub_te.tolist()
        by_sub = {}
        for s, p, t in zip(subs, pred.tolist(), yte.tolist()):
            by_sub.setdefault(s, [[], t])[0].append(p)
        correct = 0
        for s, (ps, t) in by_sub.items():
            vote = max(set(ps), key=ps.count)
            correct += int(vote == t)
        acc_subj = correct / max(1, len(by_sub))
        msg += f"  subject_vote_acc={acc_subj:.3f} ({correct}/{len(by_sub)})"
    print(msg)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/pretrain.pt")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "gaitpdb", "gaitndd", "fog", "kneepad", "inhouse"])
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    # 固定任务头训练随机性（初始化/dropout），保证评估可复现
    torch.manual_seed(int(os.environ.get("REHABFM_EVAL_SEED", "0")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "synthetic":
        ds = SyntheticRehabDataset(segments_per_subject=4, seed=1)
    elif args.dataset == "fog":
        from .adapters_fog import FoGDataset
        ds = FoGDataset(os.path.expanduser(args.data_root or "~/rehabfm/data/09_FoG-Multimodal"))
    elif args.dataset == "kneepad":
        from .adapters_kneepad import KneEPADDataset
        ds = KneEPADDataset(os.path.expanduser(
            args.data_root or "~/rehabfm/data/03_KneE-PAD/extracted/dataset"))
    elif args.dataset == "inhouse":
        ds = None  # 走固定 19/10 划分分支
    else:
        from .adapters_gait import PhysioNetGaitDataset
        default_root = {"gaitpdb": "~/rehabfm/data/12_GaitPDB",
                        "gaitndd": "~/rehabfm/data/13_GaitNDD"}[args.dataset]
        ds = PhysioNetGaitDataset(os.path.expanduser(args.data_root or default_root),
                                  dataset_idx=0)

    # ---- 预训练骨干（冻结） ----
    model = RehabFM(n_datasets=int(os.environ.get('REHABFM_N_DATASETS', '16'))).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    for p in model.parameters():
        p.requires_grad_(False)

    if args.dataset == "inhouse":
        # 用户指定划分：19 例训头，10 例留出只测不训（测试集划分.xlsx）
        from .adapters_inhouse import InHouseRehabDataset
        root = os.path.expanduser(args.data_root or "~/rehabfm/data/90_InHouse/extracted")
        dl_tr = DataLoader(InHouseRehabDataset(root, split="train"),
                           batch_size=64, shuffle=False, collate_fn=collate_fn)
        dl_te = DataLoader(InHouseRehabDataset(root, split="test"),
                           batch_size=64, shuffle=False, collate_fn=collate_fn)
        z_tr, tok_tr, y_tr, sub_tr, _ = extract_features(model, dl_tr, device)
        z_te, tok_te, y_te, sub_te, _ = extract_features(model, dl_te, device)
        z = torch.cat([z_tr, z_te]); tok = torch.cat([tok_tr, tok_te])
        y = torch.cat([y_tr, y_te]); sub = torch.cat([sub_tr, sub_te])
        mtr = torch.zeros(len(y), dtype=torch.bool); mtr[:len(y_tr)] = True
        mte = ~mtr
        loaders = (dl_tr, dl_te)
    else:
        dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        z, tok, y, sub, win_y = extract_features(model, dl, device)
        mtr, mte = subject_split(sub, y)
        loaders = (dl, None)

    def run_heads(y, tag):
        n_cls = int(y.max().item()) + 1
        print(f"[downstream:{tag}] subjects={len(torch.unique(sub))} "
              f"(train={mtr.sum().item()} segs / test={mte.sum().item()} segs) "
              f"n_cls={n_cls} | device={device}")
        print(f"  chance level ≈ {torch.bincount(y[mte]).max().item() / max(1, mte.sum().item()):.3f}")

        print("== 策略 1/2：冻结骨干 + 轻量头 ==")
        sub_te_arg = sub[mte] if args.dataset == "inhouse" else None
        fit_eval("linear probe (frozen z)",
                 lambda: nn.Linear(z.shape[1], n_cls),
                 z[mtr], y[mtr], z[mte], y[mte],
                 HP["probe_epochs"], HP["probe_lr"], HP["probe_weight_decay"],
                 sub_te=sub_te_arg)
        fit_eval("LSTM head (frozen tokens)",
                 lambda: LSTMHead(n_classes=n_cls),
                 tok[mtr], y[mtr], tok[mte], y[mte],
                 HP["lstm_epochs"], HP["lstm_lr"], sub_te=sub_te_arg)

        print("== 基线：同架构从零端到端（无预训练，表 8 必备基线）==")
        scratch = RehabFM(n_datasets=int(os.environ.get('REHABFM_N_DATASETS', '16'))).to(device)
        if loaders[1] is not None:
            zs_tr, toks_tr = extract_features(scratch, loaders[0], device)[:2]
            zs_te, toks_te = extract_features(scratch, loaders[1], device)[:2]
            zs, toks = torch.cat([zs_tr, zs_te]), torch.cat([toks_tr, toks_te])
        else:
            zs, toks = extract_features(scratch, loaders[0], device)[:2]  # 随机骨干特征（下界参考）
        fit_eval("linear probe (random z)", lambda: nn.Linear(zs.shape[1], n_cls),
                 zs[mtr], y[mtr], zs[mte], y[mte],
                 HP["probe_epochs"], HP["probe_lr"], HP["probe_weight_decay"])

    if args.dataset == "inhouse":
        # Brunnstrom 三种标签口径全部评估（特征只提一次，标签按口径重算）：
        #   stage6        标准六级分类（I-VI 期）
        #   merge12_34_56 对称合并三类（I+II / III+IV / V+VI）
        #   merge123_45_6 早期实验口径（I-III / IV-V / VI）
        from .adapters_inhouse import load_labels, brunnstrom_label
        labels_map = load_labels(os.path.join(os.path.dirname(root),
                                              "患者信息和康复标签.xlsx"))
        for scheme in ["stage6", "merge12_34_56", "merge123_45_6"]:
            y_s = torch.tensor([brunnstrom_label(labels_map[int(s) - 9000]["brunnstrom"],
                                                 scheme) for s in sub.tolist()])
            run_heads(y_s, scheme)
    else:
        run_heads(y, args.dataset)


if __name__ == "__main__":
    main()
