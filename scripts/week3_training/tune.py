r"""
tune.py - Week 3 tuning: run several model versions, compare them on VALIDATION only.

Run from C:\SL_PROJECT:
    python scripts\week3_training\tune.py                    # all experiments (~15-25 min)
    python scripts\week3_training\tune.py E1_regularise      # or only the ones named

Each experiment changes ONE idea compared with the baseline, so we can see what helps.
The test set is never used here; it is kept for the single final Q1/Q4 evaluation.

Outputs (runs\tuning\):
    results.csv         one row per experiment (appended each time you run)
    val_curves.png      validation top-1 per epoch for the experiments just run
    <experiment>\       model_best.pt, history.csv, norm_stats.npz, config.json
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR = "data/processed_v1"
SPLIT_CSV = "data/split_v1.csv"
OUT_DIR = os.path.join("runs", "tuning")
BATCH, LR, MAX_EPOCHS, PATIENCE, SEED = 32, 1e-3, 200, 30, 42

BASE = dict(hidden=128, layers=2, dropout=0.3, weight_decay=1e-4, pool="last", bidir=False,
            drop_pose_z=False, hand_relative=False)
EXPERIMENTS = {
    "E0_baseline": {},
    "E1_regularise": dict(hidden=64, dropout=0.5, weight_decay=1e-3),
    "E2_hand_shape": dict(hand_relative=True, drop_pose_z=True),
    "E3_whole_sequence": dict(pool="mean", bidir=True),
    "E4_combined": dict(hidden=64, dropout=0.5, weight_decay=1e-3, hand_relative=True,
                        drop_pose_z=True, pool="mean", bidir=True),
}

N_POSE = 23  # pose points kept by preprocess.py (POSE_KEEP = 0..22)


# ---------------- feature options (applied to the saved processed data) ----------------
def transform(X, drop_pose_z, hand_relative):
    """X: (clips, frames, 197) from preprocess.py -> new feature array."""
    N, T, _ = X.shape
    pose = X[..., :N_POSE * 3]
    hands = [X[..., 69:132], X[..., 132:195]]
    flags = X[..., 195:197]
    parts = []
    if drop_pose_z:   # pose depth is a rough estimate (values up to -8 shoulder widths)
        pose = pose.reshape(N, T, N_POSE, 3)[..., :2].reshape(N, T, N_POSE * 2)
    parts.append(pose)
    for k, hand in enumerate(hands):
        if not hand_relative:
            parts.append(hand)
            continue
        h = hand.reshape(N, T, 21, 3)
        wrist = h[..., 0, :]                                        # where the hand is
        size = np.linalg.norm(h[..., 9, :2] - h[..., 0, :2], axis=-1)  # wrist -> middle knuckle
        size = np.where(size > 1e-6, size, 1.0)[..., None, None]
        shape = (h[..., 1:, :] - h[..., :1, :]) / size              # handshape, size-free
        missing = flags[..., k] == 0
        shape[missing] = 0
        parts += [wrist, shape.reshape(N, T, 60)]
    parts.append(flags)
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ---------------- model ----------------
class SignLSTM(nn.Module):
    def __init__(self, n_feat, n_cls, hidden, layers, dropout, bidir, pool):
        super().__init__()
        self.pool = pool
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=bidir)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if bidir else 1), n_cls)

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = out.mean(dim=1) if self.pool == "mean" else out[:, -1]
        return self.head(self.drop(feat))


def topk_correct(logits, y, k):
    k = min(k, logits.shape[1])
    return (logits.topk(k, dim=1).indices == y[:, None]).any(dim=1).sum().item()


def run_epoch(model, X, y, loss_fn, opt=None):
    training = opt is not None
    model.train(training)
    idx = torch.randperm(len(X)) if training else torch.arange(len(X))
    tot = c1 = c5 = 0
    with torch.set_grad_enabled(training):
        for i in range(0, len(X), BATCH):
            b = idx[i:i + BATCH]
            logits = model(X[b])
            loss = loss_fn(logits, y[b])
            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * len(b)
            c1 += topk_correct(logits, y[b], 1)
            c5 += topk_correct(logits, y[b], 5)
    return tot / len(X), c1 / len(X), c5 / len(X)


def time_inference(model, frames, n_feat, repeats=200):
    model.eval()
    x = torch.randn(1, frames, n_feat)
    with torch.no_grad():
        for _ in range(20):
            model(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
    return (time.perf_counter() - t0) / repeats * 1000


# ---------------- one experiment ----------------
def run_experiment(name, cfg, X_all, y_all, which, n_cls):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Xf = transform(X_all, cfg["drop_pose_z"], cfg["hand_relative"])
    tr, va = (which == "train").values, (which == "val").values
    Xtr, Xva = Xf[tr], Xf[va]
    mu, sd = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1)) + 1e-6
    mu[-2:], sd[-2:] = 0.0, 1.0                     # keep the hand-present flags as 0/1
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Xtr, Xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    ytr, yva = torch.tensor(y_all[tr]), torch.tensor(y_all[va])

    frames, n_feat = Xtr.shape[1], Xtr.shape[2]
    model = SignLSTM(n_feat, n_cls, cfg["hidden"], cfg["layers"], cfg["dropout"],
                     cfg["bidir"], cfg["pool"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "norm_stats.npz"), mean=mu, std=sd)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "BATCH": BATCH, "LR": LR, "SEED": SEED, "features": int(n_feat)}, f, indent=2)

    print(f"\n----- {name}: {n_feat} features, {n_params:,} parameters -----")
    hist, best, best_ep, wait, t0 = [], -1.0, 0, 0, time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        tl, t1, _ = run_epoch(model, Xtr, ytr, loss_fn, opt)
        vl, v1, v5 = run_epoch(model, Xva, yva, loss_fn)
        hist.append(dict(epoch=ep, train_loss=tl, train_top1=t1, val_loss=vl, val_top1=v1, val_top5=v5))
        if v1 > best:
            best, best_ep, wait = v1, ep, 0
            torch.save(model.state_dict(), os.path.join(out, "model_best.pt"))
        else:
            wait += 1
        if ep % 10 == 0:
            print(f"  epoch {ep:3d} | train top1 {t1:.1%} | val top1 {v1:.1%} top5 {v5:.1%}")
        if wait >= PATIENCE:
            break
    h = pd.DataFrame(hist)
    h.to_csv(os.path.join(out, "history.csv"), index=False)
    row = h[h["epoch"] == best_ep].iloc[0]
    model.load_state_dict(torch.load(os.path.join(out, "model_best.pt")))
    ms = time_inference(model, frames, n_feat)
    result = dict(experiment=name, val_top1=round(row["val_top1"] * 100, 1),
                  val_top5=round(row["val_top5"] * 100, 1),
                  train_top1=round(row["train_top1"] * 100, 1),
                  gap=round((row["train_top1"] - row["val_top1"]) * 100, 1),
                  best_epoch=best_ep, epochs=len(h), params=n_params, ms_per_sign=round(ms, 1),
                  seconds=round(time.time() - t0), **cfg)
    print(f"  best epoch {best_ep}: val top1 {result['val_top1']}%, top5 {result['val_top5']}%, "
          f"gap {result['gap']} points, {ms:.1f} ms per sign")
    return result, h


def main():
    names = sys.argv[1:] or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {unknown}. Choose from {list(EXPERIMENTS)}")

    X_all = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_all = np.load(os.path.join(DATA_DIR, "y.npy"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), dtype={"video_id": str})
    split = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    which = meta["video_id"].map(dict(zip(split["video_id"], split["split"])))
    n_cls = len(pd.read_csv(os.path.join(DATA_DIR, "labels.csv")))
    print(f"Train {(which == 'train').sum()} | validation {(which == 'val').sum()} | "
          f"test {(which == 'test').sum()} (held out, never used here)")

    results, curves = [], {}
    for name in names:
        res, h = run_experiment(name, {**BASE, **EXPERIMENTS[name]}, X_all, y_all, which, n_cls)
        results.append(res)
        curves[name] = h

    os.makedirs(OUT_DIR, exist_ok=True)
    table = pd.DataFrame(results)
    path = os.path.join(OUT_DIR, "results.csv")
    table.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, h in curves.items():
        ax.plot(h["epoch"], h["val_top1"] * 100, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation top-1 (%)")
    ax.set_title("Validation accuracy per experiment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "val_curves.png"), dpi=90)
    plt.close(fig)

    print("\n========== TUNING RESULTS (validation only) ==========")
    cols = ["experiment", "val_top1", "val_top5", "train_top1", "gap", "best_epoch", "params", "ms_per_sign"]
    print(table.sort_values("val_top1", ascending=False)[cols].to_string(index=False))
    print("\nNote: 1 validation clip = 0.6 points, so differences under ~3 points may be noise.")
    print(f"Saved {path} and {os.path.join(OUT_DIR, 'val_curves.png')}")


if __name__ == "__main__":
    main()