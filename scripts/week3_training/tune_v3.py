r"""
tune_v3.py - Week 3 tuning, ROUND 3: confirm E6 (one LSTM layer) across seeds.

Round 1 (tune.py, runs\tuning\): E4 combined was best (42.6% validation top-1, seed 42).
Round 2 (tune_v2.py, runs\tuning_v2\): the same E4 setup scored 42.6 / 51.7 / 47.2% with
seeds 42 / 7 / 123, so chance moves results by about 9 points. E6 (E4 with one layer)
scored 50.0% at seed 42 with half the parameters. Round 3 runs E6 with seeds 7 and 123,
so E4 and E6 can be compared on their AVERAGE over the same three seeds.

Run from C:\SL_PROJECT (~20-25 min):
    python -u scripts\week3_training\tune_v3.py > report_outputs\B5_tuning_v3.txt

A progress bar per experiment shows in the terminal: [experiment k/n], epochs done,
current and best validation top-1, and "stop_in" = epochs left before early stopping.

Outputs (runs\tuning_v3\):
    results.csv, val_curves.png, <experiment>\ (model_best.pt, history.csv, norm_stats.npz, config.json)
The test set is never used here.
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
from tqdm import tqdm

DATA_DIR = "data/processed_v1"
SPLIT_CSV = "data/split_v1.csv"
OUT_DIR = os.path.join("runs", "tuning_v3")
EARLIER = [(1, os.path.join("runs", "tuning", "results.csv")),
           (2, os.path.join("runs", "tuning_v2", "results.csv"))]
BATCH, LR, MAX_EPOCHS, PATIENCE, SEED = 32, 1e-3, 200, 30, 42

BASE = dict(hidden=128, layers=2, dropout=0.3, weight_decay=1e-4, pool="last", bidir=False,
            drop_pose_z=False, hand_relative=False, seed=SEED)
E4 = dict(hidden=64, dropout=0.5, weight_decay=1e-3, hand_relative=True, drop_pose_z=True,
          pool="mean", bidir=True)
E6 = {**E4, "layers": 1}
EXPERIMENTS = {
    "E6_seed7": {**E6, "seed": 7},
    "E6_seed123": {**E6, "seed": 123},
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
def run_experiment(name, cfg, X_all, y_all, which, n_cls, label=""):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
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
        json.dump({**cfg, "BATCH": BATCH, "LR": LR, "features": int(n_feat)}, f, indent=2)

    print(f"\n----- {name}: {n_feat} features, {n_params:,} parameters -----")
    hist, best, best_ep, wait, t0 = [], -1.0, 0, 0, time.time()
    # progress bar (shown in the terminal even when results go to a file).
    # Training usually stops early, so the bar rarely reaches MAX_EPOCHS.
    bar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"{label}{name}", unit="epoch",
               ncols=100, leave=True)
    for ep in bar:
        tl, t1, _ = run_epoch(model, Xtr, ytr, loss_fn, opt)
        vl, v1, v5 = run_epoch(model, Xva, yva, loss_fn)
        hist.append(dict(epoch=ep, train_loss=tl, train_top1=t1, val_loss=vl, val_top1=v1, val_top5=v5))
        if v1 > best:
            best, best_ep, wait = v1, ep, 0
            torch.save(model.state_dict(), os.path.join(out, "model_best.pt"))
        else:
            wait += 1
        bar.set_postfix(val=f"{v1:.0%}", best=f"{best:.0%}", stop_in=PATIENCE - wait)
        if ep % 10 == 0:
            print(f"  epoch {ep:3d} | train top1 {t1:.1%} | val top1 {v1:.1%} top5 {v5:.1%}")
        if wait >= PATIENCE:
            break
    bar.close()
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
    for k, name in enumerate(names, 1):
        res, h = run_experiment(name, {**BASE, **EXPERIMENTS[name]}, X_all, y_all, which, n_cls,
                                label=f"[{k}/{len(names)}] ")
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
    ax.set_title("Validation accuracy per experiment (round 3)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "val_curves.png"), dpi=90)
    plt.close(fig)

    cols = ["round", "experiment", "seed", "val_top1", "val_top5", "train_top1", "gap", "params"]
    table["round"] = 3
    print("\n========== ROUND 3 RESULTS (validation only) ==========")
    print(table.sort_values("val_top1", ascending=False)[cols[1:]].to_string(index=False))

    frames = [pd.read_csv(p).assign(round=r) for r, p in EARLIER if os.path.exists(p)]
    allr = pd.concat(frames + [pd.read_csv(path).assign(round=3)], ignore_index=True)
    if "seed" not in allr:
        allr["seed"] = SEED
    allr["seed"] = allr["seed"].fillna(SEED).astype(int)   # round 1 used seed 42
    print("\n========== BEFORE AND AFTER: ROUNDS 1-3 ==========")
    print(allr[cols].to_string(index=False))

    # same setup across seeds: E4_combined + E4_seed*, E6_one_layer + E6_seed*
    setup = {"E4_combined": "E4", "E4_seed7": "E4", "E4_seed123": "E4",
             "E6_one_layer": "E6", "E6_seed7": "E6", "E6_seed123": "E6"}
    allr["setup"] = allr["experiment"].map(setup)
    g = allr.dropna(subset=["setup"]).groupby("setup")
    summary = pd.DataFrame({
        "runs": g.size(),
        "seeds": g["seed"].apply(lambda s: ", ".join(str(v) for v in sorted(s))),
        "val_top1_mean": g["val_top1"].mean().round(1),
        "val_top1_min": g["val_top1"].min(),
        "val_top1_max": g["val_top1"].max(),
        "val_top5_mean": g["val_top5"].mean().round(1),
        "params": g["params"].first(),
    })
    print("\n========== SAME SETUP ACROSS SEEDS (the fair comparison) ==========")
    print(summary.to_string())
    print("\nNote: 1 validation clip = 0.6 points, so differences under ~3 points may be noise.")
    print(f"Saved {path} and {os.path.join(OUT_DIR, 'val_curves.png')}")


if __name__ == "__main__":
    main()