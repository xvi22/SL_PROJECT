r"""
tune_v4.py - Week 3 tuning, ROUND 4: combine models, and train differently. VALIDATION only.

Rounds so far (validation top-1, average over seeds 42 / 7 / 123):
  E4 (2 layers) 47.2%   |   E6 (1 layer, half the size) ~48.7%   -> E6 chosen as best setup.
Round 4 tries two more ideas before the one-time test:
  Part A - ENSEMBLES: average the guesses of models already trained (no new training):
           3 x E4, 3 x E6, and all 6 together.
  Part B - E7 = E6 + movement features (how far each point moved since the previous frame)
           + label smoothing 0.1 (stops the model being over-confident), on seeds 42 / 7 / 123,
           then the 3 x E7 ensemble.
The test set is never used here.

Run from C:\SL_PROJECT (~30-40 min, mostly Part B):
    python -u scripts\week3_training\tune_v4.py > report_outputs\B6_tuning_v4.txt

Outputs (runs\tuning_v4\): results.csv (E7 runs), ensembles.csv, E7_seed*\ folders.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_v3 import (BASE, E6, LR, MAX_EPOCHS, PATIENCE, SignLSTM,  # noqa: E402
                     run_epoch, time_inference, transform)

DATA_DIR = "data/processed_v1"
SPLIT_CSV = "data/split_v1.csv"
OUT_DIR = os.path.join("runs", "tuning_v4")
SEED = 42

E7 = {**E6, "velocity": True, "smooth": 0.1}
EXPERIMENTS = {f"E7_seed{s}": {**BASE, **E7, "seed": s} for s in (42, 7, 123)}

TRAINED = {  # models from earlier rounds, for the ensembles
    "E4": [os.path.join("runs", "tuning", "E4_combined"),
           os.path.join("runs", "tuning_v2", "E4_seed7"),
           os.path.join("runs", "tuning_v2", "E4_seed123")],
    "E6": [os.path.join("runs", "tuning_v2", "E6_one_layer"),
           os.path.join("runs", "tuning_v3", "E6_seed7"),
           os.path.join("runs", "tuning_v3", "E6_seed123")],
}
EARLIER = [(1, os.path.join("runs", "tuning", "results.csv")),
           (2, os.path.join("runs", "tuning_v2", "results.csv")),
           (3, os.path.join("runs", "tuning_v3", "results.csv"))]


# ---------------- features ----------------
def add_velocity(Xf, drop_pose_z):
    """Append frame-to-frame movement of every point (flags stay last).
    Hand movement is only counted when that hand is present in both frames."""
    pose_dim = 46 if drop_pose_z else 69
    base, flags = Xf[..., :-2], Xf[..., -2:]
    vel = np.zeros_like(base)
    vel[:, 1:] = base[:, 1:] - base[:, :-1]
    for k in range(2):
        a = pose_dim + 63 * k
        both = flags[:, 1:, k] * flags[:, :-1, k]
        vel[:, 1:, a:a + 63] *= both[..., None]
    return np.concatenate([base, vel, flags], axis=-1).astype(np.float32)


def features(X_all, cfg):
    Xf = transform(X_all, cfg["drop_pose_z"], cfg["hand_relative"])
    if cfg.get("velocity"):
        Xf = add_velocity(Xf, cfg["drop_pose_z"])
    return Xf


# ---------------- Part A: ensembles ----------------
def val_probs(run_dir, X_all, va, n_cls):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    st = np.load(os.path.join(run_dir, "norm_stats.npz"))
    model = SignLSTM(cfg["features"], n_cls, cfg["hidden"], cfg["layers"], cfg["dropout"],
                     cfg["bidir"], cfg["pool"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pt")))
    model.eval()
    Xv = (features(X_all, cfg)[va] - st["mean"]) / st["std"]
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(Xv, dtype=torch.float32)), dim=1).numpy()


def score(probs, y):
    top5 = np.argsort(-probs, axis=1)[:, :5]
    return 100 * (top5[:, 0] == y).mean(), 100 * (top5 == y[:, None]).any(axis=1).mean()


def ensembles(groups, X_all, va, y_va, n_cls):
    rows, cache = [], {}
    for name, dirs in groups.items():
        probs = []
        for d in dirs:
            if d not in cache:
                cache[d] = val_probs(d, X_all, va, n_cls)
            probs.append(cache[d])
        singles = [score(p, y_va)[0] for p in probs]
        t1, t5 = score(np.mean(probs, axis=0), y_va)
        rows.append(dict(ensemble=name, models=len(dirs),
                         single_top1_mean=round(float(np.mean(singles)), 1),
                         single_top1_best=round(float(np.max(singles)), 1),
                         ensemble_top1=round(t1, 1), ensemble_top5=round(t5, 1)))
    return pd.DataFrame(rows)


# ---------------- Part B: one training run ----------------
def run_experiment(name, cfg, X_all, y_all, tr, va, n_cls, label):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    Xf = features(X_all, cfg)
    Xtr, Xva = Xf[tr], Xf[va]
    mu, sd = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1)) + 1e-6
    mu[-2:], sd[-2:] = 0.0, 1.0
    Xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xva = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
    ytr, yva = torch.tensor(y_all[tr]), torch.tensor(y_all[va])
    frames, n_feat = Xtr.shape[1], Xtr.shape[2]

    model = SignLSTM(n_feat, n_cls, cfg["hidden"], cfg["layers"], cfg["dropout"],
                     cfg["bidir"], cfg["pool"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    train_loss = nn.CrossEntropyLoss(label_smoothing=cfg.get("smooth", 0.0))
    val_loss = nn.CrossEntropyLoss()

    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "norm_stats.npz"), mean=mu, std=sd)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "LR": LR, "features": int(n_feat)}, f, indent=2)

    print(f"\n----- {name}: {n_feat} features, {n_params:,} parameters -----")
    hist, best, best_ep, wait, t0 = [], -1.0, 0, 0, time.time()
    bar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"{label}{name}", unit="epoch", ncols=100)
    for ep in bar:
        tl, t1, _ = run_epoch(model, Xtr, ytr, train_loss, opt)
        vl, v1, v5 = run_epoch(model, Xva, yva, val_loss)
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
    print(f"  best epoch {best_ep}: val top1 {row['val_top1']:.1%}, top5 {row['val_top5']:.1%}")
    return dict(experiment=name, seed=cfg["seed"], val_top1=round(row["val_top1"] * 100, 1),
                val_top5=round(row["val_top5"] * 100, 1), train_top1=round(row["train_top1"] * 100, 1),
                gap=round((row["train_top1"] - row["val_top1"]) * 100, 1), best_epoch=best_ep,
                epochs=len(h), params=n_params, ms_per_sign=round(ms, 1),
                seconds=round(time.time() - t0))


def main():
    X_all = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_all = np.load(os.path.join(DATA_DIR, "y.npy"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), dtype={"video_id": str})
    split = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    which = meta["video_id"].map(dict(zip(split["video_id"], split["split"])))
    tr, va = (which == "train").values, (which == "val").values
    n_cls = len(pd.read_csv(os.path.join(DATA_DIR, "labels.csv")))
    y_va = y_all[va]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Train {tr.sum()} | validation {va.sum()} | test {(which == 'test').sum()} "
          "(held out, never used here)")

    # ----- Part A -----
    groups = {**TRAINED, "E4 + E6 (all 6)": TRAINED["E4"] + TRAINED["E6"]}
    ens_a = ensembles(groups, X_all, va, y_va, n_cls)
    print("\n========== PART A: ENSEMBLES OF EXISTING MODELS (validation) ==========")
    print(ens_a.to_string(index=False))

    # ----- Part B -----
    names = list(EXPERIMENTS)
    results = [run_experiment(n, EXPERIMENTS[n], X_all, y_all, tr, va, n_cls,
                              f"[{k}/{len(names)}] ") for k, n in enumerate(names, 1)]
    table = pd.DataFrame(results)
    path = os.path.join(OUT_DIR, "results.csv")
    table.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    e7_dirs = [os.path.join(OUT_DIR, n) for n in names]
    ens_b = ensembles({"E7": e7_dirs, "E6 + E7 (all 6)": TRAINED["E6"] + e7_dirs},
                      X_all, va, y_va, n_cls)
    ens_all = pd.concat([ens_a, ens_b], ignore_index=True)
    ens_all.to_csv(os.path.join(OUT_DIR, "ensembles.csv"), index=False)

    # ----- summaries -----
    frames = [pd.read_csv(p).assign(round=r) for r, p in EARLIER if os.path.exists(p)]
    allr = pd.concat(frames + [table.assign(round=4)], ignore_index=True)
    allr["seed"] = allr.get("seed", pd.Series(SEED, index=allr.index)).fillna(SEED).astype(int)
    setup = {"E4_combined": "E4", "E4_seed7": "E4", "E4_seed123": "E4",
             "E6_one_layer": "E6", "E6_seed7": "E6", "E6_seed123": "E6",
             "E7_seed42": "E7", "E7_seed7": "E7", "E7_seed123": "E7"}
    allr["setup"] = allr["experiment"].map(setup)
    g = allr.dropna(subset=["setup"]).groupby("setup")
    single = pd.DataFrame({"runs": g.size(),
                           "val_top1_mean": g["val_top1"].mean().round(1),
                           "val_top1_min": g["val_top1"].min(),
                           "val_top1_max": g["val_top1"].max(),
                           "val_top5_mean": g["val_top5"].mean().round(1),
                           "params": g["params"].first()})

    print("\n========== ROUND 4, PART B: E7 RUNS (validation) ==========")
    print(table[["experiment", "seed", "val_top1", "val_top5", "train_top1", "gap", "params"]]
          .to_string(index=False))
    print("\n========== SINGLE MODELS: SAME SETUP ACROSS SEEDS ==========")
    print(single.to_string())
    print("\n========== ALL ENSEMBLES (validation) ==========")
    print(ens_all.to_string(index=False))
    print("\nNote: 1 validation clip = 0.6 points; single runs moved ~9 points between seeds.")
    print(f"Saved {path} and {os.path.join(OUT_DIR, 'ensembles.csv')}")


if __name__ == "__main__":
    main()