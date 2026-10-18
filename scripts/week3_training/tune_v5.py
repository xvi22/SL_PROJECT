r"""
tune_v5.py - v2 training: the SAME E7 setup as v1, trained on three training sets.

  w100      WLASL only, 100 words (703 clips)          -> effect of going from 80 to 100 words
  matched   WLASL + ASL Citizen, same size as the official WLASL100 training split
            that I3D / SPOTER used (1,426 vs 1,442)     -> fair data-size comparison (Q4)
  full      WLASL + all kept ASL Citizen clips (3,913) -> how far more data can take us

Everything else is exactly v1's E7 (imported from tune_v3 / tune_v4, not copied):
  hand-relative handshape, no pose depth, movement features, 1 bidirectional LSTM layer of 64,
  dropout 0.5, mean over frames, Adam lr 1e-3, weight decay 1e-3, batch 32, label smoothing 0.1,
  max 200 epochs, early stopping patience 30 on validation top-1, seeds 42 / 7 / 123,
  features standardised with the TRAINING set's mean / std, ensemble = average of the 3 seeds.
Validation = the same 204 WLASL clips for every set. The test set is NEVER used here.

Validation is reported three ways:
  all 100   all 204 validation clips, 100 possible answers
  v1 clips  the 176 validation clips of the 80 v1 words, 100 possible answers
  like v1   the same 176 clips, but only the 80 v1 words allowed as answers
            -> like-for-like with v1 (v1 E7 ensemble: 55.1% top-1 / 81.8% top-5)

Run from C:\SL_PROJECT :
  FIRST a quick test (2 epochs, 1 seed, about 1-2 minutes, separate folder):
    python -u scripts\week3_training\tune_v5.py quick
  then the real runs (all three sets; w100 + matched ~1 h, full several hours):
    python -u scripts\week3_training\tune_v5.py > report_outputs\V2_13_tuning_v5.txt
  or one set at a time (e.g. full overnight):
    python -u scripts\week3_training\tune_v5.py full > report_outputs\V2_13c_tuning_v5_full.txt
  A run that already has its finished results is skipped, so it is safe to restart.

Outputs (runs\tuning_v5\):
  <set>_seed<s>\  model_best.pt, norm_stats.npz, config.json, history.csv, val_probs.npy
  results.csv     one row per run
  ensembles.csv   one row per set (3-seed ensemble)
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
from tune_v3 import LR, MAX_EPOCHS, PATIENCE, SignLSTM, run_epoch, time_inference  # noqa: E402
from tune_v4 import BASE, E7, features  # noqa: E402

P2 = os.path.join("data", "processed_v2")
V1_LABELS = os.path.join("data", "processed_v1", "labels.csv")
SETS = {"w100": os.path.join("data", "split_v2_full.csv"),        # WLASL rows only (see below)
        "matched": os.path.join("data", "split_v2_matched.csv"),
        "full": os.path.join("data", "split_v2_full.csv")}
SEEDS = (42, 7, 123)
CFG = {**BASE, **E7}
V1_REF = "v1 E7 ensemble on the 176 v1 validation clips: 55.1% top-1 / 81.8% top-5 (80 words)"


def load_set(name, meta):
    s = pd.read_csv(SETS[name], dtype={"clip_id": str})
    if name == "w100":
        s = s[s["source"] != "asl_citizen"]
    pos = {u: i for i, u in enumerate(meta["uid"])}
    missing = [u for u in s["uid"] if u not in pos]
    if missing:
        sys.exit(f"{len(missing)} clips in {SETS[name]} are not in processed_v2 - tell Claude")
    tr = np.array([pos[u] for u in s.loc[s["split"] == "train", "uid"]])
    va = np.array([pos[u] for u in s.loc[s["split"] == "val", "uid"]])
    return tr, va


def scores(probs, y, allowed=None):
    """top-1 / top-5 in %. allowed = label numbers that may be answered (others blocked)."""
    p = probs.copy()
    if allowed is not None:
        block = np.ones(p.shape[1], bool)
        block[allowed] = False
        p[:, block] = -1
    top5 = np.argsort(-p, axis=1)[:, :5]
    return 100 * (top5[:, 0] == y).mean(), 100 * (top5 == y[:, None]).any(axis=1).mean()


def val_report(probs, y_va, v1_mask, v1_allowed):
    a1, a5 = scores(probs, y_va)
    b1, b5 = scores(probs[v1_mask], y_va[v1_mask])
    c1, c5 = scores(probs[v1_mask], y_va[v1_mask], v1_allowed)
    return {"all100_top1": round(a1, 1), "all100_top5": round(a5, 1),
            "v1clips_top1": round(b1, 1), "v1clips_top5": round(b5, 1),
            "likev1_top1": round(c1, 1), "likev1_top5": round(c5, 1)}


def train_one(name, seed, F, y, tr, va, n_cls, out_root, max_epochs, label):
    cfg = {**CFG, "seed": seed}
    out = os.path.join(out_root, f"{name}_seed{seed}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Xva = F[tr], F[va]
    mu, sd = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1)) + 1e-6
    mu[-2:], sd[-2:] = 0.0, 1.0                           # hand-present flags stay 0/1
    Xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xva = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
    ytr, yva = torch.tensor(y[tr]), torch.tensor(y[va])
    frames, n_feat = Xtr.shape[1], Xtr.shape[2]

    model = SignLSTM(n_feat, n_cls, cfg["hidden"], cfg["layers"], cfg["dropout"], cfg["bidir"], cfg["pool"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    train_loss = nn.CrossEntropyLoss(label_smoothing=cfg.get("smooth", 0.0))
    val_loss = nn.CrossEntropyLoss()
    np.savez(os.path.join(out, "norm_stats.npz"), mean=mu, std=sd)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "LR": LR, "features": int(n_feat), "n_classes": int(n_cls), "train_set": name,
                   "train_clips": int(len(tr)), "labels": os.path.join(P2, "labels.csv")}, f, indent=2)

    print(f"\n----- {name}_seed{seed}: {len(tr)} training clips, {n_feat} features, {n_params:,} parameters -----")
    hist, best, best_ep, wait, t0 = [], -1.0, 0, 0, time.time()
    bar = tqdm(range(1, max_epochs + 1), desc=f"{label}{name}_seed{seed}", unit="epoch", ncols=100,
               file=sys.stderr)
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
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xva), dim=1).numpy()
    np.save(os.path.join(out, "val_probs.npy"), probs)
    ms = time_inference(model, frames, n_feat)
    print(f"  best epoch {best_ep}: val top1 {row['val_top1']:.1%}, top5 {row['val_top5']:.1%} "
          f"| train top1 {row['train_top1']:.1%} | {(time.time() - t0) / 60:.0f} min")
    return {"set": name, "seed": seed, "train_clips": int(len(tr)), "best_epoch": int(best_ep),
            "epochs": int(len(h)), "train_top1": round(row["train_top1"] * 100, 1),
            "params": int(n_params), "ms_per_sign": round(ms, 1),
            "minutes": round((time.time() - t0) / 60, 1)}, probs


def main():
    args = sys.argv[1:]
    quick = "quick" in args
    names = [a for a in args if a in SETS] or list(SETS)
    bad = [a for a in args if a not in SETS and a != "quick"]
    if bad:
        sys.exit(f"Unknown option(s) {bad}. Use: quick, w100, matched, full")
    out_root = os.path.join("runs", "tuning_v5_quicktest" if quick else "tuning_v5")
    seeds = (42,) if quick else SEEDS
    max_epochs = 2 if quick else MAX_EPOCHS
    os.makedirs(out_root, exist_ok=True)

    X = np.load(os.path.join(P2, "X.npy"))
    y = np.load(os.path.join(P2, "y.npy"))
    meta = pd.read_csv(os.path.join(P2, "meta.csv"), dtype={"clip_id": str})
    labels = pd.read_csv(os.path.join(P2, "labels.csv"))
    n_cls = len(labels)
    v1_words = set(pd.read_csv(V1_LABELS)["word"])
    v1_allowed = labels.loc[labels["word"].isin(v1_words), "label"].to_numpy()
    print(f"{'QUICK TEST (2 epochs, 1 seed) - results mean nothing, only checks the script runs' if quick else 'v2 training'}")
    print(f"Clips in processed_v2: {len(X)} | words: {n_cls} | sets: {', '.join(names)} | seeds: {seeds}")
    print("Building E7 features (same as v1)...")
    F = features(X, CFG)

    res_path = os.path.join(out_root, "results.csv")
    done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["set", "seed"])
    jobs = [(n, s) for n in names for s in seeds]
    va_ref = None
    for k, (name, seed) in enumerate(jobs, 1):
        tr, va = load_set(name, meta)
        if va_ref is None:
            va_ref = va
        elif not np.array_equal(np.sort(va), np.sort(va_ref)):
            sys.exit("Validation clips differ between sets - tell Claude")
        y_va = y[va]
        v1_mask = meta["word"].iloc[va].isin(v1_words).to_numpy()
        run_dir = os.path.join(out_root, f"{name}_seed{seed}")
        if ((done["set"] == name) & (done["seed"] == seed)).any() and \
                os.path.exists(os.path.join(run_dir, "val_probs.npy")):
            print(f"\n[{k}/{len(jobs)}] {name}_seed{seed}: already finished, skipped")
            continue
        res, probs = train_one(name, seed, F, y, tr, va, n_cls, out_root, max_epochs, f"[{k}/{len(jobs)}] ")
        res.update(val_report(probs, y_va, v1_mask, v1_allowed))
        print(f"  validation: all 100 {res['all100_top1']}% | v1 clips {res['v1clips_top1']}% | "
              f"like v1 {res['likev1_top1']}%")
        row = pd.DataFrame([res])
        row.to_csv(res_path, mode="a", header=not os.path.exists(res_path), index=False)
        done = pd.read_csv(res_path)

    # ---- ensembles + summary -----------------------------------------------------------
    results = pd.read_csv(res_path)
    ens_rows = []
    for name in SETS:
        dirs = [os.path.join(out_root, f"{name}_seed{s}") for s in seeds]
        if not all(os.path.exists(os.path.join(d, "val_probs.npy")) for d in dirs):
            continue
        tr, va = load_set(name, meta)
        y_va = y[va]
        v1_mask = meta["word"].iloc[va].isin(v1_words).to_numpy()
        probs = np.mean([np.load(os.path.join(d, "val_probs.npy")) for d in dirs], axis=0)
        ens_rows.append({"set": name, "models": len(dirs), "train_clips": int(len(tr)),
                         **val_report(probs, y_va, v1_mask, v1_allowed)})
    ens = pd.DataFrame(ens_rows)
    if len(ens):
        ens.to_csv(os.path.join(out_root, "ensembles.csv"), index=False)

    pd.set_option("display.width", 200)
    cols = ["set", "seed", "train_clips", "all100_top1", "all100_top5", "v1clips_top1", "likev1_top1",
            "likev1_top5", "train_top1", "best_epoch", "minutes"]
    print("\n========== SINGLE MODELS (validation) ==========")
    print(results[cols].to_string(index=False))
    g = results.groupby("set")
    mean = g[["all100_top1", "all100_top5", "likev1_top1", "likev1_top5", "train_top1"]].mean().round(1)
    mean["likev1_min"], mean["likev1_max"] = g["likev1_top1"].min(), g["likev1_top1"].max()
    print("\n========== SAME SET ACROSS SEEDS (average) ==========")
    print(mean.reindex([n for n in SETS if n in mean.index]).to_string())
    if len(ens):
        print("\n========== 3-SEED ENSEMBLES (validation) ==========")
        print(ens.to_string(index=False))
    print(f"\nReference: {V1_REF}")
    print("Note: 1 validation clip = 0.5 points (204 clips) or 0.6 points (176 v1 clips).")
    print("The test set was not used. Saved " + res_path)


if __name__ == "__main__":
    main()