r"""
tune_v8.py - side experiment: extra HANDSHAPE features.

Why: v1's mistakes were mostly signs made in the same place with similar movement that differ
in FINGER detail (white/like, bird/orange, colour words). E7 already gives the finger points
relative to the wrist; this adds numbers that describe the fingers more directly, per hand:
  - 15 finger bend angles (3 joints x 5 fingers; 0 = straight, 1 = folded back)
  - 7 fingertip distances (thumb tip to the 4 other tips, and neighbouring tips), in hand sizes
= 22 per hand, 44 extra features (346 -> 390). Missing hands -> zeros (as for all hand features).

ONE change compared with the base model (E7 by default, or a size from tune_v6 such as H128):
the 44 extra features. Training uses tune_v5.train_one() (not copied). VALIDATION ONLY.
Note: a model trained with these features needs features_v8() at prediction time too
(config.json records "handshape_extra": true).

Run from C:\SL_PROJECT :
  python -u scripts\week3_training\tune_v8.py quick
  python -u scripts\week3_training\tune_v8.py > report_outputs\V2_18_tuning_v8_matched.txt
  options: full | H128 | H128_L2 | H256

Outputs (runs\tuning_v8\): <set>_<base>_HAND_seed<s>\ folders, results.csv, ensembles_*.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_v5  # noqa: E402
from tune_v5 import CFG, MAX_EPOCHS, P2, SEEDS, V1_LABELS, features, load_set, train_one, val_report  # noqa: E402
from tune_v7 import SIZES, reference_runs  # noqa: E402

OUT_ROOT = os.path.join("runs", "tuning_v8")
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
TIPS = [4, 8, 12, 16, 20]
TIP_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12), (12, 16), (16, 20)]


def hand_extra(h, present):
    """h: (N, T, 21, 3) one hand in shoulder units; present: (N, T) 0/1 -> (N, T, 22)."""
    feats = []
    for chain in FINGERS:
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
            v1, v2 = h[..., a, :] - h[..., b, :], h[..., c, :] - h[..., b, :]
            cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
            feats.append(1 - np.arccos(np.clip(cos, -1, 1)) / np.pi)   # 0 straight ... 1 folded
    size = np.linalg.norm(h[..., 9, :2] - h[..., 0, :2], axis=-1)
    size = np.where(size > 1e-6, size, 1.0)
    for a, b in TIP_PAIRS:
        feats.append(np.linalg.norm(h[..., a, :] - h[..., b, :], axis=-1) / size)
    out = np.stack(feats, axis=-1).astype(np.float32)
    out[present == 0] = 0
    return out


def features_v8(X, cfg):
    """E7 features with the 44 handshape features inserted just before the 2 flags."""
    F = features(X, cfg)
    N, T, _ = X.shape
    left = X[..., 69:132].reshape(N, T, 21, 3)
    right = X[..., 132:195].reshape(N, T, 21, 3)
    extra = np.concatenate([hand_extra(left, X[..., 195]), hand_extra(right, X[..., 196])], axis=-1)
    return np.concatenate([F[..., :-2], extra, F[..., -2:]], axis=-1).astype(np.float32)


def main():
    args = sys.argv[1:]
    quick = "quick" in args
    train_set = "full" if "full" in args else "matched"
    base = next((a for a in args if a in SIZES), "E7")
    out_root = os.path.join("runs", "tuning_v8_quicktest") if quick else OUT_ROOT
    seeds = (42,) if quick else SEEDS
    max_epochs = 2 if quick else MAX_EPOCHS
    os.makedirs(out_root, exist_ok=True)
    cfg = {**CFG, **SIZES[base], "handshape_extra": True}

    X = np.load(os.path.join(P2, "X.npy"))
    y = np.load(os.path.join(P2, "y.npy"))
    meta = pd.read_csv(os.path.join(P2, "meta.csv"), dtype={"clip_id": str})
    labels = pd.read_csv(os.path.join(P2, "labels.csv"))
    v1_words = set(pd.read_csv(V1_LABELS)["word"])
    v1_allowed = labels.loc[labels["word"].isin(v1_words), "label"].to_numpy()
    tr, va = load_set(train_set, meta)
    y_va = y[va]
    v1_mask = meta["word"].iloc[va].isin(v1_words).to_numpy()
    print(f"{'QUICK TEST - results mean nothing' if quick else 'Handshape-feature experiment'} | set: {train_set} "
          f"({len(tr)} clips) | base: {base} | seeds {seeds}")
    print("Building E7 + handshape features...")
    F = features_v8(X, cfg)
    print(f"Features per frame: {F.shape[2]} (E7: 346)")

    res_path = os.path.join(out_root, "results.csv")
    run = f"{train_set}_{base}_HAND"
    for k, seed in enumerate(seeds, 1):
        done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["run", "seed"])
        if ((done["run"] == run) & (done["seed"] == seed)).any() and \
                os.path.exists(os.path.join(out_root, f"{run}_seed{seed}", "val_probs.npy")):
            print(f"\n[{k}/{len(seeds)}] {run}_seed{seed}: already finished, skipped")
            continue
        tune_v5.CFG = cfg                                  # model size + record handshape_extra
        res, probs = train_one(run, seed, F, y, tr, va, len(labels), out_root, max_epochs, f"[{k}/{len(seeds)}] ")
        res.update(val_report(probs, y_va, v1_mask, v1_allowed))
        res.update({"run": run, "set": train_set, "base": base, "experiment": "HAND"})
        print(f"  validation: all 100 {res['all100_top1']}% | like v1 {res['likev1_top1']}%")
        pd.DataFrame([res]).to_csv(res_path, mode="a", header=not os.path.exists(res_path), index=False)
    tune_v5.CFG = CFG

    results = pd.read_csv(res_path)
    results = results[results["run"] == run]
    ref, ref_dirs = (None, None) if quick else reference_runs(train_set, base, seeds)
    if ref is not None and len(ref):
        results = pd.concat([ref.assign(experiment=f"{base} (no extra features)"), results], ignore_index=True)
    g = results.groupby("experiment", sort=False)
    summ = pd.DataFrame({"runs": g.size(), "all100_mean": g["all100_top1"].mean().round(1),
                         "likev1_mean": g["likev1_top1"].mean().round(1), "likev1_min": g["likev1_top1"].min(),
                         "likev1_max": g["likev1_top1"].max(), "top5_mean": g["all100_top5"].mean().round(1),
                         "train_top1": g["train_top1"].mean().round(1), "params": g["params"].first()})
    ens = []
    groups = ([(f"{base} (no extra features)", ref_dirs)] if ref_dirs else []) + \
             [("HAND", [os.path.join(out_root, f"{run}_seed{s}") for s in seeds])]
    for name, dirs in groups:
        if all(os.path.exists(os.path.join(d, "val_probs.npy")) for d in dirs):
            p = np.mean([np.load(os.path.join(d, "val_probs.npy")) for d in dirs], axis=0)
            ens.append({"experiment": name, **val_report(p, y_va, v1_mask, v1_allowed)})
    ens = pd.DataFrame(ens)
    if len(ens):
        ens.to_csv(os.path.join(out_root, f"ensembles_{train_set}_{base}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n========== HANDSHAPE FEATURES ON {train_set.upper()} (base {base}): average of seeds ==========")
    print(summ.to_string())
    if len(ens):
        print("\n========== 3-SEED ENSEMBLES (validation) ==========")
        print(ens[["experiment", "all100_top1", "all100_top5", "likev1_top1", "likev1_top5"]].to_string(index=False))
    print("\nHow to read: keep it only if the AVERAGE beats the base by more than ~3 points.")
    print("The test set was not used. Saved " + res_path)


if __name__ == "__main__":
    main()