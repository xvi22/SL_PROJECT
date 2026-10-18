r"""
tune_v7.py - side experiment: DATA AUGMENTATION (small random changes to training clips).

Each epoch, every TRAINING clip is changed a little at random before the model sees it.
Validation clips are never changed. ONE change per experiment compared with the base model:

  AUG_GEO     geometry + speed: size x0.9-1.1, rotation +/-10 degrees, shift +/-0.1 shoulder
              widths, and a slight speed-up (up to 10% cut from the start and/or the end,
              stretched back to 64 frames)
  AUG_MIRROR  half of the clips mirrored (left <-> right, hands swapped), like a
              left-handed signer - the same idea as mirror_frame() from Week 2

Everything else is identical to the base model (features, settings, early stopping, seeds,
training sets, the same 204 validation clips). Standardising uses the UNCHANGED training clips
(the live system will use these stats). I3D and SPOTER were trained with augmentation
(crops / flips / rotations), so this also makes the comparison with them fairer.

Base model: E7 (default) or a bigger size from tune_v6 if that won, e.g. "H128".
VALIDATION ONLY. The test set is not used.

Run from C:\SL_PROJECT :
  quick check (2 epochs, 1 seed):
    python -u scripts\week3_training\tune_v7.py quick
  matched set, base E7 (6 runs, ~1-1.5 hours):
    python -u scripts\week3_training\tune_v7.py > report_outputs\V2_16_tuning_v7_matched.txt
  options (add at the end): full | H128 | H128_L2 | H256 | AUG_GEO | AUG_MIRROR | BOTH

Outputs (runs\tuning_v7\): <set>_<base>_<experiment>_seed<s>\ folders, results.csv, ensembles_*.csv
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
from tune_v5 import CFG, P2, SEEDS, V1_LABELS, features, load_set, val_report  # noqa: E402

SIZES = {"E7": dict(hidden=64, layers=1), "H128": dict(hidden=128, layers=1),
         "H128_L2": dict(hidden=128, layers=2), "H256": dict(hidden=256, layers=1)}
EXPERIMENTS = {"AUG_GEO": dict(geo=True, mirror=False),
               "AUG_MIRROR": dict(geo=False, mirror=True),
               "BOTH": dict(geo=True, mirror=True)}
DEFAULT_EXPS = ["AUG_GEO", "AUG_MIRROR"]
OUT_ROOT = os.path.join("runs", "tuning_v7")

# layout of the 197 preprocessed numbers: pose 23 x 3 | left hand 21 x 3 | right hand 21 x 3 | 2 flags
N_POSE = 23
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
              (19, 20), (21, 22)]                                 # left/right pose points (0..22)
POSE_PERM = np.arange(N_POSE)
for a, b in POSE_PAIRS:
    POSE_PERM[a], POSE_PERM[b] = b, a


def split_parts(X):
    N, T, _ = X.shape
    pose = X[..., :69].reshape(N, T, N_POSE, 3).copy()
    left = X[..., 69:132].reshape(N, T, 21, 3).copy()
    right = X[..., 132:195].reshape(N, T, 21, 3).copy()
    flags = X[..., 195:197].copy()
    return pose, left, right, flags


def join_parts(pose, left, right, flags):
    N, T = flags.shape[:2]
    return np.concatenate([pose.reshape(N, T, -1), left.reshape(N, T, -1), right.reshape(N, T, -1),
                           flags], axis=-1).astype(np.float32)


def mirror(X):
    """Mirror whole clips: x -> -x (x is centred on the shoulder midpoint), swap left/right
    pose points, swap the two hand slots and their flags. Same idea as mirror_frame()."""
    pose, left, right, flags = split_parts(X)
    pose = pose[:, :, POSE_PERM]
    pose[..., 0] *= -1
    left, right = right, left
    left[..., 0] *= -1
    right[..., 0] *= -1
    flags = flags[..., ::-1]
    return join_parts(pose, left, right, flags)


def geo_speed(X, rng):
    """Per clip: random size, rotation (x-y plane), shift, and slight speed-up (time crop)."""
    N, T, _ = X.shape
    pose, left, right, flags = split_parts(X)
    scale = rng.uniform(0.9, 1.1, N)
    ang = np.deg2rad(rng.uniform(-10, 10, N))
    shift = rng.uniform(-0.1, 0.1, (N, 2))
    c, s = np.cos(ang), np.sin(ang)
    for part, k in ((pose, None), (left, 0), (right, 1)):
        x, yv = part[..., 0].copy(), part[..., 1].copy()
        part[..., 0] = scale[:, None, None] * (c[:, None, None] * x - s[:, None, None] * yv) + shift[:, 0, None, None]
        part[..., 1] = scale[:, None, None] * (s[:, None, None] * x + c[:, None, None] * yv) + shift[:, 1, None, None]
        part[..., 2] *= scale[:, None, None]
        if k is not None:                       # missing hands stay exactly zero
            part[flags[..., k] == 0] = 0
    out = join_parts(pose, left, right, flags)
    # speed: keep a random window covering 90-100% of the clip, stretch back to T frames
    # (nearest frame, so the hand-present flags stay consistent)
    start = rng.uniform(0, 0.1, N) * (T - 1)
    end = (1 - rng.uniform(0, 0.1, N)) * (T - 1)
    idx = np.rint(start[:, None] + (end - start)[:, None] * np.linspace(0, 1, T)[None]).astype(int)
    return out[np.arange(N)[:, None], idx]


def augment(X, rng, geo, do_mirror):
    out = X
    if geo:
        out = geo_speed(out, rng)
    if do_mirror:
        flip = rng.random(len(out)) < 0.5
        if flip.any():
            out = out.copy()
            out[flip] = mirror(out[flip])
    return out


def train_aug(run, seed, Xraw, y, tr, va, n_cls, cfg, aug, out_root, max_epochs, label):
    """Same as tune_v5.train_one, except that the training clips are augmented every epoch."""
    out = os.path.join(out_root, f"{run}_seed{seed}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    Ftr0 = features(Xraw[tr], cfg)                      # UNCHANGED training clips -> norm stats
    mu, sd = Ftr0.mean(axis=(0, 1)), Ftr0.std(axis=(0, 1)) + 1e-6
    mu[-2:], sd[-2:] = 0.0, 1.0
    Xva = torch.tensor((features(Xraw[va], cfg) - mu) / sd, dtype=torch.float32)
    ytr, yva = torch.tensor(y[tr]), torch.tensor(y[va])
    frames, n_feat = Ftr0.shape[1], Ftr0.shape[2]

    model = SignLSTM(n_feat, n_cls, cfg["hidden"], cfg["layers"], cfg["dropout"], cfg["bidir"], cfg["pool"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])
    train_loss = nn.CrossEntropyLoss(label_smoothing=cfg.get("smooth", 0.0))
    val_loss = nn.CrossEntropyLoss()
    np.savez(os.path.join(out, "norm_stats.npz"), mean=mu, std=sd)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg, "LR": LR, "features": int(n_feat), "n_classes": int(n_cls), "train_set": run,
                   "train_clips": int(len(tr)), "augmentation": aug,
                   "labels": os.path.join(P2, "labels.csv")}, f, indent=2)

    print(f"\n----- {run}_seed{seed}: {len(tr)} training clips, augmentation {aug}, {n_params:,} parameters -----")
    hist, best, best_ep, wait, t0 = [], -1.0, 0, 0, time.time()
    bar = tqdm(range(1, max_epochs + 1), desc=f"{label}{run}_seed{seed}", unit="epoch", ncols=100, file=sys.stderr)
    for ep in bar:
        Xa = augment(Xraw[tr], rng, aug["geo"], aug["mirror"])
        Xtr = torch.tensor((features(Xa, cfg) - mu) / sd, dtype=torch.float32)
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
    print(f"  best epoch {best_ep}: val top1 {row['val_top1']:.1%} | train top1 (augmented) "
          f"{row['train_top1']:.1%} | {(time.time() - t0) / 60:.0f} min")
    return {"seed": seed, "train_clips": int(len(tr)), "best_epoch": int(best_ep), "epochs": int(len(h)),
            "train_top1": round(row["train_top1"] * 100, 1), "params": int(n_params),
            "ms_per_sign": round(ms, 1), "minutes": round((time.time() - t0) / 60, 1)}, probs


def reference_runs(train_set, base, seeds):
    """Results of the same base model WITHOUT augmentation (tune_v5 for E7, tune_v6 for bigger sizes)."""
    if base == "E7":
        path, dirs = os.path.join("runs", "tuning_v5", "results.csv"), \
            [os.path.join("runs", "tuning_v5", f"{train_set}_seed{s}") for s in seeds]
        key = ("set", train_set)
    else:
        path, dirs = os.path.join("runs", "tuning_v6", "results.csv"), \
            [os.path.join("runs", "tuning_v6", f"{train_set}_{base}_seed{s}") for s in seeds]
        key = ("run", f"{train_set}_{base}")
    if not os.path.exists(path):
        return None, None
    r = pd.read_csv(path)
    return r[r[key[0]] == key[1]], dirs


def main():
    args = sys.argv[1:]
    quick = "quick" in args
    train_set = "full" if "full" in args else "matched"
    base = next((a for a in args if a in SIZES), "E7")
    exps = [a for a in args if a in EXPERIMENTS] or DEFAULT_EXPS
    bad = [a for a in args if a not in SIZES and a not in EXPERIMENTS and a not in ("quick", "full")]
    if bad:
        sys.exit(f"Unknown option(s) {bad}")
    out_root = os.path.join("runs", "tuning_v7_quicktest") if quick else OUT_ROOT
    seeds = (42,) if quick else SEEDS
    max_epochs = 2 if quick else MAX_EPOCHS
    os.makedirs(out_root, exist_ok=True)
    cfg = {**CFG, **SIZES[base]}

    X = np.load(os.path.join(P2, "X.npy"))
    y = np.load(os.path.join(P2, "y.npy"))
    meta = pd.read_csv(os.path.join(P2, "meta.csv"), dtype={"clip_id": str})
    labels = pd.read_csv(os.path.join(P2, "labels.csv"))
    v1_words = set(pd.read_csv(V1_LABELS)["word"])
    v1_allowed = labels.loc[labels["word"].isin(v1_words), "label"].to_numpy()
    tr, va = load_set(train_set, meta)
    y_va = y[va]
    v1_mask = meta["word"].iloc[va].isin(v1_words).to_numpy()
    print(f"{'QUICK TEST - results mean nothing' if quick else 'Augmentation experiment'} | set: {train_set} "
          f"({len(tr)} clips) | base: {base} | experiments: {', '.join(exps)} | seeds {seeds}")

    res_path = os.path.join(out_root, "results.csv")
    jobs = [(e, s) for e in exps for s in seeds]
    for k, (exp, seed) in enumerate(jobs, 1):
        run = f"{train_set}_{base}_{exp}"
        done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["run", "seed"])
        if ((done["run"] == run) & (done["seed"] == seed)).any() and \
                os.path.exists(os.path.join(out_root, f"{run}_seed{seed}", "val_probs.npy")):
            print(f"\n[{k}/{len(jobs)}] {run}_seed{seed}: already finished, skipped")
            continue
        res, probs = train_aug(run, seed, X, y, tr, va, len(labels), cfg, EXPERIMENTS[exp], out_root,
                               max_epochs, f"[{k}/{len(jobs)}] ")
        res.update(val_report(probs, y_va, v1_mask, v1_allowed))
        res.update({"run": run, "set": train_set, "base": base, "experiment": exp})
        print(f"  validation: all 100 {res['all100_top1']}% | like v1 {res['likev1_top1']}%")
        pd.DataFrame([res]).to_csv(res_path, mode="a", header=not os.path.exists(res_path), index=False)

    # ---- summary vs the same base model without augmentation ------------------------------
    results = pd.read_csv(res_path)
    results = results[(results["set"] == train_set) & (results["base"] == base)]
    ref, ref_dirs = (None, None) if quick else reference_runs(train_set, base, seeds)
    if ref is not None and len(ref):
        results = pd.concat([ref.assign(experiment="no augmentation"), results], ignore_index=True)
    g = results.groupby("experiment", sort=False)
    summ = pd.DataFrame({"runs": g.size(), "all100_mean": g["all100_top1"].mean().round(1),
                         "likev1_mean": g["likev1_top1"].mean().round(1),
                         "likev1_min": g["likev1_top1"].min(), "likev1_max": g["likev1_top1"].max(),
                         "top5_mean": g["all100_top5"].mean().round(1),
                         "train_top1": g["train_top1"].mean().round(1),
                         "best_epoch": g["best_epoch"].mean().round(0), "minutes": g["minutes"].mean().round(1)})
    ens = []
    groups = ([("no augmentation", ref_dirs)] if ref_dirs else []) + \
             [(e, [os.path.join(out_root, f"{train_set}_{base}_{e}_seed{s}") for s in seeds]) for e in exps]
    for name, dirs in groups:
        if all(os.path.exists(os.path.join(d, "val_probs.npy")) for d in dirs):
            p = np.mean([np.load(os.path.join(d, "val_probs.npy")) for d in dirs], axis=0)
            ens.append({"experiment": name, "models": len(dirs), **val_report(p, y_va, v1_mask, v1_allowed)})
    ens = pd.DataFrame(ens)
    if len(ens):
        ens.to_csv(os.path.join(out_root, f"ensembles_{train_set}_{base}.csv"), index=False)

    pd.set_option("display.width", 220)
    print(f"\n========== AUGMENTATION ON {train_set.upper()} (base {base}): SINGLE MODELS, average of seeds ==========")
    print(summ.to_string())
    if len(ens):
        print("\n========== 3-SEED ENSEMBLES (validation) ==========")
        print(ens[["experiment", "all100_top1", "all100_top5", "likev1_top1", "likev1_top5"]].to_string(index=False))
    print("\nHow to read: keep an augmentation only if its AVERAGE beats 'no augmentation' by more than ~3 points.")
    print("train_top1 is measured on the CHANGED clips, so it is naturally a bit lower with augmentation.")
    print("The test set was not used. Saved " + res_path)


if __name__ == "__main__":
    main()