r"""
tune_v6.py - side experiment: is a BIGGER model better now that there is more data?

ONE change compared with E7 (tune_v5.py): the model size. Everything else is identical
(same features, dropout 0.5, weight decay 1e-3, label smoothing 0.1, Adam lr 1e-3, batch 32,
early stopping 30, seeds 42 / 7 / 123, same training sets, same 204 validation clips).
The training code itself is tune_v5.train_one() - not copied, so nothing else can differ.

  E7       64 units, 1 layer   (already trained in runs\tuning_v5 - used as the reference)
  H128    128 units, 1 layer
  H128_L2 128 units, 2 layers
  H256    256 units, 1 layer

VALIDATION ONLY. The test set is not used. The v2 models in runs\tuning_v5 are not touched.

Run from C:\SL_PROJECT :
  quick check first (2 epochs, 1 seed, ~1-2 min):
    python -u scripts\week3_training\tune_v6.py quick
  then the matched set (9 runs, roughly 1-2 hours):
    python -u scripts\week3_training\tune_v6.py > report_outputs\V2_15_tuning_v6_matched.txt
  optionally the full set as well (longer):
    python -u scripts\week3_training\tune_v6.py full > report_outputs\V2_15b_tuning_v6_full.txt
  Finished runs are skipped, so it is safe to restart.

Outputs (runs\tuning_v6\): <set>_<experiment>_seed<s>\ folders, results.csv, ensembles.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_v5  # noqa: E402
from tune_v5 import CFG, MAX_EPOCHS, P2, SEEDS, V1_LABELS, features, load_set, train_one, val_report  # noqa: E402

EXPERIMENTS = {"H128": dict(hidden=128, layers=1),
               "H128_L2": dict(hidden=128, layers=2),
               "H256": dict(hidden=256, layers=1)}
OUT_ROOT = os.path.join("runs", "tuning_v6")
REF_ROOT = os.path.join("runs", "tuning_v5")


def main():
    args = sys.argv[1:]
    quick = "quick" in args
    train_set = "full" if "full" in args else "matched"
    out_root = os.path.join("runs", "tuning_v6_quicktest") if quick else OUT_ROOT
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
    tr, va = load_set(train_set, meta)
    y_va = y[va]
    v1_mask = meta["word"].iloc[va].isin(v1_words).to_numpy()
    print(f"{'QUICK TEST - results mean nothing' if quick else 'Bigger-model experiment'} | "
          f"training set: {train_set} ({len(tr)} clips) | validation: {len(va)} clips | seeds {seeds}")
    print("Building E7 features (same as v1/v2)...")
    F = features(X, CFG)

    res_path = os.path.join(out_root, "results.csv")
    jobs = [(e, s) for e in EXPERIMENTS for s in seeds]
    for k, (exp, seed) in enumerate(jobs, 1):
        run = f"{train_set}_{exp}"
        run_dir = os.path.join(out_root, f"{run}_seed{seed}")
        done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["run", "seed"])
        if ((done["run"] == run) & (done["seed"] == seed)).any() and \
                os.path.exists(os.path.join(run_dir, "val_probs.npy")):
            print(f"\n[{k}/{len(jobs)}] {run}_seed{seed}: already finished, skipped")
            continue
        tune_v5.CFG = {**CFG, **EXPERIMENTS[exp]}       # the ONLY change: model size
        res, probs = train_one(run, seed, F, y, tr, va, n_cls, out_root, max_epochs, f"[{k}/{len(jobs)}] ")
        res.update(val_report(probs, y_va, v1_mask, v1_allowed))
        res.update({"run": run, "experiment": exp, "hidden": EXPERIMENTS[exp]["hidden"],
                    "layers": EXPERIMENTS[exp]["layers"]})
        print(f"  validation: all 100 {res['all100_top1']}% | like v1 {res['likev1_top1']}%")
        pd.DataFrame([res]).to_csv(res_path, mode="a", header=not os.path.exists(res_path), index=False)
    tune_v5.CFG = CFG

    # ---- summary: compare with E7 on the same training set ------------------------------
    results = pd.read_csv(res_path)
    results = results[results["set"].str.startswith(train_set)]
    rows = []
    ref = os.path.join(REF_ROOT, "results.csv")
    if os.path.exists(ref) and not quick:
        r = pd.read_csv(ref)
        r = r[r["set"] == train_set].assign(experiment="E7 (v2)", hidden=64, layers=1)
        results = pd.concat([r, results], ignore_index=True)
    g = results.groupby("experiment", sort=False)
    summ = pd.DataFrame({
        "units": g["hidden"].first(), "layers": g["layers"].first(), "params": g["params"].first(),
        "runs": g.size(),
        "all100_mean": g["all100_top1"].mean().round(1), "likev1_mean": g["likev1_top1"].mean().round(1),
        "likev1_min": g["likev1_top1"].min(), "likev1_max": g["likev1_top1"].max(),
        "top5_mean": g["all100_top5"].mean().round(1), "train_top1": g["train_top1"].mean().round(1),
        "ms_per_sign": g["ms_per_sign"].mean().round(1), "minutes": g["minutes"].mean().round(1)})

    ens = []
    for exp in ["E7 (v2)"] + list(EXPERIMENTS):
        if exp == "E7 (v2)":
            dirs = [os.path.join(REF_ROOT, f"{train_set}_seed{s}") for s in seeds]
        else:
            dirs = [os.path.join(out_root, f"{train_set}_{exp}_seed{s}") for s in seeds]
        if quick and exp == "E7 (v2)":
            continue
        if all(os.path.exists(os.path.join(d, "val_probs.npy")) for d in dirs):
            p = np.mean([np.load(os.path.join(d, "val_probs.npy")) for d in dirs], axis=0)
            ens.append({"experiment": exp, "models": len(dirs), **val_report(p, y_va, v1_mask, v1_allowed)})
    ens = pd.DataFrame(ens)
    if len(ens):
        ens.to_csv(os.path.join(out_root, f"ensembles_{train_set}.csv"), index=False)

    pd.set_option("display.width", 220)
    print(f"\n========== MODEL SIZE ON THE {train_set.upper()} SET: SINGLE MODELS (validation, average of seeds) ==========")
    print(summ.to_string())
    if len(ens):
        print("\n========== 3-SEED ENSEMBLES (validation) ==========")
        print(ens[["experiment", "all100_top1", "all100_top5", "likev1_top1", "likev1_top5"]].to_string(index=False))
    print("\nHow to read: a bigger model is only worth it if its AVERAGE is clearly higher than E7")
    print("(more than ~3 points, given seed luck) - otherwise keep the smaller, faster E7.")
    print("train_top1 much higher than validation = memorising (bigger models do this more).")
    print("The test set was not used. Saved " + res_path)


if __name__ == "__main__":
    main()