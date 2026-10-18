r"""
evaluate_test_v2.py - the ONE-TIME test of the three v2 models (w100, matched, full).

The models were chosen and frozen on validation only (tune_v5.py). Nothing is trained or
changed here, and nothing about these three models will be changed because of these results.
The script REFUSES to run a second time (if runs\final_test_v2\ exists), like evaluate_test.py.

Test set = 105 WLASL clips (data\split_wlasl100.csv), the same for every model:
  all 100   all 105 test clips, 100 possible answers
  v1 clips  the 85 test clips of the 80 v1 words (exactly v1's test set), 100 possible answers
  like v1   the same 85 clips, only the 80 v1 words allowed as answers
            -> like-for-like with v1's one-time test: 49.4% top-1 / 76.5% top-5
  new 20    the 20 test clips of the 20 new words (1 per word - very rough)

Proof first: the v1 ensemble is run again on its own 85 test clips. It must give exactly
49.4% / 76.5% again (same models, same clips, same preprocessing). This re-run decides nothing;
it only proves the v2 test uses the identical clips and pipeline.

It also reports: single models, model size and speed (from training), and confidence
("not sure" threshold 0.3, as planned for the live system).

Uses tune_v4.val_probs() - the same prediction code as v1.

Run from C:\SL_PROJECT :
  python -u scripts\week3_training\evaluate_test_v2.py > report_outputs\V2_14_final_test_v2.txt

Outputs (runs\final_test_v2\):
  summary.csv                 one row per model (ensembles and singles)
  predictions_<set>.csv       ensemble prediction per test clip (word, guess, confidence, correct)
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_v4 import val_probs  # noqa: E402  (same prediction code as the v1 ensemble)

P2 = os.path.join("data", "processed_v2")
SPLIT = os.path.join("data", "split_wlasl100.csv")
V1_LABELS = os.path.join("data", "processed_v1", "labels.csv")
V1_RUNS = [os.path.join("runs", "tuning_v4", f"E7_seed{s}") for s in (42, 7, 123)]
V5 = os.path.join("runs", "tuning_v5")
SETS = ["w100", "matched", "full"]
SEEDS = (42, 7, 123)
OUT = os.path.join("runs", "final_test_v2")
NOT_SURE = 0.3


def topk(probs, y, allowed=None):
    p = probs.copy()
    if allowed is not None:
        block = np.ones(p.shape[1], bool)
        block[allowed] = False
        p[:, block] = -1
    t5 = np.argsort(-p, axis=1)[:, :5]
    return round(100 * (t5[:, 0] == y).mean(), 1), round(100 * (t5 == y[:, None]).any(1).mean(), 1)


def report(probs, y, v1_mask, new_mask, v1_allowed):
    a = topk(probs, y)
    b = topk(probs[v1_mask], y[v1_mask])
    c = topk(probs[v1_mask], y[v1_mask], v1_allowed)
    d = topk(probs[new_mask], y[new_mask])
    return {"all100_top1": a[0], "all100_top5": a[1], "v1clips_top1": b[0], "v1clips_top5": b[1],
            "likev1_top1": c[0], "likev1_top5": c[1], "new20_top1": d[0], "new20_top5": d[1]}


def confidence(probs, y):
    conf = probs.max(1)
    right = probs.argmax(1) == y
    answered = conf >= NOT_SURE
    return {"conf_right": round(float(conf[right].mean()), 2) if right.any() else np.nan,
            "conf_wrong": round(float(conf[~right].mean()), 2) if (~right).any() else np.nan,
            "answered_pct": round(100 * answered.mean(), 1),
            "correct_when_answered_pct": round(100 * right[answered].mean(), 1) if answered.any() else np.nan}


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} already exists: the one-time v2 test has been done. "
                 "It must not be run again (that would turn the test set into a tuning set).")
    for s in SETS:
        for seed in SEEDS:
            d = os.path.join(V5, f"{s}_seed{seed}")
            if not os.path.exists(os.path.join(d, "model_best.pt")):
                sys.exit(f"Missing model {d} - finish tune_v5.py first")

    X = np.load(os.path.join(P2, "X.npy"))
    y = np.load(os.path.join(P2, "y.npy"))
    meta = pd.read_csv(os.path.join(P2, "meta.csv"), dtype={"clip_id": str})
    labels = pd.read_csv(os.path.join(P2, "labels.csv"))
    split = pd.read_csv(SPLIT, dtype={"video_id": str})
    test_ids = set(split.loc[split["split"] == "test", "video_id"])
    is_wl = meta["source"].isin(["wlasl_v1", "wlasl_new20"])
    test = (is_wl & meta["clip_id"].isin(test_ids)).to_numpy()
    v1_lab = pd.read_csv(V1_LABELS)
    v1_words = set(v1_lab["word"])
    v1_allowed = labels.loc[labels["word"].isin(v1_words), "label"].to_numpy()
    y_te = y[test]
    words_te = meta.loc[test, "word"].to_numpy()
    v1_mask = np.isin(words_te, list(v1_words))
    new_mask = ~v1_mask
    print(f"Test clips: {test.sum()} ({v1_mask.sum()} of the 80 v1 words = v1's test set, "
          f"{new_mask.sum()} of the 20 new words)")
    if v1_mask.sum() != 85:
        sys.exit("Expected 85 v1 test clips - tell Claude before going on")

    # ---- proof: v1 ensemble on its own test clips ------------------------------------
    print("\nRe-running the v1 ensemble on its 85 test clips (proof of identical clips and pipeline)...")
    v1_test = test & meta["word"].isin(v1_words).to_numpy()
    p_v1 = np.mean([val_probs(d, X, v1_test, len(v1_lab)) for d in
                    tqdm(V1_RUNS, desc="v1 models", unit="model", file=sys.stderr)], axis=0)
    v1_label_of = dict(zip(v1_lab["word"], v1_lab["label"]))
    y_v1 = meta.loc[v1_test, "word"].map(v1_label_of).to_numpy()
    v1_t1, v1_t5 = topk(p_v1, y_v1)
    ok = (v1_t1, v1_t5) == (49.4, 76.5)
    print(f"v1 ensemble: {v1_t1}% / {v1_t5}%  (expected 49.4% / 76.5%) -> "
          f"{'IDENTICAL' if ok else 'DIFFERENT - stop and tell Claude'}")
    if not ok:
        sys.exit(1)

    # ---- v2 models -------------------------------------------------------------------------
    res = pd.read_csv(os.path.join(V5, "results.csv"))
    rows, ens_probs = [], {}
    jobs = [(s, seed) for s in SETS for seed in SEEDS]
    probs_of = {}
    for s, seed in tqdm(jobs, desc="v2 models on test", unit="model", file=sys.stderr):
        probs_of[(s, seed)] = val_probs(os.path.join(V5, f"{s}_seed{seed}"), X, test, len(labels))
    os.makedirs(OUT)          # from here on the test counts as used
    for s in SETS:
        r = res[res["set"] == s]
        for seed in SEEDS:
            p = probs_of[(s, seed)]
            rows.append({"model": f"{s}_seed{seed}", "set": s, "kind": "single",
                         "train_clips": int(r["train_clips"].iat[0]),
                         **report(p, y_te, v1_mask, new_mask, v1_allowed), **confidence(p, y_te),
                         "params": int(r["params"].iat[0]),
                         "ms_per_sign": float(r.loc[r["seed"] == seed, "ms_per_sign"].iat[0])})
        pe = np.mean([probs_of[(s, seed)] for seed in SEEDS], axis=0)
        ens_probs[s] = pe
        rows.append({"model": f"{s}_ensemble", "set": s, "kind": "ensemble",
                     "train_clips": int(r["train_clips"].iat[0]),
                     **report(pe, y_te, v1_mask, new_mask, v1_allowed), **confidence(pe, y_te),
                     "params": int(r["params"].iat[0]) * len(SEEDS),
                     "ms_per_sign": round(float(r["ms_per_sign"].sum()), 1)})
        pred = pe.argmax(1)
        pd.DataFrame({"uid": meta.loc[test, "uid"].to_numpy(), "word": words_te,
                      "v1_word": v1_mask, "guess": labels.set_index("label").loc[pred, "word"].to_numpy(),
                      "confidence": pe.max(1).round(3), "correct": pred == y_te}).to_csv(
            os.path.join(OUT, f"predictions_{s}.csv"), index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # ---- print -------------------------------------------------------------------------------
    pd.set_option("display.width", 220)
    ens = summ[summ.kind == "ensemble"]
    sing = summ[summ.kind == "single"]
    print("\n========== ONE-TIME TEST: 3-SEED ENSEMBLES ==========")
    print(ens[["model", "train_clips", "likev1_top1", "likev1_top5", "all100_top1", "all100_top5",
               "new20_top1", "new20_top5"]].to_string(index=False))
    print(f"\nv1 ensemble on the same 85 clips (like v1): {v1_t1}% top-1 / {v1_t5}% top-5")
    print("\n========== SINGLE MODELS ==========")
    print(sing[["model", "likev1_top1", "likev1_top5", "all100_top1", "all100_top5"]].to_string(index=False))
    g = sing.groupby("set")[["likev1_top1", "all100_top1"]].agg(["mean", "min", "max"]).round(1)
    print("\nSingle models per set (mean / min / max):")
    print(g.reindex(SETS).to_string())
    print("\n========== CONFIDENCE ('not sure' below 0.3) - ensembles, all 105 clips ==========")
    print(ens[["model", "conf_right", "conf_wrong", "answered_pct", "correct_when_answered_pct"]].to_string(index=False))
    print("\n========== SIZE AND SPEED ==========")
    print(ens[["model", "params", "ms_per_sign"]].to_string(index=False))
    print("\nNotes: 1 test clip = 1.2 points (85 v1 clips) / 1.0 point (105 clips) / 5 points (20 new-word clips).")
    print("Published WLASL100 baselines (official test set, not ours): I3D 65.89%, SPOTER 63.18% top-1.")
    print(f"Saved {OUT}\\summary.csv and predictions_<set>.csv. Do not run this script again.")


if __name__ == "__main__":
    main()