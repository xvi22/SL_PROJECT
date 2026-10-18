r"""
evaluate_test_best.py - the ONE-TIME test of the "best version" (H128 + handshape features).

Recipe decided on VALIDATION ONLY before this test: 128 units, 1 bidirectional LSTM layer,
E7 features + 44 handshape features (390), no augmentation, 3-seed ensemble. Two models:
  best_matched   runs\tuning_v8\matched_H128_HAND_seed{42,7,123}  (same data size as I3D / SPOTER)
  best_full      runs\tuning_v8\full_H128_HAND_seed{42,7,123}     (all kept ASL Citizen data)
Nothing is trained or changed here, and nothing will be changed because of these results.
The script REFUSES to run a second time (if runs\final_test_best\ exists).

Proof first (validation only): the models' predictions are recomputed for the 204 validation
clips and must equal the val_probs.npy saved during training. This proves the prediction code
(features_v8 + norm stats + model) is exactly the training one - before the test is touched.

Test set: the same 105 WLASL clips as the v2 test (data\split_wlasl100.csv):
  like v1 (85 v1 clips, only the 80 v1 words allowed), all 100 (105 clips), new 20 (20 clips).

Run from C:\SL_PROJECT :
  python -u scripts\week3_training\evaluate_test_best.py > report_outputs\V2_20_final_test_best.txt

Outputs (runs\final_test_best\): summary.csv, predictions_<model>.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_v3 import SignLSTM  # noqa: E402
from tune_v5 import P2, SEEDS, V1_LABELS, load_set  # noqa: E402
from tune_v8 import features_v8  # noqa: E402

SPLIT = os.path.join("data", "split_wlasl100.csv")
GROUPS = {"best_matched": ("matched", [os.path.join("runs", "tuning_v8", f"matched_H128_HAND_seed{s}") for s in SEEDS]),
          "best_full": ("full", [os.path.join("runs", "tuning_v8", f"full_H128_HAND_seed{s}") for s in SEEDS])}
OUT = os.path.join("runs", "final_test_best")
NOT_SURE = 0.3
EARLIER = {"v1 (E7, 80 words)": {"likev1_top1": 49.4, "likev1_top5": 76.5},
           "v2-matched (E7)": {"likev1_top1": 62.4, "likev1_top5": 89.4, "all100_top1": 62.9, "all100_top5": 89.5},
           "v2-full (E7)": {"likev1_top1": 69.4, "likev1_top5": 89.4, "all100_top1": 70.5, "all100_top5": 88.6}}


def predict(run_dir, X, idx, n_cls):
    """Probabilities for clips X[idx] (in the given order) - same steps as training."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    st = np.load(os.path.join(run_dir, "norm_stats.npz"))
    model = SignLSTM(cfg["features"], n_cls, cfg["hidden"], cfg["layers"], cfg["dropout"], cfg["bidir"], cfg["pool"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pt"), map_location="cpu"))
    model.eval()
    F = (features_v8(X[idx], cfg) - st["mean"]) / st["std"]
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(F, dtype=torch.float32)), dim=1).numpy()


def topk(probs, y, allowed=None):
    p = probs.copy()
    if allowed is not None:
        block = np.ones(p.shape[1], bool)
        block[allowed] = False
        p[:, block] = -1
    t5 = np.argsort(-p, axis=1)[:, :5]
    return round(100 * (t5[:, 0] == y).mean(), 1), round(100 * (t5 == y[:, None]).any(1).mean(), 1)


def report(p, y, v1m, newm, allowed):
    a, b, c = topk(p, y), topk(p[v1m], y[v1m], allowed), topk(p[newm], y[newm])
    return {"likev1_top1": b[0], "likev1_top5": b[1], "all100_top1": a[0], "all100_top5": a[1],
            "new20_top1": c[0], "new20_top5": c[1]}


def confidence(p, y):
    conf, right = p.max(1), p.argmax(1) == y
    ans = conf >= NOT_SURE
    return {"conf_right": round(float(conf[right].mean()), 2) if right.any() else np.nan,
            "conf_wrong": round(float(conf[~right].mean()), 2) if (~right).any() else np.nan,
            "answered_pct": round(100 * ans.mean(), 1),
            "correct_when_answered_pct": round(100 * right[ans].mean(), 1) if ans.any() else np.nan}


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} already exists: the one-time test of the best version has been done. Do not run it again.")
    for name, (_, dirs) in GROUPS.items():
        for d in dirs:
            if not os.path.exists(os.path.join(d, "model_best.pt")):
                sys.exit(f"Missing model {d} - train it first (tune_v8.py H128 / H128 full)")

    X = np.load(os.path.join(P2, "X.npy"))
    y = np.load(os.path.join(P2, "y.npy"))
    meta = pd.read_csv(os.path.join(P2, "meta.csv"), dtype={"clip_id": str})
    labels = pd.read_csv(os.path.join(P2, "labels.csv"))
    n_cls = len(labels)
    v1_words = set(pd.read_csv(V1_LABELS)["word"])
    allowed = labels.loc[labels["word"].isin(v1_words), "label"].to_numpy()

    # ---- proof on validation (test not touched yet) ------------------------------------
    print("Proof: recomputing validation predictions and comparing with those saved in training...")
    worst = 0.0
    for name, (train_set, dirs) in GROUPS.items():
        _, va = load_set(train_set, meta)                      # same order as in training
        for d in tqdm(dirs, desc=f"{name} (validation)", unit="model", file=sys.stderr):
            diff = float(np.abs(predict(d, X, va, n_cls) - np.load(os.path.join(d, "val_probs.npy"))).max())
            worst = max(worst, diff)
    print(f"Largest difference: {worst:.6f} -> {'IDENTICAL' if worst < 1e-4 else 'DIFFERENT - stop and tell Claude'}")
    if worst >= 1e-4:
        sys.exit(1)

    # ---- the one-time test --------------------------------------------------------------
    split = pd.read_csv(SPLIT, dtype={"video_id": str})
    test_ids = set(split.loc[split["split"] == "test", "video_id"])
    te = np.flatnonzero((meta["source"].isin(["wlasl_v1", "wlasl_new20"]) & meta["clip_id"].isin(test_ids)).to_numpy())
    y_te = y[te]
    words_te = meta["word"].to_numpy()[te]
    v1m = np.isin(words_te, list(v1_words))
    print(f"\nTest clips: {len(te)} ({int(v1m.sum())} v1 words, {int((~v1m).sum())} new words)")
    if v1m.sum() != 85:
        sys.exit("Expected 85 v1 test clips - tell Claude")
    probs = {}
    for name, (_, dirs) in GROUPS.items():
        for d in tqdm(dirs, desc=f"{name} (test)", unit="model", file=sys.stderr):
            probs[(name, d)] = predict(d, X, te, n_cls)
    os.makedirs(OUT)                                        # from here on the test counts as used

    res8 = pd.read_csv(os.path.join("runs", "tuning_v8", "results.csv"))
    rows = []
    for name, (train_set, dirs) in GROUPS.items():
        r = res8[res8["run"] == f"{train_set}_H128_HAND"]
        for s, d in zip(SEEDS, dirs):
            p = probs[(name, d)]
            rows.append({"model": f"{name}_seed{s}", "kind": "single", **report(p, y_te, v1m, ~v1m, allowed),
                         **confidence(p, y_te)})
        pe = np.mean([probs[(name, d)] for d in dirs], axis=0)
        rows.append({"model": f"{name}_ensemble", "kind": "ensemble", "train_clips": int(r["train_clips"].iat[0]),
                     **report(pe, y_te, v1m, ~v1m, allowed), **confidence(pe, y_te),
                     "params": int(r["params"].iat[0]) * len(dirs),
                     "ms_per_sign": round(float(r["ms_per_sign"].sum()), 1)})
        pred = pe.argmax(1)
        pd.DataFrame({"uid": meta["uid"].to_numpy()[te], "word": words_te, "v1_word": v1m,
                      "guess": labels.set_index("label").loc[pred, "word"].to_numpy(),
                      "confidence": pe.max(1).round(3), "correct": pred == y_te}).to_csv(
            os.path.join(OUT, f"predictions_{name}.csv"), index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    pd.set_option("display.width", 220)
    ens = summ[summ.kind == "ensemble"]
    print("\n========== ONE-TIME TEST: BEST VERSION (3-seed ensembles) ==========")
    print(ens[["model", "train_clips", "likev1_top1", "likev1_top5", "all100_top1", "all100_top5",
               "new20_top1"]].to_string(index=False))
    print("\nEarlier one-time tests on the same clips:")
    for k, v in EARLIER.items():
        print(f"  {k:20s} " + ", ".join(f"{a} {b}%" for a, b in v.items()))
    sing = summ[summ.kind == "single"]
    print("\n========== SINGLE MODELS ==========")
    print(sing[["model", "likev1_top1", "likev1_top5", "all100_top1", "all100_top5"]].to_string(index=False))
    print("\n========== CONFIDENCE ('not sure' below 0.3), ensembles, 105 clips ==========")
    print(ens[["model", "conf_right", "conf_wrong", "answered_pct", "correct_when_answered_pct"]].to_string(index=False))
    print("\n========== SIZE AND SPEED ==========")
    print(ens[["model", "params", "ms_per_sign"]].to_string(index=False))
    print("\nNotes: 1 test clip = 1.2 points (85) / 1.0 point (105). Published WLASL100 (official test set, not ours):")
    print("I3D 65.89%, SPOTER 63.18% top-1. Several choices were made on validation, so validation was optimistic.")
    print(f"Saved {OUT}. Do not run this script again.")


if __name__ == "__main__":
    main()