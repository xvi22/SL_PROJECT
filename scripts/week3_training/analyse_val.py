r"""
analyse_val.py - Week 3: WHY does the model get the score it gets? (VALIDATION only)

Uses the chosen final model (E7 ensemble = 3 x E7, seeds 42 / 7 / 123) on the validation
set and looks for the factors behind right and wrong answers:
  - which words fail and which pairs get confused (incl. the colour words)
  - training data per word (clips, signers)
  - data quality (hand detection, gaps filled, frame rate, video resolution)
  - kind of sign (one- vs two-handed, sign length)
  - new signers (signer never seen in training) - matters for the live tests (Q2/Q3)
  - confidence: how accurate the model is when it is sure (sets the live "not sure" threshold)
The test set is not used.

Run from C:\SL_PROJECT (about 1 minute):
    python scripts\week3_training\analyse_val.py > report_outputs\B7_analysis_val.txt

Outputs (runs\analysis_val\): per_clip.csv, factors.csv, factors.png, confidence.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data/processed_v1"
SPLIT_CSV = "data/split_v1.csv"
INDEX_V3 = "data/v3/landmarks_index_v3.csv"
OUT_DIR = os.path.join("runs", "analysis_val")
E7_DIRS = [os.path.join("runs", "tuning_v4", f"E7_seed{s}") for s in (42, 7, 123)]
COLOURS = {"black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red",
           "white", "yellow"}


def wilson(k, n, z=1.96):
    """95% margin of error for an accuracy of k out of n (Wilson interval)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def by_group(df, col, label):
    rows = []
    for g, sub in df.groupby(col, observed=True):
        k, n = int(sub["correct"].sum()), len(sub)
        lo, hi = wilson(k, n)
        rows.append(dict(factor=label, group=str(g), clips=n, accuracy=round(100 * k / n, 1),
                         ci_low=round(lo, 1), ci_high=round(hi, 1)))
    return rows


def analyse(df, probs, words, lines):
    """df: one row per validation clip with factor columns + 'true','pred','conf','correct'."""
    n = len(df)
    acc = df["correct"].mean() * 100
    top5 = (np.argsort(-probs, axis=1)[:, :5] == df["true"].values[:, None]).any(axis=1).mean() * 100
    lines += ["========== VALIDATION ERROR ANALYSIS: E7 ENSEMBLE ==========",
              f"Validation clips: {n} | top-1 {acc:.1f}% | top-5 {top5:.1f}%",
              "Groups below come with a 95% range; overlapping ranges = no clear difference.", ""]

    # ---- 1. words and confusions
    pw = df.groupby("word")["correct"].agg(["sum", "count"])
    never = pw[pw["sum"] == 0].index.tolist()
    always = pw[pw["sum"] == pw["count"]].index.tolist()
    lines.append(f"Words always right on validation: {len(always)} of {len(pw)}")
    lines.append(f"Words never right on validation: {len(never)}: " + ", ".join(never))
    wrong = df[~df["correct"]]
    pairs = wrong.groupby(["word", "pred_word"]).size().sort_values(ascending=False)
    lines.append("\nMost common confusions (true -> guessed):")
    for (t, g), k in pairs.head(12).items():
        lines.append(f"  {t} -> {g}" + (f" ({k}x)" if k > 1 else ""))
    col = df[df["word"].isin(COLOURS)]
    if len(col):
        lines.append(f"\nColour words: {int(col['correct'].sum())} of {len(col)} correct "
                     f"({100 * col['correct'].mean():.0f}%); guessed as another colour when wrong: "
                     f"{int(col[~col['correct']]['pred_word'].isin(COLOURS).sum())} of "
                     f"{int((~col['correct']).sum())}")

    # ---- 2. factors
    rows = []
    rows += by_group(df, "train_clips_grp", "training clips for the word")
    rows += by_group(df, "train_signers_grp", "training signers for the word")
    rows += by_group(df, "signer_seen", "signer seen in training")
    rows += by_group(df, "hands_grp", "hand detection in clip")
    rows += by_group(df, "filled_grp", "hand gaps filled")
    rows += by_group(df, "handed", "one/two-handed (estimated)")
    rows += by_group(df, "length_grp", "sign length")
    rows += by_group(df, "fps_grp", "frame rate")
    rows += by_group(df, "res_grp", "video height")
    fac = pd.DataFrame(rows)
    lines.append("\n========== FACTORS (accuracy %, with 95% range) ==========")
    for f, sub in fac.groupby("factor", sort=False):
        lines.append(f"{f}:")
        for _, r in sub.iterrows():
            lines.append(f"  {r['group']:<22} {r['clips']:4d} clips  {r['accuracy']:5.1f}%  "
                         f"({r['ci_low']:.0f}-{r['ci_high']:.0f})")

    # ---- 3. confidence
    lines.append("\n========== CONFIDENCE (for the live 'not sure' threshold) ==========")
    lines.append(f"Mean confidence when right {df[df.correct]['conf'].mean():.2f}, "
                 f"when wrong {df[~df.correct]['conf'].mean():.2f}")
    conf_rows = []
    for t in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        sub = df[df["conf"] >= t]
        if len(sub):
            conf_rows.append((t, 100 * len(sub) / n, 100 * sub["correct"].mean()))
            lines.append(f"  answer only if confidence >= {t:.1f}: answers {100 * len(sub) / n:5.1f}% "
                         f"of signs, {100 * sub['correct'].mean():5.1f}% of those correct")
    return fac, conf_rows


def plots(fac, conf_rows, out_dir):
    factors = list(dict.fromkeys(fac["factor"]))
    cols = 3
    rows = int(np.ceil(len(factors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    for ax, f in zip(axes.ravel(), factors):
        sub = fac[fac["factor"] == f]
        x = np.arange(len(sub))
        ax.bar(x, sub["accuracy"], color="steelblue")
        ax.errorbar(x, sub["accuracy"], yerr=[sub["accuracy"] - sub["ci_low"],
                                              sub["ci_high"] - sub["accuracy"]],
                    fmt="none", color="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n={c})" for g, c in zip(sub["group"], sub["clips"])], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f, fontsize=10)
        ax.set_ylabel("validation top-1 (%)")
    for ax in axes.ravel()[len(factors):]:
        ax.axis("off")
    fig.suptitle("E7 ensemble, validation: accuracy by factor (bars = 95% range)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "factors.png"), dpi=90)
    plt.close(fig)

    t, cov, acc = zip(*conf_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cov, acc, marker="o")
    for ti, c, a in conf_rows:
        ax.annotate(f">= {ti:.1f}", (c, a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("signs answered (%) - the rest would show 'not sure'")
    ax.set_ylabel("accuracy of answered signs (%)")
    ax.set_title("Confidence threshold trade-off (validation)")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence.png"), dpi=90)
    plt.close(fig)


def build_table(meta, split, index_v3, X_all, which, probs, words):
    va = (which == "val").values
    tr = (which == "train").values
    v = meta[va].reset_index(drop=True).copy()
    v["true"] = v["label"].values
    v["pred"] = probs.argmax(axis=1)
    v["pred_word"] = [words[i] for i in v["pred"]]
    v["conf"] = probs.max(axis=1)
    v["correct"] = v["pred"] == v["true"]

    trm = meta[tr]
    n_train = trm.groupby("word").size()
    n_sign = trm.groupby("word")["signer_id"].nunique()
    v["train_clips"] = v["word"].map(n_train)
    v["train_signers"] = v["word"].map(n_sign)
    v["train_clips_grp"] = pd.cut(v["train_clips"], [0, 6, 7, 8, 100], labels=["<=6", "7", "8", "9+"])
    v["train_signers_grp"] = pd.cut(v["train_signers"], [0, 4, 6, 100], labels=["<=4", "5-6", "7+"])
    v["signer_seen"] = np.where(v["signer_id"].isin(set(trm["signer_id"])),
                                "seen (other words)", "never seen")
    v["hands_grp"] = pd.cut(v["any_hand_share"], [0, 0.8, 0.95, 1.01],
                            labels=["<80%", "80-95%", ">95%"])
    v["filled_grp"] = np.where(v["frames_filled"] > 0, "some gaps filled", "no gaps")
    both = (X_all[va][..., -2] > 0.5) & (X_all[va][..., -1] > 0.5)
    v["handed"] = np.where(both.mean(axis=1) >= 0.5, "two-handed", "one-handed")
    v["length_grp"] = pd.qcut(v["span_frames"], 3, labels=["short", "medium", "long"], duplicates="drop")
    fps = v["video_id"].map(dict(zip(index_v3["video_id"], index_v3["fps"])))
    v["fps_grp"] = pd.cut(fps, [0, 27, 40, 100], labels=["24-25", "~30", "50"])
    v["res_grp"] = pd.cut(v["height"], [0, 300, 500, 5000], labels=["<=300", "301-500", ">500"])
    return v


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tqdm import tqdm
    from tune_v4 import val_probs  # same features and model as in tuning

    X_all = np.load(os.path.join(DATA_DIR, "X.npy"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), dtype={"video_id": str})
    split = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    index_v3 = pd.read_csv(INDEX_V3, dtype={"video_id": str})
    words = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))["word"].tolist()
    which = meta["video_id"].map(dict(zip(split["video_id"], split["split"])))
    va = (which == "val").values

    probs = np.mean([val_probs(d, X_all, va, len(words))
                     for d in tqdm(E7_DIRS, desc="Loading E7 models", unit="model")], axis=0)
    df = build_table(meta, split, index_v3, X_all, which, probs, words)
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []
    fac, conf_rows = analyse(df, probs, words, lines)
    plots(fac, conf_rows, OUT_DIR)
    df.drop(columns=["true", "pred"]).to_csv(os.path.join(OUT_DIR, "per_clip.csv"), index=False)
    fac.to_csv(os.path.join(OUT_DIR, "factors.csv"), index=False)
    print("\n".join(lines))
    print(f"\nSaved {OUT_DIR}\\per_clip.csv, factors.csv, factors.png, confidence.png")


if __name__ == "__main__":
    main()