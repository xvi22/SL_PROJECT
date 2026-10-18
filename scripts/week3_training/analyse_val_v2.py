r"""
analyse_val_v2.py - factor analysis on VALIDATION: v1 (E7) vs the best model (best-full).

Same idea as Figure 10 (analyse_val.py for v1), now comparing both models on the same
validation clips. VALIDATION ONLY - the test set is closed and not used.

  v1        runs\tuning_v4\E7_seed{42,7,123}   (80 words; ensemble)
  best      runs\tuning_v8\full_H128_HAND_seed{42,7,123} (100 words; ensemble; predictions saved in training)

Accuracy (top-1) is compared on the 176 validation clips of the 80 v1 words, with only the 80 v1
words allowed as answers ("like v1"), so both models face exactly the same question.
New-word clips (28) are shown for the best model only.

Factors (per clip): training clips for the word, training signers for the word, signer seen in
training, hand detection, hand gaps filled, one/two-handed (estimated), sign length, frame rate,
video height - as in Figure 10 - plus: ASL Citizen clips for the word, and v1 word vs new word.
Also: clips fixed / broken by the new model, words that improved most, words still worst,
and the most common confusions left.

Bars = 95% range (Wilson interval). Overlapping ranges = the difference may be chance.

Run from C:\SL_PROJECT :
  python -u scripts\week3_training\analyse_val_v2.py > report_outputs\V2_21_analysis_val_v2.txt

Outputs (runs\analysis_val_v2\): factors_v1_vs_best.png, clips.csv, factors.csv, words.csv, confusions.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_v4 import val_probs  # noqa: E402
from tune_v5 import P2, SEEDS, load_set  # noqa: E402

OUT = os.path.join("runs", "analysis_val_v2")
V1_RUNS = [os.path.join("runs", "tuning_v4", f"E7_seed{s}") for s in SEEDS]
BEST_RUNS = [os.path.join("runs", "tuning_v8", f"full_H128_HAND_seed{s}") for s in SEEDS]
V1_LABELS = os.path.join("data", "processed_v1", "labels.csv")
SPLIT_V1 = os.path.join("data", "split_v1.csv")
SPLIT_FULL = os.path.join("data", "split_v2_full.csv")
FPS_FILES = [os.path.join("data", "v3", "landmarks_index_v3.csv"),
             os.path.join("data", "wlasl_new", "landmarks_index_v3.csv")]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def binned(values, edges, labels):
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True).astype(str)


def main():
    os.makedirs(OUT, exist_ok=True)
    X = np.load(os.path.join(P2, "X.npy"))
    meta = pd.read_csv(os.path.join(P2, "meta.csv"), dtype={"clip_id": str})
    labels = pd.read_csv(os.path.join(P2, "labels.csv"))
    word_of = dict(zip(labels["label"], labels["word"]))
    v1_lab = pd.read_csv(V1_LABELS)
    v1_words = set(v1_lab["word"])
    v1_label_of = dict(zip(v1_lab["word"], v1_lab["label"]))
    allowed = labels.loc[labels["word"].isin(v1_words), "label"].to_numpy()

    # ---- predictions -----------------------------------------------------------------------
    _, va = load_set("full", meta)                                      # order used in training
    best_p = np.mean([np.load(os.path.join(d, "val_probs.npy")) for d in
                      tqdm(BEST_RUNS, desc="Best model (saved predictions)", unit="model", file=sys.stderr)], axis=0)
    clips = meta.iloc[va].reset_index(drop=True).copy()
    clips["row"] = va
    y = clips["label"].to_numpy()
    clips["best_guess"] = [word_of[i] for i in best_p.argmax(1)]
    clips["best_conf"] = best_p.max(1).round(3)
    clips["best_correct"] = best_p.argmax(1) == y
    p80 = best_p.copy()
    block = np.ones(p80.shape[1], bool)
    block[allowed] = False
    p80[:, block] = -1
    clips["best_likev1_correct"] = p80.argmax(1) == y

    is_v1 = clips["word"].isin(v1_words).to_numpy()
    mask = np.zeros(len(X), bool)
    mask[clips.loc[is_v1, "row"].to_numpy()] = True
    v1_p = np.mean([val_probs(d, X, mask, len(v1_lab)) for d in
                    tqdm(V1_RUNS, desc="v1 model (recomputed)", unit="model", file=sys.stderr)], axis=0)
    order = np.argsort(clips.loc[is_v1, "row"].to_numpy())                # val_probs returns index order
    v1_rows = clips.index[is_v1].to_numpy()[order]
    inv_v1 = {i: w for w, i in v1_label_of.items()}
    clips["v1_guess"] = ""
    clips["v1_correct"] = pd.Series([pd.NA] * len(clips), dtype="object")
    clips.loc[v1_rows, "v1_guess"] = [inv_v1[i] for i in v1_p.argmax(1)]
    clips.loc[v1_rows, "v1_correct"] = list(v1_p.argmax(1) == clips.loc[v1_rows, "word"].map(v1_label_of).to_numpy())

    # ---- factors -----------------------------------------------------------------------------
    s1 = pd.read_csv(SPLIT_V1, dtype={"video_id": str})
    tr1 = s1[s1["split"] == "train"]
    sf = pd.read_csv(SPLIT_FULL, dtype={"clip_id": str})
    trf = sf[sf["split"] == "train"]
    clips["v1_train_clips"] = clips["word"].map(tr1.groupby("word").size()).fillna(0).astype(int)
    clips["v1_train_signers"] = clips["word"].map(tr1.groupby("word")["signer_id"].nunique()).fillna(0).astype(int)
    clips["best_train_clips"] = clips["word"].map(trf.groupby("word").size()).fillna(0).astype(int)
    clips["best_train_signers"] = clips["word"].map(trf.groupby("word")["signer"].nunique()).fillna(0).astype(int)
    clips["asc_clips"] = clips["word"].map(trf[trf["source"] == "asl_citizen"].groupby("word").size()).fillna(0).astype(int)
    wl_train_signers = set(trf.loc[trf["source"] != "asl_citizen", "signer"].astype(str))
    clips["signer_seen"] = np.where(clips["signer"].astype(str).isin(wl_train_signers), "seen (other words)", "never seen")
    fps = pd.concat([pd.read_csv(f, dtype={"video_id": str}) for f in FPS_FILES if os.path.exists(f)], ignore_index=True)
    if "fps" in fps:
        clips["fps"] = clips["clip_id"].map(dict(zip(fps["video_id"], fps["fps"])))
    else:
        clips["fps"] = np.nan
    Xv = X[clips["row"].to_numpy()]
    both = (Xv[..., 195] * Xv[..., 196]).mean(1)                           # share of frames with both hands
    clips["handed"] = np.where(both > 0.3, "two-handed", "one-handed")

    F = {}
    F["training clips for the word"] = ("v1_train_clips", "best_train_clips", [-1, 6, 8, 20, 40, 1e9],
                                        ["<=6", "7-8", "9-20", "21-40", "41+"])
    F["training signers for the word"] = ("v1_train_signers", "best_train_signers", [-1, 4, 6, 12, 30, 1e9],
                                          ["<=4", "5-6", "7-12", "13-30", "31+"])
    F["ASL Citizen clips for the word"] = (None, "asc_clips", [-1, 0, 30, 60, 1e9], ["0", "1-30", "31-60", "61+"])
    F["hand detection in clip"] = ("any_hand_share", "any_hand_share", [-1, 0.8, 0.95, 2], ["<80%", "80-95%", ">95%"])
    F["hand gaps filled"] = ("frames_filled", "frames_filled", [-1, 0, 1e9], ["no gaps", "some gaps filled"])
    F["sign length"] = ("span_frames", "span_frames", [-1, 35, 50, 1e9], ["short (<=35)", "medium", "long (>50)"])
    F["frame rate"] = ("fps", "fps", [0, 26, 40, 1e9], ["24-25", "~30", "50+"])
    F["video height"] = ("height", "height", [0, 300, 500, 1e9], ["<=300", "301-500", ">500"])
    cat = {"signer seen in training": "signer_seen", "one/two-handed (estimated)": "handed"}

    rows = []

    def add(fname, group_col, model, ok, subset):
        for g, sub in subset.groupby(group_col):
            k, n = int(sub[ok].sum()), len(sub)
            acc, lo, hi = wilson(k, n)
            rows.append({"factor": fname, "group": g, "model": model, "n": n, "acc": round(acc, 1),
                         "lo": round(lo, 1), "hi": round(hi, 1)})

    v1c = clips[is_v1].copy()
    v1c["v1_correct"] = v1c["v1_correct"].astype(bool)
    for fname, (c1, cb, edges, labs) in F.items():
        if c1 is not None:
            v1c[f"g1_{fname}"] = binned(v1c[c1], edges, labs)
            add(fname, f"g1_{fname}", "v1", "v1_correct", v1c[v1c[f"g1_{fname}"] != "nan"])
        v1c[f"gb_{fname}"] = binned(v1c[cb], edges, labs)
        add(fname, f"gb_{fname}", "best", "best_likev1_correct", v1c[v1c[f"gb_{fname}"] != "nan"])
    for fname, col in cat.items():
        add(fname, col, "v1", "v1_correct", v1c)
        add(fname, col, "best", "best_likev1_correct", v1c)
    clips["word_type"] = np.where(is_v1, "v1 word (80)", "new word (20)")
    add("v1 word vs new word", "word_type", "best", "best_correct", clips)
    fac = pd.DataFrame(rows)
    fac.to_csv(os.path.join(OUT, "factors.csv"), index=False)
    clips.to_csv(os.path.join(OUT, "clips.csv"), index=False)

    # ---- figure --------------------------------------------------------------------------------
    order_f = list(F) + list(cat) + ["v1 word vs new word"]
    group_order = {f: (F[f][3] if f in F else None) for f in order_f}
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax, f in zip(axes.flat, order_f):
        d = fac[fac.factor == f]
        groups = group_order[f] or sorted(d["group"].unique())
        groups = [g for g in groups if g in set(d["group"])]
        x = np.arange(len(groups))
        for k, (model, colour) in enumerate([("v1", "#9e9e9e"), ("best", "#2f6fb0")]):
            dm = d[d.model == model].set_index("group").reindex(groups)
            if dm["acc"].isna().all():
                continue
            off = (k - 0.5) * 0.38
            ax.bar(x + off, dm["acc"], 0.36, color=colour, label=model)
            ax.errorbar(x + off, dm["acc"], yerr=[dm["acc"] - dm["lo"], dm["hi"] - dm["acc"]], fmt="none", ecolor="black", capsize=3, lw=1)
        nb = d[d.model == "best"].set_index("group").reindex(groups)["n"]
        n1 = d[d.model == "v1"].set_index("group").reindex(groups)["n"]
        ax.set_xticks(x)
        def nlab(a, b):
            fa = "-" if pd.isna(a) else str(int(a))
            fb = "-" if pd.isna(b) else str(int(b))
            return f"n={fb}" if fa == "-" else f"n={fa}/{fb}"
        ax.set_xticklabels([f"{g}\n({nlab(a, b)})" for g, a, b in zip(groups, n1, nb)], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f, fontsize=11)
        ax.set_ylabel("validation top-1 (%)")
        ax.legend(fontsize=8, loc="lower right")
    for ax in list(axes.flat)[len(order_f):]:
        ax.axis("off")
    fig.suptitle("Validation accuracy by factor: v1 (grey) vs best model (blue); bars = 95% range; "
                 "n = clips (v1/best; groups can differ because training counts changed)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(OUT, "factors_v1_vs_best.png")
    plt.savefig(fig_path, dpi=100)
    plt.close(fig)

    # ---- clip by clip, per word, confusions ----------------------------------------------------
    a, b = v1c["v1_correct"].to_numpy(), v1c["best_likev1_correct"].to_numpy()
    print("=" * 70)
    print("1. OVERALL (validation, 176 clips of the 80 v1 words, 80 words allowed)")
    print("=" * 70)
    print(f"v1:   {a.mean():.1%}   best: {b.mean():.1%}")
    print(f"Both right {int((a & b).sum())} | fixed by best {int((~a & b).sum())} | broken by best {int((a & ~b).sum())} | "
          f"both wrong {int((~a & ~b).sum())}")
    print(f"New-word clips (best only, 100 words allowed): {clips.loc[~is_v1, 'best_correct'].mean():.1%} "
          f"of {int((~is_v1).sum())}")

    print("\n" + "=" * 70)
    print("2. ACCURACY BY FACTOR (v1 vs best; acc [95% range], n)")
    print("=" * 70)
    for f in order_f:
        print(f"\n{f}:")
        d = fac[fac.factor == f]
        for g in (group_order[f] or sorted(d["group"].unique())):
            parts = []
            for model in ("v1", "best"):
                r = d[(d.model == model) & (d.group == g)]
                if len(r):
                    r = r.iloc[0]
                    parts.append(f"{model} {r.acc:5.1f}% [{r.lo:.0f}-{r.hi:.0f}] n={r.n}")
            if parts:
                print(f"  {g:22s} " + " | ".join(parts))

    w = v1c.groupby("word").agg(clips=("word", "size"), v1=("v1_correct", "mean"), best=("best_likev1_correct", "mean"),
                                v1_signers=("v1_train_signers", "first"), best_signers=("best_train_signers", "first"),
                                asc_clips=("asc_clips", "first"))
    w["change"] = w["best"] - w["v1"]
    new_w = clips[~is_v1].groupby("word").agg(clips=("word", "size"), best=("best_correct", "mean"),
                                              best_signers=("best_train_signers", "first"), asc_clips=("asc_clips", "first"))
    allw = pd.concat([w, new_w])
    allw.round(2).to_csv(os.path.join(OUT, "words.csv"))
    pd.set_option("display.width", 200)
    print("\n" + "=" * 70)
    print("3. WORDS (validation)")
    print("=" * 70)
    print("Improved most (v1 -> best):")
    print((w.sort_values(["change", "clips"], ascending=False).head(10) * [1, 100, 100, 1, 1, 1, 100]).round(0).to_string())
    print("\nStill worst for the best model (all 100 words, at least 2 clips):")
    worst = allw[allw["clips"] >= 2].sort_values(["best", "clips"], ascending=[True, False]).head(10)
    print(worst[["clips", "best", "best_signers", "asc_clips"]].assign(best=lambda d: (d.best * 100).round(0)).to_string())
    colours = ["black", "blue", "brown", "orange", "pink", "purple", "white"]
    cw = v1c[v1c["word"].isin(colours)]
    if len(cw):
        print(f"\nColour words ({len(cw)} clips): v1 {cw['v1_correct'].mean():.0%} -> best {cw['best_likev1_correct'].mean():.0%}")

    wrong = clips[~clips["best_correct"]]
    conf = wrong.groupby(["word", "best_guess"]).size().sort_values(ascending=False).rename("times").reset_index()
    conf.to_csv(os.path.join(OUT, "confusions.csv"), index=False)
    print("\nMost common confusions left (best model, all 100 words):")
    print(conf.head(12).to_string(index=False))
    print(f"\nSaved {fig_path} and CSVs in {OUT}. Validation only - the test set was not used.")


if __name__ == "__main__":
    main()