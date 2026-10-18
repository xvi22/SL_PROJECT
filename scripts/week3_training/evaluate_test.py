r"""
evaluate_test.py - Weeks 3-4: the ONE-TIME final test evaluation (Q1 and Q4).

Plan fixed BEFORE running (17 Oct 2026), after tuning rounds 1-4 on validation only:
  - Final model: the E7 ENSEMBLE = three E7 models (seeds 42, 7, 123), whose predicted
    probabilities are averaged. E7 = one bidirectional LSTM layer (64 units), dropout 0.5,
    weight decay 1e-3, whole-sequence average, handshape-from-wrist features, no pose depth,
    movement features, label smoothing 0.1.
  - Q1 HEADLINE = test top-1 of the E7 ensemble.
  - Also reported: the three single E7 models' test top-1 (average and range), the fair
    comparison with published single models (Pose-GRU, SPOTER).
  - Live/demo model = the E7 ensemble.
  - Q4 = model size (parameters) and processing time per sign (median of many runs, CPU).

Run ONCE from C:\SL_PROJECT:
    python scripts\week3_training\evaluate_test.py > report_outputs\C1_final_test.txt

It refuses to run again if runs\final_test\ already exists (protects the "test once" rule).

Outputs (runs\final_test\): results.txt, per_clip.csv, confusion.png
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_v4 import SignLSTM, features  # same features and model as in tuning  # noqa: E402

DATA_DIR = "data/processed_v1"
SPLIT_CSV = "data/split_v1.csv"
OUT_DIR = os.path.join("runs", "final_test")
MODELS = {f"E7 seed {s}": os.path.join("runs", "tuning_v4", f"E7_seed{s}") for s in (42, 7, 123)}
COLOURS = {"black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red",
           "white", "yellow"}


def load(run_dir, n_cls):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    st = np.load(os.path.join(run_dir, "norm_stats.npz"))
    model = SignLSTM(cfg["features"], n_cls, cfg["hidden"], cfg["layers"], cfg["dropout"],
                     cfg["bidir"], cfg["pool"])
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best.pt")))
    model.eval()
    return model, cfg, st["mean"], st["std"]


def top_k(probs, y):
    order = np.argsort(-probs, axis=1)[:, :5]
    return 100 * (order[:, 0] == y).mean(), 100 * (order == y[:, None]).any(axis=1).mean()


def time_ensemble(models, frames, n_feat, repeats=300):
    """Median and 95th percentile time (ms) to run ALL models on one sign."""
    x = torch.randn(1, frames, n_feat)
    times = []
    with torch.no_grad():
        for _ in range(20):
            for m in models:
                m(x)
        for _ in range(repeats):
            t0 = time.perf_counter()
            for m in models:
                torch.softmax(m(x), dim=1)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f"{OUT_DIR} already exists: the test set has already been evaluated. "
                         "The test set is used once only. (Delete the folder only if the run "
                         "crashed before finishing.)")
    for name, d in MODELS.items():
        if not os.path.exists(os.path.join(d, "model_best.pt")):
            raise SystemExit(f"Missing model for {name}: {d}")

    X_all = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_all = np.load(os.path.join(DATA_DIR, "y.npy"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"), dtype={"video_id": str})
    split = pd.read_csv(SPLIT_CSV, dtype={"video_id": str})
    words = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))["word"].tolist()
    which = meta["video_id"].map(dict(zip(split["video_id"], split["split"])))
    te = (which == "test").values
    yt = y_all[te]
    os.makedirs(OUT_DIR)

    probs, singles, models, n_params = {}, {}, [], 0
    for name, run_dir in tqdm(MODELS.items(), desc="Testing E7 models", unit="model"):
        model, cfg, mu, sd = load(run_dir, len(words))
        Xt = (features(X_all, cfg)[te] - mu) / sd
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(Xt, dtype=torch.float32)), dim=1).numpy()
        probs[name] = p
        singles[name] = top_k(p, yt)
        models.append(model)
        n_params += sum(q.numel() for q in model.parameters())
        frames, n_feat = Xt.shape[1], Xt.shape[2]

    ens = np.mean(list(probs.values()), axis=0)
    e1, e5 = top_k(ens, yt)
    s1 = np.array([v[0] for v in singles.values()])
    s5 = np.array([v[1] for v in singles.values()])
    ms, p95 = time_ensemble(models, frames, n_feat)
    n_test = int(te.sum())

    lines = ["========== FINAL TEST RESULTS (test set, used once) ==========",
             f"Test clips: {n_test} across {len(set(yt))} words (1 clip = {100 / n_test:.1f} points)", "",
             f"Q1 HEADLINE - E7 ensemble: top-1 {e1:.1f}%, top-5 {e5:.1f}%", ""]
    for name, (t1, t5) in singles.items():
        lines.append(f"  single {name:12s}: top-1 {t1:.1f}%  top-5 {t5:.1f}%")
    lines += [f"  single E7 models: top-1 mean {s1.mean():.1f}% (range {s1.min():.1f}-{s1.max():.1f}), "
              f"top-5 mean {s5.mean():.1f}%",
              "Q1 targets: Satisfactory > 45% top-1, Excellent >= 60% top-1", "",
              f"Q4: {n_params:,} parameters in total (3 x {n_params // 3:,}); processing time for the "
              f"whole ensemble median {ms:.1f} ms, 95th percentile {p95:.1f} ms per sign on this CPU",
              "Q4 targets: Satisfactory < 1 s, Excellent < 200 ms", ""]

    pred = ens.argmax(axis=1)
    conf = ens.max(axis=1)
    per_clip = pd.DataFrame({"video_id": meta.loc[te, "video_id"].values,
                             "true_word": [words[i] for i in yt],
                             "guess": [words[i] for i in pred],
                             "confidence": conf.round(3),
                             "correct": pred == yt})
    per_clip.to_csv(os.path.join(OUT_DIR, "per_clip.csv"), index=False)
    wrong = per_clip[~per_clip["correct"]]
    pairs = wrong.groupby(["true_word", "guess"]).size().sort_values(ascending=False)
    lines.append(f"Ensemble: {len(wrong)} of {n_test} test clips wrong. Confusions (true -> guessed):")
    for (t, g), k in pairs.head(12).items():
        lines.append(f"  {t} -> {g}" + (f" ({k}x)" if k > 1 else ""))
    col = per_clip[per_clip["true_word"].isin(COLOURS)]
    if len(col):
        lines.append(f"Colour words: {int(col['correct'].sum())} of {len(col)} test clips correct")
        for _, r in col[~col["correct"]].iterrows():
            lines.append(f"  {r['true_word']} -> guessed {r['guess']}")
    lines.append(f"Mean confidence when right {conf[pred == yt].mean():.2f}, "
                 f"when wrong {conf[pred != yt].mean():.2f}")
    sub = per_clip[per_clip["confidence"] >= 0.3]
    if len(sub):
        lines.append(f"With the 'not sure' threshold 0.3: answers {100 * len(sub) / n_test:.1f}% of signs, "
                     f"{100 * sub['correct'].mean():.1f}% of those correct")

    cm = np.zeros((len(words), len(words)), int)
    for t, p in zip(yt, pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90, fontsize=6)
    ax.set_yticklabels(words, fontsize=6)
    ax.set_xlabel("guessed word")
    ax.set_ylabel("true word")
    ax.set_title("Confusion matrix, test set, E7 ensemble (diagonal = correct)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion.png"), dpi=110)
    plt.close(fig)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved {OUT_DIR}\\results.txt, per_clip.csv, confusion.png")


if __name__ == "__main__":
    main()