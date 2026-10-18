r"""
split.py - Weeks 3-4 groundwork: split the processed clips into train / validation / test.

Run from C:\SL_PROJECT after preprocess.py:
    python split.py

Rules (from the report, section 2):
  - our own split, not WLASL's original one
  - stratified per word: about 70% train, 20% validation, 10% test
  - at least 1 clip per group for every word
  - signers kept in one group where possible: for each word, whole signers are
    placed in test first, then validation, the rest go to train. If a word has too
    few signers to do this, one signer's clips are split and the overlap is recorded
    (stated as a limitation in the report).
  - fixed random seed, so the split is the same every time it is run

The TEST set stays untouched until the final Q1/Q4 evaluation. Tune on validation only.

Output: split_v1.csv  (video_id, word, label, signer_id, split, signer_overlap)
"""

import os

import numpy as np
import pandas as pd

META_CSV = os.path.join("data/processed_v1", "meta.csv")
OUT_CSV = "data/split_v1.csv"
SEED = 42
TEST_SHARE, VAL_SHARE = 0.10, 0.20


def take_signers(groups, target, rng):
    """Pick whole signer groups (smallest first) until target clips are reached.
    If no whole group fits, split the smallest remaining group (signer overlap).
    Returns (chosen video_ids, remaining groups, overlap_signer or None)."""
    chosen, overlap = [], None
    # smallest groups first; ties broken randomly (but reproducibly)
    order = sorted(groups.keys(), key=lambda s: (len(groups[s]), rng.random()))
    for s in order:
        if len(chosen) + len(groups[s]) <= target:
            chosen += groups.pop(s)
        if len(chosen) == target:
            break
    if len(chosen) < target and groups:
        s = min(groups, key=lambda k: len(groups[k]))
        need = target - len(chosen)
        clips = groups[s]
        rng.shuffle(clips)
        chosen += clips[:need]
        groups[s] = clips[need:]
        if not groups[s]:
            groups.pop(s)
        else:
            overlap = s
    return chosen, groups, overlap


def main():
    meta = pd.read_csv(META_CSV, dtype={"video_id": str})
    rng = np.random.default_rng(SEED)
    split_of, overlap_words = {}, []

    for word, g in meta.groupby("word", sort=True):
        n = len(g)
        n_test = max(1, round(TEST_SHARE * n))
        n_val = max(1, round(VAL_SHARE * n))
        groups = {s: list(v) for s, v in g.groupby("signer_id")["video_id"]}

        test, groups, ov_t = take_signers(groups, n_test, rng)
        val, groups, ov_v = take_signers(groups, n_val, rng)
        train = [v for clips in groups.values() for v in clips]
        if not train:  # safety: never leave train empty
            train.append(val.pop())

        for v in test:
            split_of[v] = "test"
        for v in val:
            split_of[v] = "val"
        for v in train:
            split_of[v] = "train"
        if ov_t is not None or ov_v is not None:
            overlap_words.append(word)

    meta["split"] = meta["video_id"].map(split_of)
    # signer_overlap: this clip's signer appears in more than one group for this word
    n_groups = meta.groupby(["word", "signer_id"])["split"].transform("nunique")
    meta["signer_overlap"] = n_groups > 1
    out = meta[["video_id", "word", "label", "signer_id", "split", "signer_overlap"]]
    out.to_csv(OUT_CSV, index=False)

    print("========== SPLIT SUMMARY ==========")
    counts = out["split"].value_counts()
    total = len(out)
    for s in ["train", "val", "test"]:
        print(f"{s:>5}: {counts.get(s, 0):4d} clips ({counts.get(s, 0) / total:.0%})")
    per = out.groupby(["word", "split"]).size().unstack(fill_value=0)
    print(f"\nPer word: train min {per['train'].min()}, val min {per['val'].min()}, "
          f"test min {per['test'].min()} (every word has at least 1 in each group: "
          f"{bool((per > 0).all().all())})")
    print(f"Words where one signer had to be split across groups: {len(overlap_words)} of "
          f"{out['word'].nunique()}")
    if overlap_words:
        print("  " + ", ".join(overlap_words))
    tt = out[out["signer_overlap"]]
    print(f"Clips whose signer also appears in another group (same word): {len(tt)}")
    print("\nThe TEST set is now fixed. Do not train or tune on it; use it once, at the end.")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()