r"""
v2 step 6a: split the WLASL clips of the 20 NEW words into train / validation / test.

Same rules as scripts\week3_training\split.py (it IMPORTS take_signers() from there):
  ~70 / 20 / 10 per word, at least 1 clip per group, whole signers kept in one group
  where possible, seed 42.
The 80 v1 words are NOT re-split: their rows are copied unchanged from data\split_v1.csv,
so v1's validation and test sets stay exactly the same.

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\split_new20.py > report_outputs\V2_08_split_new20.txt

Outputs:
  data\split_new20.csv      the 20 new words only
  data\split_wlasl100.csv   all 100 words: v1 rows (unchanged) + new rows
                            columns: video_id, word, signer_id, split, signer_overlap, source
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join("scripts", "week3_training"))
from split import take_signers, SEED, TEST_SHARE, VAL_SHARE  # noqa: E402  (same rules as v1)

NEW_INDEX = os.path.join("data", "wlasl_new", "landmarks_index_v3.csv")
SPLIT_V1 = os.path.join("data", "split_v1.csv")
OUT_NEW = os.path.join("data", "split_new20.csv")
OUT_ALL = os.path.join("data", "split_wlasl100.csv")


def main():
    idx = pd.read_csv(NEW_INDEX, dtype={"video_id": str})
    idx = idx[idx["usable"]]
    rng = np.random.default_rng(SEED)
    split_of, overlap_words = {}, []
    for word, g in tqdm(idx.groupby("word", sort=True), desc="Splitting new words", unit="word",
                        file=sys.stderr):
        n = len(g)
        n_test = max(1, round(TEST_SHARE * n))
        n_val = max(1, round(VAL_SHARE * n))
        groups = {s: list(v) for s, v in g.groupby("signer_id")["video_id"]}
        test, groups, ov_t = take_signers(groups, n_test, rng)
        val, groups, ov_v = take_signers(groups, n_val, rng)
        train = [v for clips in groups.values() for v in clips]
        if not train:
            train.append(val.pop())
        for v in test:
            split_of[v] = "test"
        for v in val:
            split_of[v] = "val"
        for v in train:
            split_of[v] = "train"
        if ov_t is not None or ov_v is not None:
            overlap_words.append(word)

    new = idx[["video_id", "word", "signer_id"]].copy()
    new["split"] = new["video_id"].map(split_of)
    new["signer_overlap"] = new.groupby(["word", "signer_id"])["split"].transform("nunique") > 1
    new["source"] = "wlasl_new20"
    new.to_csv(OUT_NEW, index=False)

    v1 = pd.read_csv(SPLIT_V1, dtype={"video_id": str})
    v1 = v1[["video_id", "word", "signer_id", "split", "signer_overlap"]].assign(source="wlasl_v1")
    clash = set(v1.word) & set(new.word)
    if clash:
        sys.exit(f"ERROR: words in both v1 and new split: {sorted(clash)}")
    allw = pd.concat([v1, new], ignore_index=True)
    allw.to_csv(OUT_ALL, index=False)

    print("========== SPLIT OF THE 20 NEW WORDS ==========")
    c = new["split"].value_counts()
    for s in ["train", "val", "test"]:
        print(f"{s:>5}: {c.get(s, 0):4d} clips")
    per = new.groupby(["word", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    print("\nPer word:")
    print(per.to_string())
    print(f"\nEvery new word has at least 1 clip in each group: {bool((per > 0).all().all())}")
    print(f"Words where one signer had to be split across groups: {len(overlap_words)}"
          + (f" ({', '.join(overlap_words)})" if overlap_words else ""))
    print(f"Clips whose signer also appears in another group (same word): {int(new.signer_overlap.sum())}")

    print("\n========== ALL 100 WORDS (WLASL only) ==========")
    c = allw["split"].value_counts()
    for s in ["train", "val", "test"]:
        print(f"{s:>5}: {c.get(s, 0):4d} clips")
    print(f"v1 rows copied unchanged: {len(v1)} (val {int((v1.split == 'val').sum())}, "
          f"test {int((v1.split == 'test').sum())})")
    print("\nThe new TEST clips are now fixed too. Never train or tune on them.")
    print(f"Saved {OUT_NEW} and {OUT_ALL}")


if __name__ == "__main__":
    main()