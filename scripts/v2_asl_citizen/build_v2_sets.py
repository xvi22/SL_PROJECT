r"""
v2 step 8: build the two v2 training sets (no training yet - only WHICH clips go where).

  v2-full     = WLASL training clips + ALL kept ASL Citizen clips
  v2-matched  = WLASL training clips + just enough ASL Citizen clips so that every word has
                the SAME number of training clips as in WLASL's official training split
                (what I3D and SPOTER were trained on). Fair data-size comparison for Q4.

Rules (the same for both):
  - validation and test = WLASL only, from data\split_wlasl100.csv (v1's 176 / 85 unchanged)
  - ASL Citizen clips go into TRAINING only, and only if:
      usable (hand in >= 20% of frames), no gap > 10 frames in the sign span,
      and their gloss is "yes" in data\asl_citizen\variant_final.csv
  - v2-matched picks its ASL Citizen clips at random with a fixed seed (42), so the
    selection is the same every time and is not hand-picked

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\build_v2_sets.py > report_outputs\V2_12_build_v2_sets.txt

Outputs:
  data\split_v2_full.csv       uid, source, clip_id, word, gloss, signer, split
  data\split_v2_matched.csv    same columns
  data\v2_targets.csv          per word: official WLASL train count, our WLASL train,
                               ASL Citizen available / used (matched, full)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

P2_META = os.path.join("data", "processed_v2", "meta.csv")
SPLIT = os.path.join("data", "split_wlasl100.csv")
FINAL = os.path.join("data", "asl_citizen", "variant_final.csv")
WLASL_JSON = os.path.join("wlasl-processed", "WLASL_v0.3.json")
OUT_FULL = os.path.join("data", "split_v2_full.csv")
OUT_MATCHED = os.path.join("data", "split_v2_matched.csv")
OUT_TARGETS = os.path.join("data", "v2_targets.csv")
SEED = 42
COLS = ["uid", "source", "clip_id", "word", "gloss", "signer", "split"]


def main():
    meta = pd.read_csv(P2_META, dtype={"clip_id": str})
    split = pd.read_csv(SPLIT, dtype={"video_id": str})
    final = pd.read_csv(FINAL, dtype=str).fillna("")
    if (final["keep"] == "check").any():
        left = final[final["keep"] == "check"]
        sys.exit(f"Still 'check' in {FINAL}: {', '.join(left.word + ' ' + left.gloss)}. Decide these first.")
    kept = set(zip(final.loc[final["keep"] == "yes", "word"], final.loc[final["keep"] == "yes", "gloss"]))

    # ---- WLASL rows: train / val / test exactly as in split_wlasl100 -------------
    wl = meta[meta["source"].isin(["wlasl_v1", "wlasl_new20"])].copy()
    wl["split"] = wl["clip_id"].map(dict(zip(split["video_id"], split["split"])))
    if wl["split"].isna().any():
        sys.exit(f"{int(wl['split'].isna().sum())} WLASL clips have no split - tell Claude")

    # ---- ASL Citizen candidates (training only) -----------------------------------
    asc = meta[meta["source"] == "asl_citizen"].copy()
    n_all = len(asc)
    gap = asc["longest_gap"] > 10
    not_kept = ~pd.Series([(w, g) in kept for w, g in zip(asc["word"], asc["gloss"])], index=asc.index)
    asc_ok = asc[~gap & ~not_kept].copy()
    asc_ok["split"] = "train"
    print("ASL Citizen clips (usable after extraction): {}".format(n_all))
    print(f"  removed - gap > 10 frames: {int(gap.sum())}")
    print(f"  removed - gloss not kept (wrong version): {int((not_kept & ~gap).sum())}")
    print(f"  available for training: {len(asc_ok)}")

    # ---- official WLASL training counts (what I3D / SPOTER trained on) ------------
    with open(WLASL_JSON, encoding="utf-8") as f:
        wlasl100 = json.load(f)[:100]
    official = {e["gloss"]: sum(1 for i in e["instances"] if i.get("split") == "train") for e in wlasl100}

    rng = np.random.default_rng(SEED)
    rows, matched_parts = [], []
    for w in tqdm(sorted(wl["word"].unique()), desc="Building sets", unit="word", file=sys.stderr):
        ours = int(((wl["word"] == w) & (wl["split"] == "train")).sum())
        avail = asc_ok[asc_ok["word"] == w]
        target = official.get(w, 0)
        need = max(0, target - ours)
        take = min(need, len(avail))
        pick = avail.iloc[rng.permutation(len(avail))[:take]] if take else avail.iloc[0:0]
        matched_parts.append(pick)
        rows.append({"word": w, "official_train": target, "our_wlasl_train": ours,
                     "asc_available": len(avail), "asc_used_matched": take, "asc_used_full": len(avail),
                     "matched_total": ours + take, "full_total": ours + len(avail),
                     "short_of_target": target - (ours + take)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT_TARGETS, index=False)

    full = pd.concat([wl[COLS], asc_ok[COLS]], ignore_index=True)
    matched = pd.concat([wl[COLS]] + [p[COLS] for p in matched_parts], ignore_index=True)
    full.to_csv(OUT_FULL, index=False)
    matched.to_csv(OUT_MATCHED, index=False)

    # ---- summary -------------------------------------------------------------------
    def signers_per_word(df):
        tr = df[df["split"] == "train"]
        return tr.groupby("word")["signer"].nunique()

    v1_train = wl[(wl["source"] == "wlasl_v1") & (wl["split"] == "train")]
    sets = {"v1 (80 words, WLASL)": v1_train,
            "WLASL only (100 words)": wl[wl["split"] == "train"],
            "v2-matched": matched[matched["split"] == "train"],
            "v2-full": full[full["split"] == "train"]}
    print("\n========== TRAINING SETS ==========")
    print(f"{'set':26s} {'clips':>6s} {'per word min/median':>20s} {'signers per word min/median':>28s}")
    for name, df in sets.items():
        pw = df.groupby("word").size()
        sp = df.groupby("word")["signer"].nunique()
        print(f"{name:26s} {len(df):6d} {f'{pw.min()} / {pw.median():.0f}':>20s} "
              f"{f'{sp.min()} / {sp.median():.0f}':>28s}")
    print(f"\nOfficial WLASL100 training clips (I3D / SPOTER): {t.official_train.sum()} "
          f"(per word min {t.official_train.min()} / median {t.official_train.median():.0f})")
    print(f"v2-matched training clips: {t.matched_total.sum()}  "
          f"(WLASL {t.our_wlasl_train.sum()} + ASL Citizen {t.asc_used_matched.sum()})")
    short = t[t.short_of_target > 0]
    print(f"Words that could not reach the official count: {len(short)}"
          + (": " + ", ".join(f"{r.word} ({r.matched_total}/{r.official_train})" for r in short.itertuples())
             if len(short) else ""))
    over = t[t.our_wlasl_train > t.official_train]
    if len(over):
        print(f"Words where our WLASL training clips already exceed the official count: "
              + ", ".join(f"{r.word} ({r.our_wlasl_train}/{r.official_train})" for r in over.itertuples()))
    print(f"v2-full training clips: {t.full_total.sum()}  "
          f"(WLASL {t.our_wlasl_train.sum()} + ASL Citizen {t.asc_used_full.sum()})")

    val = wl[wl["split"] == "val"]
    test = wl[wl["split"] == "test"]
    print(f"\nValidation (WLASL only, same in both sets): {len(val)} clips, {val.word.nunique()} words "
          f"(v1 words {int((val.source == 'wlasl_v1').sum())})")
    print(f"Test (WLASL only, same in both sets): {len(test)} clips, {test.word.nunique()} words "
          f"(v1 words {int((test.source == 'wlasl_v1').sum())})  <- not used until the one final test")
    no_asc = sorted(t.loc[t.asc_available == 0, "word"])
    print(f"Words with no ASL Citizen clips: {', '.join(no_asc) if no_asc else 'none'}")
    print(f"\nSaved {OUT_FULL}, {OUT_MATCHED} and {OUT_TARGETS}")


if __name__ == "__main__":
    main()