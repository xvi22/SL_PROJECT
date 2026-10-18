r"""
v2 step 6b: preprocess ALL clips of the 100 words (WLASL 80 + WLASL new 20 + ASL Citizen).

Same steps as v1, guaranteed: it IMPORTS preprocess_clip() from
scripts\week3_training\preprocess.py (pixel units -> trim to sign span +/- 3 -> fill gaps
<= 10 frames -> normalise to the shoulders -> resample to 64 frames -> 197 features).

Proof that nothing changed: the 80 v1 WLASL words are processed again and compared with
data\processed_v1\X.npy. They must come out identical (difference 0).

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\preprocess_v2.py > report_outputs\V2_09_preprocess_v2.txt

Outputs (data\processed_v2\):
  X.npy        (clips, 64, 197) float32
  y.npy        label per clip (100 words, alphabetical)
  meta.csv     uid, source, clip_id, word, gloss, signer, label, sizes, span info,
               asl_split, longest_gap (ASL Citizen only)
  labels.csv   label -> word
  config.json  settings (same as v1)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join("scripts", "week3_training"))
from preprocess import preprocess_clip, SEQ_LEN, MARGIN, MAX_GAP, POSE_KEEP  # noqa: E402

V1_PROCESSED = os.path.join("data", "processed_v1")
OUT_DIR = os.path.join("data", "processed_v2")
SOURCES = {
    "wlasl_v1": dict(index=os.path.join("data", "v3", "landmarks_index_v3.csv"),
                     npy=os.path.join("data", "v3", "landmarks_v3"), id="video_id",
                     signer="signer_id", sizes=os.path.join("data", "video_sizes.csv")),
    "wlasl_new20": dict(index=os.path.join("data", "wlasl_new", "landmarks_index_v3.csv"),
                        npy=os.path.join("data", "wlasl_new", "landmarks_v3"), id="video_id",
                        signer="signer_id", sizes=None),
    "asl_citizen": dict(index=os.path.join("data", "asl_citizen", "span_stats.csv"),
                        npy=os.path.join("data", "asl_citizen", "landmarks_v3"), id="clip_id",
                        signer="signer", sizes=None),
}


def load_source(name, cfg):
    idx = pd.read_csv(cfg["index"], dtype={cfg["id"]: str})
    idx = idx[idx["usable"]].reset_index(drop=True)
    if cfg["sizes"]:   # v1 WLASL: sizes come from the cache made by preprocess.py
        sz = pd.read_csv(cfg["sizes"], dtype={"video_id": str})
        idx = idx.drop(columns=[c for c in ("width", "height") if c in idx], errors="ignore")
        idx = idx.merge(sz, on="video_id", how="left")
    return idx


def main():
    X, rows, skipped = [], [], []
    for name, cfg in SOURCES.items():
        idx = load_source(name, cfg)
        for r in tqdm(idx.to_dict("records"), desc=f"Preprocessing {name}", unit="clip", file=sys.stderr):
            cid = str(r[cfg["id"]])
            w, h = r.get("width"), r.get("height")
            if not w or not h or pd.isna(w) or pd.isna(h):
                skipped.append((name, cid, "no video size"))
                continue
            raw = np.load(os.path.join(cfg["npy"], f"{cid}.npy"))
            feats, info = preprocess_clip(raw, int(w), int(h))
            if feats is None:
                skipped.append((name, cid, "no hands or no pose"))
                continue
            X.append(feats)
            rows.append({"uid": f"{name}:{cid}", "source": name, "clip_id": cid, "word": r["word"],
                         "gloss": r.get("gloss", r["word"]), "signer": r[cfg["signer"]],
                         "width": int(w), "height": int(h),
                         "asl_split": r.get("asl_split", ""), "longest_gap": r.get("longest_gap", np.nan),
                         **info})

    X = np.stack(X)
    meta = pd.DataFrame(rows)
    words = sorted(meta["word"].unique())
    label_of = {w: i for i, w in enumerate(words)}
    meta["label"] = meta["word"].map(label_of)
    y = meta["label"].to_numpy(dtype=np.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    meta.to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False)
    pd.DataFrame({"label": range(len(words)), "word": words}).to_csv(
        os.path.join(OUT_DIR, "labels.csv"), index=False)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"sources": {k: v["npy"] for k, v in SOURCES.items()}, "SEQ_LEN": SEQ_LEN,
                   "MARGIN": MARGIN, "MAX_GAP": MAX_GAP, "POSE_KEEP": POSE_KEEP,
                   "features": int(X.shape[2]), "same_as": "processed_v1 (preprocess_clip imported)"},
                  f, indent=2)

    print("========== PREPROCESSING SUMMARY (v2) ==========")
    print(f"Clips processed: {len(X)} | skipped: {len(skipped)}")
    for s in skipped[:15]:
        print(f"  skipped {s[0]} {s[1]}: {s[2]}")
    print(f"Array X: {X.shape} | words: {len(words)}")
    print("\nClips per source:")
    print(meta.groupby("source").agg(clips=("uid", "count"), words=("word", "nunique"),
                                     span_median=("span_frames", "median"),
                                     hand_after=("any_hand_share", "mean")).round(3).to_string())

    # ---- proof: v1 words unchanged ---------------------------------------------
    print("\n========== CHECK: v1 WLASL clips identical to processed_v1? ==========")
    v1_meta = pd.read_csv(os.path.join(V1_PROCESSED, "meta.csv"), dtype={"video_id": str})
    v1_X = np.load(os.path.join(V1_PROCESSED, "X.npy"))
    pos_v2 = {cid: i for i, cid in enumerate(meta["clip_id"]) if meta["source"].iat[i] == "wlasl_v1"}
    common = [(i, pos_v2[v]) for i, v in enumerate(v1_meta["video_id"]) if v in pos_v2]
    diff = max(float(np.abs(v1_X[i] - X[j]).max()) for i, j in common) if common else float("nan")
    print(f"v1 clips: {len(v1_meta)} | found in v2: {len(common)} | largest difference: {diff:.6f}")
    print("IDENTICAL - preprocessing unchanged" if len(common) == len(v1_meta) and diff == 0
          else "WARNING: v1 clips differ - tell Claude before going on")
    print(f"\nSaved {OUT_DIR}\\X.npy, y.npy, meta.csv, labels.csv, config.json")


if __name__ == "__main__":
    main()