r"""
v2 step 4b: extract + clean the WLASL clips of the 20 NEW words (WLASL100 minus our v1 80).

Why: Week 2 only extracted the 80 v1 words. The 20 new words need their WLASL clips
for training, validation and test, and also for the automatic variant check.

Same settings as the 80 words, guaranteed: imports extract_clip() from
scripts\week2_extraction\extract_all_v2.py and clean_frame() from clean_hands.py.
v1/v2/v3 data of the 80 words is NOT touched; outputs go to a new folder.

Safe to stop and restart (skips clips that already have a .npy).

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\extract_wlasl_new20.py > report_outputs\V2_05_extract_wlasl_new.txt

Outputs:
  data\wlasl_new\landmarks_v2\<video_id>.npy     raw (same as WLASL v2)
  data\wlasl_new\landmarks_v3\<video_id>.npy     cleaned (same as WLASL v3)
  data\wlasl_new\landmarks_index_v3.csv          same columns as data\v3\landmarks_index_v3.csv (+ width, height)
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join("scripts", "week2_extraction"))
from extract_all_v2 import extract_clip, clip_stats          # noqa: E402
from clean_hands import clean_frame, hand_shares              # noqa: E402

WORD_LIST = os.path.join("data", "week1", "word_list.csv")
WLASL_JSON = os.path.join("wlasl-processed", "WLASL_v0.3.json")
VIDEO_DIR = os.path.join("wlasl-processed", "videos")
BASE = os.path.join("data", "wlasl_new")
RAW_DIR, V3_DIR = os.path.join(BASE, "landmarks_v2"), os.path.join(BASE, "landmarks_v3")
INDEX = os.path.join(BASE, "landmarks_index_v3.csv")
FAIL_LOG = os.path.join(BASE, "extraction_failures.txt")
MIN_HAND_SHARE = 0.20


def save_npy(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(V3_DIR, exist_ok=True)
    v1 = pd.read_csv(WORD_LIST)
    col = "word" if "word" in v1.columns else v1.columns[0]
    v1_words = set(v1[col].astype(str).str.strip())
    with open(WLASL_JSON, encoding="utf-8") as f:
        wlasl100 = json.load(f)[:100]
    new_words = [e["gloss"] for e in wlasl100 if e["gloss"] not in v1_words]
    print(f"New words ({len(new_words)}): {', '.join(new_words)}")

    clips = []
    for e in wlasl100:
        if e["gloss"] not in new_words:
            continue
        for inst in e["instances"]:
            vid = str(inst["video_id"])
            if os.path.exists(os.path.join(VIDEO_DIR, f"{vid}.mp4")):
                clips.append({"video_id": vid, "word": e["gloss"], "signer_id": inst.get("signer_id"),
                              "wlasl_split": inst.get("split"), "frame_start": inst.get("frame_start"),
                              "frame_end": inst.get("frame_end")})
    print(f"Clips to process: {len(clips)}")

    rows, failures = [], []
    for c in tqdm(clips, desc="Extracting", unit="clip", file=sys.stderr):
        vid = c["video_id"]
        path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        raw_path, v3_path = os.path.join(RAW_DIR, f"{vid}.npy"), os.path.join(V3_DIR, f"{vid}.npy")
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if os.path.exists(raw_path):
                raw = np.load(raw_path)
            else:
                raw, fps, _ = extract_clip(path)
                save_npy(raw_path, raw)
            clean = np.empty_like(raw)
            dups = disc = moved = 0
            for i, row in enumerate(raw):
                clean[i], d, m, u = clean_frame(row)
                disc, moved, dups = disc + d, moved + m, dups + u
            if not os.path.exists(v3_path):
                save_npy(v3_path, clean)
            rows.append({**c, "fps": round(fps, 2), "width": width, "height": height,
                         **clip_stats(raw), **hand_shares(clean),
                         "duplicates_removed": dups, "hands_discarded": disc, "hands_moved": moved})
        except Exception as e:
            failures.append(f"{vid}\t{c['word']}\t{type(e).__name__}: {e}")

    index = pd.DataFrame(rows)
    index["usable"] = index["any_hand_share"] >= MIN_HAND_SHARE
    index.to_csv(INDEX, index=False)
    if failures:
        with open(FAIL_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))

    print("\n========== SUMMARY ==========")
    print(f"Clips extracted: {len(index)} | failed: {len(failures)}")
    print(f"Usable clips: {index.usable.sum()} of {len(index)}")
    print(f"Detection: pose {index.pose_share.mean():.1%}, any hand (cleaned) {index.any_hand_share.mean():.1%}")
    print(f"fps values: {sorted(index.fps.unique().tolist())}")
    per_word = index[index.usable].groupby("word").agg(clips=("video_id", "count"),
                                                       signers=("signer_id", "nunique"))
    print("\nUsable WLASL clips per new word:")
    print(per_word.to_string())
    print(f"\nSaved {RAW_DIR}, {V3_DIR} and {INDEX}")


if __name__ == "__main__":
    main()