r"""
extract_all.py - Week 2: extract landmarks for every clip of the 80 selected words.

Run from C:\SL_PROJECT with the venv active (after test_one_clip.py has worked,
so the model files are already in models\):
    python extract_all.py

Safe to stop and restart: clips that already have a .npy file are skipped.

Outputs:
    landmarks\<video_id>.npy      one array per clip, shape (frames, 225)
    landmarks_index.csv           one row per clip: word, signer, frames, detection rates
    extraction_failures.txt       clips that could not be processed (if any)

Feature layout per frame (225 numbers), same as test_one_clip.py:
    pose 33 x (x,y,z) = cols 0..98 | 'Left' hand 21 x 3 = cols 99..161 |
    'Right' hand 21 x 3 = cols 162..224. Undetected parts are stored as zeros.

Clips are stored WHOLE (no frame_start/frame_end trimming). Both values are
recorded in the index so trimming can be decided later without re-extracting.
"""

import json
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision
from tqdm import tqdm

# ---------------- settings ----------------
WORD_LIST = "data/week1/word_list.csv"
WLASL_JSON = os.path.join("wlasl-processed", "WLASL_v0.3.json")
VIDEO_DIR = os.path.join("wlasl-processed", "videos")
MODEL_DIR = "models"
OUT_DIR = "data/v1/landmarks"
INDEX_CSV = "data/v1/landmarks_index.csv"
FAIL_LOG = "data/v1/extraction_failures.txt"

# A clip counts as "usable" if a hand is found in at least this share of frames.
# Kept low on purpose: clips include resting frames with hands out of view.
MIN_HAND_SHARE = 0.20
MIN_CLIPS_PER_WORD = 8  # same floor as word selection

POSE_SLICE, LEFT_SLICE, RIGHT_SLICE = slice(0, 99), slice(99, 162), slice(162, 225)


def load_clip_list():
    """Return one dict per available clip of the selected words."""
    words_df = pd.read_csv(WORD_LIST)
    col = "word" if "word" in words_df.columns else words_df.columns[0]
    words = set(words_df[col].astype(str).str.strip())

    with open(WLASL_JSON, encoding="utf-8") as f:
        wlasl = json.load(f)

    clips = []
    for entry in wlasl:
        if entry["gloss"] not in words:
            continue
        for inst in entry["instances"]:
            vid = inst["video_id"]
            if os.path.exists(os.path.join(VIDEO_DIR, f"{vid}.mp4")):
                clips.append({
                    "video_id": vid,
                    "word": entry["gloss"],
                    "signer_id": inst.get("signer_id"),
                    "wlasl_split": inst.get("split"),  # original split, reference only
                    "frame_start": inst.get("frame_start"),
                    "frame_end": inst.get("frame_end"),
                })
    found = {c["word"] for c in clips}
    missing = words - found
    if missing:
        print(f"WARNING: no clips found for {len(missing)} words: {sorted(missing)}")
    return clips


def make_landmarkers():
    # A new pair per clip: VIDEO mode needs timestamps that only go up,
    # and every clip starts again at 0.
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return hand, pose


def extract_clip(video_path):
    """Return (features [frames, 225], fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hand_lm, pose_lm = make_landmarkers()
    rows, idx, last_ts = [], 0, -1
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            ts = max(int(idx * 1000 / fps), last_ts + 1)
            last_ts = ts
            hand_res = hand_lm.detect_for_video(image, ts)
            pose_res = pose_lm.detect_for_video(image, ts)

            row = np.zeros(225, dtype=np.float32)
            if pose_res.pose_landmarks:
                row[POSE_SLICE] = [v for p in pose_res.pose_landmarks[0] for v in (p.x, p.y, p.z)]
            for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
                vals = [v for p in lms for v in (p.x, p.y, p.z)]
                if handed[0].category_name == "Left":
                    row[LEFT_SLICE] = vals
                else:
                    row[RIGHT_SLICE] = vals
            rows.append(row)
            idx += 1
    finally:
        cap.release()
        hand_lm.close()
        pose_lm.close()
    if not rows:
        raise IOError("video has no readable frames")
    return np.stack(rows), fps


def clip_stats(features):
    pose_ok = features[:, POSE_SLICE].any(axis=1)
    left_ok = features[:, LEFT_SLICE].any(axis=1)
    right_ok = features[:, RIGHT_SLICE].any(axis=1)
    return {
        "num_frames": len(features),
        "pose_share": round(float(pose_ok.mean()), 3),
        "left_share": round(float(left_ok.mean()), 3),
        "right_share": round(float(right_ok.mean()), 3),
        "any_hand_share": round(float((left_ok | right_ok).mean()), 3),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = load_clip_list()
    print(f"Clips to process: {len(clips)} (expected 864)")

    rows, failures = [], []
    for clip in tqdm(clips, desc="Extracting", unit="clip"):
        vid = clip["video_id"]
        npy_path = os.path.join(OUT_DIR, f"{vid}.npy")
        video_path = os.path.join(VIDEO_DIR, f"{vid}.mp4")
        try:
            fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) or 25.0
            if os.path.exists(npy_path):  # already done on an earlier run
                features = np.load(npy_path)
            else:
                features, fps = extract_clip(video_path)
                np.save(npy_path, features)
            rows.append({**clip, "fps": round(fps, 2), **clip_stats(features)})
        except Exception as e:  # keep going; log the clip
            failures.append(f"{vid}\t{clip['word']}\t{e}")

    index = pd.DataFrame(rows)
    index["usable"] = index["any_hand_share"] >= MIN_HAND_SHARE
    index.to_csv(INDEX_CSV, index=False)
    if failures:
        with open(FAIL_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))

    # ---------------- summary for the report ----------------
    print("\n========== SUMMARY ==========")
    print(f"Clips extracted: {len(index)} | failed: {len(failures)}"
          + (f" (see {FAIL_LOG})" if failures else ""))
    print(f"Usable clips (hand in >= {MIN_HAND_SHARE:.0%} of frames): "
          f"{index['usable'].sum()} of {len(index)}")

    fr = index["num_frames"]
    print(f"\nClip length (frames): min {fr.min()}, median {int(fr.median())}, "
          f"90th percentile {int(fr.quantile(0.9))}, max {fr.max()}")
    print(f"Clip fps values: {sorted(index['fps'].unique().tolist())}")

    print(f"\nDetection (mean share of frames): pose {index['pose_share'].mean():.1%}, "
          f"any hand {index['any_hand_share'].mean():.1%}")

    trimmed = index[index["frame_end"].fillna(-1) != -1]
    print(f"\nClips with a frame_end value in the JSON: {len(trimmed)} of {len(index)}")
    if len(trimmed):
        expected = trimmed["frame_end"] - trimmed["frame_start"] + 1
        match = (abs(trimmed["num_frames"] - expected) <= 2).mean()
        print(f"  Their video length already matches frame_end - frame_start + 1: {match:.0%}"
              " (high = Kaggle videos are already cut; do NOT trim again)")

    per_word = index[index["usable"]].groupby("word").size()
    per_word = per_word.reindex(sorted(index["word"].unique()), fill_value=0)
    low = per_word[per_word < MIN_CLIPS_PER_WORD]
    print(f"\nUsable clips per word: min {per_word.min()}, mean {per_word.mean():.1f}")
    if len(low):
        print(f"Words below {MIN_CLIPS_PER_WORD} usable clips: " + ", ".join(
            f"{w} ({n})" for w, n in low.items()))
    else:
        print(f"No word dropped below {MIN_CLIPS_PER_WORD} usable clips.")

    worst = index.nsmallest(5, "any_hand_share")[["video_id", "word", "any_hand_share"]]
    print("\nLowest hand detection (check these by eye):")
    print(worst.to_string(index=False))
    print(f"\nSaved {OUT_DIR}\\*.npy and {INDEX_CSV}")


if __name__ == "__main__":
    main()