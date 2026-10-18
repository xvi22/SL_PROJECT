r"""
v2 step 4: extract + clean landmarks for the ASL Citizen candidate clips.

What it does, in plain words:
  For every downloaded ASL Citizen video (all versions, e.g. DRINK1 AND DRINK2):
    1. extract the 225 landmark numbers per frame  -> data\asl_citizen\landmarks\    (like WLASL v2)
    2. clean the hands with the v3 rule             -> data\asl_citizen\landmarks_v3\ (like WLASL v3)
  The wrong versions are dropped LATER (after the automatic variant check).

Same settings as WLASL, guaranteed: this script does not copy the code. It IMPORTS
extract_clip() from scripts\week2_extraction\extract_all_v2.py and clean_frame() from
scripts\week2_extraction\clean_hands.py, so the exact same extraction and cleaning rule is used.

FAST: several clips are processed at the same time (default 8 workers; the laptop has
24 CPU cores). Each clip is still extracted exactly the same way, so the data is identical.

Safe to stop and restart: clips that already have a .npy are skipped.
(Files are saved under a temporary name first, so a crash never leaves a broken .npy.)

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\extract_asl_citizen.py > report_outputs\V2_04_extract_asl_citizen.txt
  (optional: add a number at the end to change the workers, e.g. ... extract_asl_citizen.py 12)

Outputs:
  data\asl_citizen\landmarks\<clip_id>.npy        raw, shape (frames, 225)
  data\asl_citizen\landmarks_v3\<clip_id>.npy     cleaned, same shape
  data\asl_citizen\landmarks_index_v3.csv         one row per clip (word, gloss, signer, sizes, detection rates, usable)
  data\asl_citizen\extraction_failures.txt        clips that could not be processed (if any)
"""
import os
os.environ["GLOG_minloglevel"] = "2"     # hide MediaPipe's harmless warning spam
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join("scripts", "week2_extraction"))
from extract_all_v2 import extract_clip, clip_stats          # noqa: E402  (same as WLASL v2)
from clean_hands import clean_frame, hand_shares              # noqa: E402  (same v3 rule)

BASE = os.path.join("data", "asl_citizen")
CAND = os.path.join(BASE, "candidates.csv")
RAW_DIR = os.path.join(BASE, "landmarks")
V3_DIR = os.path.join(BASE, "landmarks_v3")
INDEX = os.path.join(BASE, "landmarks_index_v3.csv")
FAIL_LOG = os.path.join(BASE, "extraction_failures.txt")
MIN_HAND_SHARE = 0.20      # same usable bar as WLASL


def save_npy(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def clean_clip(raw):
    out = np.empty_like(raw)
    disc = moved = dups = 0
    for i, row in enumerate(raw):
        out[i], d, m, u = clean_frame(row)
        disc += d
        moved += m
        dups += u
    return out, dups, disc, moved


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = (cap.get(cv2.CAP_PROP_FPS) or 25.0,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def process_clip(c):
    """Extract + clean ONE clip. Runs inside a worker. Returns (row, error, newly_extracted)."""
    raw_path = os.path.join(RAW_DIR, f"{c['clip_id']}.npy")
    v3_path = os.path.join(V3_DIR, f"{c['clip_id']}.npy")
    try:
        fps, width, height = video_info(c["path"])
        new = False
        if os.path.exists(raw_path):
            raw = np.load(raw_path)
        else:
            raw, fps, _ = extract_clip(c["path"])
            save_npy(raw_path, raw)
            new = True
        clean, dups, disc, moved = clean_clip(raw)
        if not os.path.exists(v3_path):
            save_npy(v3_path, clean)
        raw_stats = clip_stats(raw)
        row = {
            "clip_id": c["clip_id"], "word": c["word"], "gloss": c["gloss"], "match_type": c["match_type"],
            "signer": c["signer"], "asl_split": c["asl_split"], "video": c["video"],
            "fps": round(fps, 2), "width": width, "height": height,
            "num_frames": raw_stats["num_frames"], "pose_share": raw_stats["pose_share"],
            "raw_any_hand_share": raw_stats["any_hand_share"],
            **hand_shares(clean),
            "duplicates_removed": dups, "hands_discarded": disc, "hands_moved": moved,
        }
        return row, None, new
    except Exception as e:
        return None, f"{c['clip_id']}\t{c['word']}\t{c['gloss']}\t{type(e).__name__}: {e}", False


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(V3_DIR, exist_ok=True)
    cand = pd.read_csv(CAND)
    cand = cand[cand["downloaded"]].copy()
    cand["clip_id"] = cand["video"].map(lambda v: os.path.splitext(os.path.basename(v))[0])
    already = sum(os.path.exists(os.path.join(V3_DIR, f"{c}.npy")) for c in cand.clip_id)
    print(f"Clips: {len(cand)} ({cand.word.nunique()} words, {cand.gloss.nunique()} glosses). "
          f"Already done: {already}. To do: {len(cand) - already}. Workers: {workers}")

    jobs = cand[["clip_id", "word", "gloss", "match_type", "signer", "asl_split", "video", "path"]] \
        .to_dict("records")
    rows, failures = [], []
    t0, n_new = time.time(), 0
    with Pool(workers) as pool:
        bar = tqdm(pool.imap_unordered(process_clip, jobs, chunksize=1), total=len(jobs),
                   desc="Extracting", unit="clip", file=sys.stderr)
        for row, err, new in bar:
            if row is not None:
                rows.append(row)
            if err is not None:
                failures.append(err)
            n_new += new
            if n_new:
                bar.set_postfix(new=n_new, failed=len(failures))

    index = pd.DataFrame(rows)
    index["usable"] = index["any_hand_share"] >= MIN_HAND_SHARE
    index.to_csv(INDEX, index=False)
    if failures:
        with open(FAIL_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(failures))

    # ---------------- summary ----------------
    print("\n========== SUMMARY ==========")
    print(f"Clips extracted: {len(index)} | failed: {len(failures)}"
          + (f" (see {FAIL_LOG}; run again to retry)" if failures else ""))
    if n_new:
        print(f"Newly extracted this run: {n_new} in {(time.time() - t0) / 60:.0f} min "
              f"({(time.time() - t0) / n_new:.2f} s per clip overall, {workers} workers)")
    print(f"Usable clips (hand in >= {MIN_HAND_SHARE:.0%} of frames): "
          f"{index.usable.sum()} of {len(index)}")
    fr = index.num_frames
    print(f"\nClip length (frames): min {fr.min()}, median {int(fr.median())}, "
          f"90th percentile {int(fr.quantile(0.9))}, max {fr.max()}")
    print(f"fps values: {sorted(index.fps.round(0).unique().tolist())}")
    sizes = index.groupby(["width", "height"]).size().sort_values(ascending=False)
    print(f"Video sizes (most common first): " +
          ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.head(6).items()))
    print(f"\nDetection (mean share of frames): pose {index.pose_share.mean():.1%}, "
          f"any hand raw {index.raw_any_hand_share.mean():.1%} -> cleaned {index.any_hand_share.mean():.1%}")
    print(f"(WLASL for comparison: pose 99.7%, any hand 66.4% after cleaning)")
    print(f"Cleaning: duplicates removed {index.duplicates_removed.sum()} in "
          f"{(index.duplicates_removed > 0).sum()} clips, hands discarded {index.hands_discarded.sum()} in "
          f"{(index.hands_discarded > 0).sum()} clips, moved {index.hands_moved.sum()} in "
          f"{(index.hands_moved > 0).sum()} clips")
    low_pose = index[index.pose_share < 0.9]
    print(f"Clips with pose in < 90% of frames (body cropped / too close?): {len(low_pose)}")

    per_gloss = index[index.usable].groupby(["word", "gloss"]).agg(
        usable=("clip_id", "count"), signers=("signer", "nunique"))
    print(f"\nUsable clips per gloss: min {per_gloss.usable.min()}, median {per_gloss.usable.median():.0f}")
    lost = index[~index.usable]
    if len(lost):
        print(f"Unusable clips per word: " + ", ".join(
            f"{w} ({n})" for w, n in lost.word.value_counts().items()))
    worst = index.nsmallest(5, "any_hand_share")[["clip_id", "word", "gloss", "any_hand_share", "pose_share"]]
    print("\nLowest hand detection (check these by eye):")
    print(worst.to_string(index=False))
    print(f"\nSaved {RAW_DIR}, {V3_DIR} and {INDEX}")


if __name__ == "__main__":
    main()