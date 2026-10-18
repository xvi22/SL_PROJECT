r"""
v2 step 4 (check): why is ASL Citizen extraction slow, and is shrinking frames safe?

What it does, in plain words:
  1. Looks at 40 random ASL Citizen videos and 40 random WLASL videos (no MediaPipe):
     size (pixels), frames, fps.  -> tells us WHY ASL Citizen is slower.
  2. For 6 ASL Citizen clips, extracts landmarks twice:
       a) full size (exactly as now)
       b) frames shrunk to 640 pixels on the long side
     and compares speed AND how different the coordinates are.
     If the coordinates are almost the same, shrinking is safe and much faster.

Takes about 3-5 minutes. Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\check_speed.py > report_outputs\V2_04a_check_speed.txt
"""
import os
os.environ["GLOG_minloglevel"] = "2"     # hide MediaPipe's harmless warning spam
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join("scripts", "week2_extraction"))
from extract_all_v2 import extract_clip, make_landmarkers, assign_hands   # noqa: E402

CAND = os.path.join("data", "asl_citizen", "candidates.csv")
WLASL_VIDEOS = os.path.join("wlasl-processed", "videos")
SHRINK_TO = 640
POSE_SLICE, LEFT_SLICE, RIGHT_SLICE = slice(0, 99), slice(99, 162), slice(162, 225)


def props(path):
    cap = cv2.VideoCapture(path)
    p = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
         "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 1)}
    cap.release()
    return p


def extract_clip_shrunk(video_path, max_side=SHRINK_TO):
    """Same as extract_clip() in extract_all_v2.py, but each frame is shrunk first.
    Coordinates are fractions of the image, so they keep the same meaning."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    hand_lm, pose_lm = make_landmarkers()
    rows, idx, last_ts = [], 0, -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = max(int(idx * 1000 / fps), last_ts + 1)
            last_ts = ts
            hand_res = hand_lm.detect_for_video(image, ts)
            pose_res = pose_lm.detect_for_video(image, ts)
            row = np.zeros(225, dtype=np.float32)
            pose_pts = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
            if pose_pts is not None:
                row[POSE_SLICE] = [v for p in pose_pts for v in (p.x, p.y, p.z)]
            labels = [hh[0].category_name for hh in hand_res.handedness]
            left, right = assign_hands(list(hand_res.hand_landmarks), labels, pose_pts)
            if left is not None:
                row[LEFT_SLICE] = [v for p in left for v in (p.x, p.y, p.z)]
            if right is not None:
                row[RIGHT_SLICE] = [v for p in right for v in (p.x, p.y, p.z)]
            rows.append(row)
            idx += 1
    finally:
        cap.release()
        hand_lm.close()
        pose_lm.close()
    return np.stack(rows)


def compare(a, b, width, height):
    """How far apart are the two versions? In shoulder widths (same unit as the cleaning rule)."""
    pa, pb = a[:, POSE_SLICE].reshape(-1, 33, 3), b[:, POSE_SLICE].reshape(-1, 33, 3)
    both = pa.any(axis=(1, 2)) & pb.any(axis=(1, 2))
    px = np.array([width, height])
    sw = np.median(np.linalg.norm((pa[both, 11, :2] - pa[both, 12, :2]) * px, axis=1))
    pose_gap = np.linalg.norm((pa[both, :, :2] - pb[both, :, :2]) * px, axis=2).mean() / sw
    hand_gaps = []
    for sl in (LEFT_SLICE, RIGHT_SLICE):
        ha, hb = a[:, sl].reshape(-1, 21, 3), b[:, sl].reshape(-1, 21, 3)
        m = ha.any(axis=(1, 2)) & hb.any(axis=(1, 2))
        if m.any():
            hand_gaps.append(np.linalg.norm((ha[m, :, :2] - hb[m, :, :2]) * px, axis=2).mean() / sw)
    hand_a = (a[:, LEFT_SLICE].any(1) | a[:, RIGHT_SLICE].any(1)).mean()
    hand_b = (b[:, LEFT_SLICE].any(1) | b[:, RIGHT_SLICE].any(1)).mean()
    return pose_gap, (np.mean(hand_gaps) if hand_gaps else np.nan), hand_a, hand_b


def main():
    rng = np.random.default_rng(42)
    cand = pd.read_csv(CAND)
    cand = cand[cand["downloaded"]].drop_duplicates("path")

    # ---- 1. Video properties --------------------------------------------------
    asc_paths = cand.sample(40, random_state=42)["path"].tolist()
    wl_files = sorted(os.listdir(WLASL_VIDEOS))
    wl_paths = [os.path.join(WLASL_VIDEOS, f) for f in rng.choice(wl_files, 40, replace=False)]
    stats = []
    for src, paths in [("ASL Citizen", asc_paths), ("WLASL", wl_paths)]:
        for p in tqdm(paths, desc=f"Reading {src} video info", unit="video", file=sys.stderr):
            stats.append({"source": src, **props(p)})
    stats = pd.DataFrame(stats)
    stats["pixels_k"] = stats.width * stats.height / 1000
    print("=" * 70)
    print("1. VIDEO PROPERTIES (40 random videos each)")
    print("=" * 70)
    print(stats.groupby("source")[["width", "height", "pixels_k", "frames", "fps"]]
          .median().round(1).to_string())
    for src in ["ASL Citizen", "WLASL"]:
        s = stats[stats.source == src]
        common = s.groupby(["width", "height"]).size().sort_values(ascending=False).head(4)
        print(f"{src} most common sizes: " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in common.items()))
    a, w = stats[stats.source == "ASL Citizen"], stats[stats.source == "WLASL"]
    print(f"\nASL Citizen vs WLASL (medians): {a.frames.median() / w.frames.median():.1f}x the frames, "
          f"{a.pixels_k.median() / w.pixels_k.median():.1f}x the pixels per frame")

    # ---- 2. Full size vs shrunk ------------------------------------------------
    print("\n" + "=" * 70)
    print(f"2. FULL SIZE vs SHRUNK TO {SHRINK_TO} px (6 ASL Citizen clips)")
    print("=" * 70)
    rows = []
    for p in tqdm(asc_paths[:6], desc="Timing full vs shrunk", unit="clip", file=sys.stderr):
        pr = props(p)
        t = time.time()
        full, _, _ = extract_clip(p)
        t_full = time.time() - t
        t = time.time()
        small = extract_clip_shrunk(p)
        t_small = time.time() - t
        pose_gap, hand_gap, hand_full, hand_small = compare(full, small, pr["width"], pr["height"])
        rows.append({"clip": os.path.basename(p)[:28], "size": f"{pr['width']}x{pr['height']}",
                     "frames": len(full), "sec_full": round(t_full, 1), "sec_shrunk": round(t_small, 1),
                     "pose_gap_sw": round(pose_gap, 3), "hand_gap_sw": round(hand_gap, 3),
                     "hand_full": round(hand_full, 3), "hand_shrunk": round(hand_small, 3)})
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    print(f"\nMean time per clip: full {r.sec_full.mean():.1f} s, shrunk {r.sec_shrunk.mean():.1f} s "
          f"({r.sec_full.mean() / max(r.sec_shrunk.mean(), 1e-9):.1f}x faster)")
    print(f"Mean coordinate difference (shoulder widths): pose {r.pose_gap_sw.mean():.3f}, "
          f"hands {r.hand_gap_sw.mean():.3f}")
    print("  For scale: the cleaning rule treats hands < 0.15 shoulder widths apart as the SAME hand;")
    print("  real hand-to-wrist distances in WLASL had a median of 0.08.")
    print(f"Hand found in: full {r.hand_full.mean():.1%} of frames, shrunk {r.hand_shrunk.mean():.1%}")
    est = 4421 * r.sec_shrunk.mean() / 3600
    print(f"\nEstimated total for 4,421 clips, one at a time: full {4421 * r.sec_full.mean() / 3600:.1f} h, "
          f"shrunk {est:.1f} h")
    try:
        cores = os.cpu_count()
        print(f"CPU cores on this laptop: {cores} (running several clips at once could cut this further)")
    except Exception:
        pass


if __name__ == "__main__":
    main()