r"""
preprocess.py - Weeks 3-4 groundwork: turn the v3 landmarks into training sequences.

Run from C:\SL_PROJECT (about 1-2 minutes):
    python preprocess.py                  # process all usable v3 clips
    python preprocess.py 69241 17016      # also draw check images for these clips

Steps for every clip (the order agreed in Week 2):
  1. pixel units      x * width, y * height, z * width (so x and y use the same scale;
                      WLASL videos have different aspect ratios)
  2. trim             keep the sign span (first to last frame with a hand) + MARGIN frames
  3. fill gaps        hand gaps of up to MAX_GAP frames are filled by straight-line
                      interpolation; missing pose frames copy the nearest pose frame
  4. normalise        centre x, y on the shoulder midpoint and divide by shoulder width
                      (per-clip medians), so position on screen and distance from the
                      camera no longer matter
  5. resample         stretch or squeeze every clip to SEQ_LEN frames (handles mixed fps
                      and signing speed)
  6. features         pose points POSE_KEEP + both hands + 2 "hand present" flags

Outputs (in processed_v1\):
    X.npy          shape (clips, SEQ_LEN, features), float32
    y.npy          label number per clip
    meta.csv       one row per clip: video_id, word, label, signer, span, hand shares
    labels.csv     label number -> word
    config.json    the settings used (the live pipeline must use the same ones)
    checks\        check images for chosen clips

The live pipeline (Week 5) can reuse preprocess_clip() with trim=False, because live
recording already starts at the movement and the hold frames are dropped.
"""

import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- settings ----------------
IN_DIR, IN_INDEX = "data/v3/landmarks_v3", "data/v3/landmarks_index_v3.csv"
VIDEO_DIR = os.path.join("wlasl-processed", "videos")
SIZES_CSV = "data/video_sizes.csv"          # cache of video width/height
OUT_DIR = "data/processed_v1"

SEQ_LEN = 64        # frames per sequence after resampling (sign spans: median 40, p90 60)
MARGIN = 3          # extra frames kept before and after the sign span
MAX_GAP = 10        # longest hand gap (frames) filled by interpolation
POSE_KEEP = list(range(0, 23))   # head, shoulders, arms; drops hips/legs (23-32),
                                 # which are mostly off-screen guesses in WLASL

N_POSE, N_HAND = 33, 21
POSE_SL, LEFT_SL, RIGHT_SL = slice(0, 99), slice(99, 162), slice(162, 225)
L_SHOULDER, R_SHOULDER = 11, 12


# ---------------- per-clip processing ----------------
def to_pixels(raw, width, height):
    """(frames, 225) fractions -> (frames, 75, 3) pixels; undetected points stay 0."""
    pts = raw.reshape(len(raw), 75, 3).astype(np.float64).copy()
    found = pts.any(axis=2)
    pts[..., 0] *= width
    pts[..., 1] *= height
    pts[..., 2] *= width
    pts[~found] = 0
    return pts


def fill_pose(pose, ok):
    """Copy the nearest frame with a pose into frames without one."""
    if ok.all() or not ok.any():
        return pose
    idx = np.flatnonzero(ok)
    nearest = idx[np.abs(np.arange(len(pose))[:, None] - idx[None, :]).argmin(axis=1)]
    return pose[nearest]


def fill_hand_gaps(hand, ok, max_gap):
    """Linear interpolation over gaps of up to max_gap frames between detections."""
    hand, ok = hand.copy(), ok.copy()
    idx = np.flatnonzero(ok)
    filled = 0
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                hand[a + k] = (1 - t) * hand[a] + t * hand[b]
            ok[a + 1:b] = True
            filled += gap
    return hand, ok, filled


def resample(arr, ok, n):
    """Resample (frames, ...) to n frames. Linear between two valid frames,
    otherwise the nearest frame; frames with nothing valid nearby become 0."""
    T = len(arr)
    if T == 1:
        return np.repeat(arr, n, axis=0), np.repeat(ok, n)
    pos = np.linspace(0, T - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    a = (pos - i0).reshape(-1, *([1] * (arr.ndim - 1)))
    both = ok[i0] & ok[i1]
    near = np.where(pos - i0 < 0.5, i0, i1)
    out = np.where(both.reshape(a.shape), (1 - a) * arr[i0] + a * arr[i1], arr[near])
    out_ok = both | ok[near]
    out[~out_ok] = 0
    return out, out_ok


def preprocess_clip(raw, width, height, trim=True):
    """Return (features [SEQ_LEN, F], info dict) or (None, info) if no hands at all."""
    pts = to_pixels(raw, width, height)
    pose, left, right = pts[:, :33], pts[:, 33:54], pts[:, 54:75]
    pose_ok = pose.any(axis=(1, 2))
    left_ok, right_ok = left.any(axis=(1, 2)), right.any(axis=(1, 2))
    hand_ok = left_ok | right_ok
    info = {"frames": len(raw)}
    if not hand_ok.any() or not pose_ok.any():
        return None, info

    # 2. trim to the sign span
    if trim:
        idx = np.flatnonzero(hand_ok)
        s, e = max(idx[0] - MARGIN, 0), min(idx[-1] + MARGIN, len(raw) - 1)
    else:
        s, e = 0, len(raw) - 1
    pose, left, right = pose[s:e + 1], left[s:e + 1], right[s:e + 1]
    pose_ok, left_ok, right_ok = pose_ok[s:e + 1], left_ok[s:e + 1], right_ok[s:e + 1]
    info.update(span_start=int(s), span_end=int(e), span_frames=int(e - s + 1))

    # 3. fill gaps
    pose = fill_pose(pose, pose_ok)
    left, left_ok, fl = fill_hand_gaps(left, left_ok, MAX_GAP)
    right, right_ok, fr = fill_hand_gaps(right, right_ok, MAX_GAP)
    info["frames_filled"] = int(fl + fr)

    # 4. normalise (per-clip medians: steadier than frame by frame)
    mid = (pose[:, L_SHOULDER, :2] + pose[:, R_SHOULDER, :2]) / 2
    width_sh = np.linalg.norm(pose[:, L_SHOULDER, :2] - pose[:, R_SHOULDER, :2], axis=1)
    centre = np.median(mid, axis=0)
    scale = float(np.median(width_sh[width_sh > 0])) if (width_sh > 0).any() else 1.0
    for part, ok in ((pose, np.ones(len(pose), bool)), (left, left_ok), (right, right_ok)):
        part[..., :2] -= centre
        part /= scale
        part[~ok] = 0

    # 5. resample
    pose, _ = resample(pose, np.ones(len(pose), bool), SEQ_LEN)
    left, left_ok = resample(left, left_ok, SEQ_LEN)
    right, right_ok = resample(right, right_ok, SEQ_LEN)

    # 6. features
    feats = np.concatenate([
        pose[:, POSE_KEEP].reshape(SEQ_LEN, -1),
        left.reshape(SEQ_LEN, -1),
        right.reshape(SEQ_LEN, -1),
        left_ok[:, None].astype(float),
        right_ok[:, None].astype(float),
    ], axis=1).astype(np.float32)
    info.update(left_share=round(float(left_ok.mean()), 3),
                right_share=round(float(right_ok.mean()), 3),
                any_hand_share=round(float((left_ok | right_ok).mean()), 3))
    return feats, info


# ---------------- helpers ----------------
def video_sizes(video_ids):
    """Width/height per video, cached in video_sizes.csv after the first run."""
    sizes = {}
    if os.path.exists(SIZES_CSV):
        df = pd.read_csv(SIZES_CSV, dtype={"video_id": str})
        sizes = {v: (w, h) for v, w, h in zip(df["video_id"], df["width"], df["height"])}
    missing = [v for v in video_ids if v not in sizes]
    for v in tqdm(missing, desc="Reading video sizes", unit="video", disable=not missing):
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"{v}.mp4"))
        sizes[v] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    if missing:
        pd.DataFrame([(v, w, h) for v, (w, h) in sizes.items()],
                     columns=["video_id", "width", "height"]).to_csv(SIZES_CSV, index=False)
    return sizes


def draw_check(vid, word, feats, out_dir):
    """Processed skeleton at the start, middle and end of the sequence."""
    n_pose = len(POSE_KEEP)
    pose = feats[:, :n_pose * 3].reshape(SEQ_LEN, n_pose, 3)
    left = feats[:, n_pose * 3:n_pose * 3 + 63].reshape(SEQ_LEN, 21, 3)
    right = feats[:, n_pose * 3 + 63:n_pose * 3 + 126].reshape(SEQ_LEN, 21, 3)
    arm = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    has_hand = np.flatnonzero(feats[:, -2:].any(axis=1))
    steps = [has_hand[0], has_hand[len(has_hand) // 2], has_hand[-1]] if len(has_hand) else \
        [0, SEQ_LEN // 2, SEQ_LEN - 1]
    for ax, t in zip(axes, steps):
        for a, b in arm:
            ax.plot(pose[t, [a, b], 0], pose[t, [a, b], 1], color="goldenrod", lw=2)
        ax.scatter(pose[t, :11, 0], pose[t, :11, 1], s=6, color="goldenrod")  # face points
        for h, c in ((left, "c"), (right, "m")):
            if h[t].any():
                ax.scatter(h[t, :, 0], h[t, :, 1], s=8, color=c)
        ax.set_xlim(-2, 2)
        ax.set_ylim(2, -2.5)  # y grows downwards, as in the image
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(f"step {t} of {SEQ_LEN}")
    fig.suptitle(f"{vid} ({word}) after preprocessing: centred on the shoulders, "
                 f"1 unit = shoulder width (cyan = left slot, magenta = right slot)")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{vid}_processed.png")
    plt.savefig(path, dpi=90)
    plt.close(fig)
    return path


# ---------------- main ----------------
def main():
    index = pd.read_csv(IN_INDEX, dtype={"video_id": str})
    index = index[index["usable"]].reset_index(drop=True)
    sizes = video_sizes(index["video_id"].tolist())
    words = sorted(index["word"].unique())
    label_of = {w: i for i, w in enumerate(words)}
    draw_ids = set(sys.argv[1:])

    X, y, rows, skipped, drawn = [], [], [], [], []
    for _, clip in tqdm(index.iterrows(), total=len(index), desc="Preprocessing", unit="clip"):
        vid = clip["video_id"]
        w, h = sizes[vid]
        if not w or not h:
            skipped.append((vid, "no video size"))
            continue
        raw = np.load(os.path.join(IN_DIR, f"{vid}.npy"))
        feats, info = preprocess_clip(raw, w, h)
        if feats is None:
            skipped.append((vid, "no hands or no pose"))
            continue
        X.append(feats)
        y.append(label_of[clip["word"]])
        rows.append({"video_id": vid, "word": clip["word"], "label": label_of[clip["word"]],
                     "signer_id": clip["signer_id"], "width": w, "height": h, **info})
        if vid in draw_ids:
            drawn.append(draw_check(vid, clip["word"], feats, os.path.join(OUT_DIR, "checks")))

    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = np.stack(X), np.array(y, dtype=np.int64)
    meta = pd.DataFrame(rows)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    meta.to_csv(os.path.join(OUT_DIR, "meta.csv"), index=False)
    pd.DataFrame({"label": range(len(words)), "word": words}).to_csv(
        os.path.join(OUT_DIR, "labels.csv"), index=False)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"source": IN_DIR, "SEQ_LEN": SEQ_LEN, "MARGIN": MARGIN, "MAX_GAP": MAX_GAP,
                   "POSE_KEEP": POSE_KEEP, "features": int(X.shape[2]),
                   "feature_layout": f"pose {len(POSE_KEEP)}x3, left hand 21x3, right hand 21x3, "
                                     "left present, right present",
                   "units": "x,y centred on shoulder midpoint; all divided by shoulder width"},
                  f, indent=2)

    print("\n========== PREPROCESSING SUMMARY ==========")
    print(f"Clips processed: {len(X)} | skipped: {len(skipped)}")
    for vid, why in skipped:
        print(f"  skipped {vid}: {why}")
    print(f"Array X: {X.shape}  (clips, frames, features) | words: {len(words)}")
    print(f"Sign span used (frames, incl. margin): median {int(meta['span_frames'].median())}, "
          f"max {meta['span_frames'].max()} -> all resampled to {SEQ_LEN}")
    print(f"Hand frames filled by interpolation: {meta['frames_filled'].sum()} "
          f"(gaps up to {MAX_GAP} frames)")
    print(f"Frames with a hand after processing: mean {meta['any_hand_share'].mean():.1%}")
    per_word = meta.groupby("word").size()
    print(f"Clips per word: min {per_word.min()}, mean {per_word.mean():.1f}")
    print(f"Value range after normalising: {X[..., :-2].min():.2f} to {X[..., :-2].max():.2f} "
          "(shoulder widths)")
    print(f"Video sizes: {meta.groupby(['width', 'height']).size().to_dict()}")
    print(f"\nSaved {OUT_DIR}\\X.npy, y.npy, meta.csv, labels.csv, config.json")
    for p in drawn:
        print(f"Check image: {p}")


if __name__ == "__main__":
    main()