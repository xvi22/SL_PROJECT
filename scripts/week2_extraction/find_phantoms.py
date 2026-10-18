r"""
find_phantoms.py - find "phantom hands" in the v2 landmarks (read-only check).

A phantom hand is a false detection, e.g. part of the face detected as a hand
(seen in clip 24636, "give"). Test: a real hand's wrist (hand point 0) should be
close to the matching pose wrist (pose 15 for the left slot, 16 for the right).

Distance is measured in SHOULDER WIDTHS, so it does not depend on how far the
signer is from the camera. Nothing is changed; this only measures.

Run from C:\SL_PROJECT:
    python find_phantoms.py                     # the numbers for all clips
    python find_phantoms.py 64088 17016 34733   # also draw the most suspect frames

Saves phantom_stats.csv (one row per clip). With video IDs, it also saves
phantoms\<video_id>_phantoms.png showing that clip's 3 most suspect frames.
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANDMARK_DIR = "data/v2/landmarks_v2"
VIDEO_DIR = os.path.join("wlasl-processed", "videos")
OUT_DIR = "data/v2/phantoms"
MIN_CLIPS_PER_WORD = 8
INDEX_CSV = "data/v2/landmarks_index_v2.csv"
THRESHOLDS = [0.4, 0.6, 1.0]  # shoulder widths; we pick one after seeing the results
FLAG_AT = 0.6                 # used for the per-clip table below

SLOTS = {"left": (slice(99, 162), 15), "right": (slice(162, 225), 16)}


def hand_wrist_distances(features):
    """Return an array of distances (shoulder widths), one per detected hand."""
    pose = features[:, 0:99].reshape(-1, 33, 3)
    shoulder = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)
    out = []
    for sl, pose_wrist in SLOTS.values():
        hand = features[:, sl].reshape(-1, 21, 3)
        found = hand.any(axis=(1, 2)) & pose.any(axis=(1, 2)) & (shoulder > 0)
        d = np.linalg.norm(hand[found, 0, :2] - pose[found, pose_wrist, :2], axis=1)
        out.append(d / shoulder[found])
    return np.concatenate(out)


def frame_distances(features):
    """Per frame, the larger hand-to-wrist distance (shoulder widths), or 0 if no hand."""
    pose = features[:, 0:99].reshape(-1, 33, 3)
    shoulder = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)
    worst = np.zeros(len(features))
    for sl, pose_wrist in SLOTS.values():
        hand = features[:, sl].reshape(-1, 21, 3)
        ok = hand.any(axis=(1, 2)) & pose.any(axis=(1, 2)) & (shoulder > 0)
        d = np.zeros(len(features))
        d[ok] = np.linalg.norm(hand[ok, 0, :2] - pose[ok, pose_wrist, :2], axis=1) / shoulder[ok]
        worst = np.maximum(worst, d)
    return worst


def draw_suspects(video_id, word):
    features = np.load(os.path.join(LANDMARK_DIR, f"{video_id}.npy"))
    worst = frame_distances(features)
    picks = [int(i) for i in np.argsort(worst)[::-1][:3] if worst[i] > FLAG_AT]
    if not picks:
        print(f"{video_id}: no frames above {FLAG_AT}")
        return
    picks.sort()
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"{video_id}.mp4"))
    frames, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in picks:
            frames[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5), squeeze=False)
    for ax, i in zip(axes[0], picks):
        h, w = frames[i].shape[:2]
        ax.imshow(frames[i])
        pose = features[i, 0:99].reshape(33, 3)
        for a, b in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
            ax.plot([pose[a, 0] * w, pose[b, 0] * w], [pose[a, 1] * h, pose[b, 1] * h], color="yellow", lw=1.5)
        for name, (sl, pw) in SLOTS.items():
            colour = "cyan" if name == "left" else "magenta"
            ax.scatter([pose[pw, 0] * w], [pose[pw, 1] * h], s=120, facecolors="none",
                       edgecolors=colour, linewidths=2.5)  # ring = pose wrist
            hand = features[i, sl].reshape(21, 3)
            if hand.any():
                ax.scatter(hand[:, 0] * w, hand[:, 1] * h, s=8, color=colour)  # dots = hand
        ax.set_title(f"frame {i}: {worst[i]:.1f} shoulder widths")
        ax.axis("off")
    fig.suptitle(f"{video_id} ({word}) - rings = pose wrists, dots = detected hand "
                 f"(cyan = left slot, magenta = right slot)")
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{video_id}_phantoms.png")
    plt.savefig(out, dpi=90)
    plt.close(fig)
    print(f"{video_id} ({word}): frames {picks} -> {out}")


def main():
    index = pd.read_csv(INDEX_CSV, dtype={"video_id": str})
    all_d, rows = [], []
    for vid, word in zip(index["video_id"], index["word"]):
        d = hand_wrist_distances(np.load(os.path.join(LANDMARK_DIR, f"{vid}.npy")))
        all_d.append(d)
        rows.append({
            "video_id": vid, "word": word, "hand_detections": len(d),
            "phantom": int((d > FLAG_AT).sum()),
            "phantom_share": round(float((d > FLAG_AT).mean()), 3) if len(d) else 0.0,
            "median_dist": round(float(np.median(d)), 3) if len(d) else np.nan,
        })
    all_d = np.concatenate(all_d)
    df = pd.DataFrame(rows)
    df.to_csv("data/v2/phantom_stats.csv", index=False)

    print("========== PHANTOM HAND CHECK (v2) ==========")
    print(f"Hand detections checked: {len(all_d)}")
    p = np.percentile(all_d, [50, 90, 95, 99])
    print(f"Hand wrist to pose wrist (shoulder widths): median {p[0]:.2f}, "
          f"90th {p[1]:.2f}, 95th {p[2]:.2f}, 99th {p[3]:.2f}")
    print("\nDetections further than:")
    for t in THRESHOLDS:
        n = int((all_d > t).sum())
        clips = int(sum((np.asarray(r["phantom"]) > 0) for r in rows)) if t == FLAG_AT else None
        print(f"  {t:.1f} shoulder widths: {n} ({n / len(all_d):.1%})"
              + (f"  in {clips} clips" if clips is not None else ""))

    bad = df[df["phantom_share"] > 0.5]
    print(f"\nClips where over half the hand detections are > {FLAG_AT} away: {len(bad)}")
    print("\nClips with the most suspect detections:")
    print(df.nlargest(10, "phantom")[["video_id", "word", "hand_detections", "phantom",
                                       "phantom_share"]].to_string(index=False))
    words_bad = sorted(set(bad["word"]))
    if words_bad:
        print("\nIf those mostly-phantom clips were dropped:")
        for w in words_bad:
            n = int((index["word"] == w).sum())
            k = int((bad["word"] == w).sum())
            note = "OK" if n - k >= MIN_CLIPS_PER_WORD else f"BELOW {MIN_CLIPS_PER_WORD}"
            print(f"  {w}: {n} clips -> {n - k} ({note})")
    print("\nSaved phantom_stats.csv")

    ids = sys.argv[1:]
    if ids:
        print("\n========== SUSPECT FRAMES ==========")
        words = dict(zip(df["video_id"], df["word"]))
        for vid in ids:
            draw_suspects(vid, words.get(vid, "?"))


if __name__ == "__main__":
    main()