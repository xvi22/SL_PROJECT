r"""
check_clips_v2.py - Week 2 sanity check of the v2 extracted landmarks.

Run from C:\SL_PROJECT (after extract_all_v2.py):
    python check_clips_v2.py                 # checks the 5 lowest clips
    python check_clips_v2.py 26712 06471     # or any video IDs you choose

Reads the v2 extraction (landmarks_v2\, landmarks_index_v2.csv) and writes to
span_stats_v2.csv and checks_v2\. The original check_clips.py is kept unchanged
for the v1 extraction (landmarks\, checks\).

Part 1 (all clips, from the .npy files only):
    "Sign span" = first frame with a hand to last frame with a hand.
    Prints how much of each clip is resting time, and how well hands are
    tracked INSIDE the sign span (this is what matters for training).

Part 2 (chosen clips): saves checks\<video_id>_check.png with
    - a detection timeline (pose / 'Left' hand / 'Right' hand per frame)
    - landmarks drawn on 3 frames inside the sign span
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = "v2"
LANDMARK_DIR, INDEX_CSV, OUT_DIR, SPAN_CSV = "data/v2/landmarks_v2", "data/v2/landmarks_index_v2.csv", "data/v2/checks_v2", "data/v2/span_stats_v2.csv"
VIDEO_DIR = os.path.join("wlasl-processed", "videos")
DEFAULT_IDS = ["26712", "32325", "06471", "64283", "05749"]  # lowest 5 from the summary

POSE_SLICE, LEFT_SLICE, RIGHT_SLICE = slice(0, 99), slice(99, 162), slice(162, 225)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def span_stats(features):
    """Detection inside the sign span (first to last frame with any hand)."""
    left = features[:, LEFT_SLICE].any(axis=1)
    right = features[:, RIGHT_SLICE].any(axis=1)
    hand = left | right
    idx = np.flatnonzero(hand)
    if len(idx) == 0:
        return None
    start, end = idx[0], idx[-1]
    inside = hand[start:end + 1]
    # longest run of frames with no hand inside the span
    gaps, run = [0], 0
    for h in inside:
        run = 0 if h else run + 1
        gaps.append(run)
    return {
        "frames": len(features),
        "span_start": int(start),
        "span_end": int(end),
        "span_frames": int(end - start + 1),
        "rest_share": round(1 - (end - start + 1) / len(features), 3),
        "hand_in_span": round(float(inside.mean()), 3),
        "longest_gap": int(max(gaps)),
    }


def part1_all_clips():
    index = pd.read_csv(INDEX_CSV, dtype={"video_id": str})
    rows = []
    for vid, word in zip(index["video_id"], index["word"]):
        s = span_stats(np.load(os.path.join(LANDMARK_DIR, f"{vid}.npy")))
        if s:
            rows.append({"video_id": vid, "word": word, **s})
    df = pd.DataFrame(rows)
    df.to_csv(SPAN_CSV, index=False)

    print(f"========== PART 1: ALL CLIPS ({VERSION}) ==========")
    print(f"Resting time (outside the sign span): mean {df['rest_share'].mean():.1%} of each clip")
    print(f"Sign span length (frames): min {df['span_frames'].min()}, "
          f"median {int(df['span_frames'].median())}, "
          f"90th percentile {int(df['span_frames'].quantile(0.9))}, max {df['span_frames'].max()}")
    print(f"Hand detected INSIDE the sign span: mean {df['hand_in_span'].mean():.1%}")
    weak = df[df["hand_in_span"] < 0.8].sort_values("hand_in_span")
    print(f"Clips with hands found in under 80% of their sign span: {len(weak)}")
    if len(weak):
        print(weak[["video_id", "word", "hand_in_span", "longest_gap"]].head(10).to_string(index=False))
    print(f"Saved {SPAN_CSV}\n")
    return df


def read_frames(video_id, wanted):
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"{video_id}.mp4"))
    frames, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return frames


def part2_check(video_id, word):
    features = np.load(os.path.join(LANDMARK_DIR, f"{video_id}.npy"))
    s = span_stats(features)
    n = len(features)
    a, b = (s["span_start"], s["span_end"]) if s else (0, n - 1)
    picks = sorted({a, (a + b) // 2, b})
    frames = read_frames(video_id, set(picks))

    fig = plt.figure(figsize=(15, 7))
    for k, i in enumerate(picks):
        ax = fig.add_subplot(2, 3, k + 1)
        if i in frames:
            h, w = frames[i].shape[:2]
            ax.imshow(frames[i])
            for sl, n_pts, conns, colour in [
                (POSE_SLICE, 33, POSE_CONNECTIONS, "yellow"),
                (LEFT_SLICE, 21, HAND_CONNECTIONS, "cyan"),
                (RIGHT_SLICE, 21, HAND_CONNECTIONS, "magenta"),
            ]:
                pts = features[i, sl].reshape(n_pts, 3)
                if pts.any():
                    for p, q in conns:
                        ax.plot([pts[p, 0] * w, pts[q, 0] * w], [pts[p, 1] * h, pts[q, 1] * h],
                                color=colour, linewidth=1.5)
        ax.set_title(f"frame {i}")
        ax.axis("off")

    ax = fig.add_subplot(2, 1, 2)
    t = np.arange(n)
    for row, (sl, label, colour) in enumerate([
        (POSE_SLICE, "pose", "gold"), (LEFT_SLICE, "'Left' hand", "c"), (RIGHT_SLICE, "'Right' hand", "m")
    ]):
        found = features[:, sl].any(axis=1)
        ax.scatter(t[found], np.full(found.sum(), row), s=12, color=colour, marker="s")
    if s:
        ax.axvspan(a, b, color="green", alpha=0.1, label="sign span")
        ax.legend(loc="upper right")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["pose", "'Left' hand", "'Right' hand"])
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlim(-1, n)
    ax.set_xlabel("frame (a square = detected in that frame)")
    title = f"[{VERSION}] {video_id} ({word}): {n} frames"
    if s:
        title += (f" | sign span {a}-{b} | hands found in {s['hand_in_span']:.0%} of span"
                  f" | longest gap {s['longest_gap']} frames")
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{video_id}_check.png")
    plt.savefig(out, dpi=90)
    plt.close(fig)
    return s, out


def main():
    df = part1_all_clips()
    ids = sys.argv[1:] or DEFAULT_IDS
    words = dict(zip(df["video_id"], df["word"]))
    print("========== PART 2: CHOSEN CLIPS ==========")
    for vid in ids:
        s, out = part2_check(vid, words.get(vid, "?"))
        if s:
            print(f"{vid} ({words.get(vid, '?')}): {s['frames']} frames, sign span "
                  f"{s['span_start']}-{s['span_end']}, resting {s['rest_share']:.0%}, "
                  f"hands in span {s['hand_in_span']:.0%}, longest gap {s['longest_gap']} -> {out}")
        else:
            print(f"{vid}: no hands detected at all -> {out}")


if __name__ == "__main__":
    main()