r"""
clean_hands.py - Week 2 (v3): clean the hand data in the v2 landmarks.

Run from C:\SL_PROJECT (no re-extraction needed, about 1-2 minutes):
    python clean_hands.py                            # clean all clips
    python clean_hands.py 34733 28108 24636 64088    # also draw before/after images

Why: the v2 check (find_phantoms.py) found three problems:
  1. phantom hands: part of the face detected as a hand (24636 "give")
  2. duplicate hands: the same real hand detected twice (28108 "hot")
  3. stolen slot: a phantom took the real hand's slot (34733 "man"),
     because v2 matched hands to wrists by best TOTAL fit

New rule (v3), per frame:
  - two detections lying almost on top of each other are the SAME hand
    (a duplicate): keep only the one closer to a pose wrist;
  - each pose wrist takes its NEAREST detected hand,
  - but only if that hand is within MAX_DIST shoulder widths of the wrist;
  - closest pairs are matched first; a hand near no wrist is discarded.
The v2 files already hold every hand MediaPipe found (max 2 per frame),
so the rule is applied to them directly. v1 and v2 stay unchanged.

Outputs:
    landmarks_v3\<video_id>.npy     cleaned arrays, same layout (frames, 225)
    landmarks_index_v3.csv          index with updated detection rates
    clean_checks\<id>_v2_v3.png     before/after images (if IDs are given)
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

IN_DIR, IN_INDEX = "data/v2/landmarks_v2", "data/v2/landmarks_index_v2.csv"
OUT_DIR, OUT_INDEX = "data/v3/landmarks_v3", "data/v3/landmarks_index_v3.csv"
VIDEO_DIR = os.path.join("wlasl-processed", "videos")
CHECK_DIR = "data/v3/clean_checks"

MAX_DIST = 1.0          # shoulder widths; real hands: median 0.08, 95% within 0.24.
                        # 0.6 was tried first but discarded the real tapping hand in
                        # "doctor" (17016), where the hands touch; face phantoms were 2.0+.
DUP_DIST = 0.15         # shoulder widths; mean point-to-point gap below this = same hand
MIN_HAND_SHARE = 0.20   # same "usable" bar as extraction
MIN_CLIPS_PER_WORD = 8

POSE_SLICE = slice(0, 99)
SLOTS = [(slice(99, 162), 15), (slice(162, 225), 16)]  # (left slot, pose wrist 15), (right slot, 16)


def clean_frame(row):
    """Return (cleaned row, hands_discarded, hands_moved, duplicates) for one frame."""
    pose = row[POSE_SLICE].reshape(33, 3)
    hands = [(k, row[sl].reshape(21, 3)) for k, (sl, _) in enumerate(SLOTS) if row[sl].any()]
    if not hands:
        return row, 0, 0, 0
    shoulder = np.linalg.norm(pose[11, :2] - pose[12, :2])
    if not pose.any() or shoulder == 0:  # rare: no pose to check against, keep as is
        return row, 0, 0, 0

    def nearest_wrist(h):
        return min(np.linalg.norm(h[0, :2] - pose[w, :2]) for _, w in SLOTS)

    dup = 0
    if len(hands) == 2:
        gap = np.linalg.norm(hands[0][1][:, :2] - hands[1][1][:, :2], axis=1).mean() / shoulder
        if gap < DUP_DIST:  # same hand detected twice: keep the one nearer a wrist
            hands = [min(hands, key=lambda kh: nearest_wrist(kh[1]))]
            dup = 1

    # all (distance, hand, slot) pairs, closest first
    pairs = sorted(
        (np.linalg.norm(h[0, :2] - pose[wrist, :2]) / shoulder, i, slot)
        for i, (_, h) in enumerate(hands)
        for slot, (_, wrist) in enumerate(SLOTS)
    )
    out = row.copy()
    for sl, _ in SLOTS:
        out[sl] = 0
    used_hands, used_slots, moved = set(), set(), 0
    for d, i, slot in pairs:
        if d > MAX_DIST or i in used_hands or slot in used_slots:
            continue
        out[SLOTS[slot][0]] = hands[i][1].ravel()
        used_hands.add(i)
        used_slots.add(slot)
        moved += int(hands[i][0] != slot)
    return out, len(hands) - len(used_hands), moved, dup


def hand_shares(f):
    left = f[:, SLOTS[0][0]].any(axis=1)
    right = f[:, SLOTS[1][0]].any(axis=1)
    return {"left_share": round(float(left.mean()), 3),
            "right_share": round(float(right.mean()), 3),
            "any_hand_share": round(float((left | right).mean()), 3)}


def draw_before_after(vid, word, before, after, changed):
    """Draw up to 3 changed frames: v2 on top, v3 below."""
    picks = sorted(changed[:3])
    if not picks:
        print(f"{vid}: no frames changed")
        return
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"{vid}.mp4"))
    frames, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in picks:
            frames[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

    fig, axes = plt.subplots(2, len(picks), figsize=(5 * len(picks), 8), squeeze=False)
    for col, i in enumerate(picks):
        for r, (data, label) in enumerate([(before, "v2"), (after, "v3")]):
            ax = axes[r][col]
            h, w = frames[i].shape[:2]
            ax.imshow(frames[i])
            pose = data[i, POSE_SLICE].reshape(33, 3)
            for a, b in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
                ax.plot([pose[a, 0] * w, pose[b, 0] * w], [pose[a, 1] * h, pose[b, 1] * h],
                        color="yellow", lw=1.5)
            for (sl, wrist), colour in zip(SLOTS, ["cyan", "magenta"]):
                ax.scatter([pose[wrist, 0] * w], [pose[wrist, 1] * h], s=120,
                           facecolors="none", edgecolors=colour, linewidths=2.5)
                hand = data[i, sl].reshape(21, 3)
                if hand.any():
                    ax.scatter(hand[:, 0] * w, hand[:, 1] * h, s=8, color=colour)
            ax.set_title(f"{label} - frame {i}")
            ax.axis("off")
    fig.suptitle(f"{vid} ({word}): top v2, bottom v3 | rings = pose wrists, dots = hand "
                 f"(cyan = left slot, magenta = right slot)")
    plt.tight_layout()
    os.makedirs(CHECK_DIR, exist_ok=True)
    out = os.path.join(CHECK_DIR, f"{vid}_v2_v3.png")
    plt.savefig(out, dpi=90)
    plt.close(fig)
    print(f"{vid} ({word}): frames {picks} -> {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    index = pd.read_csv(IN_INDEX, dtype={"video_id": str})
    draw_ids = set(sys.argv[1:])
    rows, to_draw = [], []
    tot_disc = tot_moved = tot_dup = frames_changed = 0

    for _, clip in tqdm(index.iterrows(), total=len(index), desc="Cleaning", unit="clip"):
        vid = clip["video_id"]
        before = np.load(os.path.join(IN_DIR, f"{vid}.npy"))
        after = np.empty_like(before)
        disc = moved = dups = 0
        changed = []
        for i, row in enumerate(before):
            after[i], d, m, u = clean_frame(row)
            disc += d
            moved += m
            dups += u
            if d or m or u:
                changed.append(i)
        np.save(os.path.join(OUT_DIR, f"{vid}.npy"), after)
        tot_disc += disc
        tot_moved += moved
        tot_dup += dups
        frames_changed += len(changed)
        new = {**clip.to_dict(), **hand_shares(after),
               "duplicates_removed": dups, "hands_discarded": disc, "hands_moved": moved}
        new["usable"] = new["any_hand_share"] >= MIN_HAND_SHARE
        rows.append(new)
        if vid in draw_ids:
            to_draw.append((vid, clip["word"], before, after, changed))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_INDEX, index=False)

    print("\n========== CLEANING SUMMARY (v2 -> v3) ==========")
    print(f"Rule: duplicates (< {DUP_DIST}) removed, then nearest hand per pose wrist, "
          f"within {MAX_DIST} shoulder widths")
    print(f"Frames changed: {frames_changed}")
    print(f"Duplicate hands removed: {tot_dup} in {(out['duplicates_removed'] > 0).sum()} clips")
    print(f"Hands discarded (too far from any wrist): {tot_disc} in "
          f"{(out['hands_discarded'] > 0).sum()} clips")
    print(f"Hands moved to the other slot: {tot_moved} in {(out['hands_moved'] > 0).sum()} clips")
    print(f"Any hand detected (mean share of frames): v2 {index['any_hand_share'].mean():.1%} "
          f"-> v3 {out['any_hand_share'].mean():.1%}")

    unusable = out[~out["usable"]]
    print(f"\nClips now below the usable bar ({MIN_HAND_SHARE:.0%} of frames with a hand): "
          f"{len(unusable)}")
    if len(unusable):
        print(unusable[["video_id", "word", "any_hand_share"]].to_string(index=False))
    per_word = out[out["usable"]].groupby("word").size().reindex(
        sorted(out["word"].unique()), fill_value=0)
    low = per_word[per_word < MIN_CLIPS_PER_WORD]
    print(f"Usable clips per word: min {per_word.min()}, mean {per_word.mean():.1f}")
    print("Words below 8 usable clips: " + (", ".join(f"{w} ({n})" for w, n in low.items())
                                           if len(low) else "none"))

    print("\nClips with the most hands discarded:")
    print(out.nlargest(5, "hands_discarded")[["video_id", "word", "duplicates_removed",
                                               "hands_discarded", "hands_moved"]].to_string(index=False))
    print(f"\nSaved {OUT_DIR}\\*.npy and {OUT_INDEX}")

    if to_draw:
        print("\n========== BEFORE / AFTER IMAGES ==========")
        for args in to_draw:
            draw_before_after(*args)


if __name__ == "__main__":
    main()