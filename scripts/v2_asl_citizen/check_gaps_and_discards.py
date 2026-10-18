r"""
v2 step 5b: two follow-up checks on the ASL Citizen landmarks.

  1. DISCARDS PER WORD: does the cleaning remove hands much more often for some words?
     (BIRD showed a real hand at the mouth being removed because the pose wrist was wrong.
     If face signs top this list, that is a pattern, not a one-off.)
     Draws one typical clip for each of the 4 words with the most discards.
  2. LONG GAPS: clips where the hands disappear for more than 10 frames in the middle of
     the sign span. Draws the 4 worst, with frames picked at: start of span, last frame
     before the gap, first frame after the gap, end of span.

Uses span_stats.csv from check_asl_citizen_data.py (run that first) and the same drawing code.

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\check_gaps_and_discards.py > report_outputs\V2_07_gaps_and_discards.txt

Outputs:
  data\asl_citizen\checks_b\discard_<word>_<clip_id>.png
  data\asl_citizen\checks_b\gap_<clip_id>.png
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join("scripts", "v2_asl_citizen"))
from check_asl_citizen_data import draw_frame, timeline, LEFT, RIGHT  # noqa: E402

BASE = os.path.join("data", "asl_citizen")
SPANS = os.path.join(BASE, "span_stats.csv")
RAW, V3, VIDEOS = (os.path.join(BASE, d) for d in ("landmarks", "landmarks_v3", "videos"))
OUT = os.path.join(BASE, "checks_b")


def longest_gap(clean):
    hand = clean[:, LEFT].any(1) | clean[:, RIGHT].any(1)
    idx = np.flatnonzero(hand)
    best = (0, 0, 0)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a - 1 > best[0]:
            best = (b - a - 1, a, b)        # gap length, last hand frame before, first after
    return idx[0], idx[-1], best


def draw(r, picks, why, name):
    raw = np.load(os.path.join(RAW, f"{r.clip_id}.npy"))
    clean = np.load(os.path.join(V3, f"{r.clip_id}.npy"))
    cap = cv2.VideoCapture(os.path.join(VIDEOS, os.path.basename(r.video)))
    want, frames, i = set(int(p) for p in picks), {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f
        i += 1
    cap.release()
    tops, bottoms = [], []
    for p in picks:
        p = int(min(p, len(raw) - 1))
        base = frames.get(p, np.zeros((480, 640, 3), np.uint8))
        s = 240 / base.shape[0]
        for img, store, label in [(draw_frame(base.copy(), raw[p]), tops, "raw"),
                                  (draw_frame(base.copy(), clean[p], raw[p]), bottoms, "v3")]:
            img = cv2.resize(img, (int(img.shape[1] * s), 240))
            cv2.putText(img, f"{label} frame {p}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            store.append(img)
    grid = np.vstack([np.hstack(tops), np.hstack(bottoms)])
    title = np.full((56, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"{r.clip_id} ({r.word} / {r.gloss}) - {why}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(title, f"{len(raw)} frames, {r.fps:.0f} fps | hands raw {r.raw_any_hand_share:.0%} -> v3 "
                       f"{r.any_hand_share:.0%} | grey X = discarded", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.imwrite(os.path.join(OUT, name), np.vstack([title, grid, timeline(raw, clean, grid.shape[1])]))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(SPANS)
    use = df[df.usable]
    jobs = []

    # ---- 1. discards per word ----------------------------------------------
    print("=" * 70)
    print("1. DISCARDED HANDS PER WORD (usable clips)")
    print("=" * 70)
    pw = use.groupby("word").agg(clips=("clip_id", "count"),
                                 discards_per_clip=("hands_discarded", "mean"),
                                 clips_with_discards=("hands_discarded", lambda s: (s > 0).mean()),
                                 hand_in_span=("hand_in_span", "mean")).sort_values(
        "discards_per_clip", ascending=False)
    pw["discards_per_clip"] = pw.discards_per_clip.round(2)
    pw["clips_with_discards"] = (pw.clips_with_discards * 100).round(0).astype(int).astype(str) + "%"
    pw["hand_in_span"] = (pw.hand_in_span * 100).round(1).astype(str) + "%"
    print(f"Overall: {use.hands_discarded.mean():.2f} discards per clip")
    print("\nTop 15 words:")
    print(pw.head(15).to_string())
    print("\nBottom 5 words:")
    print(pw.tail(5).to_string())
    for w in pw.head(4).index:
        sub = use[(use.word == w) & (use.hands_discarded > 0)]
        if len(sub):
            med = sub.hands_discarded.median()
            r = sub.iloc[(sub.hands_discarded - med).abs().argsort().iloc[0]]
            jobs.append(("discard", r, w))

    # ---- 2. long gaps -------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. LONG GAPS INSIDE THE SIGN SPAN (usable clips)")
    print("=" * 70)
    gaps = use[use.longest_gap > 10]
    print(f"Clips with a gap > 10 frames: {len(gaps)} of {len(use)} ({len(gaps) / len(use):.1%})")
    if len(gaps):
        print(f"  gap length: median {gaps.longest_gap.median():.0f} frames, max {gaps.longest_gap.max()}")
        print(f"  these clips' span: median {gaps.span_frames.median():.0f} frames "
              f"(all usable clips: {use.span_frames.median():.0f})")
        print(f"  words with the most: " + ", ".join(
            f"{w} ({n})" for w, n in gaps.word.value_counts().head(8).items()))
        print(f"  signers with the most: " + ", ".join(
            f"{s} ({n})" for s, n in gaps.signer.value_counts().head(5).items()))
        for _, r in gaps.nlargest(4, "longest_gap").iterrows():
            jobs.append(("gap", r, None))

    # ---- images ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. CHECK IMAGES")
    print("=" * 70)
    for kind, r, w in tqdm(jobs, desc="Check images", unit="image", file=sys.stderr):
        try:
            clean = np.load(os.path.join(V3, f"{r.clip_id}.npy"))
            raw = np.load(os.path.join(RAW, f"{r.clip_id}.npy"))
            if kind == "gap":
                s, e, (g, a, b) = longest_gap(clean)
                picks = [s, a, b, e]
                why = f"gap of {g} frames ({a + 1}-{b - 1})"
                name = f"gap_{r.clip_id}.png"
            else:
                changed = np.flatnonzero((raw[:, 99:] != clean[:, 99:]).any(1))
                picks = changed[np.linspace(0, len(changed) - 1, 4).astype(int)] if len(changed) else [0, 1, 2, 3]
                why = f"typical discards for '{w}' ({int(r.hands_discarded)} discarded)"
                name = f"discard_{w}_{r.clip_id}.png"
            draw(r, picks, why, name)
            print(f"  {name}")
        except Exception as ex:
            print(f"  could not draw {r.clip_id}: {type(ex).__name__}: {ex}")
    print(f"\nSaved images in {OUT}")


if __name__ == "__main__":
    main()