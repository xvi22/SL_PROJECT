r"""
v2 step 5: check the ASL Citizen landmarks by eye and by numbers (like Week 2's check_clips.py).

What it does, in plain words:
  1. SIGN SPAN numbers (first to last frame with a hand), computed with the SAME code for
     ASL Citizen and for our WLASL v3 data, so they can be compared fairly:
       - how much of each clip is resting time (no hands at start/end)
       - how often a hand is found INSIDE the sign span  (WLASL Week 2: 96.8%)
       - sign span length in frames and seconds
  2. CHECK IMAGES for clips worth looking at:
       - the 4 clips with the lowest hand detection
       - the 4 clips where cleaning discarded the most hands
       - the 3 clips with the lowest pose detection
       - 3 random normal clips (to see what "normal" looks like)
     Each image: 4 frames. Top row = raw (before cleaning), bottom row = cleaned (v3).
     Under it: a timeline of where hands were found (raw vs cleaned).

Run from C:\SL_PROJECT :
  python -u scripts\v2_asl_citizen\check_asl_citizen_data.py > report_outputs\V2_06_check_asl_citizen.txt

Outputs:
  data\asl_citizen\span_stats.csv            sign span numbers per ASL Citizen clip
  data\asl_citizen\checks\<clip_id>.png      check images
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = os.path.join("data", "asl_citizen")
ASC_INDEX = os.path.join(BASE, "landmarks_index_v3.csv")
ASC_RAW, ASC_V3 = os.path.join(BASE, "landmarks"), os.path.join(BASE, "landmarks_v3")
ASC_VIDEOS = os.path.join(BASE, "videos")
WL_INDEX = os.path.join("data", "v3", "landmarks_index_v3.csv")
WL_V3 = os.path.join("data", "v3", "landmarks_v3")
CHECK_DIR = os.path.join(BASE, "checks")

POSE, LEFT, RIGHT = slice(0, 99), slice(99, 162), slice(162, 225)
ARM_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
CYAN, MAGENTA, GREY, YELLOW = (255, 255, 0), (255, 0, 255), (150, 150, 150), (0, 255, 255)


def span_stats(f, fps):
    hand = f[:, LEFT].any(axis=1) | f[:, RIGHT].any(axis=1)
    n = len(f)
    if not hand.any():
        return {"span_frames": 0, "span_sec": 0.0, "resting_share": 1.0,
                "hand_in_span": 0.0, "longest_gap": 0}
    idx = np.flatnonzero(hand)
    s, e = idx[0], idx[-1]
    inside = hand[s:e + 1]
    gaps, run = [], 0
    for h in inside:
        run = 0 if h else run + 1
        gaps.append(run)
    return {"span_frames": int(e - s + 1), "span_sec": round((e - s + 1) / fps, 2),
            "resting_share": round(1 - (e - s + 1) / n, 3),
            "hand_in_span": round(float(inside.mean()), 3), "longest_gap": int(max(gaps))}


def describe(df, name):
    print(f"\n{name}: {len(df)} usable clips")
    print(f"  resting time (no hands at start/end): {df.resting_share.mean():.1%} of each clip")
    print(f"  hand found inside the sign span:     {df.hand_in_span.mean():.1%} of span frames")
    print(f"  sign span frames: min {df.span_frames.min()} / median {df.span_frames.median():.0f} / "
          f"p90 {df.span_frames.quantile(0.9):.0f} / max {df.span_frames.max()}")
    print(f"  sign span seconds: median {df.span_sec.median():.2f} / p90 {df.span_sec.quantile(0.9):.2f}")
    print(f"  clips with hand in < 80% of span: {(df.hand_in_span < 0.8).sum()}  "
          f"| longest gap > 10 frames: {(df.longest_gap > 10).sum()}")


def draw_frame(img, row, mark_discarded_from=None):
    h, w = img.shape[:2]
    pose = row[POSE].reshape(33, 3)
    if pose.any():
        for a, b in ARM_LINKS:
            cv2.line(img, (int(pose[a, 0] * w), int(pose[a, 1] * h)),
                     (int(pose[b, 0] * w), int(pose[b, 1] * h)), YELLOW, 2)
        for wr, col in [(15, CYAN), (16, MAGENTA)]:
            cv2.circle(img, (int(pose[wr, 0] * w), int(pose[wr, 1] * h)), 12, col, 2)
    for sl, col in [(LEFT, CYAN), (RIGHT, MAGENTA)]:
        hand = row[sl].reshape(21, 3)
        if hand.any():
            for x, y, _ in hand:
                cv2.circle(img, (int(x * w), int(y * h)), 3, col, -1)
    if mark_discarded_from is not None:      # raw hands that are gone after cleaning -> grey X
        kept = [row[sl].reshape(21, 3) for sl in (LEFT, RIGHT) if row[sl].any()]
        for sl in (LEFT, RIGHT):
            raw_hand = mark_discarded_from[sl].reshape(21, 3)
            if raw_hand.any() and not any(np.allclose(raw_hand, k) for k in kept):
                cx, cy = int(raw_hand[:, 0].mean() * w), int(raw_hand[:, 1].mean() * h)
                cv2.drawMarker(img, (cx, cy), GREY, cv2.MARKER_TILTED_CROSS, 40, 4)
    return img


def timeline(raw, clean, width):
    rows = []
    for f, label in [(raw, "raw"), (clean, "v3")]:
        bar = np.full((18, width, 3), 40, np.uint8)
        n = len(f)
        for i in range(n):
            x0, x1 = int(i * width / n), max(int((i + 1) * width / n), int(i * width / n) + 1)
            if f[i, LEFT].any():
                bar[:9, x0:x1] = CYAN
            if f[i, RIGHT].any():
                bar[9:, x0:x1] = MAGENTA
        cv2.putText(bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(bar)
    return np.vstack(rows)


def check_image(r, why):
    raw = np.load(os.path.join(ASC_RAW, f"{r.clip_id}.npy"))
    clean = np.load(os.path.join(ASC_V3, f"{r.clip_id}.npy"))
    hand = clean[:, LEFT].any(1) | clean[:, RIGHT].any(1)
    changed = np.flatnonzero((raw[:, 99:] != clean[:, 99:]).any(1))
    if why == "most hands discarded" and len(changed):
        picks = changed[np.linspace(0, len(changed) - 1, 4).astype(int)]
    elif hand.any():
        s, e = np.flatnonzero(hand)[[0, -1]]
        picks = np.linspace(s, e, 4).astype(int)
    else:
        picks = np.linspace(0, len(raw) - 1, 4).astype(int)
    cap = cv2.VideoCapture(os.path.join(ASC_VIDEOS, os.path.basename(r.video)))
    frames, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in set(picks):
            frames[i] = f
        i += 1
    cap.release()
    cell_h = 240
    tops, bottoms = [], []
    for p in picks:
        base = frames.get(p, np.zeros((480, 640, 3), np.uint8))
        s = cell_h / base.shape[0]
        a = draw_frame(base.copy(), raw[min(p, len(raw) - 1)])
        b = draw_frame(base.copy(), clean[min(p, len(clean) - 1)], raw[min(p, len(raw) - 1)])
        for img, store, label in [(a, tops, "raw"), (b, bottoms, "v3")]:
            img = cv2.resize(img, (int(img.shape[1] * s), cell_h))
            cv2.putText(img, f"{label} frame {p}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            store.append(img)
    grid = np.vstack([np.hstack(tops), np.hstack(bottoms)])
    title = np.full((56, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"{r.clip_id} ({r.word} / {r.gloss}) - {why}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(title, f"hands raw {r.raw_any_hand_share:.0%} -> v3 {r.any_hand_share:.0%}, pose {r.pose_share:.0%}"
                       f" | rings = pose wrists, dots = hands (cyan left slot, magenta right), grey X = discarded",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    out = np.vstack([title, grid, timeline(raw, clean, grid.shape[1])])
    cv2.imwrite(os.path.join(CHECK_DIR, f"{r.clip_id}.png"), out)


def main():
    os.makedirs(CHECK_DIR, exist_ok=True)
    asc = pd.read_csv(ASC_INDEX)
    wl = pd.read_csv(WL_INDEX, dtype={"video_id": str})

    # ---- 1. sign spans ------------------------------------------------------
    rows = []
    for r in tqdm(asc.itertuples(index=False), total=len(asc), desc="ASL Citizen spans", unit="clip",
                  file=sys.stderr):
        f = np.load(os.path.join(ASC_V3, f"{r.clip_id}.npy"))
        rows.append({"clip_id": r.clip_id, **span_stats(f, r.fps)})
    asc = asc.merge(pd.DataFrame(rows), on="clip_id")
    asc.to_csv(os.path.join(BASE, "span_stats.csv"), index=False)

    wrows = []
    wl_use = wl[wl["usable"]]
    for r in tqdm(wl_use.itertuples(index=False), total=len(wl_use), desc="WLASL v3 spans", unit="clip",
                  file=sys.stderr):
        f = np.load(os.path.join(WL_V3, f"{r.video_id}.npy"))
        wrows.append(span_stats(f, r.fps))
    wl_spans = pd.DataFrame(wrows)

    print("=" * 70)
    print("1. SIGN SPANS (same code for both datasets, cleaned v3 data, usable clips)")
    print("=" * 70)
    describe(wl_spans, "WLASL v3")
    describe(asc[asc.usable], "ASL Citizen v3")
    a = asc[asc.usable]
    print(f"\nASL Citizen hand detection: whole clip {a.any_hand_share.mean():.1%} -> inside sign span "
          f"{a.hand_in_span.mean():.1%}")
    print(f"fps below 20 (choppy): {(asc.fps < 20).sum()} clips | 50+ fps: {(asc.fps >= 50).sum()} clips")
    print(f"Discarded hands per clip: ASL Citizen {asc.hands_discarded.sum() / len(asc):.2f} "
          f"(WLASL v3: 368 / 863 = 0.43)")
    top_disc = asc.groupby("signer").hands_discarded.sum().sort_values(ascending=False)
    print(f"Discarded hands by signer (top 5 of {asc.signer.nunique()}): " +
          ", ".join(f"{s} ({n})" for s, n in top_disc.head(5).items()) +
          f"  | share of all discards from these 5: {top_disc.head(5).sum() / max(top_disc.sum(), 1):.0%}")

    # ---- 2. check images ----------------------------------------------------
    picks = []
    picks += [(r, "lowest hand detection") for r in asc.nsmallest(4, "any_hand_share").itertuples(index=False)]
    picks += [(r, "most hands discarded") for r in asc.nlargest(4, "hands_discarded").itertuples(index=False)]
    picks += [(r, "lowest pose detection") for r in asc.nsmallest(3, "pose_share").itertuples(index=False)]
    normal = asc[asc.usable & (asc.hands_discarded == 0)]
    picks += [(r, "random normal clip") for r in normal.sample(min(3, len(normal)), random_state=42)
              .itertuples(index=False)]
    print("\n" + "=" * 70)
    print("2. CHECK IMAGES")
    print("=" * 70)
    for r, why in tqdm(picks, desc="Check images", unit="image", file=sys.stderr):
        try:
            check_image(r, why)
            print(f"  {why:22s} {r.clip_id} ({r.word}/{r.gloss}): hands v3 {r.any_hand_share:.0%}, "
                  f"discarded {r.hands_discarded}, pose {r.pose_share:.0%}")
        except Exception as e:
            print(f"  could not draw {r.clip_id}: {type(e).__name__}: {e}")
    print(f"\nSaved {os.path.join(BASE, 'span_stats.csv')} and images in {CHECK_DIR}")


if __name__ == "__main__":
    main()