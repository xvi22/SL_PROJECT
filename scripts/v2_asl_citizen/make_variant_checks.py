"""
v2 step 3b: comparison images to check which ASL Citizen gloss is the SAME sign as WLASL.

Why: ASL Citizen often has several numbered versions of a word (e.g. DRINK1, DRINK2).
Each one is a different sign. We must keep only the version(s) that match WLASL's sign,
otherwise we would teach the model the wrong sign.

What it does, in plain words:
  For each word, it makes ONE image with several rows:
    - top rows:    2 WLASL clips of the word (different signers)   <- the "correct" sign
    - next rows:   2 clips of EACH ASL Citizen candidate gloss (different signers)
  Each row shows 6 frames spread across the middle of the clip (start and end are
  usually resting hands, so they are skipped). You compare the rows by eye.

  It also makes data\\asl_citizen\\variant_choice.csv, a table to fill in:
  for each word + gloss, put yes or no in the "keep" column.
  Exact single matches are pre-filled "yes"; variants and near matches are left blank.
  If the file already exists it is NOT overwritten (so your answers are safe).

Run from C:\\SL_PROJECT\\ :
  python -u scripts\\v2_asl_citizen\\make_variant_checks.py            (the 33 words that need checking)
  python -u scripts\\v2_asl_citizen\\make_variant_checks.py all        (all 100 words)

Outputs:
  data\\asl_citizen\\variant_checks\\<word>.png    (words to check)
  data\\asl_citizen\\exact_checks\\<word>.png      (exact matches, only with "all")
  data\\asl_citizen\\variant_choice.csv
"""
import os
import sys
import json

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = os.path.join("data", "asl_citizen")
MATCH = os.path.join(BASE, "word_match.csv")
CAND = os.path.join(BASE, "candidates.csv")
CHOICE = os.path.join(BASE, "variant_choice.csv")
WLASL_JSON = os.path.join("wlasl-processed", "WLASL_v0.3.json")
WLASL_VIDEOS = os.path.join("wlasl-processed", "videos")

N_FRAMES = 6        # frames per row
CELL = 170          # each frame is shrunk to fit a CELL x CELL square
LABEL_W = 230       # width of the text label on the left of each row
CLIPS_PER_SOURCE = 2


def grab_frames(path, n=N_FRAMES):
    """Read n frames spread over the middle 70% of the video (skip resting start/end)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for idx in np.linspace(total * 0.15, total * 0.85, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, img = cap.read()
            frames.append(img if ok else None)
    cap.release()
    return frames + [None] * (n - len(frames))


def fit(img):
    """Shrink a frame to fit a CELL x CELL grey square, keeping its shape."""
    cell = np.full((CELL, CELL, 3), 60, np.uint8)
    if img is None:
        cv2.putText(cell, "no frame", (30, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return cell
    h, w = img.shape[:2]
    s = CELL / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    y, x = (CELL - small.shape[0]) // 2, (CELL - small.shape[1]) // 2
    cell[y:y + small.shape[0], x:x + small.shape[1]] = small
    return cell


def make_row(path, line1, line2, colour):
    label = np.full((CELL, LABEL_W, 3), 255, np.uint8)
    cv2.rectangle(label, (0, 0), (8, CELL), colour, -1)
    cv2.putText(label, line1, (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(label, line2, (16, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cells = [fit(f) for f in grab_frames(path)]
    row = np.hstack([label] + cells)
    cv2.line(row, (0, CELL - 1), (row.shape[1], CELL - 1), (255, 255, 255), 2)
    return row


def pick_spread(df, signer_col, n):
    """Pick up to n clips from different signers."""
    return df.drop_duplicates(signer_col).head(n)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "variants"
    match = pd.read_csv(MATCH)
    cand = pd.read_csv(CAND)
    cand = cand[cand["downloaded"]]

    # WLASL clips we have, per word
    with open(WLASL_JSON) as f:
        entries = json.load(f)[:100]
    have = {os.path.splitext(n)[0] for n in os.listdir(WLASL_VIDEOS)}
    wlasl = {}
    for e in entries:
        rows = [{"video_id": str(i["video_id"]), "signer": i.get("signer_id")}
                for i in e["instances"] if str(i["video_id"]) in have]
        wlasl[e["gloss"].strip().lower()] = pd.DataFrame(rows, columns=["video_id", "signer"])

    # ---- choice table (never overwrite the user's answers) ------------------
    if not os.path.exists(CHOICE):
        g = (cand.groupby(["word", "gloss", "match_type"])
                 .agg(clips=("video", "count"), signers=("signer", "nunique")).reset_index())
        g["keep"] = np.where(g["match_type"] == "exact", "yes", "")
        g["note"] = ""
        g.to_csv(CHOICE, index=False)
        print(f"Made {CHOICE} (fill in the 'keep' column with yes / no)")
    else:
        print(f"{CHOICE} already exists, not overwritten")

    # ---- which words get an image -------------------------------------------
    to_check = match[match.match_type != "exact"]
    jobs = [(w, "variant_checks") for w in to_check.word]
    if mode == "all":
        jobs += [(w, "exact_checks") for w in match[match.match_type == "exact"].word]
    print(f"Making {len(jobs)} comparison images")

    made, problems = 0, []
    for word, folder in tqdm(jobs, desc="Comparison images", unit="word", file=sys.stderr):
        out_dir = os.path.join(BASE, folder)
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        w_df = pick_spread(wlasl.get(word, pd.DataFrame(columns=["video_id", "signer"])),
                           "signer", CLIPS_PER_SOURCE)
        for _, r in w_df.iterrows():
            rows.append(make_row(os.path.join(WLASL_VIDEOS, f"{r.video_id}.mp4"),
                                 "WLASL", f"clip {r.video_id}, signer {r.signer}", (0, 160, 0)))
        if not rows:
            problems.append(f"{word}: no WLASL clip found")
        word_cand = cand[cand.word == word]
        if word_cand.empty:
            problems.append(f"{word}: no ASL Citizen candidate downloaded")
        palette = [(200, 80, 0), (0, 0, 200), (160, 0, 160), (0, 140, 200)]
        for k, (gloss, gdf) in enumerate(word_cand.groupby("gloss")):
            for _, r in pick_spread(gdf, "signer", CLIPS_PER_SOURCE).iterrows():
                rows.append(make_row(r.path, gloss, f"signer {r.signer}", palette[k % 4]))
        if not rows:
            continue
        title = np.full((50, rows[0].shape[1], 3), 255, np.uint8)
        cv2.putText(title, f"{word.upper()}  -  which ASL Citizen rows match the green WLASL rows?",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(out_dir, f"{word}.png"), np.vstack([title] + rows))
        made += 1

    print(f"\nImages made: {made} of {len(jobs)}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print(f"Saved in {os.path.join(BASE, 'variant_checks')}"
          + (f" and {os.path.join(BASE, 'exact_checks')}" if mode == "all" else ""))


if __name__ == "__main__":
    main()