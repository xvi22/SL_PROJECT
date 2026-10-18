r"""
compare_v1_v2.py - compare the v1 (MediaPipe label) and v2 (pose wrist) extractions.

Run from C:\SL_PROJECT after both extractions exist:
    python compare_v1_v2.py

For every frame it compares which hand slots are filled in v1 and in v2:
  - swapped:   v1 had the hand in one slot, v2 put it in the other
               (MediaPipe's label was overruled by the wrist rule)
  - recovered: v1 had one hand, v2 has two
               (both hands had the same label in v1, so one was overwritten)
Saves compare_v1_v2.csv (one row per clip).
"""

import os

import numpy as np
import pandas as pd

V1_DIR, V2_DIR = "data/v1/landmarks", "data/v2/landmarks_v2"
INDEX_CSV = "data/v2/landmarks_index_v2.csv"
LEFT_SLICE, RIGHT_SLICE = slice(99, 162), slice(162, 225)


def slots(features):
    return features[:, LEFT_SLICE].any(axis=1), features[:, RIGHT_SLICE].any(axis=1)


def main():
    index = pd.read_csv(INDEX_CSV, dtype={"video_id": str})
    rows = []
    for vid, word in zip(index["video_id"], index["word"]):
        v1 = np.load(os.path.join(V1_DIR, f"{vid}.npy"))
        v2 = np.load(os.path.join(V2_DIR, f"{vid}.npy"))
        if len(v1) != len(v2):
            print(f"WARNING: {vid} has {len(v1)} frames in v1 but {len(v2)} in v2; skipped")
            continue
        l1, r1 = slots(v1)
        l2, r2 = slots(v2)
        swapped = ((l1 & ~r1) & (r2 & ~l2)) | ((r1 & ~l1) & (l2 & ~r2))
        recovered = (l1 ^ r1) & (l2 & r2)
        rows.append({
            "video_id": vid, "word": word, "frames": len(v1),
            "swapped_frames": int(swapped.sum()),
            "recovered_frames": int(recovered.sum()),
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/v2/compare_v1_v2.csv", index=False)
    total = df["frames"].sum()
    sw, rc = df["swapped_frames"], df["recovered_frames"]

    print("========== v1 vs v2 ==========")
    print(f"Clips compared: {len(df)} | frames: {total}")
    print(f"Frames where the hand changed slot (label overruled): {sw.sum()} "
          f"({sw.sum() / total:.1%} of frames) in {(sw > 0).sum()} clips")
    print(f"Frames where a lost second hand was recovered:        {rc.sum()} "
          f"({rc.sum() / total:.1%} of frames) in {(rc > 0).sum()} clips")
    print("\nClips with the most slot changes:")
    print(df.nlargest(5, "swapped_frames")[["video_id", "word", "frames", "swapped_frames"]]
          .to_string(index=False))
    hat = df[df["video_id"] == "26712"]
    if len(hat):
        print(f"\nhat (26712): {int(hat['swapped_frames'].iloc[0])} frames changed slot")
    print("\nSaved compare_v1_v2.csv")


if __name__ == "__main__":
    main()