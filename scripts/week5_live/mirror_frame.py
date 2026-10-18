r"""
mirror_frame.py - left-handed mode for the live pipeline (Week 5).

When the toggle is on, each raw frame (225 numbers, same layout as the landmark files)
is mirrored BEFORE cleaning, preprocessing and the LSTM, so a left-handed signer looks
like a right-handed one. No retraining needed.

    from mirror_frame import mirror_frame
    if left_handed_mode:
        row = mirror_frame(row)

Three things happen:
  1. x -> 1 - x for every detected point (raw coordinates are fractions of the image)
  2. the left-hand and right-hand slots swap
  3. left/right pose points swap (e.g. left shoulder 11 <-> right shoulder 12)
Undetected points stay zero.

Run this file on its own to self-test:  python mirror_frame.py
"""

import numpy as np

# MediaPipe pose left/right pairs (33-point model)
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
              (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def mirror_frame(row):
    """Mirror one raw 225-number frame (left-handed mode)."""
    pose = row[0:99].reshape(33, 3).copy()
    left = row[99:162].reshape(21, 3).copy()
    right = row[162:225].reshape(21, 3).copy()
    for a, b in POSE_PAIRS:
        pose[[a, b]] = pose[[b, a]]
    left, right = right, left                      # swap hand slots
    for part in (pose, left, right):
        found = part.any(axis=1)                   # keep undetected points as zeros
        part[found, 0] = 1 - part[found, 0]        # flip x
    return np.concatenate([pose.ravel(), left.ravel(), right.ravel()]).astype(row.dtype)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    row = rng.random(225).astype(np.float32)
    row[99:162] = 0                                  # left hand missing
    m = mirror_frame(row)
    assert np.allclose(mirror_frame(m), row), "mirroring twice must give the original"
    assert not m[162:225].any(), "missing left hand must become a missing right hand"
    assert np.isclose(m[11 * 3], 1 - row[12 * 3]), "left shoulder x = 1 - right shoulder x"
    print("mirror_frame self-test passed")