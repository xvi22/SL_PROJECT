r"""
test_one_clip.py - Week 2, step 1: extract landmarks from ONE WLASL clip.

Uses the MediaPipe 1.1.0 Tasks API (HandLandmarker + PoseLandmarker).
The old mp.solutions.holistic API does not exist in 1.1.0.

Run from C:\SL_PROJECT with the venv active:
    python test_one_clip.py

Outputs (in landmarks_test\):
    <video_id>.npy        array of shape (frames, 225)
    <video_id>_check.png  landmarks drawn on 3 frames, for a visual check
"""

import os
import urllib.request

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# ---------------- settings ----------------
VIDEO_ID = "69241"  # a "book" clip we have (from find_word.py)
VIDEO_DIR = os.path.join("wlasl-processed", "videos")
MODEL_DIR = "models"
OUT_DIR = "data/test/landmarks_test"

MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

# Feature layout per frame (225 numbers):
#   pose:       33 points x (x, y, z) = 99   -> columns   0..98
#   left hand:  21 points x (x, y, z) = 63   -> columns  99..161
#   right hand: 21 points x (x, y, z) = 63   -> columns 162..224
# A part that is not detected in a frame is stored as zeros.
N_POSE, N_HAND = 33, 21
POSE_SLICE = slice(0, 99)
LEFT_SLICE = slice(99, 162)
RIGHT_SLICE = slice(162, 225)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
POSE_CONNECTIONS = [  # shoulders, arms, hips only
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
]


def get_models():
    """Download the two model files the first time the script runs."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)


def make_landmarkers():
    hand_opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker_full.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return (vision.HandLandmarker.create_from_options(hand_opts),
            vision.PoseLandmarker.create_from_options(pose_opts))


def extract(video_path):
    """Return (features array [frames, 225], list of RGB frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    hand_lm, pose_lm = make_landmarkers()
    rows, frames = [], []
    idx, last_ts = 0, -1

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs strictly increasing timestamps in milliseconds
        ts = max(int(idx * 1000 / fps), last_ts + 1)
        last_ts = ts

        hand_res = hand_lm.detect_for_video(image, ts)
        pose_res = pose_lm.detect_for_video(image, ts)

        row = np.zeros(225, dtype=np.float32)
        if pose_res.pose_landmarks:
            row[POSE_SLICE] = [v for p in pose_res.pose_landmarks[0] for v in (p.x, p.y, p.z)]
        for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            vals = [v for p in lms for v in (p.x, p.y, p.z)]
            # MediaPipe's "Left"/"Right" label assumes a mirrored (selfie) image.
            # We store the raw label; the live pipeline must feed frames the
            # same way (unmirrored) so the labels stay consistent.
            if handed[0].category_name == "Left":
                row[LEFT_SLICE] = vals
            else:
                row[RIGHT_SLICE] = vals

        rows.append(row)
        frames.append(rgb)
        idx += 1

    cap.release()
    hand_lm.close()
    pose_lm.close()
    return np.stack(rows), frames, fps


def plot_check(features, frames, out_path):
    """Draw landmarks on the first, middle and last frame."""
    picks = [0, len(frames) // 2, len(frames) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, i in zip(axes, picks):
        h, w = frames[i].shape[:2]
        ax.imshow(frames[i])
        parts = [
            (features[i, POSE_SLICE].reshape(N_POSE, 3), POSE_CONNECTIONS, "yellow"),
            (features[i, LEFT_SLICE].reshape(N_HAND, 3), HAND_CONNECTIONS, "cyan"),
            (features[i, RIGHT_SLICE].reshape(N_HAND, 3), HAND_CONNECTIONS, "magenta"),
        ]
        for pts, conns, colour in parts:
            if not pts.any():  # part not detected in this frame
                continue
            xs, ys = pts[:, 0] * w, pts[:, 1] * h
            for a, b in conns:
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=colour, linewidth=1.5)
        ax.set_title(f"frame {i}")
        ax.axis("off")
    fig.suptitle("yellow = pose, cyan = 'Left' hand, magenta = 'Right' hand")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    get_models()
    os.makedirs(OUT_DIR, exist_ok=True)
    video_path = os.path.join(VIDEO_DIR, f"{VIDEO_ID}.mp4")

    features, frames, fps = extract(video_path)
    np.save(os.path.join(OUT_DIR, f"{VIDEO_ID}.npy"), features)
    plot_check(features, frames, os.path.join(OUT_DIR, f"{VIDEO_ID}_check.png"))

    n = len(features)
    pose_ok = features[:, POSE_SLICE].any(axis=1)
    left_ok = features[:, LEFT_SLICE].any(axis=1)
    right_ok = features[:, RIGHT_SLICE].any(axis=1)
    print(f"\nClip {VIDEO_ID}: {n} frames at {fps:.1f} fps ({n / fps:.1f} s)")
    print(f"Array shape: {features.shape}")
    print(f"Pose detected:        {pose_ok.mean():6.1%} of frames")
    print(f"'Left' hand detected: {left_ok.mean():6.1%} of frames")
    print(f"'Right' hand detected:{right_ok.mean():6.1%} of frames")
    print(f"Any hand detected:    {(left_ok | right_ok).mean():6.1%} of frames")
    print(f"\nSaved {OUT_DIR}\\{VIDEO_ID}.npy and {VIDEO_ID}_check.png")


if __name__ == "__main__":
    main()