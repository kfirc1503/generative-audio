"""Dlib face-landmark extraction → 68×2 motion vectors (port of upstream face_landmarks.py).

Imports dlib/cv2/imutils lazily so this module loads even if those aren't installed yet.
Run-time deps: dlib, opencv-python, imutils. Pretrained model:
/storage/kfir/data/audio_and_video/dlib_models/shape_predictor_68_face_landmarks.dat
"""
from pathlib import Path
from typing import Tuple

import numpy as np


def extract_face_landmarks(video_path: str, predictor_path: str, refresh_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Run Dlib detector + 68-pt predictor over each frame. Returns (landmarks, face_rects)."""
    import cv2
    import dlib
    from imutils import face_utils

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    tracker = dlib.correlation_tracker()

    cap = cv2.VideoCapture(video_path)
    tracking = False
    since_detect = 0
    rect = None
    landmarks, rects = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if tracking and since_detect < refresh_size:
            quality = tracker.update(gray)
            if quality < 8.75:
                tracking = False
            else:
                since_detect += 1

        if not (tracking and since_detect < refresh_size):
            since_detect = 0
            detected = detector(gray, 1)
            if detected:
                rect = detected[0]
                tracker.start_track(frame, rect)
                tracking = True

        if rect is not None:
            shape = predictor(gray, rect)
            landmarks.append(face_utils.shape_to_np(shape))
            rects.append(face_utils.rect_to_bb(rect))

    cap.release()
    return np.asarray(landmarks), np.asarray(rects)


def motion_vector(landmarks: np.ndarray) -> np.ndarray:
    """Frame-to-frame delta (paper's video features)."""
    feat = np.zeros_like(landmarks, dtype=np.float32)
    feat[1:] = landmarks[1:] - landmarks[:-1]
    return feat


def upsample_to_audio_rate(features: np.ndarray, n_audio_frames: int) -> np.ndarray:
    """Upsample 25 fps video features to ~83.33 fps audio frame rate via nearest-frame repeat."""
    n_video = features.shape[0]
    if n_video == 0 or n_audio_frames == 0:
        return np.zeros((n_audio_frames, *features.shape[1:]), dtype=features.dtype)
    idx = np.minimum((np.arange(n_audio_frames) * n_video) // n_audio_frames, n_video - 1)
    return features[idx]


def process_speaker_videos(
    video_dir: str,
    out_dir: str,
    predictor_path: str,
    file_ext: str = "mpg",
    refresh_size: int = 8,
):
    """Extract landmarks for every video in video_dir, save .npy files in out_dir,
    plus video_feat_mean.npy / video_feat_std.npy for normalization."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_paths = sorted(Path(video_dir).glob(f"*.{file_ext}"))

    feat_sum = np.zeros((68, 2), dtype=np.float64)
    feat_sq_sum = np.zeros((68, 2), dtype=np.float64)
    n_frames = 0
    for v in video_paths:
        landmarks, _ = extract_face_landmarks(str(v), predictor_path, refresh_size)
        np.save(out_dir / f"{v.stem}.npy", landmarks)
        feat = motion_vector(landmarks)
        feat_sum += feat.sum(axis=0)
        feat_sq_sum += (feat ** 2).sum(axis=0)
        n_frames += len(feat)

    mean = feat_sum / n_frames
    std = np.sqrt(feat_sq_sum / n_frames - mean ** 2)
    np.save(out_dir / "video_feat_mean.npy", mean.astype(np.float32))
    np.save(out_dir / "video_feat_std.npy", std.astype(np.float32))
