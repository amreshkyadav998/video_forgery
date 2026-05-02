"""
face_detect.py
Face detection using OpenCV Haar cascade.
Deepfake artifacts concentrate around the face; cropping focuses all
downstream features on the manipulated region and removes irrelevant
background, neck, and hair.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
        if _cascade.empty():
            raise RuntimeError("Failed to load Haar cascade from OpenCV data")
    return _cascade


def crop_face(frame, target_size=(224, 224), pad_ratio=0.3):
    """
    Detect the largest face in *frame* and return a padded crop
    resized to *target_size*.  Falls back to a centre crop if no
    face is found.

    Parameters
    ----------
    frame      : (H, W, 3) float32 in [0, 1]
    target_size: output size
    pad_ratio  : extra padding around the face box (fraction of face size)
    """
    cascade = _get_cascade()
    h, w = frame.shape[:2]

    gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
    )

    if len(faces) > 0:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(fw, fh) * pad_ratio)
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(w, fx + fw + pad)
        y2 = min(h, fy + fh + pad)
        crop = frame[y1:y2, x1:x2]
    else:
        s = int(min(h, w) * 0.75)
        cy, cx = h // 2, w // 2
        crop = frame[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2]

    crop = cv2.resize(crop, target_size)
    return crop.astype(np.float32)


def crop_faces_batch(frames, target_size=(224, 224)):
    """Crop faces from an array of frames, logging detection rate."""
    cascade = _get_cascade()
    results = []
    detected = 0

    for frame in frames:
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
        if len(faces) > 0:
            detected += 1
        results.append(crop_face(frame, target_size))

    pct = detected / len(frames) * 100 if len(frames) else 0
    logger.info(f"Face detection: {detected}/{len(frames)} ({pct:.1f}%) faces found")
    return np.array(results, dtype=np.float32)
