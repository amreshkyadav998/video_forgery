"""
predict_video.py
End-to-end inference on a single video file.
Centroid-based out-of-distribution detection calibrated from actual
training feature statistics prevents false FAKE on external videos.
"""

import os
import cv2
import numpy as np
import logging

from src.srm import apply_srm, extract_srm_stats
from src.svm_model import load_svm
from src.face_detect import crop_face

logger = logging.getLogger(__name__)

IMG_SIZE = 224
MIN_FACE_RATIO = 0.25
OOD_DIST_THRESHOLD = 25.0
FAKE_THRESHOLD = 0.60

_cached_extractor = None
_cached_cnn = None
_cached_svm = None
_cached_train_stats = None


def _get_models(cnn_model_path):
    global _cached_extractor, _cached_cnn
    if _cached_extractor is None:
        import tensorflow as tf
        from src.cnn_model import build_feature_extractor as build_ext
        logger.info(f"Loading CNN classifier from {cnn_model_path} ...")
        _cached_cnn = tf.keras.models.load_model(cnn_model_path)
        _cached_extractor = build_ext(_cached_cnn)
    return _cached_cnn, _cached_extractor


def _get_svm(svm_path):
    global _cached_svm
    if _cached_svm is None:
        _cached_svm = load_svm(svm_path)
    return _cached_svm


def _get_train_stats(stats_path="models/train_stats.npz"):
    global _cached_train_stats
    if _cached_train_stats is None and os.path.isfile(stats_path):
        _cached_train_stats = dict(np.load(stats_path))
        logger.info(f"Loaded training stats from {stats_path}")
    return _cached_train_stats


def extract_frames(video_path, max_frames=20):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return np.array([])

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 300
    step = max(1, total // max_frames)

    idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        idx += step

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return np.array(frames, dtype=np.float32)


def _crop_faces_with_count(frames):
    from src.face_detect import _get_cascade
    cascade = _get_cascade()
    crops = []
    detected = 0
    for f in frames:
        gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
        if len(faces) > 0:
            detected += 1
        crops.append(crop_face(f, target_size=(224, 224)))
    pct = detected / len(frames) * 100 if frames.size else 0
    logger.info(f"Face detection: {detected}/{len(frames)} ({pct:.1f}%)")
    return np.array(crops, dtype=np.float32), detected


def _check_ood_centroid(features, stats_path="models/train_stats.npz"):
    """
    Centroid-based OOD detection calibrated from training data.

    For each frame, compute Euclidean distance to the REAL and FAKE
    class centroids. The minimum distance is the frame's distance to
    its nearest known class.

    Training data shows max in-distribution distance = ~20.
    If median min-distance exceeds OOD_DIST_THRESHOLD, the video's
    features are unlike anything seen during training.
    """
    stats = _get_train_stats(stats_path)
    if stats is None:
        logger.warning("No training stats found, skipping OOD check.")
        return False, {}

    real_c = stats["real_centroid"]
    fake_c = stats["fake_centroid"]

    dist_to_real = np.linalg.norm(features - real_c, axis=1)
    dist_to_fake = np.linalg.norm(features - fake_c, axis=1)
    min_dist = np.minimum(dist_to_real, dist_to_fake)

    median_min = float(np.median(min_dist))
    mean_min = float(min_dist.mean())
    max_min = float(min_dist.max())

    logger.info(
        f"OOD centroid check: median_min_dist={median_min:.2f}, "
        f"mean_min_dist={mean_min:.2f}, max_min_dist={max_min:.2f} "
        f"(threshold={OOD_DIST_THRESHOLD})"
    )

    is_ood = median_min > OOD_DIST_THRESHOLD

    if is_ood:
        logger.info("Video is OUT OF DISTRIBUTION (features far from both "
                     "REAL and FAKE training clusters) -> defaulting to REAL.")

    return is_ood, {"median_min_dist": median_min, "mean_min_dist": mean_min}


def predict_video(video_path,
                  cnn_model_path="models/cnn_classifier.keras",
                  svm_path="models/svm_model.pkl"):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = extract_frames(video_path)
    if frames.size == 0:
        raise ValueError("No frames extracted from the video.")

    face_crops, n_faces = _crop_faces_with_count(frames)
    face_ratio = n_faces / len(frames)

    if face_ratio < MIN_FACE_RATIO:
        logger.info(f"Too few faces ({n_faces}/{len(frames)}). "
                     "Defaulting to REAL.")
        return "REAL", 0.50

    cnn, extractor = _get_models(cnn_model_path)

    cnn_raw = extractor.predict(face_crops, batch_size=32, verbose=0)
    srm_frames = np.array([apply_srm(f) for f in face_crops], dtype=np.float32)
    cnn_srm = extractor.predict(srm_frames, batch_size=32, verbose=0)
    srm_stats = extract_srm_stats(srm_frames)
    combined = np.concatenate([cnn_raw, cnn_srm, srm_stats], axis=1)

    stats_path = os.path.join(os.path.dirname(cnn_model_path), "train_stats.npz")
    is_ood, _ = _check_ood_centroid(combined, stats_path)
    if is_ood:
        return "REAL", 0.52

    svm = _get_svm(svm_path)

    cnn_probs = cnn.predict(face_crops, batch_size=32, verbose=0).ravel()
    cnn_mean = float(cnn_probs.mean())

    svm_probs = svm.predict_proba(combined)[:, 1]
    svm_mean = float(svm_probs.mean())

    logger.info(f"CNN prob_fake={cnn_mean:.4f}  |  SVM prob_fake={svm_mean:.4f}")

    weighted_prob = 0.7 * cnn_mean + 0.3 * svm_mean

    if weighted_prob >= FAKE_THRESHOLD:
        label = "FAKE"
        confidence = weighted_prob
    else:
        label = "REAL"
        confidence = 1.0 - weighted_prob

    logger.info(f"Prediction: {label}  (confidence: {confidence:.2%})")
    return label, confidence
