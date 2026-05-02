"""
load_data.py
Loads .mp4 videos from the UADFV dataset, extracts frames,
resizes, normalizes, and returns feature arrays with labels
plus video IDs for proper video-level train/test splitting.
"""

import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 224
MAX_FRAMES = 30


def extract_frames(video_path, max_frames=MAX_FRAMES):
    """
    Extract up to `max_frames` evenly-spaced frames from a video file.
    Each frame is resized to (IMG_SIZE, IMG_SIZE) and normalised to [0, 1].
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 900

    step = max(1, total_frames // max_frames)
    frame_idx = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        frame_idx += step

    cap.release()
    return frames


def load_videos_from_folder(folder_path, label, max_frames=MAX_FRAMES):
    """
    Scan a folder for .mp4 files, extract frames, label them,
    and track which video each frame belongs to.

    Returns
    -------
    frames_list  : list of np.ndarray
    labels_list  : list of int
    vid_ids_list : list of str   (one ID per frame, identifying source video)
    video_count  : int
    """
    frames_list = []
    labels_list = []
    vid_ids_list = []
    video_count = 0

    if not os.path.isdir(folder_path):
        logger.error(f"Directory not found: {folder_path}")
        return frames_list, labels_list, vid_ids_list, video_count

    # Collect .mp4 files recursively (covers subfolders like fakenew/, realnew/)
    video_files = []
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in sorted(filenames):
            if fname.lower().endswith(".mp4"):
                video_files.append(os.path.join(dirpath, fname))
    video_files = sorted(video_files)

    logger.info(f"Found {len(video_files)} .mp4 videos under {folder_path} "
                f"(including subfolders)")

    for video_path in video_files:
        vf = os.path.relpath(video_path, folder_path)
        frames = extract_frames(video_path, max_frames=max_frames)

        if len(frames) == 0:
            logger.warning(f"No frames extracted from {vf}, skipping.")
            continue

        vid_id = f"{label}_{os.path.basename(video_path)}"
        frames_list.extend(frames)
        labels_list.extend([label] * len(frames))
        vid_ids_list.extend([vid_id] * len(frames))
        video_count += 1
        logger.info(f"  [{video_count}] {vf} -> {len(frames)} frames")

    logger.info(f"Loaded {len(frames_list)} frames from {video_count} videos "
                f"(label={label}) in {folder_path}")
    return frames_list, labels_list, vid_ids_list, video_count


def load_dataset(dataset_path="dataset/UADFV", max_frames_per_video=MAX_FRAMES):
    """
    Load the full UADFV dataset.

    Returns
    -------
    X         : np.ndarray  shape (N, 128, 128, 3)  float32 [0, 1]
    y         : np.ndarray  shape (N,)               int32
    video_ids : np.ndarray  shape (N,)               str  (source video per frame)
    """
    real_path = os.path.join(dataset_path, "real")
    fake_path = os.path.join(dataset_path, "fake")

    logger.info("=" * 50)
    logger.info("Loading REAL videos ...")
    real_frames, real_labels, real_vids, real_count = load_videos_from_folder(
        real_path, label=0, max_frames=max_frames_per_video)

    logger.info("Loading FAKE videos ...")
    fake_frames, fake_labels, fake_vids, fake_count = load_videos_from_folder(
        fake_path, label=1, max_frames=max_frames_per_video)

    all_frames = real_frames + fake_frames
    all_labels = real_labels + fake_labels
    all_vids = real_vids + fake_vids

    if len(all_frames) == 0:
        logger.error("No frames loaded -- check your dataset path and .mp4 files.")
        return np.array([]), np.array([]), np.array([])

    X = np.array(all_frames, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    video_ids = np.array(all_vids)

    del all_frames, all_labels, all_vids, real_frames, fake_frames

    total_videos = real_count + fake_count
    logger.info("=" * 50)
    logger.info(f"Total videos processed : {total_videos}  "
                f"(Real: {real_count}, Fake: {fake_count})")
    logger.info(f"Total frames extracted : {X.shape[0]}  "
                f"(Real: {int(np.sum(y == 0))}, Fake: {int(np.sum(y == 1))})")
    logger.info(f"Frame shape            : {X.shape[1:]}")
    logger.info("=" * 50)

    return X, y, video_ids
