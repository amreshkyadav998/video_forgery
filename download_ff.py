"""
download_ff.py
Downloads FaceForensics++ (c23) from Kaggle using kagglehub,
then organises videos into:
  dataset/UADFV/fake/fakenew/   <- manipulated (fake) videos
  dataset/UADFV/real/realnew/   <- original (real) videos

Target: 1000+ videos in each folder.

Usage:
    pip install kagglehub
    python download_ff.py
"""

import os
import shutil
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

FAKE_DEST = os.path.join("dataset", "UADFV", "fake", "fakenew")
REAL_DEST = os.path.join("dataset", "UADFV", "real", "realnew")

# Actual folder structure inside FaceForensics++_C23/
FF_BASE_SUBDIR = "FaceForensics++_C23"

FF_FAKE_SUBDIRS = [
    "FaceForensics++_C23/Deepfakes",
    "FaceForensics++_C23/DeepFakeDetection",
    "FaceForensics++_C23/Face2Face",
    "FaceForensics++_C23/FaceShifter",
    "FaceForensics++_C23/FaceSwap",
    "FaceForensics++_C23/NeuralTextures",
]

FF_REAL_SUBDIRS = [
    "FaceForensics++_C23/original",
]

MAX_FAKE_VIDEOS = 1500
MAX_REAL_VIDEOS = 1500


def collect_mp4s(root, subdirs):
    """Walk through subdirs under root and collect all .mp4 file paths."""
    collected = []
    for subdir in subdirs:
        folder = os.path.join(root, subdir)
        if not os.path.isdir(folder):
            # Try walking the root to find any .mp4 recursively as fallback
            logger.warning(f"Expected folder not found: {folder}")
            continue
        for dirpath, _, filenames in os.walk(folder):
            for fname in sorted(filenames):
                if fname.lower().endswith(".mp4"):
                    collected.append(os.path.join(dirpath, fname))
    return collected


def fallback_collect_mp4s(root, keyword):
    """
    Fallback: walk entire downloaded root and collect .mp4s
    whose path contains the keyword ('manipulated' or 'original').
    """
    collected = []
    for dirpath, _, filenames in os.walk(root):
        if keyword.lower() in dirpath.lower():
            for fname in sorted(filenames):
                if fname.lower().endswith(".mp4"):
                    collected.append(os.path.join(dirpath, fname))
    return collected


def copy_videos(src_list, dest_folder, max_count, label):
    os.makedirs(dest_folder, exist_ok=True)
    existing = len([f for f in os.listdir(dest_folder)
                    if f.lower().endswith(".mp4")])
    logger.info(f"Already in {dest_folder}: {existing} videos")

    copied = 0
    skipped = 0
    for src_path in src_list:
        if copied >= max_count:
            break
        fname = os.path.basename(src_path)
        # Prefix with label to avoid filename collisions with UADFV
        dest_name = f"ff_{label}_{fname}"
        dest_path = os.path.join(dest_folder, dest_name)
        if os.path.exists(dest_path):
            skipped += 1
            continue
        shutil.copy2(src_path, dest_path)
        copied += 1
        if copied % 100 == 0:
            logger.info(f"  Copied {copied}/{max_count} {label} videos ...")

    logger.info(f"Done: copied {copied} new {label} videos "
                f"(skipped {skipped} already existing) -> {dest_folder}")
    total = len([f for f in os.listdir(dest_folder)
                 if f.lower().endswith(".mp4")])
    logger.info(f"Total {label} videos in {dest_folder}: {total}")
    return copied


def main():
    logger.info("=" * 60)
    logger.info("  FaceForensics++ Dataset Downloader")
    logger.info("=" * 60)

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading FF++ c23 dataset from Kaggle ...")
    logger.info("(This may take a while — dataset is several GB)")
    path = kagglehub.dataset_download("xdxd003/ff-c23")
    logger.info(f"Downloaded to: {path}")

    # Print folder structure to help debug if subdirs don't match
    logger.info("Scanning downloaded folder structure ...")
    top_entries = []
    for dirpath, dirnames, filenames in os.walk(path):
        depth = dirpath.replace(path, "").count(os.sep)
        if depth <= 3:
            indent = "  " * depth
            top_entries.append(f"{indent}{os.path.basename(dirpath)}/")
        if depth >= 3:
            dirnames.clear()
    for e in top_entries[:60]:
        logger.info(e)

    # --- Collect FAKE videos ---
    logger.info("\nCollecting FAKE (manipulated) videos ...")
    fake_videos = collect_mp4s(path, FF_FAKE_SUBDIRS)

    if not fake_videos:
        logger.warning("Standard FF++ paths not found, trying fallback scan ...")
        fake_videos = fallback_collect_mp4s(path, "manipulated")

    logger.info(f"Found {len(fake_videos)} fake .mp4 files in FF++ dataset")

    # --- Collect REAL videos ---
    logger.info("\nCollecting REAL (original) videos ...")
    real_videos = collect_mp4s(path, FF_REAL_SUBDIRS)

    if not real_videos:
        logger.warning("Standard FF++ paths not found, trying fallback scan ...")
        real_videos = fallback_collect_mp4s(path, "original")

    logger.info(f"Found {len(real_videos)} real .mp4 files in FF++ dataset")

    # --- Copy to destination ---
    logger.info("\nCopying FAKE videos ...")
    copy_videos(fake_videos, FAKE_DEST, MAX_FAKE_VIDEOS, "fake")

    logger.info("\nCopying REAL videos ...")
    copy_videos(real_videos, REAL_DEST, MAX_REAL_VIDEOS, "real")

    logger.info("\n" + "=" * 60)
    logger.info("  Done! Summary:")
    logger.info(f"  Fake videos -> {FAKE_DEST}")
    logger.info(f"  Real videos -> {REAL_DEST}")
    logger.info("  Now retrain by running: python main.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
