"""
srm.py
Steganalysis Rich Model (SRM) high-pass filter for noise-residual extraction
plus statistical feature computation for the SVM classifier.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

SRM_KERNEL = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0, -1,  2, -1,  0],
    [ 0,  2, -4,  2,  0],
    [ 0, -1,  2, -1,  0],
    [ 0,  0,  0,  0,  0]
], dtype=np.float32)


def apply_srm(image):
    """Apply the SRM high-pass filter with min-max normalisation to [0,1]."""
    try:
        filtered = cv2.filter2D(image, -1, SRM_KERNEL)
        fmin = filtered.min()
        fmax = filtered.max()
        if fmax - fmin > 1e-8:
            filtered = (filtered - fmin) / (fmax - fmin)
        else:
            filtered = np.zeros_like(filtered)
        return filtered.astype(np.float32)
    except Exception as e:
        logger.error(f"SRM filtering failed: {e}")
        return image


def apply_srm_batch(images):
    """Apply SRM to a batch of images → (N, H, W, 3)."""
    logger.info(f"Applying SRM filter to {len(images)} images ...")
    filtered = np.array([apply_srm(img) for img in images], dtype=np.float32)
    logger.info("SRM filtering complete.")
    return filtered


def extract_srm_stats(srm_images):
    """
    Compute per-image statistical features from SRM noise residuals.

    For each of the 3 colour channels, compute:
      mean, std, max, min, 25th percentile, 75th percentile

    Returns
    -------
    np.ndarray  shape (N, 18)  —  6 stats × 3 channels
    """
    logger.info(f"Extracting SRM stats from {len(srm_images)} images ...")
    features = []
    for img in srm_images:
        row = []
        for c in range(img.shape[-1]):
            ch = img[:, :, c].ravel()
            row.extend([
                float(np.mean(ch)),
                float(np.std(ch)),
                float(np.max(ch)),
                float(np.min(ch)),
                float(np.percentile(ch, 25)),
                float(np.percentile(ch, 75)),
            ])
        features.append(row)
    result = np.array(features, dtype=np.float32)
    logger.info(f"SRM features shape: {result.shape}")
    return result
