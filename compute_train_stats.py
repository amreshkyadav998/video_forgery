"""
Quick script to compute training feature statistics from existing
trained models + UADFV dataset.  No retraining required.
Saves models/train_stats.npz for OOD detection.
"""

import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join("dataset", "UADFV")
MAX_FRAMES = 30


def main():
    from src.load_data import load_dataset
    from src.face_detect import crop_faces_batch
    from src.srm import apply_srm_batch, extract_srm_stats

    import tensorflow as tf
    from src.cnn_model import build_feature_extractor

    logger.info("Loading UADFV dataset ...")
    X, y, video_ids = load_dataset(DATASET_PATH, max_frames_per_video=MAX_FRAMES)
    logger.info(f"Loaded {len(X)} frames  (real={int((y==0).sum())}, fake={int((y==1).sum())})")

    logger.info("Cropping faces ...")
    X_face = crop_faces_batch(X)
    del X

    logger.info("Loading trained CNN ...")
    cnn = tf.keras.models.load_model(os.path.join("models", "cnn_classifier.keras"))
    extractor = build_feature_extractor(cnn)

    logger.info("Extracting CNN features (raw) ...")
    feats_raw = extractor.predict(X_face, batch_size=32, verbose=0)

    logger.info("Applying SRM + extracting CNN features (SRM) ...")
    X_srm = apply_srm_batch(X_face)
    feats_srm = extractor.predict(X_srm, batch_size=32, verbose=0)
    srm_stats = extract_srm_stats(X_srm)
    del X_face, X_srm

    combined = np.concatenate([feats_raw, feats_srm, srm_stats], axis=1)
    logger.info(f"Combined features shape: {combined.shape}")

    real_mask = y == 0
    fake_mask = y == 1

    stats_path = os.path.join("models", "train_stats.npz")
    np.savez(
        stats_path,
        real_centroid=combined[real_mask].mean(axis=0),
        fake_centroid=combined[fake_mask].mean(axis=0),
        real_std=combined[real_mask].std(axis=0) + 1e-8,
        fake_std=combined[fake_mask].std(axis=0) + 1e-8,
        global_mean=combined.mean(axis=0),
        global_std=combined.std(axis=0) + 1e-8,
    )
    logger.info(f"Saved training stats -> {stats_path}")

    real_feats = combined[real_mask]
    fake_feats = combined[fake_mask]
    r_centroid = real_feats.mean(axis=0)
    f_centroid = fake_feats.mean(axis=0)

    real_dists_to_real = np.linalg.norm(real_feats - r_centroid, axis=1)
    real_dists_to_fake = np.linalg.norm(real_feats - f_centroid, axis=1)
    fake_dists_to_real = np.linalg.norm(fake_feats - r_centroid, axis=1)
    fake_dists_to_fake = np.linalg.norm(fake_feats - f_centroid, axis=1)

    logger.info(f"REAL frames -> dist to real centroid: "
                f"mean={real_dists_to_real.mean():.2f}, max={real_dists_to_real.max():.2f}")
    logger.info(f"REAL frames -> dist to fake centroid: "
                f"mean={real_dists_to_fake.mean():.2f}, max={real_dists_to_fake.max():.2f}")
    logger.info(f"FAKE frames -> dist to real centroid: "
                f"mean={fake_dists_to_real.mean():.2f}, max={fake_dists_to_real.max():.2f}")
    logger.info(f"FAKE frames -> dist to fake centroid: "
                f"mean={fake_dists_to_fake.mean():.2f}, max={fake_dists_to_fake.max():.2f}")

    max_in_dist = max(real_dists_to_real.max(), fake_dists_to_fake.max())
    logger.info(f"Max in-distribution distance to own centroid: {max_in_dist:.2f}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
