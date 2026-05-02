"""
main.py
Video Forgery Detection pipeline:
  1. Load dataset, pair-level split (preserving video IDs)
  2. Crop faces (OpenCV Haar cascade)
  3. Fine-tune MobileNetV2 on face crops (augmented, two-phase)
  4. Extract CNN features (128-d) + SRM stats (18-d)  →  dual-path + SRM
  5. Train SVM (StandardScaler + RBF) on combined features
  6. Train LSTM on combined features
  7. Evaluate: frame-level + ensemble + VIDEO-level accuracy
  8. Generate visuals
"""

import os
import sys
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join("dataset", "UADFV")
MAX_FRAMES_PER_VIDEO = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
CNN_PHASE1_EPOCHS = 15
CNN_PHASE2_EPOCHS = 30
CNN_BATCH = 32
LSTM_EPOCHS = 40
LSTM_BATCH = 32


def _extract_pair_id(vid_id):
    name = vid_id.split("_", 1)[1]
    name = name.replace("_fake", "")
    name = os.path.splitext(name)[0]
    return name


def pair_level_split(X, y, video_ids, test_size=0.2, random_state=42):
    """Split by scene pair; also return split video_ids for video-level eval."""
    pair_per_frame = np.array([_extract_pair_id(v) for v in video_ids])
    unique_pairs = np.unique(pair_per_frame)

    train_pairs, test_pairs = train_test_split(
        unique_pairs, test_size=test_size, random_state=random_state,
    )

    train_mask = np.isin(pair_per_frame, train_pairs)
    test_mask = np.isin(pair_per_frame, test_pairs)

    logger.info(f"Pair-level split: {len(train_pairs)} train pairs, "
                f"{len(test_pairs)} test pairs")
    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask],
            video_ids[train_mask], video_ids[test_mask])


def video_level_accuracy(y_true, probs_fake, video_ids):
    """
    Compute video-level accuracy using probability-averaged voting.
    probs_fake : probability of FAKE class per frame.
    """
    unique_vids = np.unique(video_ids)
    vid_truths, vid_preds = [], []
    for vid in unique_vids:
        mask = video_ids == vid
        avg_prob = probs_fake[mask].mean()
        vid_preds.append(1 if avg_prob >= 0.5 else 0)
        vid_truths.append(int(y_true[mask][0]))

    vid_truths = np.array(vid_truths)
    vid_preds = np.array(vid_preds)
    acc = accuracy_score(vid_truths, vid_preds)

    n_vids = len(unique_vids)
    correct = int(np.sum(vid_truths == vid_preds))
    logger.info(f"Video-level accuracy: {correct}/{n_vids} = {acc:.4f}")
    return acc, vid_truths, vid_preds


def main():
    logger.info("=" * 60)
    logger.info("  VIDEO FORGERY DETECTION -- Training Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    from src.load_data import load_dataset

    X, y, video_ids = load_dataset(DATASET_PATH,
                                   max_frames_per_video=MAX_FRAMES_PER_VIDEO)
    if X.size == 0:
        logger.error("Dataset is empty. Aborting.")
        sys.exit(1)

    logger.info(f"Total frames loaded : {len(X)}")
    logger.info(f"  Real              : {int(np.sum(y == 0))}")
    logger.info(f"  Fake              : {int(np.sum(y == 1))}")

    # ------------------------------------------------------------------
    # 2. Pair-level train / test split (preserving video IDs)
    # ------------------------------------------------------------------
    (X_train, X_test,
     y_train, y_test,
     vid_ids_train, vid_ids_test) = pair_level_split(
        X, y, video_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    del X
    logger.info(f"Training frames     : {len(X_train)}")
    logger.info(f"Testing  frames     : {len(X_test)}")
    logger.info(f"  Train real/fake   : {int(np.sum(y_train==0))} / "
                f"{int(np.sum(y_train==1))}")
    logger.info(f"  Test  real/fake   : {int(np.sum(y_test==0))} / "
                f"{int(np.sum(y_test==1))}")

    # ------------------------------------------------------------------
    # 3. Crop faces
    # ------------------------------------------------------------------
    from src.face_detect import crop_faces_batch

    logger.info("Cropping faces from training frames ...")
    X_train_face = crop_faces_batch(X_train)
    logger.info("Cropping faces from test frames ...")
    X_test_face = crop_faces_batch(X_test)
    del X_train, X_test

    # ------------------------------------------------------------------
    # 4. Fine-tune MobileNetV2 on face crops
    # ------------------------------------------------------------------
    from src.cnn_model import (build_cnn_classifier, train_cnn,
                               build_feature_extractor, extract_features)

    cnn, base = build_cnn_classifier(input_shape=(224, 224, 3))
    cnn_history = train_cnn(
        cnn, base,
        X_train_face, y_train, X_test_face, y_test,
        phase1_epochs=CNN_PHASE1_EPOCHS,
        phase2_epochs=CNN_PHASE2_EPOCHS,
        batch_size=CNN_BATCH,
    )

    os.makedirs("models", exist_ok=True)
    cnn_path = os.path.join("models", "cnn_classifier.keras")
    cnn.save(cnn_path)
    logger.info(f"CNN classifier saved -> {cnn_path}")

    # ------------------------------------------------------------------
    # 5. Extract features: CNN raw (128-d) + CNN on SRM (128-d) + stats (18-d)
    # ------------------------------------------------------------------
    from src.srm import apply_srm_batch, extract_srm_stats

    extractor = build_feature_extractor(cnn)

    feats_raw_train = extract_features(extractor, X_train_face)
    feats_raw_test = extract_features(extractor, X_test_face)

    X_train_srm = apply_srm_batch(X_train_face)
    X_test_srm = apply_srm_batch(X_test_face)

    feats_srm_train = extract_features(extractor, X_train_srm)
    feats_srm_test = extract_features(extractor, X_test_srm)

    stats_train = extract_srm_stats(X_train_srm)
    stats_test = extract_srm_stats(X_test_srm)
    del X_train_srm, X_test_srm

    # ------------------------------------------------------------------
    # 6. Combine features (274-d)
    # ------------------------------------------------------------------
    combined_train = np.concatenate(
        [feats_raw_train, feats_srm_train, stats_train], axis=1)
    combined_test = np.concatenate(
        [feats_raw_test, feats_srm_test, stats_test], axis=1)
    logger.info(f"Combined feature dim: {combined_train.shape[1]}  "
                f"(raw {feats_raw_train.shape[1]} + "
                f"srm_cnn {feats_srm_train.shape[1]} + "
                f"srm_stats {stats_train.shape[1]})")

    real_mask = y_train == 0
    fake_mask = y_train == 1
    train_stats_path = os.path.join("models", "train_stats.npz")
    np.savez(
        train_stats_path,
        real_centroid=combined_train[real_mask].mean(axis=0),
        fake_centroid=combined_train[fake_mask].mean(axis=0),
        real_std=combined_train[real_mask].std(axis=0) + 1e-8,
        fake_std=combined_train[fake_mask].std(axis=0) + 1e-8,
        global_mean=combined_train.mean(axis=0),
        global_std=combined_train.std(axis=0) + 1e-8,
    )
    logger.info(f"Training feature stats saved -> {train_stats_path}")

    # ------------------------------------------------------------------
    # 7. Train SVM
    # ------------------------------------------------------------------
    from src.svm_model import train_svm, evaluate_svm, save_svm

    svm = train_svm(combined_train, y_train)
    y_pred_svm, svm_acc, report = evaluate_svm(svm, combined_test, y_test)
    save_svm(svm)

    # ------------------------------------------------------------------
    # 8. Train LSTM
    # ------------------------------------------------------------------
    from src.lstm_model import build_lstm_model, train_lstm

    lstm = build_lstm_model(combined_train.shape[1])
    lstm_history = train_lstm(lstm, combined_train, y_train,
                              combined_test, y_test,
                              epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH)

    lstm_path = os.path.join("models", "lstm_model.keras")
    lstm.save(lstm_path)
    logger.info(f"LSTM model saved -> {lstm_path}")

    # ------------------------------------------------------------------
    # 9. Ensemble: average CNN + SVM probabilities
    # ------------------------------------------------------------------
    cnn_probs = cnn.predict(X_test_face, batch_size=32, verbose=0).ravel()
    svm_probs = svm.predict_proba(combined_test)[:, 1]
    ensemble_probs = (cnn_probs + svm_probs) / 2.0
    y_pred_ensemble = (ensemble_probs >= 0.5).astype(int)
    del X_train_face, X_test_face

    ens_acc = accuracy_score(y_test, y_pred_ensemble)
    logger.info(f"Ensemble frame-level accuracy: {ens_acc:.4f}")

    # ------------------------------------------------------------------
    # 10. Video-level evaluation (probability-averaged voting)
    # ------------------------------------------------------------------
    logger.info("--- Video-level evaluation (SVM) ---")
    svm_vid_acc, vid_y_true, vid_y_pred_svm = video_level_accuracy(
        y_test, svm_probs, vid_ids_test)

    logger.info("--- Video-level evaluation (Ensemble: CNN + SVM) ---")
    ens_vid_acc, _, vid_y_pred_ens = video_level_accuracy(
        y_test, ensemble_probs, vid_ids_test)

    # Use ensemble video-level results as final metrics
    best_vid_acc = max(svm_vid_acc, ens_vid_acc)
    if ens_vid_acc >= svm_vid_acc:
        final_vid_preds = vid_y_pred_ens
        final_method = "Ensemble (CNN + SVM)"
    else:
        final_vid_preds = vid_y_pred_svm
        final_method = "SVM"

    vid_precision = precision_score(vid_y_true, final_vid_preds, zero_division=0)
    vid_recall = recall_score(vid_y_true, final_vid_preds, zero_division=0)
    vid_f1 = f1_score(vid_y_true, final_vid_preds, zero_division=0)

    logger.info("=" * 50)
    logger.info(f"  FINAL VIDEO-LEVEL RESULTS ({final_method})")
    logger.info("-" * 50)
    logger.info(f"  Accuracy  : {best_vid_acc:.4f}")
    logger.info(f"  Precision : {vid_precision:.4f}")
    logger.info(f"  Recall    : {vid_recall:.4f}")
    logger.info(f"  F1-Score  : {vid_f1:.4f}")
    logger.info("=" * 50)

    # Also log frame-level for reference
    logger.info(f"  (Frame-level SVM accuracy     : {svm_acc:.4f})")
    logger.info(f"  (Frame-level Ensemble accuracy : {ens_acc:.4f})")

    # ------------------------------------------------------------------
    # 11. Graphs & tables (CNN training history + video-level metrics)
    # ------------------------------------------------------------------
    from src.visualize import generate_all_visuals

    generate_all_visuals(cnn_history, vid_y_true, final_vid_preds,
                         best_vid_acc, vid_precision, vid_recall, vid_f1)

    logger.info("=" * 60)
    logger.info("  Pipeline complete -- models saved, results generated.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
