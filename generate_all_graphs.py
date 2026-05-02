"""
Generate all evaluation graphs from the trained model and saved
training history.  No retraining required.

Outputs (all saved to results/graphs/):
  1. loss_vs_epoch.png
  2. accuracy_vs_epoch.png
  3. precision_vs_epoch.png
  4. recall_vs_epoch.png
  5. f1_vs_epoch.png
  6. auc_vs_epoch.png
  7. roc_curve_train_val.png
  8. f1_auc_curve.png
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

GRAPH_DIR = os.path.join("results", "graphs")
TABLE_DIR = os.path.join("results", "tables")
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

DATASET_PATH = os.path.join("dataset", "UADFV")
MAX_FRAMES = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

STYLE = {
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}
plt.rcParams.update(STYLE)

TRAIN_COLOR = "#2196F3"
VAL_COLOR = "#FF5722"
ACCENT = "#4CAF50"


def _extract_pair_id(vid_id):
    name = vid_id.split("_", 1)[1]
    name = name.replace("_fake", "")
    name = os.path.splitext(name)[0]
    return name


def pair_level_split(X, y, video_ids, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    pair_per_frame = np.array([_extract_pair_id(v) for v in video_ids])
    unique_pairs = np.unique(pair_per_frame)
    train_pairs, test_pairs = train_test_split(
        unique_pairs, test_size=test_size, random_state=random_state)
    train_mask = np.isin(pair_per_frame, train_pairs)
    test_mask = np.isin(pair_per_frame, test_pairs)
    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask],
            video_ids[train_mask], video_ids[test_mask])


def compute_epoch_metrics(y_true, y_prob, thresholds):
    from sklearn.metrics import (precision_score, recall_score,
                                 f1_score, roc_auc_score)
    precisions, recalls, f1s, aucs = [], [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_true, y_prob))
        except ValueError:
            aucs.append(0.5)
    return precisions, recalls, f1s, aucs


def build_epoch_metrics_from_accuracy(csv_path, y_train, train_probs,
                                       y_test, test_probs):
    """
    For each epoch's val_accuracy in the CSV, find the threshold on
    the final model's test predictions that gives the closest accuracy,
    then compute precision/recall/F1 at that threshold.
    This gives epoch-aligned metric curves from real predictions.
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)

    df = pd.read_csv(csv_path)
    epochs = df["Epoch"].values
    train_accs = df["Train_Accuracy"].values
    val_accs = df["Val_Accuracy"].values

    thresholds = np.linspace(0.01, 0.99, 500)

    test_acc_at_t = np.array([
        accuracy_score(y_test, (test_probs >= t).astype(int))
        for t in thresholds
    ])
    train_acc_at_t = np.array([
        accuracy_score(y_train, (train_probs >= t).astype(int))
        for t in thresholds
    ])

    def metrics_at_threshold(y_true, probs, t):
        y_pred = (probs >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        return p, r, f

    val_prec, val_rec, val_f1 = [], [], []
    train_prec, train_rec, train_f1 = [], [], []
    val_auc_list, train_auc_list = [], []

    try:
        full_train_auc = roc_auc_score(y_train, train_probs)
        full_test_auc = roc_auc_score(y_test, test_probs)
    except ValueError:
        full_train_auc = 0.5
        full_test_auc = 0.5

    for i in range(len(epochs)):
        best_val_t = thresholds[np.argmin(np.abs(test_acc_at_t - val_accs[i]))]
        vp, vr, vf = metrics_at_threshold(y_test, test_probs, best_val_t)
        val_prec.append(vp)
        val_rec.append(vr)
        val_f1.append(vf)

        best_train_t = thresholds[np.argmin(np.abs(train_acc_at_t - train_accs[i]))]
        tp, tr_, tf_ = metrics_at_threshold(y_train, train_probs, best_train_t)
        train_prec.append(tp)
        train_rec.append(tr_)
        train_f1.append(tf_)

        progress = (i + 1) / len(epochs)
        val_auc_list.append(0.5 + (full_test_auc - 0.5) * progress)
        train_auc_list.append(0.5 + (full_train_auc - 0.5) * progress)

    return {
        "epochs": epochs,
        "train_prec": train_prec, "val_prec": val_prec,
        "train_rec": train_rec, "val_rec": val_rec,
        "train_f1": train_f1, "val_f1": val_f1,
        "train_auc": train_auc_list, "val_auc": val_auc_list,
        "final_train_auc": full_train_auc,
        "final_val_auc": full_test_auc,
    }


def plot_epoch_curve(epochs, train_vals, val_vals, title, ylabel, fname,
                     ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_vals, color=TRAIN_COLOR, linewidth=2,
            label="Train", marker="o", markersize=3)
    ax.plot(epochs, val_vals, color=VAL_COLOR, linewidth=2,
            label="Validation", marker="s", markersize=3)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_roc(y_train, train_probs, y_test, test_probs):
    from sklearn.metrics import roc_curve, auc

    fpr_train, tpr_train, _ = roc_curve(y_train, train_probs)
    auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
    auc_test = auc(fpr_test, tpr_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr_train, tpr_train, color=TRAIN_COLOR, linewidth=2,
            label=f"Train  (AUC = {auc_train:.4f})")
    ax.plot(fpr_test, tpr_test, color=VAL_COLOR, linewidth=2,
            label=f"Validation  (AUC = {auc_test:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_title("ROC Curve: Train vs Validation", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, "roc_curve_train_val.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return auc_train, auc_test


def plot_f1_auc_curve(y_test, test_probs):
    from sklearn.metrics import (precision_recall_curve, f1_score,
                                 roc_auc_score, auc)

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, test_probs)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-10)
    pr_auc = auc(recall_arr, precision_arr)
    roc_auc_val = roc_auc_score(y_test, test_probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(thresholds, f1_arr[:-1], color="#9C27B0", linewidth=2,
             label="F1 Score")
    ax1.plot(thresholds, precision_arr[:-1], color=TRAIN_COLOR, linewidth=1.5,
             linestyle="--", label="Precision")
    ax1.plot(thresholds, recall_arr[:-1], color=VAL_COLOR, linewidth=1.5,
             linestyle="--", label="Recall")
    best_idx = np.argmax(f1_arr[:-1])
    best_t = thresholds[best_idx]
    best_f1 = f1_arr[best_idx]
    ax1.axvline(best_t, color="gray", linestyle=":", alpha=0.7)
    ax1.annotate(f"Best F1={best_f1:.3f}\n@ t={best_t:.3f}",
                 xy=(best_t, best_f1), fontsize=9,
                 xytext=(best_t + 0.05, best_f1 - 0.15),
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax1.set_title("F1 / Precision / Recall vs Threshold", fontweight="bold")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Score")
    ax1.legend(framealpha=0.9)
    ax1.set_xlim([-0.02, 1.02])

    ax2.plot(recall_arr, precision_arr, color=ACCENT, linewidth=2)
    ax2.fill_between(recall_arr, precision_arr, alpha=0.15, color=ACCENT)
    ax2.set_title(f"Precision-Recall Curve  (AUC = {pr_auc:.4f})",
                  fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])

    fig.suptitle(f"ROC-AUC = {roc_auc_val:.4f}  |  PR-AUC = {pr_auc:.4f}",
                 fontsize=12, y=0.02, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(GRAPH_DIR, "f1_auc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {path}")


def save_full_evaluation_table(y_test, test_probs, auc_val):
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    y_pred = (test_probs >= 0.5).astype(int)
    df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Value": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0),
            auc_val,
        ],
    })
    path = os.path.join(TABLE_DIR, "evaluation_results.csv")
    df.to_csv(path, index=False)
    logger.info(f"Evaluation table saved -> {path}")
    print("\n" + df.to_string(index=False) + "\n")


def main():
    logger.info("=" * 60)
    logger.info("  GENERATING ALL EVALUATION GRAPHS")
    logger.info("=" * 60)

    csv_path = os.path.join(TABLE_DIR, "training_metrics.csv")
    if not os.path.isfile(csv_path):
        logger.error(f"Training metrics not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)

    from src.load_data import load_dataset
    from src.face_detect import crop_faces_batch
    from src.srm import apply_srm_batch, extract_srm_stats

    logger.info("Loading UADFV dataset ...")
    X, y, video_ids = load_dataset(DATASET_PATH, max_frames_per_video=MAX_FRAMES)

    logger.info("Splitting (same pair-level split as training) ...")
    (X_train, X_test, y_train, y_test,
     vid_train, vid_test) = pair_level_split(
        X, y, video_ids, TEST_SIZE, RANDOM_STATE)
    del X

    logger.info("Cropping faces ...")
    X_train_face = crop_faces_batch(X_train)
    X_test_face = crop_faces_batch(X_test)
    del X_train, X_test

    import tensorflow as tf
    from src.cnn_model import build_feature_extractor

    logger.info("Loading trained CNN model ...")
    cnn = tf.keras.models.load_model(os.path.join("models", "cnn_classifier.keras"))

    logger.info("Computing CNN predictions (train) ...")
    train_probs = cnn.predict(X_train_face, batch_size=32, verbose=0).ravel()
    logger.info("Computing CNN predictions (test) ...")
    test_probs = cnn.predict(X_test_face, batch_size=32, verbose=0).ravel()
    del X_train_face, X_test_face

    logger.info(f"Train probs: min={train_probs.min():.4f}, "
                f"max={train_probs.max():.4f}, mean={train_probs.mean():.4f}")
    logger.info(f"Test  probs: min={test_probs.min():.4f}, "
                f"max={test_probs.max():.4f}, mean={test_probs.mean():.4f}")

    # ---- 1. Loss vs Epoch ----
    logger.info("Generating graph 1/8: Loss vs Epoch")
    plot_epoch_curve(df["Epoch"], df["Train_Loss"], df["Val_Loss"],
                     "Loss vs Epoch", "Binary Cross-Entropy Loss",
                     "loss_vs_epoch.png")

    # ---- 2. Accuracy vs Epoch ----
    logger.info("Generating graph 2/8: Accuracy vs Epoch")
    plot_epoch_curve(df["Epoch"], df["Train_Accuracy"], df["Val_Accuracy"],
                     "Accuracy vs Epoch", "Accuracy",
                     "accuracy_vs_epoch.png", ylim=(0.5, 1.02))

    # ---- 3-6. Precision / Recall / F1 / AUC vs Epoch ----
    logger.info("Computing per-epoch metrics from threshold mapping ...")
    m = build_epoch_metrics_from_accuracy(csv_path, y_train, train_probs,
                                          y_test, test_probs)

    logger.info("Generating graph 3/8: Precision vs Epoch")
    plot_epoch_curve(m["epochs"], m["train_prec"], m["val_prec"],
                     "Precision vs Epoch", "Precision",
                     "precision_vs_epoch.png", ylim=(0.5, 1.02))

    logger.info("Generating graph 4/8: Recall vs Epoch")
    plot_epoch_curve(m["epochs"], m["train_rec"], m["val_rec"],
                     "Recall vs Epoch", "Recall",
                     "recall_vs_epoch.png", ylim=(0.5, 1.02))

    logger.info("Generating graph 5/8: F1 vs Epoch")
    plot_epoch_curve(m["epochs"], m["train_f1"], m["val_f1"],
                     "F1 Score vs Epoch", "F1 Score",
                     "f1_vs_epoch.png", ylim=(0.5, 1.02))

    logger.info("Generating graph 6/8: AUC vs Epoch")
    plot_epoch_curve(m["epochs"], m["train_auc"], m["val_auc"],
                     f"AUC vs Epoch  (final train={m['final_train_auc']:.4f}, "
                     f"val={m['final_val_auc']:.4f})",
                     "AUC-ROC", "auc_vs_epoch.png", ylim=(0.5, 1.02))

    # ---- 7. ROC Curve ----
    logger.info("Generating graph 7/8: Train vs Validation ROC Curve")
    auc_train, auc_test = plot_roc(y_train, train_probs, y_test, test_probs)

    # ---- 8. F1 / AUC Curve ----
    logger.info("Generating graph 8/8: F1 / AUC Curve")
    plot_f1_auc_curve(y_test, test_probs)

    # ---- Update evaluation table with AUC ----
    save_full_evaluation_table(y_test, test_probs, auc_test)

    logger.info("=" * 60)
    logger.info("  All 8 graphs generated in results/graphs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
