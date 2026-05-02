"""
visualize.py
Generate and save training graphs and tabular results.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

GRAPH_DIR = os.path.join("results", "graphs")
TABLE_DIR = os.path.join("results", "tables")


def _ensure_dirs():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def plot_accuracy(history, save_path=None):
    """Accuracy vs Epoch for training and validation."""
    _ensure_dirs()
    save_path = save_path or os.path.join(GRAPH_DIR, "accuracy_vs_epoch.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    plt.title("Accuracy vs Epoch", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Accuracy graph saved -> {save_path}")


def plot_loss(history, save_path=None):
    """Loss vs Epoch for training and validation."""
    _ensure_dirs()
    save_path = save_path or os.path.join(GRAPH_DIR, "loss_vs_epoch.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Loss vs Epoch", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Loss graph saved -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Confusion matrix heatmap."""
    _ensure_dirs()
    save_path = save_path or os.path.join(GRAPH_DIR, "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved -> {save_path}")


def save_training_metrics_table(history, save_path=None):
    """Save per-epoch training metrics as a CSV table."""
    _ensure_dirs()
    save_path = save_path or os.path.join(TABLE_DIR, "training_metrics.csv")

    df = pd.DataFrame({
        "Epoch": range(1, len(history.history["accuracy"]) + 1),
        "Train_Accuracy": history.history["accuracy"],
        "Val_Accuracy": history.history["val_accuracy"],
        "Train_Loss": history.history["loss"],
        "Val_Loss": history.history["val_loss"],
    })
    df.to_csv(save_path, index=False)
    logger.info(f"Training metrics table saved -> {save_path}")
    print("\n" + "=" * 60)
    print("TRAINING METRICS (per epoch)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")
    return df


def save_evaluation_table(accuracy, precision, recall, f1, save_path=None):
    """Save final evaluation metrics as CSV."""
    _ensure_dirs()
    save_path = save_path or os.path.join(TABLE_DIR, "evaluation_results.csv")

    df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, precision, recall, f1],
    })
    df.to_csv(save_path, index=False)
    logger.info(f"Evaluation table saved -> {save_path}")
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(df.to_string(index=False))
    print("=" * 40 + "\n")
    return df


def generate_all_visuals(history, y_true, y_pred,
                         accuracy, precision, recall, f1):
    """Convenience wrapper to produce every artefact at once."""
    plot_accuracy(history)
    plot_loss(history)
    plot_confusion_matrix(y_true, y_pred)
    save_training_metrics_table(history)
    save_evaluation_table(accuracy, precision, recall, f1)
    logger.info("All visualisations and tables generated.")
