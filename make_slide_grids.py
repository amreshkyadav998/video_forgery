"""
make_slide_grids.py

Combine existing graphs from results/graphs/ into composite images
designed to fill TWO slides each for the EXPERIMENTS and RESULTS pages,
so individual plots remain large and clearly readable.

Output files (all saved to results/graphs/):
    EXPERIMENTS slide
        experiments_grid_1.png   (Loss | Accuracy | Precision)
        experiments_grid_2.png   (Recall | F1 | AUC)

    RESULTS slide
        results_grid_1.png       (ROC Curve | Confusion Matrix)
        results_grid_2.png       (F1 / PR Curve | Accuracy Convergence)

Run:
    python make_slide_grids.py
"""

import os
import sys
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

GRAPH_DIR = os.path.join("results", "graphs")


def load_image(name):
    path = os.path.join(GRAPH_DIR, name)
    if not os.path.isfile(path):
        logger.warning(f"Missing: {path}")
        return None
    return mpimg.imread(path)


def make_grid(image_names, captions, rows, cols, out_name,
              suptitle, figsize, title_fontsize=18, suptitle_fontsize=22):
    """Generic grid builder: places images in a rows x cols layout."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, name, cap in zip(axes, image_names, captions):
        img = load_image(name)
        if img is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"(missing)\n{name}", ha="center", va="center",
                    fontsize=12, color="red")
            continue
        ax.imshow(img)
        ax.set_title(cap, fontsize=title_fontsize, fontweight="bold", pad=10)
        ax.axis("off")

    for ax in axes[len(image_names):]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=suptitle_fontsize,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(GRAPH_DIR, out_name)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved -> {out_path}")
    return out_path


def main():
    logger.info("=" * 60)
    logger.info("  Building 2-page composite images for the PPT")
    logger.info("=" * 60)

    # ---------- EXPERIMENTS — Page 1 (3 plots, large) ----------
    make_grid(
        image_names=[
            "loss_vs_epoch.png",
            "accuracy_vs_epoch.png",
            "precision_vs_epoch.png",
        ],
        captions=[
            "Loss vs Epoch",
            "Accuracy vs Epoch",
            "Precision vs Epoch",
        ],
        rows=1, cols=3,
        out_name="experiments_grid_1.png",
        suptitle="Experiments — Training Curves (Part 1 of 2)",
        figsize=(22, 8),
    )

    # ---------- EXPERIMENTS — Page 2 (3 plots, large) ----------
    make_grid(
        image_names=[
            "recall_vs_epoch.png",
            "f1_vs_epoch.png",
            "auc_vs_epoch.png",
        ],
        captions=[
            "Recall vs Epoch",
            "F1-Score vs Epoch",
            "AUC-ROC vs Epoch",
        ],
        rows=1, cols=3,
        out_name="experiments_grid_2.png",
        suptitle="Experiments — Training Curves (Part 2 of 2)",
        figsize=(22, 8),
    )

    # ---------- RESULTS — Page 1 (ROC + Confusion Matrix) ----------
    cm_file = "confusion_matrix1.png" if os.path.isfile(
        os.path.join(GRAPH_DIR, "confusion_matrix1.png")
    ) else "confusion_matrix.png"

    make_grid(
        image_names=[
            "roc_curve_train_val.png",
            cm_file,
        ],
        captions=[
            "ROC Curve — Train vs Validation",
            "Confusion Matrix (Test Set)",
        ],
        rows=1, cols=2,
        out_name="results_grid_1.png",
        suptitle="Results — Final Model Evaluation (Part 1 of 2)",
        figsize=(20, 9),
    )

    # ---------- RESULTS — Page 2 (F1/PR + Accuracy convergence) ----------
    make_grid(
        image_names=[
            "f1_auc_curve.png",
            "accuracy_vs_epoch.png",
        ],
        captions=[
            "F1 / Precision-Recall Curve",
            "Final Accuracy Convergence",
        ],
        rows=1, cols=2,
        out_name="results_grid_2.png",
        suptitle="Results — Final Model Evaluation (Part 2 of 2)",
        figsize=(20, 9),
    )

    logger.info("=" * 60)
    logger.info("  Done. Drop these into your slides:")
    logger.info("    EXPERIMENTS slide 1 -> results/graphs/experiments_grid_1.png")
    logger.info("    EXPERIMENTS slide 2 -> results/graphs/experiments_grid_2.png")
    logger.info("    RESULTS slide 1     -> results/graphs/results_grid_1.png")
    logger.info("    RESULTS slide 2     -> results/graphs/results_grid_2.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
