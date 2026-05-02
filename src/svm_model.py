"""
svm_model.py
SVM classifier wrapped in a Pipeline with StandardScaler.
Scaling is critical — pretrained CNN features have wildly different
scales per dimension; unscaled features cripple the RBF kernel.
"""

import os
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "svm_model.pkl")


def build_svm():
    """Pipeline: StandardScaler → SVC(RBF, C=10, probability=True)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])


def train_svm(X_train, y_train):
    """Fit the Pipeline (scaler + SVM) and return it."""
    logger.info("Training SVM pipeline (StandardScaler + RBF SVM) ...")
    clf = build_svm()
    clf.fit(X_train, y_train)
    logger.info("SVM training complete.")
    return clf


def evaluate_svm(clf, X_test, y_test):
    """Evaluate and log a classification report."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Real", "Fake"])
    logger.info(f"SVM Accuracy: {acc:.4f}")
    logger.info(f"\n{report}")
    return y_pred, acc, report


def save_svm(clf, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)
    logger.info(f"SVM pipeline saved to {path}")


def load_svm(path=MODEL_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SVM model not found at {path}")
    clf = joblib.load(path)
    logger.info(f"SVM pipeline loaded from {path}")
    return clf
