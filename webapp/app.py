"""
app.py
Flask web application for video forgery detection.
Upload a video and get a REAL / FAKE prediction.
"""

import os
import sys
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.predict_video as pv
from src.predict_video import predict_video

pv._cached_cnn = None
pv._cached_extractor = None
pv._cached_svm = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "video_forgery_secret_key"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv"}
CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_classifier.keras")
SVM_PATH = os.path.join(PROJECT_ROOT, "models", "svm_model.pkl")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        flash("No file selected. Please upload a video.", "error")
        return redirect(url_for("index"))

    file = request.files["video"]
    if file.filename == "":
        flash("No file selected. Please upload a video.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file format. Use mp4, avi, mov, mkv, or wmv.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    logger.info(f"Uploaded video saved to {filepath}")

    try:
        label, confidence = predict_video(
            filepath,
            cnn_model_path=CNN_MODEL_PATH,
            svm_path=SVM_PATH,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for("index"))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template("index.html",
                           result=label,
                           confidence=f"{confidence:.2%}")


if __name__ == "__main__":
    logger.info("Starting Flask server on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
