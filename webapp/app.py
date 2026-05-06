"""
app.py
Flask web application for video forgery detection.
Upload a video and get a REAL / FAKE prediction.

While inference runs, intermediate stage images (raw frames, face crops,
SRM noise residuals) are persisted under ``results/images/<run_id>/`` and
displayed back to the user in a gallery.
"""

import os
import sys
import logging
import re
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    abort,
)
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
RESULTS_IMAGES_DIR = os.path.join(PROJECT_ROOT, "results", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_IMAGES_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_IMAGES_DIR"] = RESULTS_IMAGES_DIR

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv"}
CNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_classifier.keras")
SVM_PATH = os.path.join(PROJECT_ROOT, "models", "svm_model.pkl")

STAGE_LABELS = [
    ("frames", "Extracted Frames"),
    ("faces", "Face Crops"),
    ("srm", "SRM Noise Residual"),
]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _make_run_id(filename):
    """Build a unique, filesystem-safe id like ``20260506_201233_video1``."""
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "run"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{stem}"


def _collect_stage_images(run_id):
    """
    Walk ``results/images/<run_id>/`` and return a list of stage groups
    suitable for rendering in the template:

        [{"label": "Extracted Frames",
          "images": ["/results/images/<run_id>/frames/frame_00.png", ...]},
         ...]
    """
    run_dir = os.path.join(RESULTS_IMAGES_DIR, run_id)
    groups = []
    for stage_name, stage_label in STAGE_LABELS:
        stage_dir = os.path.join(run_dir, stage_name)
        if not os.path.isdir(stage_dir):
            continue
        files = sorted(
            f for f in os.listdir(stage_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not files:
            continue
        urls = [
            url_for("results_image",
                    run_id=run_id,
                    stage=stage_name,
                    filename=f)
            for f in files
        ]
        groups.append({"label": stage_label, "images": urls})
    return groups


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/results/images/<run_id>/<stage>/<path:filename>")
def results_image(run_id, stage, filename):
    """Serve a saved intermediate image from ``results/images/``."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", run_id) or \
       not re.fullmatch(r"[A-Za-z0-9_\-]+", stage):
        abort(404)
    directory = os.path.join(RESULTS_IMAGES_DIR, run_id, stage)
    if not os.path.isdir(directory):
        abort(404)
    return send_from_directory(directory, filename)


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

    run_id = _make_run_id(filename)
    save_dir = os.path.join(RESULTS_IMAGES_DIR, run_id)
    os.makedirs(save_dir, exist_ok=True)

    try:
        label, confidence = predict_video(
            filepath,
            cnn_model_path=CNN_MODEL_PATH,
            svm_path=SVM_PATH,
            save_dir=save_dir,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for("index"))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    stage_groups = _collect_stage_images(run_id)
    logger.info(f"Run {run_id}: collected {sum(len(g['images']) for g in stage_groups)} "
                f"intermediate images across {len(stage_groups)} stages")

    return render_template("index.html",
                           result=label,
                           confidence=f"{confidence:.2%}",
                           confidence_pct=round(confidence * 100, 2),
                           stage_groups=stage_groups,
                           run_id=run_id,
                           video_name=filename)


if __name__ == "__main__":
    logger.info("Starting Flask server on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
