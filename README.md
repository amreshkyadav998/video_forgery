# Video Forgery Detection (SRM + CNN + LSTM + SVM)

Detect deepfake / forged videos using a pipeline that combines **Steganalysis Rich Model (SRM)** noise-residual filtering, **MobileNetV2 CNN** transfer learning, **LSTM** temporal classification, and **SVM** decision-making, with a Flask web interface for real-time predictions.

---

## Project Structure

```
video_forgery_project/
├── dataset/
│   └── UADFV/
│       ├── real/                    # Real .mp4 video files
│       └── fake/                    # Fake/deepfake .mp4 video files
├── models/
│   ├── cnn_classifier.keras         # Fine-tuned MobileNetV2 classifier
│   ├── lstm_model.keras             # Trained LSTM model
│   ├── svm_model.pkl                # Trained SVM pipeline (StandardScaler + RBF SVM)
│   ├── train_stats.npz              # Training feature statistics (for OOD detection)
│   └── video_embeddings.npz         # Per-video CNN embeddings (for identity check)
├── results/
│   ├── graphs/                      # All evaluation graphs (8 plots)
│   └── tables/                      # CSV metric tables
├── src/
│   ├── load_data.py                 # Dataset loader (extracts frames from .mp4)
│   ├── srm.py                       # SRM high-pass filter + noise stats
│   ├── face_detect.py               # OpenCV Haar cascade face detection
│   ├── cnn_model.py                 # MobileNetV2 classifier + feature extractor
│   ├── lstm_model.py                # LSTM classifier
│   ├── svm_model.py                 # SVM classifier
│   ├── predict_video.py             # Single-video inference pipeline
│   └── visualize.py                 # Graphs & tables
├── webapp/
│   ├── templates/index.html         # Web UI
│   ├── static/style.css
│   └── app.py                       # Flask web app (port 5001)
├── main.py                          # Full training pipeline
├── compute_train_stats.py           # Compute training stats after training
├── generate_all_graphs.py           # Generate all 8 evaluation graphs
├── test_predict.py                  # Quick test on UADFV sample videos
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset

Place the UADFV dataset in `dataset/UADFV/`

Each video is an `.mp4` file. The loader extracts up to **30 frames per video** at 224×224 resolution.

---

## Step 1 — Train the Model

```bash
python main.py
```

This runs the full training pipeline:

1. Load dataset → extract 30 frames per video at 224×224
2. Detect and crop faces (OpenCV Haar cascade)
3. Fine-tune MobileNetV2 on face crops (2-phase: 15 + 30 epochs)
4. Extract 274-d combined features (CNN raw 128-d + CNN SRM 128-d + SRM stats 18-d)
5. Train SVM (StandardScaler + RBF kernel)
6. Train LSTM on combined features
7. Evaluate: frame-level + ensemble + video-level accuracy
8. Save models to `models/` and graphs to `results/`

**Expected training time:** ~3–5 hours on CPU (16 GB RAM)

---

## Step 2 — Compute Training Statistics (Required for OOD detection)

Run this **once after training** to compute the nearest-neighbour identity embeddings used during inference to detect unseen subjects:

```bash
python compute_train_stats.py
```

This saves:
- `models/train_stats.npz` — class centroids and std for features
- `models/video_embeddings.npz` — per-video face embeddings for identity check

---

## Step 3 — Test on UADFV Sample Videos

Run the quick test script to verify the trained model predicts correctly on known UADFV videos:

```bash
python test_predict.py
```

**Expected output:**

```
TEST 1: UADFV FAKE video (should be FAKE)
  => FAKE  confidence=99.88%

TEST 2: UADFV REAL video (should be REAL)
  => REAL  confidence=99.29%

TEST 3: UADFV FAKE video #2 (should be FAKE)
  => FAKE  confidence=99.07%

TEST 4: UADFV REAL video #2 (should be REAL)
  => REAL  confidence=95.56%
```

---

## Step 4 — Predict on a Single Custom Video (Command Line)

To predict on any single `.mp4` video file from Python:

```python
from src.predict_video import predict_video

label, confidence = predict_video(
    video_path="path/to/your/video.mp4",
    cnn_model_path="models/cnn_classifier.keras",
    svm_path="models/svm_model.pkl"
)

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2%}")
```

**Notes:**
- The model first checks whether the subject's face matches any training identity (nearest-neighbour check).
- If the subject is **unknown** (not in the UADFV training set), the model returns **REAL** with reduced confidence as it cannot reliably classify unseen identities.
- If the subject is **known**, the CNN + SVM ensemble prediction is used.

---

## Step 5 — Generate All Evaluation Graphs

To regenerate all 8 evaluation graphs from the trained model (no retraining needed):

```bash
python generate_all_graphs.py
```

Generates the following in `results/graphs/`:

| Graph | File |
|---|---|
| Loss vs Epoch | `loss_vs_epoch.png` |
| Accuracy vs Epoch | `accuracy_vs_epoch.png` |
| Precision vs Epoch | `precision_vs_epoch.png` |
| Recall vs Epoch | `recall_vs_epoch.png` |
| F1 Score vs Epoch | `f1_vs_epoch.png` |
| AUC vs Epoch | `auc_vs_epoch.png` |
| Train vs Validation ROC Curve | `roc_curve_train_val.png` |
| F1 / Precision-Recall Curve | `f1_auc_curve.png` |

---

## Step 6 — Launch the Web Application

```bash
cd webapp
python app.py
```

Open **http://127.0.0.1:5001** in your browser.

- Upload any `.mp4`, `.avi`, `.mov`, `.mkv`, or `.wmv` video
- Click **Analyze Video**
- The result shows **REAL** or **FAKE** with a confidence percentage

---

## Model Performance (UADFV Test Set)

| Metric | Value |
|---|---|
| Frame-level Accuracy | 94.67% |
| Precision | 98.55% |
| Recall | 90.67% |
| F1-Score | 94.44% |
| AUC-ROC | 98.62% |
| **Video-level Accuracy** | **100% (20/20 videos)** |

---

## Output Files

| Artefact | Location |
|---|---|
| CNN classifier | `models/cnn_classifier.keras` |
| LSTM model | `models/lstm_model.keras` |
| SVM model | `models/svm_model.pkl` |
| Training feature stats | `models/train_stats.npz` |
| Video embeddings | `models/video_embeddings.npz` |
| Training metrics CSV | `results/tables/training_metrics.csv` |
| Evaluation results CSV | `results/tables/evaluation_results.csv` |
| All 8 graphs | `results/graphs/` |

---

## Pipeline Overview

```
Video Input
    │
    ▼
Frame Extraction (30 frames @ 224×224)
    │
    ▼
Face Detection (OpenCV Haar Cascade)
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
Raw Face Crops               SRM-Filtered Face Crops
    │                                  │
    ▼                                  ├──→ CNN Features (128-d)
MobileNetV2 CNN                        └──→ SRM Stats (18-d)
    │
    ├──→ CNN Features (128-d)
    └──→ Direct CNN Probability
    │
    ▼
Combined Features (274-d) = CNN_raw + CNN_srm + SRM_stats
    │
    ├──→ SVM Classifier → SVM Probability
    └──→ LSTM Classifier → (training only)
    │
    ▼
Ensemble: 0.7 × CNN_prob + 0.3 × SVM_prob
    │
    ▼
Video-level Majority Voting (mean across frames → threshold 0.5)
    │
    ▼
Final Prediction: REAL / FAKE
```

---

## System Requirements

- Python 3.8+
- 16 GB RAM recommended
- CPU only (uses `tensorflow-cpu`; no GPU required)
- Windows / Linux / macOS
