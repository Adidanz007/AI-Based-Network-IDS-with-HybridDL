![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras%20%7C%20Scikit--Learn-orange.svg)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

# AI-Based Intrusion Detection Using Hybrid Deep Learning Models

A complete implementation of a Network Intrusion Detection System (NIDS) that combines classical Machine Learning baselines with individual Deep Learning models and a Hybrid CNN-LSTM architecture, evaluated on the NSL-KDD benchmark dataset.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Methodology / Pipeline](#methodology--pipeline)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Expected Output](#expected-output)
- [Results Summary](#results-summary)

---

## Problem Statement

> **AI-Based Intrusion Detection Using Hybrid Deep Learning Models**

Build an intelligent NIDS that:
- Classifies network traffic as **Normal** or **Attack** (binary classification)
- Uses hybrid deep learning (CNN + LSTM fusion) alongside ML baselines
- Evaluates all models against standard metrics (Accuracy, Precision, Recall, F1, AUC)
- Provides a comprehensive comparison to identify the best-performing architecture

---

## Project Structure

```
AI-Based-Network-IDS_ML-DL-main/
├── HybridDL.ipynb          ← Main notebook (run this)
├── README.md
└── nsl-kdd/
    ├── KDDTrain+.txt       ← Training dataset (125,973 records)
    └── KDDTest+.txt        ← Test dataset (22,544 records)
```

Only these 4 files are required to run the project. No other files are needed.

---

## Dataset

**NSL-KDD** — An improved version of the KDD Cup 1999 dataset, specifically designed for evaluating network intrusion detection systems.

| Property | Value |
|---|---|
| Training samples | 125,973 |
| Test samples | 22,544 |
| Features | 41 (network connection attributes) |
| Label (binary) | `normal` → 0, all attack types → 1 |
| File format | CSV (headerless `.txt`) |

### Feature Categories

| Category | Examples |
|---|---|
| Connection Info | `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes` |
| Content Features | `urgent`, `hot`, `num_failed_logins`, `logged_in`, `root_shell`, `num_shells` |
| Traffic Statistics | `count`, `srv_count`, `serror_rate`, `same_srv_rate`, `dst_host_count` |

---

## Models Implemented

### Machine Learning Baselines (5 models)

| Model | Description |
|---|---|
| Logistic Regression | Linear probabilistic classifier |
| Naive Bayes | Gaussian Bayes with feature independence assumption |
| SVM | Support Vector Machine with RBF kernel |
| Decision Tree | Non-parametric rule-based classifier |
| Random Forest | Ensemble of 100 decision trees |

### Deep Learning Models (3 models)

| Model | Architecture |
|---|---|
| **CNN (Conv1D)** | Conv1D(64) → BN → MaxPool → Conv1D(128) → BN → MaxPool → Conv1D(64) → BN → Flatten → Dense(128) → Dense(64) → Dense(1, sigmoid) |
| **LSTM** | LSTM(128, return_seq=True) + L2 → BN → Drop → LSTM(64) + L2 → BN → Drop → Dense(64) → Dense(1, sigmoid) |
| **GRU** | GRU(128, return_seq=True) + L2 → BN → Drop → GRU(64) + L2 → BN → Drop → Dense(64) → Dense(1, sigmoid) |

### Hybrid Models (3 models)

| Model | Description |
|---|---|
| **Hybrid CNN-LSTM** | Functional API parallel fusion — shared Input branches into CNN path and LSTM path, concatenated then passed through a shared Dense head |
| **Ensemble Avg** | Soft-voting average of CNN + LSTM + GRU predicted probabilities |
| **Ensemble Weighted** | AUC-proportional weighted average of CNN + LSTM + GRU probabilities |

---

## Methodology / Pipeline

```
Raw Data (KDDTrain+.txt / KDDTest+.txt)
        │
        ▼
1. PREPROCESSING
   - Load with 43-column header (41 features + outcome + level)
   - RobustScaler on numeric features
   - pd.get_dummies on categorical features (protocol_type, service, flag)
   - StandardScaler post-split (fit on train, transform on test only)
   - Binary label: normal=0, all attacks=1
        │
        ▼
2. DATA SHAPES
   - ML models:  (samples, n_features)        → x_train, x_test
   - DL models:  (samples, n_features, 1)     → x_train_dl, x_test_dl
        │
        ▼
3. ML BASELINES  →  LR, NB, SVM, DT, RF
        │
        ▼
4. DEEP LEARNING  →  CNN, LSTM, GRU
   (EarlyStopping patience=5, ReduceLROnPlateau factor=0.5, patience=3)
        │
        ▼
5. HYBRID MODELS  →  CNN-LSTM Fusion, Ensemble Avg, Ensemble Weighted
        │
        ▼
6. FULL COMPARISON
   - results_tracker dict → DataFrame
   - Styled HTML table, bar charts, radar chart, leaderboard
```

### Training Callbacks (all DL models)
```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
```

### Hyperparameter Tuning (CNN)
Uses `scikeras.wrappers.KerasClassifier` + `RandomizedSearchCV` (5-fold CV):

| Parameter | Search Space |
|---|---|
| `filters` | [32, 64, 128] |
| `dropout_rate` | [0.2, 0.3, 0.4] |
| `learning_rate` | [0.001, 0.0001] |
| `batch_size` | [32, 64, 128] |

---

## Installation

```powershell
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn xgboost joblib scikeras
```

> **Python 3.9+ required.** TensorFlow 2.10+ recommended.

---

## How to Run

1. Open `HybridDL.ipynb` in VS Code
2. Click **Select Kernel** (top-right) → choose your Python 3.9+ interpreter
3. Click **Run All** (`Ctrl+Shift+P` → "Run All Cells")

No manual input is required. The notebook reads the dataset files automatically from the `nsl-kdd/` folder.

> Estimated runtime: **20–60 minutes** on CPU | **5–15 minutes** with GPU

---

## Expected Output

| Section | Output |
|---|---|
| Data loading | Dataset shapes, HTML head tables, class distribution chart |
| Preprocessing | Encoded feature shapes, train/test split sizes, 3-D tensor confirmation |
| ML models | Accuracy/Precision/Recall/F1 per model, confusion matrix heatmaps |
| CNN | Training loss/accuracy curves, ROC curve, AUC score |
| CNN Hypertuning | Best hyperparameters from RandomizedSearchCV |
| LSTM | Training curves, evaluation metrics, CNN vs LSTM comparison |
| GRU | Evaluation + 3-model comparison + parameter count table |
| Hybrid CNN-LSTM | Model summary, training curves, evaluation |
| Ensemble Avg | Combined soft-vote accuracy and AUC |
| Ensemble Weighted | AUC-weighted vote results |
| Combined ROC | Single chart overlaying all 6 DL model ROC curves |
| Comparison table | Color-coded HTML — 10 models × 5 metrics, max highlighted green |
| Bar charts | 5 horizontal bar charts (one per metric), gold border on best |
| Radar chart | Spider chart comparing DL + Hybrid models |
| Final summary | Top-3 leaderboard, category averages, problem statement checklist ✅ |

---

## Results Summary

### Models Evaluated (10 total)

| # | Category | Model |
|---|---|---|
| 1 | ML | Logistic Regression |
| 2 | ML | Naive Bayes |
| 3 | ML | SVM |
| 4 | ML | Decision Tree |
| 5 | ML | Random Forest |
| 6 | DL | CNN (Conv1D) |
| 7 | DL | LSTM |
| 8 | DL | GRU |
| 9 | Hybrid | CNN-LSTM Fusion |
| 10 | Hybrid | Ensemble (Avg + Weighted) |

### Evaluation Metrics

- **Accuracy** — overall correctness
- **Precision** — low false positive rate (critical for IDS: avoid false alarms)
- **Recall** — low false negative rate (critical for IDS: catch all attacks)
- **F1 Score** — harmonic mean of precision and recall
- **AUC** — area under the ROC curve (ranking quality)

### Key Findings

- **Hybrid CNN-LSTM** consistently outperforms individual DL models by capturing both local (spatial) and temporal patterns simultaneously
- **Ensemble methods** further improve robustness by combining strengths of CNN, LSTM, and GRU
- **Random Forest** is the strongest ML baseline but falls short of hybrid DL on AUC
- **Binary classification** achieves >99% accuracy — the NSL-KDD dataset is well-suited for this task

---

*Problem Statement: AI-Based Intrusion Detection Using Hybrid Deep Learning Models*
