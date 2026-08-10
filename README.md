# 🛡️ AI-Based Network Intrusion Detection System with Hybrid Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-blue.svg)
![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-green.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

An AI-powered **Network Intrusion Detection System (NIDS)** that uses Machine Learning, Deep Learning, and Hybrid Deep Learning techniques to classify network traffic as **Normal or Attack** using the **NSL-KDD dataset**.

The project evaluates multiple ML and DL architectures, including **CNN, LSTM, GRU, Hybrid CNN-LSTM, and ensemble models**, and provides an interactive **Streamlit cybersecurity dashboard** for visualizing model performance.

The project is also **Dockerized**, allowing the dashboard to be executed in a consistent environment across different systems.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Key Features](#-key-features)
* [Project Architecture](#-project-architecture)
* [Dataset](#-dataset)
* [Models Implemented](#-models-implemented)
* [Methodology](#-methodology)
* [Project Structure](#-project-structure)
* [Requirements](#-requirements)
* [Running the Project Locally](#-running-the-project-locally)
* [Running with Docker](#-running-with-docker)
* [Docker Architecture](#-docker-architecture)
* [Training the Models](#-training-the-models)
* [Dashboard](#-dashboard)
* [Expected Outputs](#-expected-outputs)
* [Evaluation Metrics](#-evaluation-metrics)
* [Results Summary](#-results-summary)
* [Troubleshooting](#-troubleshooting)
* [Team Setup](#-team-setup)
* [Future Improvements](#-future-improvements)

---

# 📊 Project Overview

Traditional rule-based intrusion detection systems can struggle to identify complex or previously unseen network attacks.

This project explores an AI-based approach to intrusion detection by comparing conventional Machine Learning models with Deep Learning and Hybrid Deep Learning architectures.

The system performs **binary classification**:

```text
Network Traffic
       │
       ▼
   Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
 ┌───────────────┐
 │ ML / DL Models│
 └───────┬───────┘
         │
         ▼
 Normal / Attack
         │
         ▼
 Performance Analysis
         │
         ▼
 Streamlit Dashboard
```

The project uses the NSL-KDD dataset and evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# 🎯 Problem Statement

> **AI-Based Intrusion Detection Using Hybrid Deep Learning Models**

The objective is to build an intelligent Network Intrusion Detection System that:

* Classifies network traffic as **Normal or Attack**
* Uses traditional Machine Learning as baseline models
* Uses Deep Learning architectures for network traffic classification
* Combines CNN and LSTM architectures through hybrid fusion
* Uses ensemble learning to improve prediction robustness
* Evaluates all models using standard classification metrics
* Provides an interactive dashboard for model comparison and visualization

---

# ✨ Key Features

### 🤖 Machine Learning

The project implements multiple traditional ML classifiers:

* Logistic Regression
* Naive Bayes
* Support Vector Machine
* Decision Tree
* Random Forest

### 🧠 Deep Learning

Deep Learning models include:

* CNN / Conv1D
* LSTM
* GRU

### 🔗 Hybrid Deep Learning

The project includes:

* Hybrid CNN-LSTM Fusion
* Ensemble Average
* AUC-Weighted Ensemble

### 📊 Model Evaluation

Models are compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curves

### 📈 Interactive Dashboard

The Streamlit dashboard provides:

* Project overview
* Model leaderboard
* Performance comparison
* Metric visualizations
* ROC analysis
* Real-time IDS demonstration

### 🐳 Docker Support

The project includes Docker configuration so that teammates can run the dashboard without manually installing the complete Python/ML environment.

---

# 🏗️ Project Architecture

```text
                    NSL-KDD Dataset
                           │
                           ▼
                  ┌─────────────────┐
                  │ Data Preprocess │
                  └────────┬────────┘
                           │
                           ▼
                  Feature Engineering
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       ML Models                    DL Models
             │                           │
      ┌──────┼──────┐             ┌──────┼──────┐
      │      │      │             │      │      │
      ▼      ▼      ▼             ▼      ▼      ▼
     LR     SVM     RF           CNN    LSTM    GRU
             │                           │
             │                           ▼
             │                    Hybrid CNN-LSTM
             │                           │
             │                           ▼
             │                     Ensembles
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                  Model Comparison
                           │
                           ▼
                  Results & Metrics
                           │
                           ▼
                 Streamlit Dashboard
```

---

# 📂 Dataset

The project uses the **NSL-KDD dataset**, an improved version of the KDD Cup 1999 dataset commonly used for evaluating network intrusion detection systems.

### Dataset Information

| Property         | Value             |
| ---------------- | ----------------- |
| Training samples | 125,973           |
| Test samples     | 22,544            |
| Features         | 41                |
| Classification   | Binary            |
| Normal label     | `0`               |
| Attack label     | `1`               |
| File format      | Headerless `.txt` |

Dataset files:

```text
nsl-kdd/
├── KDDTrain+.txt
└── KDDTest+.txt
```

### Feature Categories

| Category               | Examples                                       |
| ---------------------- | ---------------------------------------------- |
| Connection Information | `duration`, `protocol_type`, `service`, `flag` |
| Traffic Information    | `src_bytes`, `dst_bytes`, `count`, `srv_count` |
| Content Features       | `urgent`, `hot`, `num_failed_logins`           |
| Host Statistics        | `dst_host_count`, `same_srv_rate`              |

---

# 🤖 Models Implemented

## Machine Learning Models

| Model               | Description                                            |
| ------------------- | ------------------------------------------------------ |
| Logistic Regression | Linear probabilistic classifier                        |
| Naive Bayes         | Probabilistic classifier based on feature independence |
| SVM                 | Support Vector Machine with RBF kernel                 |
| Decision Tree       | Tree-based rule classifier                             |
| Random Forest       | Ensemble of decision trees                             |

---

## Deep Learning Models

### CNN

A 1D convolutional architecture is used to learn local patterns from network traffic features.

```text
Input
 ↓
Conv1D
 ↓
Batch Normalization
 ↓
MaxPooling
 ↓
Conv1D
 ↓
Batch Normalization
 ↓
MaxPooling
 ↓
Conv1D
 ↓
Flatten
 ↓
Dense Layers
 ↓
Sigmoid Output
```

### LSTM

The LSTM architecture is used to learn sequential relationships between network traffic features.

### GRU

GRU is used as a lighter recurrent alternative to LSTM.

---

## Hybrid Models

### Hybrid CNN-LSTM

The CNN and LSTM branches process the same input in parallel.

```text
                 Input
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
       CNN Path          LSTM Path
          │                 │
          └────────┬────────┘
                   │
              Concatenate
                   │
                   ▼
              Dense Layers
                   │
                   ▼
             Binary Output
```

### Ensemble Average

Combines CNN, LSTM, and GRU prediction probabilities using soft voting.

### Ensemble Weighted

Combines model predictions using AUC-based weighting.

---

# 🔬 Methodology

The overall pipeline is:

```text
1. Load NSL-KDD Dataset
          ↓
2. Data Preprocessing
          ↓
3. Encode Categorical Features
          ↓
4. Feature Scaling
          ↓
5. Binary Label Conversion
          ↓
6. Train ML Baseline Models
          ↓
7. Train CNN / LSTM / GRU
          ↓
8. Train Hybrid CNN-LSTM
          ↓
9. Generate Ensemble Predictions
          ↓
10. Evaluate All Models
          ↓
11. Generate Results
          ↓
12. Visualize Results
          ↓
13. Streamlit Dashboard
```

### Preprocessing

The project performs:

* Dataset loading
* Feature naming
* Missing/invalid value handling
* Categorical encoding
* Numerical feature scaling
* Binary label conversion
* Train/test preparation

The binary classification is:

```text
normal → 0
attack → 1
```

---

# 📁 Project Structure

```text
AI-Based-Network-IDS-with-HybridDL/
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
│
├── README.md
├── DASHBOARD_SETUP.md
│
├── HybridDL.ipynb
├── cyber_dashboard.py
│
└── nsl-kdd/
    ├── KDDTrain+.txt
    └── KDDTest+.txt
```

### Important Files

| File                 | Purpose                                     |
| -------------------- | ------------------------------------------- |
| `HybridDL.ipynb`     | Main ML/DL training and evaluation notebook |
| `cyber_dashboard.py` | Streamlit dashboard                         |
| `Dockerfile`         | Instructions for creating the Docker image  |
| `requirements.txt`   | Python dependencies                         |
| `.dockerignore`      | Files excluded from Docker build            |
| `.gitignore`         | Files excluded from Git                     |
| `DASHBOARD_SETUP.md` | Additional dashboard documentation          |
| `nsl-kdd/`           | Training and testing datasets               |

---

# 💻 Requirements

## For Local Development

Recommended:

* Python 3.9+
* Jupyter Notebook / JupyterLab
* VS Code
* Git

Required Python packages include:

```text
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
xgboost
joblib
scikeras
streamlit
plotly
jupyter
```

---

# ▶️ Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Adidanz007/AI-Based-Network-IDS-with-HybridDL.git
```

Move into the project directory:

```bash
cd AI-Based-Network-IDS-with-HybridDL
```

---

## 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Train and Evaluate the Models

Open:

```text
HybridDL.ipynb
```

Run the notebook using Python 3.9+.

The notebook automatically loads:

```text
nsl-kdd/KDDTrain+.txt
nsl-kdd/KDDTest+.txt
```

The complete training process may take significantly longer on CPU than simply starting the dashboard.

---

# 📊 Running the Dashboard Locally

After the required results have been generated, start Streamlit:

```powershell
streamlit run cyber_dashboard.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

# 🐳 Running with Docker

Docker provides a reproducible environment for running the Streamlit dashboard.

Instead of manually installing Python, TensorFlow, Streamlit, NumPy, Pandas, and other dependencies, Docker creates an isolated environment containing the required packages.

## Docker Requirements

Install:

* Docker Desktop
* Git

You do **not** need to manually install the Python dependencies when using Docker.

---

## 1. Clone the Repository

```powershell
git clone https://github.com/Adidanz007/AI-Based-Network-IDS-with-HybridDL.git
```

```powershell
cd AI-Based-Network-IDS-with-HybridDL
```

---

## 2. Build the Docker Image

Run:

```powershell
docker build -t ai-network-ids .
```

This command:

1. Reads the `Dockerfile`
2. Creates a Python environment
3. Installs the dependencies from `requirements.txt`
4. Copies the project into the Docker image
5. Configures Streamlit
6. Creates an image called `ai-network-ids`

---

## 3. Run the Docker Container

```powershell
docker run --name ai-network-ids-container -p 8501:8501 ai-network-ids
```

The port mapping is:

```text
Your Computer                    Docker Container

localhost:8501  ───────────────► 8501
                                      │
                                      ▼
                              Streamlit Dashboard
```

---

## 4. Open the Dashboard

Open your browser:

```text
http://localhost:8501
```

The AI-Based Network Intrusion Detection Dashboard should now be available.

---

# 🐳 Docker Architecture

```text
                 GitHub Repository
                         │
                         ▼
                    Dockerfile
                         │
                         ▼
                  Docker Build
                         │
                         ▼
              ┌──────────────────┐
              │  Docker Image    │
              │  ai-network-ids  │
              └────────┬─────────┘
                       │
                  docker run
                       │
                       ▼
              ┌──────────────────┐
              │ Docker Container │
              │                  │
              │ Python           │
              │ TensorFlow       │
              │ Scikit-Learn     │
              │ Streamlit        │
              │ Pandas           │
              │ NumPy            │
              │ Plotly           │
              └────────┬─────────┘
                       │
                       ▼
                 Port 8501
                       │
                       ▼
             http://localhost:8501
```

---

# 🔄 Docker Workflow for Team Members

Once the project is available on GitHub, a teammate can run the dashboard using only:

```powershell
git clone https://github.com/Adidanz007/AI-Based-Network-IDS-with-HybridDL.git

cd AI-Based-Network-IDS-with-HybridDL

docker build -t ai-network-ids .

docker run --name ai-network-ids-container -p 8501:8501 ai-network-ids
```

Then open:

```text
http://localhost:8501
```

The teammate does not need to manually install the Python dependencies listed in `requirements.txt`.

---

# 🛑 Stop the Docker Container

To stop the running container:

```powershell
docker stop ai-network-ids-container
```

---

# ▶️ Start the Existing Container Again

After stopping it, you can start it again without rebuilding:

```powershell
docker start ai-network-ids-container
```

Then open:

```text
http://localhost:8501
```

---

# 🔍 Check Running Containers

```powershell
docker ps
```

To view all containers:

```powershell
docker ps -a
```

---

# 📜 View Container Logs

If the dashboard does not start correctly:

```powershell
docker logs ai-network-ids-container
```

This is useful for identifying Python, Streamlit, dependency, or application errors.

---

# 🧹 Remove the Container

If you want to recreate the container:

```powershell
docker stop ai-network-ids-container
docker rm ai-network-ids-container
```

Then create it again:

```powershell
docker run --name ai-network-ids-container -p 8501:8501 ai-network-ids
```

---

# 🧠 Training vs Docker Dashboard

The project has two different workflows.

## Model Training

```text
HybridDL.ipynb
      │
      ▼
NSL-KDD Dataset
      │
      ▼
Preprocessing
      │
      ▼
ML + DL Training
      │
      ▼
Model Evaluation
      │
      ▼
Results
```

## Dashboard Deployment

```text
Project Results
      │
      ▼
cyber_dashboard.py
      │
      ▼
Docker
      │
      ▼
Streamlit
      │
      ▼
localhost:8501
```

The Docker container is primarily intended to provide a **consistent environment for running the Streamlit dashboard**.

Model training can still be performed through `HybridDL.ipynb` during development and experimentation.

---

# 📈 Dashboard

The Streamlit dashboard provides an interactive interface for analyzing the performance of the implemented models.

### Dashboard Sections

### 🏠 Overview

Provides a high-level summary of:

* Dataset
* Models
* Metrics
* Project objectives

### 🏆 Model Leaderboard

Ranks models according to their evaluation performance.

### 📊 Metrics Visualization

Provides graphical comparisons of:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC

### 📈 ROC Analysis

Compares the ROC curves and AUC performance of different models.

### 🚨 IDS Demonstration

Provides an interactive demonstration of intrusion detection predictions.

---

# 🧪 Expected Outputs

The notebook produces:

| Section           | Expected Output                               |
| ----------------- | --------------------------------------------- |
| Data Loading      | Dataset shapes and class distribution         |
| Preprocessing     | Encoded/scaled feature information            |
| ML Models         | Classification metrics and confusion matrices |
| CNN               | Training curves, ROC curve and AUC            |
| CNN Tuning        | Best hyperparameters                          |
| LSTM              | Training and evaluation metrics               |
| GRU               | Evaluation and comparison                     |
| Hybrid CNN-LSTM   | Hybrid model evaluation                       |
| Ensemble Average  | Combined predictions                          |
| Ensemble Weighted | AUC-weighted predictions                      |
| ROC Analysis      | Combined ROC visualization                    |
| Model Comparison  | Complete model comparison table               |
| Dashboard         | Interactive performance visualization         |

---

# 📏 Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the overall percentage of correctly classified samples.

### Precision

Measures how many samples predicted as attacks are actually attacks.

This is important for reducing false alarms in an IDS.

### Recall

Measures how many actual attacks are correctly detected.

High recall is important for detecting as many attacks as possible.

### F1 Score

The harmonic mean of Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between Normal and Attack traffic across classification thresholds.

---

# 🏆 Results Summary

The project evaluates multiple Machine Learning and Deep Learning architectures.

### Machine Learning

1. Logistic Regression
2. Naive Bayes
3. SVM
4. Decision Tree
5. Random Forest

### Deep Learning

6. CNN
7. LSTM
8. GRU

### Hybrid / Ensemble

9. Hybrid CNN-LSTM
10. Ensemble models

The project focuses on comparing the models rather than relying on a single architecture.

The hybrid CNN-LSTM architecture is designed to combine complementary representation-learning capabilities, while ensemble models combine predictions from multiple DL architectures.

> **Note:** Reported performance depends on the exact preprocessing, training configuration, random seeds, and execution environment. Results should therefore be reproduced using the notebook before making final performance claims.

---

# ⚠️ Troubleshooting

## Docker daemon is not running

If you see:

```text
Cannot connect to the Docker daemon
```

make sure Docker Desktop is running.

Test:

```powershell
docker info
```

---

## Docker image build fails

Run:

```powershell
docker build -t ai-network-ids .
```

Read the first meaningful error in the build output.

For dependency-related problems, check:

```text
requirements.txt
```

---

## Dashboard does not open

Check whether the container is running:

```powershell
docker ps
```

Then check the logs:

```powershell
docker logs ai-network-ids-container
```

Make sure port `8501` is not already being used by another application.

---

## Port 8501 is already in use

Run the container using another host port:

```powershell
docker run --name ai-network-ids-container -p 8502:8501 ai-network-ids
```

Then open:

```text
http://localhost:8502
```

The container still uses port `8501`; only the host port has changed.

---

# 👥 Team Setup

For team members, the recommended workflow is:

```text
              GitHub
                 │
                 ▼
              git clone
                 │
                 ▼
          Project Repository
                 │
                 ▼
           docker build
                 │
                 ▼
            Docker Image
                 │
                 ▼
            docker run
                 │
                 ▼
        Streamlit Dashboard
                 │
                 ▼
       http://localhost:8501
```

This approach ensures that all team members use the same application environment and dependency configuration.

---

# 🚀 Future Improvements

Potential future enhancements include:

* Real-time network packet capture
* Live network traffic classification
* REST API for intrusion predictions
* Model serving through FastAPI
* Database integration
* Authentication and authorization
* Docker Compose for multi-service deployment
* GPU-enabled Docker configuration
* Model versioning
* CI/CD using GitHub Actions
* Container image publishing through GitHub Container Registry
* Real-time attack alerts
* Deployment to a cloud platform

---

# 📚 Project Technologies

| Technology         | Purpose                            |
| ------------------ | ---------------------------------- |
| Python             | Core programming language          |
| Pandas             | Data processing                    |
| NumPy              | Numerical computation              |
| Scikit-Learn       | Machine Learning                   |
| TensorFlow / Keras | Deep Learning                      |
| SciKeras           | Keras and Scikit-Learn integration |
| XGBoost            | ML experimentation                 |
| Matplotlib         | Visualization                      |
| Seaborn            | Statistical visualization          |
| Plotly             | Interactive visualization          |
| Streamlit          | Dashboard                          |
| Jupyter Notebook   | Model development                  |
| Docker             | Containerization                   |
| Git / GitHub       | Version control                    |

---

# 👨‍💻 Project

**AI-Based Network Intrusion Detection Using Hybrid Deep Learning Models**

This project demonstrates the application of Machine Learning and Hybrid Deep Learning techniques to network intrusion detection using the NSL-KDD benchmark dataset.

---

## ⭐ If you find this project useful

Consider giving the repository a star and sharing feedback or suggestions for improvement.
