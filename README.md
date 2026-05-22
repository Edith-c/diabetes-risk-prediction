# 🩺 Diabetes Risk Prediction – End-to-End ML & LLM System

This project implements a complete Machine Learning and Large Language Model (LLM) workflow for predicting diabetes risk using patient medical information. The system combines supervised machine learning, MLflow experiment tracking, Streamlit deployment, and conversational AI integration through Ollama.

The application allows users to enter health information in natural language, automatically extracts structured medical features, predicts diabetes risk using a trained machine learning model, and generates a patient-friendly explanation of the prediction.

---

# Project Structure

```text
src/            → Training, preprocessing, evaluation, and Streamlit application  
tests/          → Unit tests for preprocessing, models, and interface  
configs/        → YAML configuration files  
models/         → Saved trained models and preprocessors  
notebooks/      → Exploratory analysis notebooks  
mlruns/         → MLflow experiment tracking artifacts  
data/           → Dataset files (excluded from Git)  
```

---

# Key Components

## Machine Learning

### Models Trained
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Decision Tree Classifier
- Support Vector Machine (SVM)

### Task
Predict diabetes risk (`Outcome`)

### Preprocessing
- Missing value handling
- Feature scaling
- Train/test split
- Data validation
- Leakage prevention

### Features Used
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

# Experiment Tracking (MLflow)

Each training run is tracked using MLflow.

### Logged Parameters
- model type
- hyperparameters
- preprocessing configuration

### Logged Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Logged Artifacts
- trained model
- preprocessing pipeline
- experiment metadata

The project includes multiple experiment runs with different configurations and automatically identifies the best-performing model using `mlflow.search_runs()`.

---

# LLM-Powered Interface

The application integrates a local Large Language Model using Ollama.

The LLM is responsible for:
- parsing natural language user input,
- extracting structured medical features,
- generating contextual explanations,
- handling incomplete or invalid inputs.

### Example User Input
```text
I am 45 years old with glucose 160, BMI 34, insulin 90, and blood pressure 80.
```

### System Workflow
Natural Language Input → Feature Extraction → Model Prediction → AI Explanation

---

# Streamlit Application

The frontend application is built using Streamlit.

### Features
- Interactive user interface
- Conversational health input
- Diabetes risk prediction
- AI-generated explanations
- Validation warnings for unrealistic values
- Edge case handling for missing features

---

# Testing (Pytest)

### Included Tests
- preprocessing validation tests
- model prediction tests
- interface parsing tests
- edge case handling tests

### Run Tests
```bash
pytest tests/ -v
```

---

# Configuration Management

Training hyperparameters are managed using YAML configuration files.

### Example
```yaml
random_forest:
  n_estimators: 100
  max_depth: 5
```

This prevents hardcoded training parameters and improves reproducibility.

---

# Version Control

### Git + .gitignore
The following files are excluded from Git:
- datasets
- trained models
- MLflow artifacts
- environment variables

### Excluded Files
```text
data/
models/
mlruns/
.env
```

---

# Docker Support

The project includes a Dockerfile for containerized deployment.

### Build Container
```bash
docker build -t diabetes-risk-app .
```

### Run Container
```bash
docker run -p 8501:8501 diabetes-risk-app
```

---

# How to Run

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Start Ollama
```bash
ollama serve
```

## Pull Llama Model
```bash
ollama pull llama3
```

## Train Models
```bash
python src/train.py
```

## Launch MLflow
```bash
mlflow ui
```

## Run Streamlit Application
```bash
streamlit run src/app.py
```

## Run Tests
```bash
pytest tests/ -v
```

---

# Pipeline Flow

Dataset → Preprocessing → Model Training → MLflow Tracking → Best Model Selection → Streamlit Interface → Ollama LLM Explanation

---

# Technologies Used

## Machine Learning
- Scikit-learn
- Pandas
- NumPy

## Experiment Tracking
- MLflow

## LLM Integration
- Ollama
- Llama 3

## Frontend
- Streamlit

## Deployment
- Docker

## Development Tools
- Git
- GitHub
- YAML
- Pytest

---

# Disclaimer

This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional healthcare advice.