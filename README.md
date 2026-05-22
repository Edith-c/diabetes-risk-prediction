# 🩺 Diabetes Risk Prediction System

## Project Description

The Diabetes Risk Prediction System is an end-to-end machine learning application that predicts the likelihood of diabetes using medical and demographic information from a patient.

This project was built using the Pima Indians Diabetes Dataset from Kaggle and combines:

- A trained machine learning prediction model
- A Large Language Model (LLM) conversational interface
- MLflow experiment tracking
- Streamlit user interface
- Ollama local LLM integration

The application is designed for educational and demonstration purposes. Users can enter health information such as glucose level, BMI, age, and blood pressure, and the system predicts diabetes risk while providing an AI-generated explanation.

### Problem Solved

Many users do not understand how medical indicators relate to diabetes risk. This application provides:

- Fast diabetes risk estimation
- Simple AI explanations
- Conversational interaction
- Educational insight into health indicators

### Intended Users

- Students learning ML engineering
- Healthcare analytics learners
- Portfolio reviewers and recruiters
- Users exploring diabetes risk factors

---

# 📂 Project Structure

```text
your-project/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── configs/
│   └── config.yaml
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_interface.py
├── notebooks/
│   └── exploration.ipynb
└── data/
    └── .gitkeep