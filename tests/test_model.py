import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


model = joblib.load(
    "models/diabetes_model.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


def sample_input():

    return pd.DataFrame([{
        "Pregnancies": 2,
        "Glucose": 140,
        "BloodPressure": 85,
        "SkinThickness": 30,
        "Insulin": 100,
        "BMI": 32.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45
    }])


def test_prediction_shape():

    X = sample_input()

    X_processed = preprocessor.transform(X)

    pred = model.predict(X_processed)

    assert pred.shape == (1,)


def test_prediction_type():

    X = sample_input()

    X_processed = preprocessor.transform(X)

    pred = model.predict(X_processed)

    assert isinstance(pred[0], (np.integer, int))