============================================================
# AI Diabetes Risk Assessment Assistant
# Pima Indians Diabetes Dataset
# Complete ML Pipeline
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import shap

# ============================================================
# 2. LOAD DATASET
# ============================================================

# Replace with your actual dataset path
df = pd.read_csv("diabetes.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# Columns where zero values are medically invalid
invalid_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace invalid zeros with NaN
for col in invalid_columns:
    df[col] = df[col].replace(0, np.nan)

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

df[invalid_columns] = imputer.fit_transform(
    df[invalid_columns]
)

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# ============================================================
# 5. FEATURE / TARGET SPLIT
# ============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Set Shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)

# ============================================================
# 8. TRAIN MODELS
# ============================================================

# ------------------------------------------------------------
# MODEL 1 — LOGISTIC REGRESSION
# ------------------------------------------------------------

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

# ------------------------------------------------------------
# MODEL 2 — RANDOM FOREST
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_model.fit(X_train, y_train)

# ------------------------------------------------------------
# MODEL 3 — XGBOOST
# ------------------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# ============================================================
# 9. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    print(f"\n==============================")
    print(f"{model_name} RESULTS")
    print(f"==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }

# ============================================================
# 10. EVALUATE ALL MODELS
# ============================================================

lr_results = evaluate_model(
    lr_model,
    X_test,
    y_test,
    "Logistic Regression"
)

rf_results = evaluate_model(
    rf_model,
    X_test,
    y_test,
    "Random Forest"
)

xgb_results = evaluate_model(
    xgb_model,
    X_test,
    y_test,
    "XGBoost"
)

# ============================================================
# 11. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame([
    lr_results,
    rf_results,
    xgb_results
])

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df)

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

best_model = xgb_model

preds = best_model.predict(X_test)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ============================================================
# 13. ROC CURVE
# ============================================================

probs = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, probs)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label="XGBoost")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")
plt.legend()

plt.show()

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

importance = best_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance")

plt.show()

print("\nFeature Importance:")
print(feature_importance)

# ============================================================
# 15. SHAP EXPLAINABILITY
# ============================================================

explainer = shap.Explainer(best_model)

shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns
)

# ============================================================
# 16. SAVE MODEL + SCALER
# ============================================================

joblib.dump(best_model, "diabetes_model.pkl")

joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")

# ============================================================
# 17. TEST SINGLE PREDICTION
# ============================================================

sample_patient = {
    "Pregnancies": 2,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 35
}

sample_df = pd.DataFrame([sample_patient])

# Replace invalid zeros
sample_df["Insulin"] = sample_df["Insulin"].replace(0, np.nan)

# Impute missing values
sample_df[invalid_columns] = imputer.transform(
    sample_df[invalid_columns]
)

# Scale features
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = best_model.predict(sample_scaled)[0]

# Predict probability
probability = best_model.predict_proba(sample_scaled)[0][1]

print("\n==============================")
print("SINGLE PATIENT PREDICTION")
print("==============================")

print("Prediction:", prediction)

print(f"Diabetes Probability: {probability:.2%}")

if prediction == 1:
    print("Result: High likelihood of diabetes.")
else:
    print("Result: Low likelihood of diabetes.")

# ============================================================
# 18. SHAP EXPLANATION FOR SINGLE PATIENT
# ============================================================

single_shap_values = explainer(sample_scaled)

print("\nGenerating SHAP explanation plot...")

shap.plots.waterfall(single_shap_values[0])

# ============================================================
# END OF PROJECT
# ============================================================

 