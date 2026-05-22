import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from preprocessing import (
    load_data,
    preprocess_data
)

from evaluate import evaluate_model

# ============================================
# LOAD CONFIG
# ============================================

with open(
    "configs/config.yaml",
    "r"
) as file:

    config = yaml.safe_load(file)

# ============================================
# CREATE DIRECTORIES
# ============================================

os.makedirs(
    "models",
    exist_ok=True
)

# ============================================
# SET MLFLOW EXPERIMENT
# ============================================

mlflow.set_experiment(
    config["mlflow"]["experiment_name"]
)

# ============================================
# LOAD DATA
# ============================================

data_path = "data/diabetes.csv"

df = load_data(data_path)

(
    X_train,
    X_test,
    y_train,
    y_test
) = preprocess_data(df)

# ============================================
# EXPERIMENT CONFIGURATIONS
# ============================================

experiments = [

    {
        "name": "LogisticRegression",

        "model": LogisticRegression(
            max_iter=1000
        ),

        "params": {
            "algorithm": "LogisticRegression",
            "max_iter": 1000
        }
    },

    {
        "name": "RandomForest_100",

        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        ),

        "params": {
            "algorithm": "RandomForest",
            "n_estimators": 100,
            "max_depth": 5
        }
    },

    {
        "name": "RandomForest_200",

        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ),

        "params": {
            "algorithm": "RandomForest",
            "n_estimators": 200,
            "max_depth": 10
        }
    },

    {
        "name": "GradientBoosting_01",

        "model": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

        "params": {
            "algorithm": "GradientBoosting",
            "n_estimators": 100,
            "learning_rate": 0.1
        }
    },

    {
        "name": "GradientBoosting_005",

        "model": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        ),

        "params": {
            "algorithm": "GradientBoosting",
            "n_estimators": 200,
            "learning_rate": 0.05
        }
    }
]

# ============================================
# TRAINING LOOP
# ============================================

results = []

best_model = None
best_score = 0

for experiment in experiments:

    model_name = experiment["name"]

    model = experiment["model"]

    params = experiment["params"]

    print(f"\nTraining {model_name}...")

    with mlflow.start_run(
        run_name=model_name
    ):

        # ============================================
        # LOG PARAMETERS
        # ============================================

        mlflow.log_params(params)

        # ============================================
        # LOG DATASET INFO
        # ============================================

        mlflow.log_param(
            "dataset_name",
            "Pima Indians Diabetes Dataset"
        )

        mlflow.log_param(
            "dataset_rows",
            len(df)
        )

        mlflow.log_param(
            "dataset_features",
            X_train.shape[1]
        )

        # ============================================
        # TRAIN MODEL
        # ============================================

        model.fit(
            X_train,
            y_train
        )

        # ============================================
        # PREDICTIONS
        # ============================================

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # ============================================
        # EVALUATION
        # ============================================

        metrics = evaluate_model(
            model_name,
            y_test,
            predictions,
            probabilities
        )

        results.append(metrics)

        # ============================================
        # LOG METRICS
        # ============================================

        for metric_name, metric_value in metrics.items():

            if metric_name != "Model":

                mlflow.log_metric(
                    metric_name.lower().replace(
                        " ",
                        "_"
                    ),
                    metric_value
                )

        # ============================================
        # LOG MODEL ARTIFACT
        # ============================================

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # ============================================
        # TRACK BEST MODEL
        # ============================================

        if metrics["ROC-AUC"] > best_score:

            best_score = metrics["ROC-AUC"]

            best_model = model

# ============================================
# SAVE BEST MODEL
# ============================================

joblib.dump(
    best_model,
    "models/diabetes_model.pkl"
)

print("\nBest model saved successfully!")

# ============================================
# RESULTS TABLE
# ============================================

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")

print(results_df)