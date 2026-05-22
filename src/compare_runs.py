import mlflow
import pandas as pd

# ============================================
# GET EXPERIMENT
# ============================================

experiment = mlflow.get_experiment_by_name(
    "Diabetes Prediction"
)

# ============================================
# SEARCH RUNS
# ============================================

runs = mlflow.search_runs(
    experiment_ids=[
        experiment.experiment_id
    ]
)

# ============================================
# SORT BY ROC-AUC
# ============================================

runs_sorted = runs.sort_values(
    by="metrics.roc_auc",
    ascending=False
)

# ============================================
# BEST RUN
# ============================================

best_run = runs_sorted.iloc[0]

print("\n===== BEST RUN =====")

print(best_run)

print("\n===== TOP EXPERIMENTS =====")

columns = [

    "tags.mlflow.runName",
    "metrics.accuracy",
    "metrics.precision",
    "metrics.recall",
    "metrics.f1_score",
    "metrics.roc_auc"
]

print(
    runs_sorted[columns]
)

print("\n===== BEST MODEL =====")

print(
    f"Run Name: "
    f"{best_run['tags.mlflow.runName']}"
)

print(
    f"ROC-AUC: "
    f"{best_run['metrics.roc_auc']:.4f}"
)