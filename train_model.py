import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("./data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Preprocess
preprocessor = StandardScaler()

X_processed = preprocessor.fit_transform(X)

# Train model
model = RandomForestClassifier()

model.fit(X_processed, y)

# Save files
joblib.dump(
    model,
    "models/diabetes_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("Done")