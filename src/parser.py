import re

def extract_features_local(text):

    features = {
        "Age": None,
        "Glucose": None,
        "BMI": None
    }

    # Age
    age_match = re.search(
        r"(\d+)\s*(?:years old|year old|yo)",
        text,
        re.IGNORECASE
    )

    # Glucose
    glucose_match = re.search(
        r"glucose\s*(\d+)",
        text,
        re.IGNORECASE
    )

    # BMI
    bmi_match = re.search(
        r"bmi\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if age_match:
        features["Age"] = int(age_match.group(1))

    if glucose_match:
        features["Glucose"] = int(glucose_match.group(1))

    if bmi_match:
        features["BMI"] = int(bmi_match.group(1))

    return features