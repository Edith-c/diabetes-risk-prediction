import json
import joblib
import pandas as pd
import streamlit as st
import ollama

# ============================================
# LOAD OLLAMA
# ============================================

client = ollama.Client(
    host="http://localhost:11434"
)

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(
    "models/diabetes_model.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ============================================
# TITLE
# ============================================

st.title(
    "🩺 Diabetes Risk Predictor"
)

st.write(
    """
    Enter your health information below.

    Example:
    Age: 35
    Pregnancies: 2
    Glucose: 148
    Blood Pressure: 72
    Skin Thickness: 35
    Insulin: 88
    BMI: 33.6
    Diabetes Pedigree Function: 0.627
    """
)

# ============================================
# USER INPUT
# ============================================

age = st.number_input(
    "Enter your age",
    min_value=1,
    max_value=120,
    value=25
)

pregnancies = st.number_input(
    "Number of pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Enter your glucose level",
    min_value=0,
    max_value=300,
    value=100
)

blood_pressure = st.number_input(
    "Enter your blood pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Enter your skin thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Enter your insulin level",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "Enter your BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    "Enter your Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

# ============================================
# REQUIRED FEATURES
# ============================================

required_features = [

    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ============================================
# EXTRACT FEATURES
# ============================================

def extract_features(text):

    prompt = f"""
    Extract diabetes information from the text.

    Return ONLY valid JSON.

    Format:

    {{
        "Pregnancies": number or null,
        "Glucose": number or null,
        "BloodPressure": number or null,
        "SkinThickness": number or null,
        "Insulin": number or null,
        "BMI": number or null,
        "DiabetesPedigreeFunction": number or null,
        "Age": number or null
    }}

    User Input:
    {text}
    """

    response = client.chat(

        model="llama3",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response["message"]["content"]
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    start = content.find("{")
    end = content.rfind("}") + 1

    json_content = content[start:end]

    return json.loads(json_content)

# ============================================
# VALIDATE INPUT
# ============================================

def validate_ranges(features):

    warnings = []

    if features["Glucose"] is not None:

        if features["Glucose"] < 50 or features["Glucose"] > 300:

            warnings.append(
                "⚠️ Unusual glucose value."
            )

    if features["BMI"] is not None:

        if features["BMI"] < 10 or features["BMI"] > 70:

            warnings.append(
                "⚠️ Unusual BMI value."
            )

    if features["Age"] is not None:

        if features["Age"] < 1 or features["Age"] > 120:

            warnings.append(
                "⚠️ Unusual age value."
            )

    return warnings

# ============================================
# PREDICT BUTTON
# ============================================

if st.button("Predict"):

    try:

        # ============================================
        # EXTRACT DATA
        # ============================================

        extracted = {

        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

        st.subheader(
            "Extracted Features"
        )

        st.json(extracted)

        # ============================================
        # CHECK MISSING FEATURES
        # ============================================

        missing = []

        for feature in required_features:

            if extracted.get(feature) is None:

                missing.append(feature)

        # ============================================
        # HANDLE MISSING VALUES
        # ============================================

        if len(missing) > 0:

            st.warning(
                f"Missing information: {', '.join(missing)}"
            )

        else:

            # ============================================
            # VALIDATE RANGES
            # ============================================

            warnings = validate_ranges(
                extracted
            )

            for warning in warnings:

                st.warning(warning)

            # ============================================
            # PREPROCESS INPUT
            # ============================================

            input_df = pd.DataFrame(
                [extracted]
            )

            processed_input = (
                preprocessor.transform(
                    input_df
                )
            )

            # ============================================
            # MODEL PREDICTION
            # ============================================

            prediction = model.predict(
                processed_input
            )[0]

            probability = model.predict_proba(
                processed_input
            )[0][1]

            # ============================================
            # RISK LEVEL
            # ============================================

            if probability >= 0.75:

                risk = "High Risk"

            elif probability >= 0.55:

                risk = "Moderate Risk"

            else:

                risk = "Low Risk"

            # ============================================
            # RESULTS
            # ============================================

            st.subheader(
                "Prediction Results"
            )

            st.write(
                f"Prediction: {prediction}"
            )

            st.write(
                f"Probability: {probability:.2%}"
            )

            st.write(
                f"Risk Level: {risk}"
            )

            # ============================================
            # AI EXPLANATION
            # ============================================

            explanation_prompt = f"""
            Explain this diabetes prediction
            in simple language.

            Prediction:
            {prediction}

            Probability:
            {probability:.2%}

            Risk:
            {risk}

            Patient Data:
            {json.dumps(extracted, indent=2)}

            Include:
            - meaning of the result
            - possible risk factors
            - reminder this is not medical advice
            """

            explanation_response = client.chat(

                model="llama3",

                messages=[
                    {
                        "role": "user",
                        "content": explanation_prompt
                    }
                ]
            )

            explanation = (
                explanation_response["message"]["content"]
            )

            st.subheader(
                "AI Explanation"
            )

            st.write(
                explanation
            )

            # ============================================
            # DISCLAIMER
            # ============================================

            st.info(
                """
                This tool is for educational purposes only.
                It is not medical advice.
                """
            )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )