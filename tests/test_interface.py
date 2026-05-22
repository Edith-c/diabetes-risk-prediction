from src.parser import extract_features_local


def test_feature_extraction():

    text = (
        "I am 45 years old with glucose 150 "
        "and BMI 33"
    )

    result = extract_features_local(text)

    assert result["Age"] == 45
    assert result["Glucose"] == 150
    assert result["BMI"] == 33


def test_missing_input_handled():

    text = "I feel tired"

    result = extract_features_local(text)

    assert result["Glucose"] is None