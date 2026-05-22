import pandas as pd
import numpy as np

from src.preprocessing import preprocess_data


def sample_df():
    return pd.DataFrame({
        "Pregnancies": [1, np.nan],
        "Glucose": [150, 120],
        "BloodPressure": [80, 70],
        "BMI": [33.2, 28.1]
    })


def test_missing_values_handled():

    df = sample_df()

    processed = preprocess_data(df)

    assert processed.isnull().sum().sum() == 0


def test_numeric_scaling():

    df = sample_df()

    processed = preprocess_data(df)

    assert processed["Glucose"].max() <= 5
    assert processed["Glucose"].min() >= -5


def test_original_dataframe_not_modified():

    df = sample_df()

    original = df.copy(deep=True)

    preprocess_data(df)

    pd.testing.assert_frame_equal(df, original)


def test_output_shape_matches_input():

    df = sample_df()

    processed = preprocess_data(df)

    assert processed.shape[0] == df.shape[0]