import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):

    df_copy = df.copy()

    imputer = SimpleImputer(strategy="mean")

    numeric_cols = df_copy.columns

    df_copy[numeric_cols] = imputer.fit_transform(
        df_copy[numeric_cols]
    )

    scaler = StandardScaler()

    df_copy[numeric_cols] = scaler.fit_transform(
        df_copy[numeric_cols]
    )

    return df_copy