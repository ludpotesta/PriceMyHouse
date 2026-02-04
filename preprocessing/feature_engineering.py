import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used in the notebook.

    Features:
        - HouseAge: YrSold - YearBuilt
        - RemodAge: YrSold - YearRemodAdd
        - TotalSF: TotalBsmtSF + 1stFlrSF + 2ndFlrSF
    """
    df = df.copy()

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = (
            df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        )

    return df
