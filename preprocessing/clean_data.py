from typing import Iterable

import pandas as pd


DEFAULT_COLS_TO_DROP = ["PoolQC", "MiscFeature", "Alley", "Fence"]

DEFAULT_NONE_COLS = [
    "MasVnrType",
    "FireplaceQu",
    "GarageQual",
    "GarageFinish",
    "GarageType",
    "GarageCond",
    "BsmtFinType1",
    "BsmtFinType2",
    "BsmtExposure",
    "BsmtCond",
    "BsmtQual",
]


def clean_data(
    df: pd.DataFrame,
    cols_to_drop: Iterable[str] = DEFAULT_COLS_TO_DROP,
    none_cols: Iterable[str] = DEFAULT_NONE_COLS,
) -> pd.DataFrame:
    """Clean the raw dataset and impute missing values.

    Args:
        df: Raw dataframe.
        cols_to_drop: Columns to drop due to high missingness.
        none_cols: Categorical columns where NaN means "None".

    Returns:
        Cleaned dataframe.
    """
    df = df.copy()

    existing_drop = [col for col in cols_to_drop if col in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    if "Electrical" in df.columns:
        mode = df["Electrical"].mode(dropna=True)
        if not mode.empty:
            df["Electrical"] = df["Electrical"].fillna(mode.iloc[0])

    return df
