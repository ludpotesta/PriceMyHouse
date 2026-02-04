from typing import Tuple

import numpy as np
import pandas as pd


def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode categorical features."""
    return pd.get_dummies(df, drop_first=drop_first)


def split_features_target(
    df: pd.DataFrame,
    target: str = "SalePrice",
    log_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target, with optional log transform.

    Args:
        df: Encoded dataframe.
        target: Target column name.
        log_target: Apply log1p to target if True.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    df = df.copy()
    if log_target:
        df[target] = np.log1p(df[target])

    X = df.drop(columns=[target])
    y = df[target]
    return X, y
