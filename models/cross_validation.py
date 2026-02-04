from pathlib import Path
from typing import Dict
import sys

import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preprocessing import (  # noqa: E402
    load_raw_data,
    clean_data,
    add_features,
    encode_categoricals,
    split_features_target,
)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

DEFAULT_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "random_state": 42,
}

DEFAULT_GBR_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": 42,
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def prepare_data(raw_path: Path = RAW_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = add_features(df)

    df_encoded = encode_categoricals(df, drop_first=True)
    X, y = split_features_target(df_encoded, target="SalePrice", log_target=True)
    return X, y


def build_models(use_xgboost: bool = True) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(**DEFAULT_RF_PARAMS),
        "Gradient Boosting": GradientBoostingRegressor(**DEFAULT_GBR_PARAMS),
    }

    if use_xgboost:
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(**DEFAULT_XGB_PARAMS)
        except Exception as exc:
            print(f"XGBoost non disponibile: {exc}")

    return models


def run_cross_validation(
    n_splits: int = 5,
    use_xgboost: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run K-Fold cross-validation and return mean RMSE/R2.

    Args:
        n_splits: Number of folds.
        use_xgboost: Whether to include XGBoost if available.
        random_state: Random seed for reproducibility.
    """
    X, y = prepare_data()

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}

    results: Dict[str, Dict[str, float]] = {}
    for name, model in build_models(use_xgboost=use_xgboost).items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "RMSE": -scores["test_rmse"].mean(),
            "R2": scores["test_r2"].mean(),
        }

    df = pd.DataFrame(results).T.sort_values(by="RMSE", ascending=True)
    return df


if __name__ == "__main__":
    df_results = run_cross_validation(n_splits=5, use_xgboost=True, random_state=42)
    print("\nRisultati Cross-Validation (5-fold):\n")
    print(df_results)
