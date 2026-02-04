from pathlib import Path
from typing import Dict, Tuple
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

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
from models.train_model import (  # noqa: E402
    train_linear_regression,
    train_random_forest,
    train_gradient_boosting,
    train_xgboost,
)
from models.evaluate_model import evaluate_model, compare_models  # noqa: E402
from models.save_load_model import save_model  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"

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


def prepare_data(
    raw_path: Path = RAW_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load raw data, preprocess, and split into train/test."""
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = add_features(df)

    df_encoded = encode_categoricals(df, drop_first=True)
    X, y = split_features_target(df_encoded, target="SalePrice", log_target=True)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_and_evaluate(
    use_xgboost: bool = True,
    save_best: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train models, evaluate, and optionally save the best model.

    Returns:
        results_df: Metrics table sorted by RMSE.
        models: Dict of trained models by name.
    """
    X_train, X_test, y_train, y_test = prepare_data()

    models: Dict[str, object] = {}
    results: Dict[str, Dict[str, float]] = {}

    # Linear Regression
    lr = train_linear_regression(X_train, y_train)
    models["Linear Regression"] = lr
    results["Linear Regression"] = evaluate_model(lr, X_test, y_test)

    # Random Forest
    rf = train_random_forest(X_train, y_train, params=DEFAULT_RF_PARAMS)
    models["Random Forest"] = rf
    results["Random Forest"] = evaluate_model(rf, X_test, y_test)

    # Gradient Boosting
    gbr = train_gradient_boosting(X_train, y_train, params=DEFAULT_GBR_PARAMS)
    models["Gradient Boosting"] = gbr
    results["Gradient Boosting"] = evaluate_model(gbr, X_test, y_test)

    # XGBoost (optional)
    if use_xgboost:
        try:
            xgb = train_xgboost(X_train, y_train, params=DEFAULT_XGB_PARAMS)
            models["XGBoost"] = xgb
            results["XGBoost"] = evaluate_model(xgb, X_test, y_test)
        except Exception as exc:
            print(f"XGBoost non disponibile: {exc}")

    results_df = compare_models(results)

    if save_best and not results_df.empty:
        best_name = results_df.index[0]
        model = models[best_name]
        save_path = MODEL_DIR / f"{best_name.lower().replace(' ', '_')}.joblib"
        save_model(model, str(save_path))

    return results_df, models


if __name__ == "__main__":
    df_results, _ = train_and_evaluate(use_xgboost=True, save_best=False)
    print("\nRisultati modelli:\n")
    print(df_results)
