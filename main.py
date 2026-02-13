from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from models.train_model import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)
from models.evaluate_model import evaluate_model, compare_models
from models.save_load_model import save_model

from preprocessing import (
    add_features,
    clean_data,
    encode_categoricals,
    load_raw_data,
    split_features_target,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "train_processed.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "artifacts" / "xgb_model.pkl"


def run_pipeline(save_reports: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df_raw = load_raw_data(RAW_PATH)

    df_clean = clean_data(df_raw)
    df_feat = add_features(df_clean)

    df_encoded = encode_categoricals(df_feat, drop_first=True)
    X, y = split_features_target(df_encoded, target="SalePrice", log_target=True)

    processed_df = X.copy()
    processed_df["SalePrice"] = y

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, index=False)

    return processed_df, X, y


def train_and_evaluate_all(X, y):
    print("\nSuddivisione train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Linear Regression
    print("\nAddestramento Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print(f"Linear Regression → RMSE: {lr_metrics['RMSE']:.4f}, R²: {lr_metrics['R2']:.4f}")
    results["Linear Regression"] = lr_metrics

    # Random Forest
    print("\nAddestramento Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest → RMSE: {rf_metrics['RMSE']:.4f}, R²: {rf_metrics['R2']:.4f}")
    results["Random Forest"] = rf_metrics

    # XGBoost
    print("\nAddestramento XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost → RMSE: {xgb_metrics['RMSE']:.4f}, R²: {xgb_metrics['R2']:.4f}")
    results["XGBoost"] = xgb_metrics

    # Salvataggio modello finale
    save_model(xgb_model, MODEL_PATH)
    print(f"\nModello finale salvato in: {MODEL_PATH}")

    return results


if __name__ == "__main__":
    print("\nAvvio pipeline di preprocessing...")
    processed_df, X, y = run_pipeline(save_reports=False)

    print("Preprocessing completato.")
    print(f"Dataset processato salvato in: {PROCESSED_PATH}")
    print(f"Shape finale: {processed_df.shape}")

    print("\nAvvio training e valutazione dei modelli...")
    results = train_and_evaluate_all(X, y)

    print("\nConfronto modelli:")
    comparison_table = compare_models(results)
    print(comparison_table)

    print("\nPipeline completata con successo.")