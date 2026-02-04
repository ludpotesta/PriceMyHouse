from pathlib import Path
from typing import Tuple

import pandas as pd

from models.train_model import train_xgboost
from models.evaluate_model import evaluate_model
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


def save_figures(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)

    if "SalePrice" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["SalePrice"], kde=True)
        plt.title("Distribuzione di SalePrice")
        plt.tight_layout()
        plt.savefig(output_dir / "saleprice_distribution.png")
        plt.close()

    corr = df.corr(numeric_only=True)
    if not corr.empty:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, cmap="coolwarm")
        plt.title("Matrice di correlazione (numeriche)")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()


def run_pipeline(save_reports: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df_raw = load_raw_data(RAW_PATH)

    df_clean = clean_data(df_raw)
    df_feat = add_features(df_clean)

    if save_reports:
        save_figures(df_feat, FIGURES_DIR)

    df_encoded = encode_categoricals(df_feat, drop_first=True)
    X, y = split_features_target(df_encoded, target="SalePrice", log_target=True)

    processed_df = X.copy()
    processed_df["SalePrice"] = y

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_PATH, index=False)

    return processed_df, X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """Train the XGBoost model, evaluate it, and save the trained model."""
    print("\nAddestramento del modello XGBoost...")
    model = train_xgboost(X, y)

    print("Valutazione del modello...")
    metrics = evaluate_model(model, X, y)

    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, MODEL_PATH)

    print(f"\nModello salvato in: {MODEL_PATH}")
    return model, metrics


if __name__ == "__main__":
    print("\nAvvio pipeline di preprocessing...")
    processed_df, X, y = run_pipeline(save_reports=False)

    print("Preprocessing completato.")
    print(f"Dataset processato salvato in: {PROCESSED_PATH}")
    print(f"Shape finale: {processed_df.shape}")

    print("\nAvvio training del modello...")
    model, metrics = train_and_evaluate(X, y)

    print("\nPipeline completata con successo.")