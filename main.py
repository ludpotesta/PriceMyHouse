from pathlib import Path
from typing import Tuple

import pandas as pd

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


def save_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Save basic EDA figures to reports/figures/."""
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
    """Run the preprocessing pipeline and return processed data.

    Returns:
        processed_df: Encoded dataset with target.
        X: Features.
        y: Target (log-transformed).
    """
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


if __name__ == "__main__":
    processed_df, X, y = run_pipeline(save_reports=False)
    print("Preprocessing completato.")
    print(f"Dataset processato salvato in: {PROCESSED_PATH}")
    print(f"Shape finale: {processed_df.shape}")
