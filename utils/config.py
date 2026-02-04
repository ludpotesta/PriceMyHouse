from pathlib import Path

# Root del progetto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Percorsi principali
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_processed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.pkl"