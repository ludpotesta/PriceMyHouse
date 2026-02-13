from pathlib import Path
from typing import Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def load_raw_data(path: Union[str, Path] = DEFAULT_RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Place train.csv in data/raw/."
        )
    return pd.read_csv(path)
