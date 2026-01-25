import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

def load_profiles(json_path: Path) -> List[Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("profiles", "data", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []

def load_lexicon(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)