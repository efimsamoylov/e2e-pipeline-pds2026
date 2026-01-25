import json
import re
from typing import Any, Dict, List

def normalize_text(text: str) -> str:
    """Normalize text: lowercase and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_profiles(json_path):
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

def select_current_job(experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple selection of the first job for inference."""
    if not experiences:
        return None
    # Logic can be improved, taking the first one for now
    return experiences[0]