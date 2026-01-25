import re
import json
import unicodedata
import pandas as pd
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime

def normalize_text(x: Any) -> str:
    """Normalize text: lowercase, strip accents, collapse whitespace."""
    if x is None:
        return ""
    s = str(x).lower()
    
    # Unify separators
    s = s.replace("&", " ").replace("/", " ").replace("|", " ")
    
    # Strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    
    # Remove punctuation, collapse whitespace
    _PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
    _WS_RE = re.compile(r"\s+", flags=re.UNICODE)
    
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def load_csv_data(path: str) -> pd.DataFrame:
    """Load and normalize CSV training data."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV {path} must contain 'text' and 'label' columns.")
    df["text"] = df["text"].fillna("").astype(str).apply(normalize_text)
    df["label"] = df["label"].fillna("").astype(str)
    return df

def load_profiles(path: str) -> List[Any]:
    """Load profiles from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("profiles", "data", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []

# --- Current Job Logic ---

def _parse_yyyy_mm(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}", s):
            return datetime.strptime(s, "%Y-%m")
        if re.fullmatch(r"\d{4}", s):
            return datetime.strptime(s, "%Y")
    except Exception:
        return None
    return None

def _is_active(exp: Dict[str, Any]) -> bool:
    st = (exp.get("status") or "").strip().upper()
    if st == "ACTIVE":
        return True
    return exp.get("endDate") in (None, "", "null")

def select_current_job(experiences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select the most relevant current job from a list of experiences."""
    if not experiences:
        return None

    exps = [e for e in experiences if isinstance(e, dict)]
    if not exps:
        return None

    actives = [e for e in exps if _is_active(e)]
    pool = actives if actives else exps

    def key_fn(e: Dict[str, Any]) -> Tuple[int, datetime]:
        d = _parse_yyyy_mm(e.get("startDate"))
        return (1 if d else 0, d or datetime.min)

    pool_sorted = sorted(pool, key=key_fn, reverse=True)
    return pool_sorted[0] if pool_sorted else None

def map_seniority(sen: str) -> str:
    """Map external seniority labels to internal training labels."""
    mapping = {
        "Professional": "Senior", # Mapping unknown class to closest match
    }
    return mapping.get(sen, sen)
