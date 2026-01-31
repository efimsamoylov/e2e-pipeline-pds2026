import json
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List

LEXICON_PATH = Path(__file__).parent.parent / "data" / "seniority_lexicon.json"


def load_lexicon():
    if LEXICON_PATH.exists():
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


SENIORITY_LEXICON = load_lexicon()

HIERARCHY = ["C-Level", "Director", "Management", "Lead", "Senior", "Junior", "Intern"]


def predict_seniority_rule(text: str, default_label: str = "Professional") -> Tuple[str, Dict[str, Any]]:
    t = str(text).lower()

    scores = {label: 0 for label in HIERARCHY}
    matched = {label: [] for label in HIERARCHY}

    for label in HIERARCHY:
        terms = SENIORITY_LEXICON.get(label, [])
        for term in terms:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, t):
                scores[label] += 1
                matched[label].append(term)

    for label in HIERARCHY:
        if scores[label] > 0:
            return label, {"matched_terms": matched[label], "all_scores": scores}

    return default_label, {"matched_pattern": None}
