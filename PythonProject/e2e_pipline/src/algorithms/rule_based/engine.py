import re
from typing import Any, Dict, List, Tuple

from ...common.text import normalize_text

SENIORITY_HIERARCHY = ["C-Level", "Director", "Management", "Lead", "Senior", "Junior", "Intern"]

def predict_department_rule(
    text: str,
    lexicon: Dict[str, List[str]],
    bigram_weight: float = 2.0,
    unigram_weight: float = 1.0,
    min_score: float = 2.0,
    default_label: str = "Other"
) -> Tuple[str, Dict[str, Any]]:
    t = normalize_text(text)

    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}

    for label, terms in lexicon.items():
        score = 0.0
        m: List[str] = []
        for term in terms:
            term_n = term.strip().lower()
            if not term_n:
                continue

            if " " in term_n:
                if term_n in t:
                    score += bigram_weight
                    m.append(term_n)
            else:
                if re.search(rf"(?<!\w){re.escape(term_n)}(?!\w)", t):
                    score += unigram_weight
                    m.append(term_n)

        scores[label] = score
        matched[label] = m

    best_label = max(scores, key=scores.get) if scores else default_label
    best_score = scores.get(best_label, 0.0)

    if best_score < min_score:
        best_label = default_label if default_label in lexicon else best_label

    debug = {
        "best_score": best_score,
        "scores": scores,
        "matched_terms": {k: v for k, v in matched.items() if v},
    }
    return best_label, debug

def predict_seniority_rule(
    text: str,
    lexicon: Dict[str, List[str]],
    default_label: str = "Professional"
) -> Tuple[str, Dict[str, Any]]:
    t = normalize_text(text)

    scores = {label: 0 for label in SENIORITY_HIERARCHY}
    matched = {label: [] for label in SENIORITY_HIERARCHY}

    for label in SENIORITY_HIERARCHY:
        terms = lexicon.get(label, [])
        for term in terms:
            pattern = rf"\b{re.escape(term.lower())}\b"
            if re.search(pattern, t):
                scores[label] += 1
                matched[label].append(term)

    for label in SENIORITY_HIERARCHY:
        if scores[label] > 0:
            return label, {"matched_terms": matched[label], "all_scores": scores}

    return default_label, {"matched_terms": [], "all_scores": scores}