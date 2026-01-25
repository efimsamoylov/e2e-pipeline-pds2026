import re
from typing import Dict, List, Tuple

import numpy as np

from ...common.text import normalize_text

SENIORITY_HIERARCHY = ["C-Level", "Director", "Management", "Lead", "Senior", "Junior", "Intern"]

def predict_department_rule(
    text: str,
    lexicon: Dict[str, List[str]],
    bigram_weight: float = 3.0,
    unigram_weight: float = 1.0,
    min_score: float = 2.0,
    default_label: str = None
) -> Tuple[str, float]:
    t = normalize_text(text)
    scores: Dict[str, float] = {}

    for label, terms in lexicon.items():
        score = 0.0
        for term in terms:
            term_n = term.strip().lower()
            if not term_n:
                continue

            if " " in term_n:
                if term_n in t:
                    score += bigram_weight
            else:
                if re.search(rf"(?<!\w){re.escape(term_n)}(?!\w)", t):
                    score += unigram_weight

        if score > 0:
            scores[label] = score

    if not scores:
        return default_label, 0.0

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score < min_score:
        return default_label, best_score

    return best_label, best_score

def predict_seniority_rule(
    text: str,
    lexicon: Dict[str, List[str]],
    default_label: str = None
) -> Tuple[str, float]:
    t = normalize_text(text)

    scores = {label: 0 for label in SENIORITY_HIERARCHY}

    for label in SENIORITY_HIERARCHY:
        terms = lexicon.get(label, [])
        for term in terms:
            pattern = rf"\b{re.escape(term.lower())}\b"
            if re.search(pattern, t):
                scores[label] += 1

    if any(scores.values()):
        best_label = max(
            scores.items(),
            key=lambda kv: (kv[1], SENIORITY_HIERARCHY.index(kv[0])),
        )[0]
        return best_label, 10.0

    for label, terms in lexicon.items():
        if label in SENIORITY_HIERARCHY:
            continue
        for term in terms:
            if re.search(rf"\b{re.escape(term.lower())}\b", t):
                return label, 5.0

    return default_label, 0.0

def predict_hybrid_smart(text, rule_func, lexicon, model, ml_threshold, fallback_label):
    rule_pred, _ = rule_func(text, lexicon, default_label=None)
    if rule_pred:
        return rule_pred, 1.0, "Rule (Lexicon)"

    probs = model.predict_proba([text])[0]
    if hasattr(probs, "cpu"):
        probs = probs.cpu().detach().numpy()
    elif hasattr(probs, "numpy"):
        probs = probs.numpy()

    max_conf = float(np.max(probs))
    pred_idx = int(np.argmax(probs))

    if hasattr(model, "labels") and model.labels:
        ml_pred = model.labels[pred_idx]
    else:
        ml_pred = str(pred_idx)

    if max_conf >= ml_threshold:
        return ml_pred, max_conf, "ML"

    return fallback_label, max_conf, "Fallback"
