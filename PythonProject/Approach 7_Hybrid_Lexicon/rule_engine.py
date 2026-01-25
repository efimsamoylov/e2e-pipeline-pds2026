import re
import json
from typing import Dict, List, Tuple, Any
from text_processing import normalize_text


def load_lexicon(path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Seniority hierarchy (from highest to lowest)
SENIORITY_HIERARCHY = ["C-Level", "Director", "Management", "Lead", "Senior", "Junior", "Intern"]


def predict_department_rule(
        text: str,
        lexicon: Dict[str, List[str]],
        bigram_weight: float = 3.0,  # Увеличим вес фраз
        unigram_weight: float = 1.0,
        min_score: float = 2.0,  # Минимальный счет, чтобы правило сработало
        default_label: str = None  # Возвращаем None, если не уверены (чтобы сработал ML)
) -> Tuple[str, float]:
    t = normalize_text(text)
    scores: Dict[str, float] = {}

    for label, terms in lexicon.items():
        score = 0.0
        for term in terms:
            term_n = term.strip().lower()
            if not term_n: continue

            if " " in term_n:  # bigram (фраза)
                if term_n in t:
                    score += bigram_weight
            else:  # unigram (слово)
                # Word boundary match (чтобы "it" не находилось внутри "audit")
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

    # Ищем по иерархии: если нашли Director, то это важнее, чем Senior
    for label in SENIORITY_HIERARCHY:
        terms = lexicon.get(label, [])
        for term in terms:
            pattern = rf"\b{re.escape(term.lower())}\b"
            if re.search(pattern, t):
                # Нашли совпадение в высоком приоритете
                return label, 10.0  # High confidence score

    # Если ничего из иерархии не нашли, ищем в остальных ключах (например Professional)
    for label, terms in lexicon.items():
        if label in SENIORITY_HIERARCHY: continue
        for term in terms:
            if re.search(rf"\b{re.escape(term.lower())}\b", t):
                return label, 5.0

    return default_label, 0.0