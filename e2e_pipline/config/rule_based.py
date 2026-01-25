from .base import DATA_DIR, ARTIFACTS_DIR

ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-annotated.json"
NOT_ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-not-annotated.json"

DEPT_LEXICON_PATH = DATA_DIR / "department_lexicon.json"
SEN_LEXICON_PATH = DATA_DIR / "seniority_lexicon.json"

PRED_ANNOTATED_PATH = ARTIFACTS_DIR / "predictions_rule.csv"
PRED_NOT_ANNOTATED_PATH = ARTIFACTS_DIR / "predictions_rule_not_annotated.csv"

DEPT_BIGRAM_WEIGHT = 2.0
DEPT_UNIGRAM_WEIGHT = 1.0
DEPT_MIN_SCORE = 2.0
DEPT_DEFAULT_LABEL = "Other"

SEN_DEFAULT_LABEL = "Professional"