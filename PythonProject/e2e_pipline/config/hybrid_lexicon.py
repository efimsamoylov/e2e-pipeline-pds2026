from .base import DATA_DIR, ARTIFACTS_DIR, MODELS_DIR

NOT_ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-not-annotated.json"
ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-annotated.json"

DEPT_LEXICON_PATH = DATA_DIR / "department_lexicon.json"
SEN_LEXICON_PATH = DATA_DIR / "seniority_lexicon.json"

OUTPUT_DIR = ARTIFACTS_DIR / "hybrid"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

DEPT_ML_THRESHOLD = 0.99
SEN_ML_THRESHOLD = 0.95