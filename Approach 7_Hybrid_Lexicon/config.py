from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
# Важно: берем модели, обученные в прошлом подходе
CHECKPOINTS_DIR = PROJECT_ROOT / "Approach 6_updated" / "checkpoints"

# Input Data
NOT_ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-not-annotated.json"

# Lexicons (Словари)
DEPT_LEXICON_PATH = DATA_DIR / "department_lexicon.json"
SEN_LEXICON_PATH = DATA_DIR / "seniority_lexicon.json"

# Output
OUTPUT_DIR = BASE_DIR / "Output"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

# Parameters
DEPT_ML_THRESHOLD = 0.99  # Порог для нейросети
SEN_ML_THRESHOLD = 0.95   # Порог для нейросети