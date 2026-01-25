
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to 'Approach 6'
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

# Input Files
DEPT_CSV_PATH = DATA_DIR / "department-v2.csv"
SEN_CSV_PATH = DATA_DIR / "seniority-v2.csv"
ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-annotated.json"
NOT_ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-not-annotated.json"

# Output Files
VAL_REPORT_PATH = OUT_DIR / "validation_report.csv"
PREDICTIONS_PATH = OUT_DIR / "predictions_not_annotated.csv"

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CONFIDENCE_PERCENTILE = 5
USE_SMOTE = False # SetFit doesn't need SMOTE