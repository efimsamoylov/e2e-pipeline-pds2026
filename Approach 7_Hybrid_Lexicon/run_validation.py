import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from setfit import SetFitModel

# Импорт конфигурации и функций (убедитесь, что пути в config.py правильные для валидации)
from config import (
    CHECKPOINTS_DIR, DATA_DIR,
    DEPT_LEXICON_PATH, SEN_LEXICON_PATH,
    DEPT_ML_THRESHOLD, SEN_ML_THRESHOLD
)
from text_processing import select_current_job, normalize_text
from rule_engine import load_lexicon, predict_department_rule, predict_seniority_rule

# Путь к аннотированному файлу (переопределяем или берем из config)
ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-annotated.json"


def predict_hybrid_smart(
        text: str,
        rule_func: callable,
        lexicon: dict,
        model,
        ml_threshold: float,
        fallback_label: str
):
    """
    Hybrid Prediction Logic:
    1. Rule-Based Check (High precision dictionary match).
    2. If no rule match -> ML Model (SetFit).
    3. If ML confidence < threshold -> Fallback.
    """

    # 1. Rules
    rule_pred, rule_score = rule_func(text, lexicon, default_label=None)
    if rule_pred:
        return rule_pred, 1.0, "Rule (Lexicon)"

    # 2. ML (SetFit)
    probs = model.predict_proba([text])[0]

    if hasattr(probs, "cpu"):
        probs = probs.cpu().detach().numpy()
    elif hasattr(probs, "numpy"):
        probs = probs.numpy()

    max_conf = float(np.max(probs))
    pred_idx = np.argmax(probs)

    if hasattr(model, 'labels') and model.labels:
        ml_pred = model.labels[pred_idx]
    else:
        ml_pred = str(pred_idx)

    # 3. Decision
    if max_conf >= ml_threshold:
        return ml_pred, max_conf, "ML"

    return fallback_label, max_conf, "Fallback"


def load_annotated_profiles(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle list of lists structure often found in annotated data
    if isinstance(data, list):
        return data
    return []


def map_seniority_ground_truth(sen_label):
    """Normalize ground truth labels to match model output classes."""
    if not sen_label: return None
    mapping = {
        "Professional": "Senior",
        # Often 'Professional' in data maps to 'Senior' in model or vice versa. Adjust as needed.
        "Entry": "Junior"
    }
    return mapping.get(sen_label, sen_label)


def run_validation():
    print("=== APPROACH 7: VALIDATION (Lexicon Rules + SetFit AI) ===")

    # 1. Load Resources
    print("\n[1] Loading Lexicons...")
    dept_lexicon = load_lexicon(DEPT_LEXICON_PATH)
    sen_lexicon = load_lexicon(SEN_LEXICON_PATH)

    print("\n[2] Loading AI Models...")
    dept_model = SetFitModel.from_pretrained(str(CHECKPOINTS_DIR / "department_model"))
    sen_model = SetFitModel.from_pretrained(str(CHECKPOINTS_DIR / "seniority_model"))

    # 2. Load Validation Data
    print("\n[3] Loading Annotated Profiles...")
    profiles = load_annotated_profiles(ANNOTATED_JSON_PATH)
    print(f"    Loaded {len(profiles)} profiles for validation.")

    # 3. Run Prediction
    print("\n[4] Running Validation Loop...")

    y_true_dept, y_pred_dept = [], []
    y_true_sen, y_pred_sen = [], []

    results = []

    for i, p in enumerate(tqdm(profiles)):
        # Extract job info
        # Check if p is a list (resume with multiple jobs) or dict
        jobs = p if isinstance(p, list) else p.get('experiences', [])
        curr_job = select_current_job(jobs)

        if not curr_job: continue

        pos_raw = curr_job.get("position", "")
        text = normalize_text(pos_raw)

        # Ground Truth
        truth_dept = curr_job.get("department")
        truth_sen = curr_job.get("seniority")

        # Skip if no ground truth
        if not truth_dept or not truth_sen: continue

        # Map seniority truth if needed (e.g. Professional -> Senior)
        truth_sen = map_seniority_ground_truth(truth_sen)

        # --- PREDICT ---
        # Department
        d_pred, d_conf, d_src = predict_hybrid_smart(
            text, predict_department_rule, dept_lexicon, dept_model, DEPT_ML_THRESHOLD, "Other"
        )

        # Seniority
        s_pred, s_conf, s_src = predict_hybrid_smart(
            text, predict_seniority_rule, sen_lexicon, sen_model, SEN_ML_THRESHOLD, "Senior"
        )

        # Store for metrics
        y_true_dept.append(truth_dept)
        y_pred_dept.append(d_pred)

        y_true_sen.append(truth_sen)
        y_pred_sen.append(s_pred)

        results.append({
            "text": text,
            "dept_true": truth_dept, "dept_pred": d_pred, "dept_src": d_src,
            "sen_true": truth_sen, "sen_pred": s_pred, "sen_src": s_src
        })

    # 4. Metrics & Reporting
    df_res = pd.DataFrame(results)

    print("\n" + "=" * 40)
    print("   VALIDATION RESULTS")
    print("=" * 40)

    print(f"\n--- DEPARTMENT METRICS (Threshold: {DEPT_ML_THRESHOLD}) ---")
    print(f"Accuracy: {accuracy_score(y_true_dept, y_pred_dept):.4f}")
    print(classification_report(y_true_dept, y_pred_dept, zero_division=0))
    print("Source Distribution:")
    print(df_res['dept_src'].value_counts())

    print(f"\n--- SENIORITY METRICS (Threshold: {SEN_ML_THRESHOLD}) ---")
    print(f"Accuracy: {accuracy_score(y_true_sen, y_pred_sen):.4f}")
    print(classification_report(y_true_sen, y_pred_sen, zero_division=0))
    print("Source Distribution:")
    print(df_res['sen_src'].value_counts())


if __name__ == "__main__":
    run_validation()