import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from setfit import SetFitModel

from config import hybrid_lexicon as cfg
from ...common.io import load_lexicon
from ...common.current_job import select_current_job
from ...common.text import normalize_text
from .engine import predict_department_rule, predict_seniority_rule, predict_hybrid_smart

def load_annotated_profiles(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []

def map_seniority_ground_truth(sen_label):
    if not sen_label:
        return None
    mapping = {"Professional": "Senior", "Entry": "Junior"}
    return mapping.get(sen_label, sen_label)

def run_validation():
    dept_lexicon = load_lexicon(cfg.DEPT_LEXICON_PATH)
    sen_lexicon = load_lexicon(cfg.SEN_LEXICON_PATH)

    dept_model = SetFitModel.from_pretrained(str(cfg.CHECKPOINTS_DIR / "department_model"))
    sen_model = SetFitModel.from_pretrained(str(cfg.CHECKPOINTS_DIR / "seniority_model"))

    profiles = load_annotated_profiles(cfg.ANNOTATED_JSON_PATH)

    y_true_dept, y_pred_dept = [], []
    y_true_sen, y_pred_sen = [], []
    results = []

    for p in tqdm(profiles):
        jobs = p if isinstance(p, list) else p.get("experiences", [])
        curr_job = select_current_job(jobs)
        if not curr_job:
            continue

        text = normalize_text(curr_job.get("position", ""))

        truth_dept = curr_job.get("department")
        truth_sen = curr_job.get("seniority")
        if not truth_dept or not truth_sen:
            continue

        truth_sen = map_seniority_ground_truth(truth_sen)

        d_pred, d_conf, d_src = predict_hybrid_smart(
            text, predict_department_rule, dept_lexicon, dept_model, cfg.DEPT_ML_THRESHOLD, "Other"
        )
        s_pred, s_conf, s_src = predict_hybrid_smart(
            text, predict_seniority_rule, sen_lexicon, sen_model, cfg.SEN_ML_THRESHOLD, "Senior"
        )

        y_true_dept.append(truth_dept)
        y_pred_dept.append(d_pred)
        y_true_sen.append(truth_sen)
        y_pred_sen.append(s_pred)

        results.append({
            "text": text,
            "dept_true": truth_dept, "dept_pred": d_pred, "dept_src": d_src,
            "sen_true": truth_sen, "sen_pred": s_pred, "sen_src": s_src,
        })

    df_res = pd.DataFrame(results)

    print("\n--- DEPARTMENT ---")
    print(f"Accuracy: {accuracy_score(y_true_dept, y_pred_dept):.4f}")
    print(classification_report(y_true_dept, y_pred_dept, zero_division=0))
    print(df_res["dept_src"].value_counts())

    print("\n--- SENIORITY ---")
    print(f"Accuracy: {accuracy_score(y_true_sen, y_pred_sen):.4f}")
    print(classification_report(y_true_sen, y_pred_sen, zero_division=0))
    print(df_res["sen_src"].value_counts())