import pandas as pd

from config import rule_based as cfg
from ...common.io import load_profiles, load_lexicon, save_df
from ...common.current_job import select_current_job
from ...common.text import build_job_text
from ...common.metrics import print_metrics
from .engine import predict_department_rule, predict_seniority_rule

def run_validation():
    dept_lexicon = load_lexicon(cfg.DEPT_LEXICON_PATH)
    sen_lexicon = load_lexicon(cfg.SEN_LEXICON_PATH)

    dept_params = {
        "bigram_weight": cfg.DEPT_BIGRAM_WEIGHT,
        "unigram_weight": cfg.DEPT_UNIGRAM_WEIGHT,
        "min_score": cfg.DEPT_MIN_SCORE,
        "default_label": cfg.DEPT_DEFAULT_LABEL,
        "sen_default": cfg.SEN_DEFAULT_LABEL,
    }

    profiles = load_profiles(cfg.ANNOTATED_JSON_PATH)

    rows = []
    y_true_dept, y_pred_dept = [], []
    y_true_sen, y_pred_sen = [], []

    for i, profile_jobs in enumerate(profiles):
        if not isinstance(profile_jobs, list) or not profile_jobs:
            continue

        job = select_current_job(profile_jobs)
        if not job:
            continue

        text = build_job_text(job)

        dept_predict_args = {k: v for k, v in dept_params.items() if k != "sen_default"}
        sen_default = dept_params.get("sen_default", "Professional")

        dept_pred, dept_dbg = predict_department_rule(text, dept_lexicon, **dept_predict_args)
        sen_pred, sen_dbg = predict_seniority_rule(text, sen_lexicon, default_label=sen_default)

        dept_true = job.get("department")
        sen_true = job.get("seniority")

        rows.append({
            "profile_idx": i,
            "organization": job.get("organization"),
            "position": job.get("position"),
            "startDate": job.get("startDate"),
            "status": job.get("status"),
            "dept_true": dept_true,
            "dept_pred": dept_pred,
            "sen_true": sen_true,
            "sen_pred": sen_pred,
            "dept_best_score": dept_dbg.get("best_score"),
        })

        if dept_true is not None:
            y_true_dept.append(str(dept_true))
            y_pred_dept.append(str(dept_pred))
        if sen_true is not None:
            y_true_sen.append(str(sen_true))
            y_pred_sen.append(str(sen_pred))

    df = pd.DataFrame(rows)
    #save_df(df, cfg.PRED_ANNOTATED_PATH)

    print_metrics(y_true_dept, y_pred_dept, "Department")
    print_metrics(y_true_sen, y_pred_sen, "Seniority")