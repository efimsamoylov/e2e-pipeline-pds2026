import pandas as pd

from config import rule_based as cfg
from ...common.io import load_profiles, load_lexicon, save_df
from ...common.current_job import select_current_job
from ...common.text import build_job_text
from .engine import predict_department_rule, predict_seniority_rule

def dept_confidence_from_debug(dept_dbg):
    scores = (dept_dbg or {}).get("scores", {}) or {}
    best_score = float((dept_dbg or {}).get("best_score") or 0.0)
    vals = sorted([float(v) for v in scores.values()], reverse=True)
    second_best = float(vals[1]) if len(vals) > 1 else 0.0
    margin = best_score - second_best
    conf = min(1.0, max(0.0, best_score / (best_score + 2.0)))
    return {
        "dept_best_score": best_score,
        "dept_second_best_score": second_best,
        "dept_margin": margin,
        "dept_confidence": conf,
    }

def sen_confidence_from_debug(sen_dbg):
    matched_terms = (sen_dbg or {}).get("matched_terms", [])
    return 0.9 if matched_terms else 0.4

def run_inference():
    dept_lexicon = load_lexicon(cfg.DEPT_LEXICON_PATH)
    sen_lexicon = load_lexicon(cfg.SEN_LEXICON_PATH)

    dept_params = {
        "bigram_weight": cfg.DEPT_BIGRAM_WEIGHT,
        "unigram_weight": cfg.DEPT_UNIGRAM_WEIGHT,
        "min_score": cfg.DEPT_MIN_SCORE,
        "default_label": cfg.DEPT_DEFAULT_LABEL,
        "sen_default": cfg.SEN_DEFAULT_LABEL,
    }

    profiles = load_profiles(cfg.NOT_ANNOTATED_JSON_PATH)

    rows = []
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

        rows.append({
            "profile_idx": i,
            "organization": job.get("organization"),
            "position": job.get("position"),
            "startDate": job.get("startDate"),
            "endDate": job.get("endDate"),
            "status": job.get("status"),
            "linkedin": job.get("linkedin"),
            "dept_pred": dept_pred,
            "sen_pred": sen_pred,
            **dept_confidence_from_debug(dept_dbg),
            "sen_confidence": sen_confidence_from_debug(sen_dbg),
        })

    df = pd.DataFrame(rows)
    save_df(df, cfg.PRED_NOT_ANNOTATED_PATH)
    print(f"Saved: {cfg.PRED_NOT_ANNOTATED_PATH}")