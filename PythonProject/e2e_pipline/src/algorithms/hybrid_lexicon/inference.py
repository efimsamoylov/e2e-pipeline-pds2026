import os
import pandas as pd
from tqdm import tqdm
from setfit import SetFitModel

from config import hybrid_lexicon as cfg
from ...common.io import load_profiles, load_lexicon, save_df
from ...common.current_job import select_current_job
from ...common.text import normalize_text
from .engine import predict_department_rule, predict_seniority_rule, predict_hybrid_smart

def run_inference():
    print("=== HYBRID (Lexicon + SetFit) ===")

    dept_lexicon = load_lexicon(cfg.DEPT_LEXICON_PATH)
    sen_lexicon = load_lexicon(cfg.SEN_LEXICON_PATH)

    dept_model = SetFitModel.from_pretrained(str(cfg.CHECKPOINTS_DIR / "department_model"))
    sen_model = SetFitModel.from_pretrained(str(cfg.CHECKPOINTS_DIR / "seniority_model"))

    profiles = load_profiles(cfg.NOT_ANNOTATED_JSON_PATH)

    results = []
    for i, p in enumerate(tqdm(profiles)):
        pid = p.get("id", i) if isinstance(p, dict) else i
        jobs = p if isinstance(p, list) else p.get("experiences", [])
        curr_job = select_current_job(jobs)

        pos_raw = curr_job.get("position", "") if curr_job else ""
        org_raw = curr_job.get("organization", "") if curr_job else ""
        text = normalize_text(pos_raw)

        if not text:
            results.append({
                "id": pid,
                "position": pos_raw,
                "dept_pred": "Unknown",
                "dept_src": "Empty",
                "sen_pred": "Unknown",
                "sen_src": "Empty",
            })
            continue

        d_pred, d_conf, d_src = predict_hybrid_smart(
            text, predict_department_rule, dept_lexicon, dept_model, cfg.DEPT_ML_THRESHOLD, "Other"
        )
        s_pred, s_conf, s_src = predict_hybrid_smart(
            text, predict_seniority_rule, sen_lexicon, sen_model, cfg.SEN_ML_THRESHOLD, "Senior"
        )

        results.append({
            "id": pid,
            "position": pos_raw,
            "organization": org_raw,
            "department_pred": d_pred,
            "department_conf": round(d_conf, 2),
            "department_source": d_src,
            "seniority_pred": s_pred,
            "seniority_conf": round(s_conf, 2),
            "seniority_source": s_src,
        })

    df = pd.DataFrame(results)
    save_df(df, cfg.PREDICTIONS_PATH)

    print(f"Saved: {cfg.PREDICTIONS_PATH}")
    print(df["department_source"].value_counts())
    print(df["seniority_source"].value_counts())