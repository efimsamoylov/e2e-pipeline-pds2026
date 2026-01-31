import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from setfit import SetFitModel

from config import (
    CHECKPOINTS_DIR, NOT_ANNOTATED_JSON_PATH, PREDICTIONS_PATH,
    DEPT_LEXICON_PATH, SEN_LEXICON_PATH,
    DEPT_ML_THRESHOLD, SEN_ML_THRESHOLD
)
from text_processing import load_profiles, select_current_job, normalize_text
from rule_engine import load_lexicon, predict_department_rule, predict_seniority_rule


def predict_hybrid_smart(
        text: str,
        rule_func: callable,
        lexicon: dict,
        model,
        ml_threshold: float,
        fallback_label: str
):

    rule_pred, rule_score = rule_func(text, lexicon, default_label=None)

    if rule_pred:
        return rule_pred, 1.0, "Rule (Lexicon)"

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

    if max_conf >= ml_threshold:
        return ml_pred, max_conf, "ML"

    return fallback_label, max_conf, "Fallback"


def run():
    print("=== APPROACH 7: HYBRID (Lexicon Rules + SetFit AI) ===")

    print("\n[1] Loading Lexicons...")
    dept_lexicon = load_lexicon(DEPT_LEXICON_PATH)
    sen_lexicon = load_lexicon(SEN_LEXICON_PATH)
    print(f"    Department terms loaded: {sum(len(v) for v in dept_lexicon.values())}")

    print("\n[2] Loading AI Models (from Approach 6 checkpoints)...")
    dept_model_path = str(CHECKPOINTS_DIR / "department_model")
    sen_model_path = str(CHECKPOINTS_DIR / "seniority_model")

    import warnings
    warnings.filterwarnings("ignore")

    dept_model = SetFitModel.from_pretrained(dept_model_path)
    sen_model = SetFitModel.from_pretrained(sen_model_path)
    print("    Models loaded successfully.")

    print("\n[3] Loading Profiles...")
    profiles = load_profiles(NOT_ANNOTATED_JSON_PATH)
    print(f"    Loaded {len(profiles)} profiles.")

    print("\n[4] Running Inference...")
    results = []

    for i, p in enumerate(tqdm(profiles)):
        pid = p.get("id", i) if isinstance(p, dict) else i
        jobs = p if isinstance(p, list) else p.get('experiences', [])
        curr_job = select_current_job(jobs)

        pos_raw = curr_job.get("position", "") if curr_job else ""
        org_raw = curr_job.get("organization", "") if curr_job else ""
        text = normalize_text(pos_raw)

        if not text:
            results.append({
                "id": pid, "position": pos_raw,
                "dept_pred": "Unknown", "dept_src": "Empty",
                "sen_pred": "Unknown", "sen_src": "Empty"
            })
            continue

        d_pred, d_conf, d_src = predict_hybrid_smart(
            text,
            predict_department_rule,
            dept_lexicon,
            dept_model,
            DEPT_ML_THRESHOLD,
            "Other"
        )

        # --- PREDICT SENIORITY ---
        s_pred, s_conf, s_src = predict_hybrid_smart(
            text,
            predict_seniority_rule,
            sen_lexicon,
            sen_model,
            SEN_ML_THRESHOLD,
            "Senior"
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
            "seniority_source": s_src
        })

    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\n✅ Done! Saved to: {PREDICTIONS_PATH}")
    print("\nStats:")
    print("Department Source:\n", df['department_source'].value_counts())
    print("\nSeniority Source:\n", df['seniority_source'].value_counts())


if __name__ == "__main__":
    run()
