import os
import pandas as pd
from tqdm import tqdm
from setfit import SetFitModel

from config import (
    DEPT_CSV_PATH, SEN_CSV_PATH, NOT_ANNOTATED_JSON_PATH,
    PREDICTIONS_PATH, RANDOM_STATE, USE_SMOTE
)
from text_processing import load_csv_data, load_profiles, select_current_job, normalize_text
from rules import rule_based_department, rule_based_seniority
from model_hybrid import train_model, predict_hybrid


def run_inference():
    print("=== STARTING INFERENCE PIPELINE ===")

    # 1. Load Pre-trained Models
    print("\n[1/3] Loading Pre-trained Models...")


    print("  - Loading Saved Department Model...")
    dept_vec = SetFitModel.from_pretrained("../e2e_pipline/models/checkpoints/department_model")
    dept_clf = None  
    dept_thresh = 0.6 

    print("  - Loading Saved Seniority Model...")
    sen_vec = SetFitModel.from_pretrained("../e2e_pipline/models/checkpoints/seniority_model")
    sen_clf = None
    sen_thresh = 0.6  

    # 2. Load Unlabeled Data
    print("\n[2/3] Loading Unlabeled Profiles...")
    profiles = load_profiles(NOT_ANNOTATED_JSON_PATH)
    print(f"  Loaded {len(profiles)} profiles.")

    # 3. Generate Predictions
    print("\n[3/3] Generating Predictions...")

    results = []

    for i, p in enumerate(tqdm(profiles)):
        # Extract ID if available, else index
        pid = p.get("id", i) if isinstance(p, dict) else i

        # Handle list vs dict structure
        jobs = p if isinstance(p, list) else p.get('experiences', [])

        curr_job = select_current_job(jobs)

        pos_raw = ""
        org_raw = ""
        text = ""

        if curr_job:
            pos_raw = curr_job.get("position", "")
            org_raw = curr_job.get("organization", "")
            text = normalize_text(pos_raw)  # Model uses only position

        if not text:
            # Empty prediction row
            results.append({
                "profile_id": pid,
                "input_text": "",
                "position": pos_raw,
                "organization": org_raw,
                "department_pred": "Unknown",
                "department_conf": 0.0,
                "department_source": "Empty",
                "seniority_pred": "Unknown",
                "seniority_conf": 0.0,
                "seniority_source": "Empty"
            })
            continue

        # Predict Department
        d_pred, d_conf, d_src = predict_hybrid(
            text, rule_based_department, dept_vec, dept_clf, dept_thresh, "Other"
        )

        # Predict Seniority
        s_pred, s_conf, s_src = predict_hybrid(
            text, rule_based_seniority, sen_vec, sen_clf, sen_thresh, "Senior"
        )

        results.append({
            "profile_id": pid,
            "input_text": text,
            "position": pos_raw,
            "organization": org_raw,
            "department_pred": d_pred,
            "department_conf": round(d_conf, 4),
            "department_source": d_src,
            "seniority_pred": s_pred,
            "seniority_conf": round(s_conf, 4),
            "seniority_source": s_src
        })

    # Save
    out_df = pd.DataFrame(results)

    # Determine the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the Output folder relative to the script
    output_dir = os.path.join(script_dir, "Output")

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the final file path
    output_path = os.path.join(output_dir, "predictions.csv")

    out_df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    print("\nSample:")
    print(out_df.head())


if __name__ == "__main__":
    run_inference()
