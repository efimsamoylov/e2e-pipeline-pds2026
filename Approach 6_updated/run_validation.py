import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from setfit import SetFitModel

from config import (
    DEPT_CSV_PATH, SEN_CSV_PATH, ANNOTATED_JSON_PATH, 
    RANDOM_STATE, USE_SMOTE
)
from text_processing import load_csv_data, load_profiles, select_current_job, normalize_text, map_seniority
from rules import rule_based_department, rule_based_seniority
from model_hybrid import train_model, predict_hybrid

def run_validation():
    print("=== STARTING VALIDATION PIPELINE ===")

    # 1. Train Models
    print("\n[1/3] Training Models (SKIP TRAINING)...")
    #print("  - Training Department Model...")
    #dept_df = load_csv_data(DEPT_CSV_PATH)
    # Added model_name="department_model"
    #dept_vec, dept_clf, dept_thresh = train_model(dept_df, RANDOM_STATE, USE_SMOTE, model_name="department_model")
    #print(f"    Department Threshold: {dept_thresh:.3f}")


    print("  - Loading Saved Department Model...")
    dept_vec = SetFitModel.from_pretrained("../e2e_pipline/models/checkpoints/department_model")
    dept_clf = None
    dept_thresh = 0.82

    #print("  - Training Seniority Model...")
    #sen_df = load_csv_data(SEN_CSV_PATH)
    # Added model_name="seniority_model"
    #sen_vec, sen_clf, sen_thresh = train_model(sen_df, RANDOM_STATE, USE_SMOTE, model_name="seniority_model")
    #print(f"    Seniority Threshold: {sen_thresh:.3f}")

    print("  - Loading Saved Seniority Model...")
    sen_vec = SetFitModel.from_pretrained("../e2e_pipline/models/checkpoints/seniority_model")
    sen_clf = None
    sen_thresh = 0.74

    print(f"    Department Threshold: {dept_thresh:.3f}")
    print(f"    Seniority Threshold: {sen_thresh:.3f}")

    # 2. Load Validation Data
    print("\n[2/3] Loading Validation Data (Annotated JSON)...")
    profiles = load_profiles(ANNOTATED_JSON_PATH)
    
    val_data = []
    for p in profiles:
        # Extract ground truth from list of jobs (assuming structure is list of lists of dicts)
        # or list of dicts. The loader handles list/dict wrapping, here we expect list of jobs.
        jobs = p if isinstance(p, list) else p.get('experiences', [])
        
        curr_job = select_current_job(jobs)
        if not curr_job: 
            continue
            
        pos = curr_job.get("position", "")
        text = normalize_text(pos)
        
        dept_true = curr_job.get("department")
        sen_true = curr_job.get("seniority")
        
        if not dept_true or not sen_true: 
            continue
            
        sen_true = map_seniority(sen_true)
        
        val_data.append({
            "text": text,
            "dept_true": dept_true,
            "sen_true": sen_true
        })
    
    df_val = pd.DataFrame(val_data)
    print(f"  Loaded {len(df_val)} valid profiles for validation.")

    # 3. Predict & Evaluate
    print("\n[3/3] Running Hybrid Prediction & Evaluation...")
    
    # Predict Department
    dept_preds = df_val["text"].apply(
        lambda t: predict_hybrid(t, rule_based_department, dept_vec, dept_clf, dept_thresh, "Other")
    )
    df_val["dept_pred"] = [x[0] for x in dept_preds]
    df_val["dept_source"] = [x[2] for x in dept_preds]

    # Predict Seniority
    sen_preds = df_val["text"].apply(
        lambda t: predict_hybrid(t, rule_based_seniority, sen_vec, sen_clf, sen_thresh, "Senior")
    )
    df_val["sen_pred"] = [x[0] for x in sen_preds]
    df_val["sen_source"] = [x[2] for x in sen_preds]

    # Metrics
    print("\n--- DEPARTMENT METRICS ---")
    acc_dept = accuracy_score(df_val["dept_true"], df_val["dept_pred"])
    print(f"Accuracy: {acc_dept:.4f}")
    print(classification_report(df_val["dept_true"], df_val["dept_pred"], zero_division=0))
    print("Source Distribution:")
    print(df_val["dept_source"].value_counts())

    print("\n--- SENIORITY METRICS ---")
    acc_sen = accuracy_score(df_val["sen_true"], df_val["sen_pred"])
    print(f"Accuracy: {acc_sen:.4f}")
    print(classification_report(df_val["sen_true"], df_val["sen_pred"], zero_division=0))
    print("Source Distribution:")
    print(df_val["sen_source"].value_counts())

if __name__ == "__main__":
    run_validation()
