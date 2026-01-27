import json
import re
from datetime import datetime

import pandas as pd

DEPT_CSV = "data/department-v2.csv"
SEN_CSV  = "data/seniority-v2.csv"
EVAL_JSON = "data/linkedin-cvs-annotated.json"

def normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)  # keep letters/numbers, replace rest by space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_start_date(s):
    """Parse 'YYYY-MM' or 'YYYY' to datetime; return very old date if missing."""
    if not s:
        return datetime(1900, 1, 1)
    s = str(s)
    for fmt in ("%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime(1900, 1, 1)

def choose_current_job(profile_positions: list) -> dict | None:
    """Pick the most recent ACTIVE job (by startDate). If no ACTIVE, return None."""
    active = [p for p in profile_positions if p.get("status") == "ACTIVE"]
    if not active:
        return None
    active_sorted = sorted(active, key=lambda p: parse_start_date(p.get("startDate")), reverse=True)
    return active_sorted[0]

def build_matcher(csv_path: str):
    """
    Build:
      - exact lookup: normalized_text -> label
      - patterns list for substring matching: (pattern_text, label) sorted by length desc
    Assumes CSV has columns: 'text' and 'label' (or similar).
    """
    df = pd.read_csv(csv_path)

    # Try to infer column names robustly
    cols = [c.lower() for c in df.columns]
    if "text" in cols:
        text_col = df.columns[cols.index("text")]
    else:
        text_col = df.columns[0]

    if "label" in cols:
        label_col = df.columns[cols.index("label")]
    else:
        label_col = df.columns[1]

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    exact = {}
    patterns = []
    for t, lab in zip(df[text_col], df[label_col]):
        nt = normalize(t)
        if not nt:
            continue
        exact[nt] = lab
        patterns.append((nt, lab))

    # Longest-first helps when multiple substrings match
    patterns.sort(key=lambda x: len(x[0]), reverse=True)
    return exact, patterns

def predict_from_title(title: str, exact: dict, patterns: list, default_label: str):
    nt = normalize(title)
    if not nt:
        return default_label

    # 1) exact match
    if nt in exact:
        return exact[nt]

    # 2) substring match (first hit in longest-first list)
    for pat, lab in patterns:
        if pat in nt:
            return lab

    return default_label

# --- Load matchers ---
dept_exact, dept_patterns = build_matcher(DEPT_CSV)
sen_exact,  sen_patterns  = build_matcher(SEN_CSV)

# --- Load evaluation set ---
with open(EVAL_JSON, "r", encoding="utf-8") as f:
    eval_profiles = json.load(f)

rows = []
for i, profile in enumerate(eval_profiles):
    current = choose_current_job(profile)
    if current is None:
        continue

    title = current.get("position", "")
    true_dept = current.get("department")
    true_sen  = current.get("seniority")

    pred_dept = predict_from_title(title, dept_exact, dept_patterns, default_label="Other")
    pred_sen  = predict_from_title(title, sen_exact,  sen_patterns,  default_label="Professional")

    rows.append({
        "profile_id": i,
        "title": title,
        "true_department": true_dept,
        "pred_department": pred_dept,
        "true_seniority": true_sen,
        "pred_seniority": pred_sen,
        "dept_correct": (pred_dept == true_dept),
        "sen_correct": (pred_sen == true_sen),
    })

res = pd.DataFrame(rows)

dept_acc = res["dept_correct"].mean()
sen_acc  = res["sen_correct"].mean()
both_acc = ((res["dept_correct"]) & (res["sen_correct"])).mean()

print(f"Department accuracy: {dept_acc:.3f} ({res['dept_correct'].sum()}/{len(res)})")
print(f"Seniority  accuracy: {sen_acc:.3f} ({res['sen_correct'].sum()}/{len(res)})")
print(f"Both correct:        {both_acc:.3f}")

# Optional: show most frequent errors
print("\nTop department confusions (true -> pred):")
print(
    res.loc[~res["dept_correct"], ["true_department","pred_department"]]
      .value_counts()
      .head(15)
)

print("\nTop seniority confusions (true -> pred):")
print(
    res.loc[~res["sen_correct"], ["true_seniority","pred_seniority"]]
      .value_counts()
      .head(15)
)

# Optional: inspect examples where it fails
fail_examples = res.loc[~(res["dept_correct"] & res["sen_correct"]), ["title","true_department","pred_department","true_seniority","pred_seniority"]]
print("\nSample failures:")
print(fail_examples.head(20).to_string(index=False))