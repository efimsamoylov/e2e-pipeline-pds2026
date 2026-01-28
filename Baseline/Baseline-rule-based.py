import json
import re
import pandas as pd
from datetime import datetime

DEPT_CSV = "data/department-v2.csv"
SEN_CSV  = "data/seniority-v2.csv"
EVAL_JSON = "data/linkedin-cvs-annotated.json"


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()

    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_start_date(s):
    if not s:
        return datetime(1900, 1, 1)
    s = str(s)
    for fmt in ("%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime(1900, 1, 1)


def get_current_job(profile_positions: list) -> dict | None:
    if not profile_positions:
        return None

    active = [p for p in profile_positions if p.get("status") == "ACTIVE"]
    if not active:
        return None

    # Сортируем: у кого дата новее - тот первый
    active_sorted = sorted(active, key=lambda p: parse_start_date(p.get("startDate")), reverse=True)
    return active_sorted[0]


def build_lookup(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    cols = [c.lower() for c in df.columns]
    text_col = df.columns[cols.index("text")] if "text" in cols else df.columns[0]
    label_col = df.columns[cols.index("label")] if "label" in cols else df.columns[1]

    lookup = {}
    for t, lab in zip(df[text_col], df[label_col]):
        if pd.isna(t) or pd.isna(lab):
            continue
        norm_t = normalize(str(t))
        if norm_t:
            lookup[norm_t] = str(lab)
    return lookup


dept_lookup = build_lookup(DEPT_CSV)
sen_lookup = build_lookup(SEN_CSV)

with open(EVAL_JSON, "r", encoding="utf-8") as f:
    profiles = json.load(f)

n = 0
dept_ok = 0
sen_ok = 0
both_ok = 0

for profile in profiles:
    job = get_current_job(profile)
    if job is None:
        continue

    raw_title = str(job.get("position", ""))
    title_norm = normalize(raw_title)

    pred_dept = dept_lookup.get(title_norm)
    pred_sen = sen_lookup.get(title_norm)

    true_dept = job.get("department")
    true_sen = job.get("seniority")

    n += 1

    d = (pred_dept == true_dept) if pred_dept is not None else False
    s = (pred_sen == true_sen) if pred_sen is not None else False

    dept_ok += int(d)
    sen_ok += int(s)
    both_ok += int(d and s)

print(f"Department accuracy (normalized, sorted by date, no fallback): {dept_ok / n:.3f}")
print(f"Seniority  accuracy (normalized, sorted by date, no fallback): {sen_ok / n:.3f}")
