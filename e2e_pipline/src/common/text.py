import re
from typing import Any, Dict

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_job_text(job: Dict[str, Any]) -> str:
    pos = str(job.get("position", "")).strip()
    org = str(job.get("organization", "")).strip()
    li = str(job.get("linkedin", "")).strip()

    parts = [f"Position: {pos}.", f"Organization: {org}."]
    if li:
        parts.append(f"LinkedIn: {li}.")
    return " ".join(parts)