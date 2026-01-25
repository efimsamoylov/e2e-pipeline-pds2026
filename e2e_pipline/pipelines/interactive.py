from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import rule_based as rbcfg
from config import hybrid_lexicon as hycfg

from src.common.io import load_lexicon
from src.common.text import build_job_text, normalize_text
from src.algorithms.rule_based.engine import predict_department_rule as rb_dept_rule
from src.algorithms.rule_based.engine import predict_seniority_rule as rb_sen_rule
from src.algorithms.rule_based.inference import dept_confidence_from_debug, sen_confidence_from_debug
from src.algorithms.hybrid_lexicon.engine import (
    predict_department_rule as hy_dept_rule,
    predict_seniority_rule as hy_sen_rule,
    predict_hybrid_smart,
)
from setfit import SetFitModel


def _load_rule_context():
    return {
        "dept_lexicon": load_lexicon(rbcfg.DEPT_LEXICON_PATH),
        "sen_lexicon": load_lexicon(rbcfg.SEN_LEXICON_PATH),
    }


def _load_hybrid_context():
    return {
        "dept_lexicon": load_lexicon(hycfg.DEPT_LEXICON_PATH),
        "sen_lexicon": load_lexicon(hycfg.SEN_LEXICON_PATH),
        "dept_model": SetFitModel.from_pretrained(str(hycfg.CHECKPOINTS_DIR / "department_model")),
        "sen_model": SetFitModel.from_pretrained(str(hycfg.CHECKPOINTS_DIR / "seniority_model")),
    }


def run_rule_based_single(job, ctx):
    text = build_job_text(job)
    dept_pred, dept_dbg = rb_dept_rule(
        text,
        ctx["dept_lexicon"],
        bigram_weight=rbcfg.DEPT_BIGRAM_WEIGHT,
        unigram_weight=rbcfg.DEPT_UNIGRAM_WEIGHT,
        min_score=rbcfg.DEPT_MIN_SCORE,
        default_label=rbcfg.DEPT_DEFAULT_LABEL,
    )
    sen_pred, sen_dbg = rb_sen_rule(
        text,
        ctx["sen_lexicon"],
        default_label=rbcfg.SEN_DEFAULT_LABEL,
    )

    return {
        "department": dept_pred,
        "department_conf": dept_confidence_from_debug(dept_dbg),
        "seniority": sen_pred,
        "seniority_conf": sen_confidence_from_debug(sen_dbg),
        "source": "Rule-Based",
    }


def run_hybrid_single(job, ctx):
    text = normalize_text(job.get("position", ""))
    if not text:
        return {
            "department": "Unknown",
            "department_conf": {"dept_confidence": 0.0},
            "seniority": "Unknown",
            "seniority_conf": 0.0,
            "source": "Empty",
        }

    dept_pred, dept_conf, dept_src = predict_hybrid_smart(
        text, hy_dept_rule, ctx["dept_lexicon"], ctx["dept_model"], hycfg.DEPT_ML_THRESHOLD, "Other"
    )
    sen_pred, sen_conf, sen_src = predict_hybrid_smart(
        text, hy_sen_rule, ctx["sen_lexicon"], ctx["sen_model"], hycfg.SEN_ML_THRESHOLD, "Senior"
    )

    return {
        "department": dept_pred,
        "department_conf": dept_conf,
        "seniority": sen_pred,
        "seniority_conf": sen_conf,
        "source": f"{dept_src}/{sen_src}",
    }


def print_result(title, res):
    print(f"\n=== {title} ===")
    print(f"Department: {res['department']}")
    print(f"Seniority:  {res['seniority']}")
    print(f"Source:     {res['source']}")
    if isinstance(res.get("department_conf"), dict):
        conf = res["department_conf"].get("dept_confidence")
        if conf is not None:
            print(f"Dept confidence: {conf:.2f}")
    else:
        print(f"Dept confidence: {res.get('department_conf')}")
    print(f"Sen confidence:  {res.get('seniority_conf')}")


def main():
    parser = argparse.ArgumentParser(description="Interactive single-position classification.")
    parser.add_argument("position", nargs="?", help="Job title")
    parser.add_argument("--organization", default="", help="Organization name (optional)")
    parser.add_argument("--linkedin", default="", help="LinkedIn URL (optional)")
    parser.add_argument(
        "--algo",
        choices=["rule_based", "hybrid_lexicon", "all"],
        default="all",
        help="Which algorithm to run",
    )
    parser.add_argument("--loop", action="store_true", help="Keep prompting for new titles")
    args = parser.parse_args()

    def conf_to_float(val):
        if isinstance(val, dict):
            return float(val.get("dept_confidence", 0.0))
        try:
            return float(val)
        except Exception:
            return 0.0

    def pick_best(results):
        dept_cands, sen_cands = [], []
        for res in results:
            dept_cands.append({
                "label": res["department"],
                "conf": conf_to_float(res.get("department_conf")),
                "source": res["source"],
            })
            sen_cands.append({
                "label": res["seniority"],
                "conf": conf_to_float(res.get("seniority_conf")),
                "source": res["source"],
            })
        best_dept = max(dept_cands, key=lambda x: x["conf"]) if dept_cands else None
        best_sen = max(sen_cands, key=lambda x: x["conf"]) if sen_cands else None
        return best_dept, best_sen

    rb_ctx = _load_rule_context() if args.algo in ("rule_based", "all") else None
    hy_ctx = _load_hybrid_context() if args.algo in ("hybrid_lexicon", "all") else None

    def process_one(title):
        job = {"position": title, "organization": args.organization, "linkedin": args.linkedin}
        results = []
        if rb_ctx:
            r = run_rule_based_single(job, rb_ctx)
            r["source"] = f"Rule-Based ({r['source']})"
            results.append(r)
            print_result("Rule-Based", r)
        if hy_ctx:
            h = run_hybrid_single(job, hy_ctx)
            h["source"] = f"Hybrid ({h['source']})"
            results.append(h)
            print_result("Hybrid (Lexicon + SetFit)", h)
        if len(results) > 1:
            best_dept, best_sen = pick_best(results)
            print("\n>>> Best guess (per task)")
            if best_dept:
                print(f"Department: {best_dept['label']} (conf={best_dept['conf']:.2f}, source={best_dept['source']})")
            if best_sen:
                print(f"Seniority:  {best_sen['label']} (conf={best_sen['conf']:.2f}, source={best_sen['source']})")

    def prompt_loop(initial=None):
        if initial:
            process_one(initial)
        while True:
            title = input("\nEnter job title (blank to exit): ").strip()
            if not title:
                break
            process_one(title)

    if args.loop or not args.position:
        prompt_loop(args.position)
        return

    # Single run with optional follow-up prompts if stdin is interactive
    process_one(args.position)
    try:
        if sys.stdin.isatty():
            prompt_loop()
    except Exception:
        # In non-interactive environments just exit after the first run
        pass

if __name__ == "__main__":
    main()
