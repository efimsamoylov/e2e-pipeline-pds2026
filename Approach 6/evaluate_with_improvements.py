
#Evaluation with improved model and strategies:
#1. Better TfidfVectorizer
#2. SMOTE for balancing
#3. Confidence-based fallback to "Other"
#4. Hierarchical classification (rule-based + ML)
import json
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

from config import DEPT_CSV_PATH, SEN_CSV_PATH, RANDOM_STATE, TEST_SIZE, BASE_DIR, DATA_DIR
from data_loader import load_labeled_csv
from text_processing import normalize_text
from model_improved import train_tfidf_logreg_improved, predict_with_unknown_fallback


def load_annotated_profiles(path: Path):
    """Load annotated profiles from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_train_df(csv_path: str) -> pd.DataFrame:
    """Prepare training data from CSV."""
    df = load_labeled_csv(csv_path).copy()
    df["text"] = df["text"].fillna("").astype(str).map(normalize_text)
    df["label"] = df["label"].fillna("").astype(str)
    return df


def map_seniority(sen: str) -> str:
    """Map seniority labels to match training data."""
    mapping = {
        "Professional": "Senior",
    }
    return mapping.get(sen, sen)


def rule_based_department(position_text: str) -> str:
    """
    Rule-based department detection for common patterns.
    Returns department label or None if no rule matches.

    IMPORTANT: Check specific departments (IT, Sales, Marketing) BEFORE "Other"!
    """
    text = position_text.lower()


    # Information Technology
    it_keywords = [
        # Core technical roles
        'software', 'developer', 'engineer', 'programmer', 'architect', 'devops',
        # Data & Analytics
        'data scientist', 'data engineer', 'data analyst', 'machine learning', 'ai ',
        # IT operations
        'system', 'network', 'database', 'infrastructure', 'cloud', 'sysadmin',
        # Security
        'security', 'cybersecurity', 'infosec',
        # Other technical
        'technical', 'technology', 'tech ', 'it ', 'qa engineer', 'test engineer',
        'frontend', 'backend', 'full stack', 'fullstack', 'web developer',
        'mobile developer', 'android', 'ios developer', 'ux engineer', 'ui engineer',
        # Specific tools/roles
        'java', 'python', 'javascript', '.net', 'c++', 'ruby', 'php',
        'applikation', 'informatik', 'ingenieur'  # German
    ]
    if any(kw in text for kw in it_keywords):
        return "Information Technology"

    # Sales
    sales_keywords = [
        'sales', 'verkauf', 'vertrieb', 'vendas', 'ventes', 'ventas',  # multilingual
        'account executive', 'account manager', 'business development',
        'commercial', 'verkaufsleiter', 'verkaufer', 'vertriebsleiter',
        'inside sales', 'outside sales', 'field sales', 'sales representative',
        'sales consultant', 'sales specialist', 'key account', 'account director'
    ]
    # Exclude cases that should be Other (executive sales roles checked earlier)
    if any(kw in text for kw in sales_keywords):
        # But exclude if it's a C-level/founder role (will be caught by "Other" below)
        if not any(exec_kw in text for exec_kw in ['ceo', 'cfo', 'cto', 'coo', 'founder', 'owner']):
            return "Sales"

    # Marketing
    marketing_keywords = [
        'marketing', 'communication', 'kommunikation', 'marketing', 'comunicacao',
        'brand', 'content', 'social media', 'digital marketing', 'online marketing',
        'seo', 'sem', 'ppc', 'campaign', 'advertising', 'public relations', 'pr ',
        'media', 'creative', 'copywriter', 'design', 'graphic', 'ux ', 'ui ',
        'webmarketing', 'growth', 'product marketing', 'field marketing',
        'event', 'evenement', 'events', 'community'
    ]
    if any(kw in text for kw in marketing_keywords):
        return "Marketing"

    # Business Development
    bd_keywords = [
        'business development', 'partnership', 'strategic alliance',
        'expansion', 'new business', 'bd manager', 'bizdev'
    ]
    if any(kw in text for kw in bd_keywords):
        return "Business Development"

    # Project Management
    pm_keywords = [
        'project manager', 'programme manager', 'program manager', 'pmo',
        'scrum master', 'agile coach', 'delivery manager', 'chef de projet',
        'project coordinator', 'project lead'
    ]
    if any(kw in text for kw in pm_keywords):
        return "Project Management"


    # Executive roles → Other
    executive_keywords = ['ceo', 'cfo', 'cto', 'coo', 'founder', 'owner', 'geschaftsfuhrer',
                         'managing director', 'prokurist', 'vorstand', 'board member']
    if any(kw in text for kw in executive_keywords):
        return "Other"

    # Academic/Medical → Other
    academic_keywords = ['professor', 'dr phil', 'physician', 'doctor', 'researcher', 'scientist']
    if any(kw in text for kw in academic_keywords):
        return "Other"

    # Operations → Other
    operations_keywords = ['operations director', 'head of operations', 'operations manager',
                          'chief operations', 'operations officer']
    if any(kw in text for kw in operations_keywords):
        return "Other"

    # Finance → Other
    finance_keywords = ['finance director', 'financial controller', 'treasurer',
                       'kreditsachbearbeiter', 'betriebswirt']
    if any(kw in text for kw in finance_keywords):
        return "Other"

    # Manual labor / Service → Other
    labor_keywords = ['waiter', 'bartender', 'lighting', 'meister', 'supervisor lab']
    if any(kw in text for kw in labor_keywords):
        return "Other"

    # Legal/Compliance → Other
    legal_keywords = ['legal', 'compliance', 'general counsel', 'attorney']
    if any(kw in text for kw in legal_keywords):
        return "Other"

    return None


def rule_based_seniority(position_text: str) -> str:
    """
    Rule-based seniority detection for clear patterns.
    Returns seniority label or None if no rule matches.
    ORDER MATTERS: Check more specific patterns first!
    """
    text = position_text.lower()


    if 'director' in text and 'associate' not in text:
        if any(kw in text for kw in ['director of', 'executive director', 'managing director',
                                       'sales director', 'finance director', 'global director',
                                       'strategic director', 'vertriebsdirektor', 'directeur']):
            return "Director"
        if text.strip() == 'director' or (text.startswith('director') or text.endswith('director')):
            return "Director"
        return "Director"

    if any(kw in text for kw in ['head of', 'vice president', 'vp ', 'vp,', 'v.p.']):
        return "Director"

    if any(kw in text for kw in ['ceo', 'cfo', 'cto', 'coo', 'founder', 'owner', 'chief']):
        return "Management"


    if 'manager' in text or 'managing' in text:
        # High-level management → Management
        high_level_mgmt = [
            'general manager', 'senior manager', 'director of management',
            'regional manager', 'country manager', 'area manager',
            'division manager', 'group manager', 'geschaftsfuhrer'  # German: CEO/Managing Director
        ]
        if any(kw in text for kw in high_level_mgmt):
            return "Management"

        # All other managers → Lead
        # Examples: "Account Manager", "Shop Manager", "Project Manager", "Product Manager"
        return "Lead"

    # Lead / Principal / Chef → Lead
    if any(kw in text for kw in [' lead ', 'principal', 'chef de', 'coordinator', 'specialist']):
        return "Lead"

    # Senior → Senior
    if 'senior' in text or 'sr ' in text or 'sr.' in text:
        return "Senior"

    # Junior / Associate / Analyst → Junior
    if any(kw in text for kw in ['junior', 'jr ', 'jr.', 'associate', 'analyst', 'trainee', 'intern']):
        return "Junior"

    return None


def extract_ground_truth_from_profile(profile):
    """Extract ground truth from profile."""
    experiences = []

    if isinstance(profile, dict):
        if isinstance(profile.get("experiences"), list):
            experiences = profile["experiences"]
        elif isinstance(profile.get("positions"), list):
            experiences = profile["positions"]
        elif isinstance(profile.get("experience"), list):
            experiences = profile["experience"]
        elif isinstance(profile.get("items"), list):
            experiences = profile["items"]
        else:
            if any(k in profile for k in ("position", "organization", "startDate", "endDate", "status")):
                experiences = [profile]
    elif isinstance(profile, list):
        experiences = profile

    from current_job import select_current_job
    current = select_current_job([e for e in experiences if isinstance(e, dict)])

    if current:
        dept = current.get("department", "")
        sen = current.get("seniority", "")
        sen = map_seniority(sen)
        return dept, sen, current.get("position", "")

    return None, None, None


def evaluate_with_improvements():
    """Main evaluation with all improvements."""
    print("=" * 80)
    print("EVALUATION WITH IMPROVEMENTS")
    print("=" * 80)

    # 1. Train improved models
    print("\n[1/4] Training improved models...")
    dept_df = prepare_train_df(DEPT_CSV_PATH)
    sen_df = prepare_train_df(SEN_CSV_PATH)

    try:
        dept_vec, dept_clf, dept_threshold = train_tfidf_logreg_improved(
            dept_df, task_name="Department", random_state=RANDOM_STATE,
            test_size=TEST_SIZE, use_smote=True
        )
    except ImportError:
        print("⚠️  SMOTE (imbalanced-learn) not installed. Install: pip install imbalanced-learn")
        print("Falling back to standard training...")
        from model import train_tfidf_logreg
        dept_vec, dept_clf = train_tfidf_logreg(
            dept_df, task_name="Department", random_state=RANDOM_STATE, test_size=TEST_SIZE
        )
        dept_threshold = 0.3

    try:
        sen_vec, sen_clf, sen_threshold = train_tfidf_logreg_improved(
            sen_df, task_name="Seniority", random_state=RANDOM_STATE,
            test_size=TEST_SIZE, use_smote=True
        )
    except ImportError:
        from model import train_tfidf_logreg
        sen_vec, sen_clf = train_tfidf_logreg(
            sen_df, task_name="Seniority", random_state=RANDOM_STATE, test_size=TEST_SIZE
        )
        sen_threshold = 0.3

    # 2. Load annotated profiles
    print("\n[2/4] Loading annotated profiles...")
    annotated_path = DATA_DIR.parent / "data" / "linkedin-cvs-annotated.json"
    profiles = load_annotated_profiles(annotated_path)
    print(f"Loaded {len(profiles)} annotated profiles")

    # 3. Predict with hybrid approach (rules + ML)
    print("\n[3/4] Making predictions (hybrid: rules + ML)...")
    results = []
    skipped = 0
    rule_based_dept_count = 0
    rule_based_sen_count = 0

    for idx, profile in enumerate(profiles):
        dept_true, sen_true, position_raw = extract_ground_truth_from_profile(profile)

        if not dept_true or not sen_true:
            skipped += 1
            continue

        text = normalize_text(position_raw)
        if not text.strip():
            skipped += 1
            continue

        # Department: Try rules first, then ML
        dept_pred = rule_based_department(text)
        dept_conf = 1.0 if dept_pred else 0.0

        if not dept_pred:
            try:
                dept_pred, dept_conf = predict_with_unknown_fallback(
                    text, dept_vec, dept_clf, dept_threshold, unknown_label="Other"
                )
            except:
                Xd = dept_vec.transform([text])
                dept_pred = dept_clf.predict(Xd)[0]
                dept_conf = float(dept_clf.predict_proba(Xd).max())
        else:
            rule_based_dept_count += 1

        # Seniority: Try rules first, then ML
        sen_pred = rule_based_seniority(text)
        sen_conf = 1.0 if sen_pred else 0.0

        if not sen_pred:
            try:
                sen_pred, sen_conf = predict_with_unknown_fallback(
                    text, sen_vec, sen_clf, sen_threshold, unknown_label="Senior"
                )
            except:
                Xs = sen_vec.transform([text])
                sen_pred = sen_clf.predict(Xs)[0]
                sen_conf = float(sen_clf.predict_proba(Xs).max())
        else:
            rule_based_sen_count += 1

        results.append({
            "profile_id": idx,
            "text": text,
            "position_raw": position_raw,
            "department_true": dept_true,
            "department_pred": dept_pred,
            "department_conf": dept_conf,
            "seniority_true": sen_true,
            "seniority_pred": sen_pred,
            "seniority_conf": sen_conf,
        })

    print(f"Processed {len(results)} profiles (skipped {skipped})")
    print(f"Rule-based department predictions: {rule_based_dept_count} ({rule_based_dept_count/len(results)*100:.1f}%)")
    print(f"Rule-based seniority predictions: {rule_based_sen_count} ({rule_based_sen_count/len(results)*100:.1f}%)")

    if not results:
        print("\nERROR: No valid annotated profiles found!")
        return

    # 4. Calculate metrics
    print("\n[4/4] Calculating metrics...")
    df = pd.DataFrame(results)

    # Department metrics
    print("\n" + "=" * 80)
    print("DEPARTMENT PREDICTION METRICS (IMPROVED)")
    print("=" * 80)
    dept_acc = accuracy_score(df["department_true"], df["department_pred"])
    print(f"\nAccuracy: {dept_acc:.4f} ({dept_acc*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(df["department_true"], df["department_pred"], zero_division=0))

    # Seniority metrics
    print("\n" + "=" * 80)
    print("SENIORITY PREDICTION METRICS (IMPROVED)")
    print("=" * 80)
    sen_acc = accuracy_score(df["seniority_true"], df["seniority_pred"])
    print(f"\nAccuracy: {sen_acc:.4f} ({sen_acc*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(df["seniority_true"], df["seniority_pred"], zero_division=0))

    # Save results
    output_path = BASE_DIR / "outputs" / "evaluation_improved.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    evaluate_with_improvements()
