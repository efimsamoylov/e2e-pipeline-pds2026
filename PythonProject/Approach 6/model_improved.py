"""
Improved model with better handling of imbalanced classes and 'Other' category.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def train_tfidf_logreg_improved(
    df: pd.DataFrame,
    task_name: str,
    random_state: int = 42,
    test_size: float = 0.2,
    use_smote: bool = True,
    confidence_threshold: float = 0.3
) -> Tuple[TfidfVectorizer, LogisticRegression, float]:
    """
    Improved training with:
    1. SMOTE for balancing classes
    2. Better TfidfVectorizer parameters
    3. Confidence threshold calibration
    """
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"[{task_name}] DataFrame must contain columns ['text','label']. "
            f"Got: {df.columns.tolist()}"
        )

    X = df["text"]
    y = df["label"]

    # Split data
    stratify = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Improved TfidfVectorizer with more features
    vec = TfidfVectorizer(
        ngram_range=(1, 3),  # Increased to capture more patterns
        min_df=1,            # Lower threshold to capture rare words
        max_features=100_000,  # More features
        sublinear_tf=True,   # Logarithmic TF scaling
        preprocessor=None
    )
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)

    # Apply SMOTE if enabled and minority class has enough samples
    if use_smote:
        try:
            min_samples = y_train.value_counts().min()
            if min_samples >= 6:  # SMOTE needs at least 6 samples
                smote = SMOTE(
                    random_state=random_state,
                    k_neighbors=min(5, min_samples - 1)
                )
                X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)
                print(f"[{task_name}] Applied SMOTE: {len(y_train)} samples after resampling")
        except Exception as e:
            print(f"[{task_name}] SMOTE failed: {e}, continuing without resampling")

    # Train with stronger regularization
    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=0.5,  # Stronger regularization
        solver='saga',  # Better for large datasets
        random_state=random_state
    )
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_val_vec)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print(f"\n===== {task_name}: Validation on CSV (train/val split) =====")
    print(f"macro-F1: {macro_f1:.4f}")
    print(classification_report(y_val, y_pred, digits=4))

    # Calculate confidence threshold
    # Changed from 20th to 5th percentile to be less aggressive with fallback
    y_proba = clf.predict_proba(X_val_vec)
    confidences = np.max(y_proba, axis=1)
    confidence_threshold = np.percentile(confidences, 5)  # 5th percentile (was 20th)
    print(f"Recommended confidence threshold (5th percentile): {confidence_threshold:.3f}")

    return vec, clf, confidence_threshold


def predict_with_unknown_fallback(
    text: str,
    vec: TfidfVectorizer,
    clf: LogisticRegression,
    confidence_threshold: float,
    unknown_label: str = "Other"
):
    """
    Predict with fallback to 'Other' for low-confidence predictions.
    """
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]
    max_conf = np.max(proba)
    pred = clf.classes_[np.argmax(proba)]

    # If confidence is too low, return "Other"
    if max_conf < confidence_threshold:
        return unknown_label, max_conf

    return pred, max_conf
