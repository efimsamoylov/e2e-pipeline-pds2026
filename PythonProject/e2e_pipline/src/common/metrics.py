from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def print_metrics(y_true, y_pred, task_name: str) -> None:
    if not y_true:
        print(f"\n{task_name}: No ground truth available")
        return

    print(f"\n=== {task_name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    labels = sorted(set(y_true) | set(y_pred))
    print(f"Labels: {labels}")
    print(f"\nConfusion matrix:\n{confusion_matrix(y_true, y_pred, labels=labels)}")
    print(f"\nClassification report:\n{classification_report(y_true, y_pred, labels=labels, zero_division=0)}")