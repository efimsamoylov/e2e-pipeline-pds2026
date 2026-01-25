import os
import numpy as np
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments
from typing import Tuple, Any


def train_model(
        df: pd.DataFrame,
        random_state: int = 42,
        use_smote: bool = False,  # Not used in SetFit
        model_name: str = "model"  # New argument for saving
) -> Tuple[Any, Any, float]:
    """
    Trains a SetFit model and saves it to 'checkpoints/{model_name}'.
    """

    # 1. Prepare Data
    from datasets import Dataset
    train_ds = Dataset.from_pandas(df[["text", "label"]])

    # 2. Load Pre-trained Model
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        labels=sorted(df["label"].unique().tolist())
    )

    # 3. Configure Training Arguments
    args = TrainingArguments(
        batch_size=16,
        num_iterations=20,
        num_epochs=1,
        seed=random_state,
        body_learning_rate=2e-5,  # Optional: explicit learning rate often helps
        head_learning_rate=2e-3  # Optional: explicit learning rate often helps
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        metric="accuracy"
    )

    print(f"Fine-tuning SetFit model ({model_name}) on {len(df)} examples...")
    trainer.train()

    # --- SAVING THE MODEL ---
    # Create checkpoints directory if it doesn't exist
    save_path = os.path.join("checkpoints", model_name)
    os.makedirs(save_path, exist_ok=True)

    # Save model weights and config
    model.save_pretrained(save_path)
    print(f"✅ Model saved to: {save_path}")
    # ------------------------

    # 5. Calculate Confidence Threshold
    probs = model.predict_proba(df["text"].tolist())
    confidences = np.max(probs.numpy(), axis=1)

    threshold = float(np.percentile(confidences, 10))
    print(f"Confidence threshold: {threshold:.4f}")

    return model, None, threshold


def predict_hybrid(
        text: str,
        rule_func: callable,
        model: Any,
        clf_ignored: Any,
        threshold: float,
        fallback_label: str
) -> Tuple[str, float, str]:
    """
    Hybrid prediction: Rules -> SetFit -> Fallback
    """

    # 1. Rules
    rule_pred = rule_func(text)
    if rule_pred:
        return rule_pred, 1.0, "Rule"

    # 2. ML (SetFit)
    # predict_proba returns a tensor
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
        ml_pred = fallback_label

        # 3. Fallback
    if max_conf < threshold:
        return fallback_label, max_conf, "Fallback"

    return ml_pred, max_conf, "ML"
