import numpy as np
import pandas as pd
from setfit import SetFitModel, Trainer
from typing import Tuple, Any

def train_model(
    df: pd.DataFrame, 
    random_state: int = 42,
    use_smote: bool = False # Not used in SetFit
) -> Tuple[Any, Any, float]:
    """
    Trains a SetFit model (Sentence Transformer Fine-tuning).
    """
    
    # 1. Prepare Data
    from datasets import Dataset
    train_ds = Dataset.from_pandas(df[["text", "label"]])

    # 2. Load Pre-trained Model (Multilingual)
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        labels=sorted(df["label"].unique().tolist())
    )

    # 3. Train
    # SetFit first fine-tunes embeddings using contrastive learning, then trains a head
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        # loss_class argument removed as it causes TypeError in newer versions
        metric="accuracy",
        batch_size=16,
        num_iterations=20,
        num_epochs=1,
        seed=random_state
    )
    
    print(f"Fine-tuning SetFit model on {len(df)} examples...")
    trainer.train()
    
    # 4. Calculate Confidence Threshold
    probs = model.predict_proba(df["text"].tolist())
    confidences = np.max(probs.numpy(), axis=1)
    
    # Set threshold at 10th percentile
    threshold = float(np.percentile(confidences, 10))
    print(f"Model trained. Calculated confidence threshold: {threshold:.4f}")

    return model, None, threshold

def predict_hybrid(
    text: str,
    rule_func: callable,
    model: Any, # SetFitModel
    clf_ignored: Any, # Not used here
    threshold: float,
    fallback_label: str
) -> Tuple[str, float, str]:
    """
    Hybrid prediction:
    1. Try Rule-Based
    2. Try SetFit Model
    3. If ML confidence < threshold -> Fallback Label
    """

    # 1. Rules
    rule_pred = rule_func(text)
    if rule_pred:
        return rule_pred, 1.0, "Rule"

    # 2. ML (SetFit)
    probs = model.predict_proba([text])[0]
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
