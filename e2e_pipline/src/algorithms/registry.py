from typing import Callable, Dict

from .rule_based.inference import run_inference as rb_inference
from .rule_based.validation import run_validation as rb_validation
from .hybrid_lexicon.inference import run_inference as hy_inference
from .hybrid_lexicon.validation import run_validation as hy_validation

INFERENCE_REGISTRY: Dict[str, Callable[[], None]] = {
    "rule_based": rb_inference,
    "hybrid_lexicon": hy_inference,
}

VALIDATION_REGISTRY: Dict[str, Callable[[], None]] = {
    "rule_based": rb_validation,
    "hybrid_lexicon": hy_validation,
}