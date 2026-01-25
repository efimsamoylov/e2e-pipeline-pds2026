from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.algorithms.registry import INFERENCE_REGISTRY, VALIDATION_REGISTRY

def run_all():
    for name in INFERENCE_REGISTRY:
        print(f"\n>>> Running inference: {name}")
        INFERENCE_REGISTRY[name]()

    for name in VALIDATION_REGISTRY:
        print(f"\n>>> Running validation: {name}")
        VALIDATION_REGISTRY[name]()

if __name__ == "__main__":
    run_all()