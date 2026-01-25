from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
from src.algorithms.registry import VALIDATION_REGISTRY



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=VALIDATION_REGISTRY.keys(), required=True)
    args = parser.parse_args()

    VALIDATION_REGISTRY[args.algo]()

if __name__ == "__main__":
    main()