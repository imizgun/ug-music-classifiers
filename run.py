"""
    python run.py
    python run.py --config my_exp.yaml
    python run.py --only rf mlp
"""

import argparse
import sys

from src.config import load_config
from src.experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Mood Classification")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    all_models = ["random_forest", "knn", "knn_librosa", "mlp", "mlp_emb", "cnn", "lstm"]
    parser.add_argument(
        "--only", nargs="+",
        choices=all_models,
        help="Run only specified models (overrides config enabled flags)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.only:
        for name in all_models:
            getattr(cfg.models, name).enabled = name in args.only

    run(cfg)
    sys.exit(0)


if __name__ == "__main__":
    main()
