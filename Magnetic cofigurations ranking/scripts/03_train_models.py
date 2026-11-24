from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from altermagnet_screening.model_pipeline import train_models  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the structure-only and spin-aware ALIGNN models.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 50:
        parser.error("--epochs must be at least 50.")
    train_models(ROOT, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
