from __future__ import annotations

import argparse
from pathlib import Path

from frog_challenge.config import TPUConfig
from frog_challenge.tpu import run_tpu_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TPU neural models for the EY frog challenge.")
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TPUConfig(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    run_tpu_suite(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
