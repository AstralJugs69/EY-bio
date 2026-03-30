from __future__ import annotations

import argparse
import logging
from pathlib import Path

from frog_challenge.bootstrap import ensure_feature_artifacts
from frog_challenge.config import TPUConfig
from frog_challenge.tpu import run_gpu_suite
from frog_challenge.utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPU neural models for the EY frog challenge.")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tpu"))
    parser.add_argument("--pseudo-absence-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    LOGGER.info(
        "GPU entrypoint starting | data_root=%s | feature_dir=%s | output_dir=%s | batch_size=%s | epochs=%s | patience=%s",
        args.data_root,
        args.feature_dir,
        args.output_dir,
        args.batch_size,
        args.epochs,
        args.patience,
    )
    ensure_feature_artifacts(args.feature_dir, args.data_root, pseudo_absence_cache_dir=args.pseudo_absence_cache_dir)
    config = TPUConfig(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    artifacts = run_gpu_suite(config)
    LOGGER.info("GPU entrypoint finished | summary=%s | best_model=%s", artifacts.summary_path, artifacts.best_model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
