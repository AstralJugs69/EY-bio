from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from baseline_models import finalize_submission
from frog_challenge.bootstrap import ensure_feature_artifacts
from frog_challenge.config import FeatureBuildConfig, ModelConfig, TPUConfig
from frog_challenge.features import build_feature_artifacts
from frog_challenge.modeling import run_baseline_suite
from frog_challenge.tpu import run_gpu_suite
from frog_challenge.utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified runner for Kaggle stages.")
    parser.add_argument("--stage", choices=("feature", "baseline", "gpu", "tpu", "finalize"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--tpu-dir", type=Path, default=Path("artifacts/tpu"))
    parser.add_argument("--tpu-artifact-dir", type=Path, default=None)
    parser.add_argument("--pseudo-absence-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    train_path = args.train_path or (args.data_root / "Training_Data.csv")
    test_path = args.test_path or (args.data_root / "Test.csv")
    return train_path, test_path


def main() -> int:
    configure_logging()
    args = parse_args()
    train_path, test_path = resolve_data_paths(args)
    LOGGER.info(
        "Kaggle runner starting | stage=%s | data_root=%s | feature_dir=%s | baseline_dir=%s | accelerator_dir=%s",
        args.stage,
        args.data_root,
        args.feature_dir,
        args.baseline_dir,
        args.tpu_dir,
    )

    if args.stage == "feature":
        artifacts = build_feature_artifacts(
            FeatureBuildConfig(
                train_path=train_path,
                test_path=test_path,
                output_dir=args.feature_dir,
            )
        )
        print(json.dumps(artifacts.to_dict(), indent=2))
        return 0

    if args.stage == "baseline":
        ensure_feature_artifacts(
            args.feature_dir,
            args.data_root,
            train_path=train_path,
            test_path=test_path,
            pseudo_absence_cache_dir=args.pseudo_absence_cache_dir,
        )
        artifacts = run_baseline_suite(
            ModelConfig(
                feature_dir=args.feature_dir,
                output_dir=args.baseline_dir,
            )
        )
        final_choice = finalize_submission(
            feature_dir=args.feature_dir,
            output_dir=args.baseline_dir,
            tpu_artifact_dir=args.tpu_artifact_dir,
            baseline_artifacts=artifacts,
        )
        print(
            json.dumps(
                {
                    "summary_path": str(artifacts.summary_path),
                    "submission_path": str(artifacts.submission_path),
                    "best_model_name": artifacts.best_model_name,
                    "best_oof_f1": artifacts.best_oof_f1,
                    "final_choice": final_choice,
                },
                indent=2,
            )
        )
        LOGGER.info("Kaggle runner finished baseline stage")
        return 0

    if args.stage in {"gpu", "tpu"}:
        ensure_feature_artifacts(
            args.feature_dir,
            args.data_root,
            train_path=train_path,
            test_path=test_path,
            pseudo_absence_cache_dir=args.pseudo_absence_cache_dir,
        )
        artifacts = run_gpu_suite(
            TPUConfig(
                feature_dir=args.feature_dir,
                output_dir=args.tpu_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
            )
        )
        final_choice = finalize_submission(
            feature_dir=args.feature_dir,
            output_dir=args.baseline_dir,
            tpu_artifact_dir=args.tpu_dir,
        )
        print(
            json.dumps(
                {
                    "summary_path": str(artifacts.summary_path),
                    "best_model_name": artifacts.best_model_name,
                    "best_oof_f1": artifacts.best_oof_f1,
                    "final_choice": final_choice,
                },
                indent=2,
            )
        )
        LOGGER.info("Kaggle runner finished gpu stage")
        return 0

    ensure_feature_artifacts(
        args.feature_dir,
        args.data_root,
        train_path=train_path,
        test_path=test_path,
        pseudo_absence_cache_dir=args.pseudo_absence_cache_dir,
    )
    final_choice = finalize_submission(
        feature_dir=args.feature_dir,
        output_dir=args.baseline_dir,
        tpu_artifact_dir=args.tpu_artifact_dir or args.tpu_dir,
    )
    print(json.dumps(final_choice, indent=2))
    LOGGER.info("Kaggle runner finished finalize stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
