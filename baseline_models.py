from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from frog_challenge.config import ModelConfig
from frog_challenge.modeling import ModelRunArtifacts, load_feature_tables, run_baseline_suite
from frog_challenge.tpu import predict_saved_tpu_ensemble
from frog_challenge.utils import ensure_directory, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPU baseline models for the EY frog challenge.")
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--tpu-artifact-dir", type=Path, default=None)
    return parser.parse_args()


def _write_final_submission(
    ids: pd.Series,
    probabilities,
    threshold: float,
    output_path: Path,
) -> None:
    submission = pd.DataFrame(
        {
            "ID": ids,
            "Target": (probabilities >= threshold).astype(int),
        }
    )
    submission.to_csv(output_path, index=False)


def _load_baseline_artifacts(output_dir: Path) -> ModelRunArtifacts:
    summary_path = output_dir / "baseline_summary.json"
    summary = read_json(summary_path)
    return ModelRunArtifacts(
        summary_path=summary_path,
        submission_path=output_dir / "submission.csv",
        best_model_name=str(summary["best_model_name"]),
        best_threshold=float(summary["best_threshold"]),
        best_oof_f1=float(summary["best_oof_f1"]),
    )


def finalize_submission(
    feature_dir: Path,
    output_dir: Path,
    tpu_artifact_dir: Path | None = None,
    baseline_artifacts: ModelRunArtifacts | None = None,
) -> dict[str, object]:
    output_dir = ensure_directory(output_dir)
    if baseline_artifacts is None:
        if (output_dir / "baseline_summary.json").exists():
            baseline_artifacts = _load_baseline_artifacts(output_dir)
        else:
            baseline_artifacts = run_baseline_suite(ModelConfig(feature_dir=feature_dir, output_dir=output_dir))

    _, test_frame, _ = load_feature_tables(feature_dir)
    final_submission_path = output_dir / "final_submission.csv"

    baseline_summary = read_json(baseline_artifacts.summary_path)
    baseline_submission = pd.read_csv(baseline_artifacts.submission_path)
    baseline_submission.to_csv(final_submission_path, index=False)
    final_choice = {
        "source": "baseline",
        "summary_path": str(baseline_artifacts.summary_path),
        "submission_path": str(final_submission_path),
        "best_model_name": baseline_artifacts.best_model_name,
        "best_oof_f1": baseline_artifacts.best_oof_f1,
        "best_threshold": baseline_artifacts.best_threshold,
    }

    if tpu_artifact_dir is not None and tpu_artifact_dir.exists():
        tpu_summary = read_json(tpu_artifact_dir / "tpu_summary.json")
        if float(tpu_summary["best_oof_f1"]) > float(baseline_summary["best_oof_f1"]):
            probabilities, threshold, tpu_summary = predict_saved_tpu_ensemble(tpu_artifact_dir, test_frame)
            _write_final_submission(test_frame["ID"], probabilities, threshold, final_submission_path)
            final_choice = {
                "source": "tpu",
                "summary_path": str(tpu_artifact_dir / "tpu_summary.json"),
                "submission_path": str(final_submission_path),
                "best_model_name": tpu_summary["best_model_name"],
                "best_oof_f1": float(tpu_summary["best_oof_f1"]),
                "best_threshold": float(tpu_summary["best_threshold"]),
            }

    write_json(output_dir / "final_selection.json", final_choice)
    return final_choice


def main() -> int:
    args = parse_args()
    config = ModelConfig(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
    )
    baseline_artifacts = run_baseline_suite(config)
    finalize_submission(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        tpu_artifact_dir=args.tpu_artifact_dir,
        baseline_artifacts=baseline_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
