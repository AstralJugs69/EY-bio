from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from frog_challenge.modeling import load_feature_tables
from frog_challenge.utils import configure_logging, ensure_directory, read_dataframe_parquet, read_json, write_json


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate extra submission sweeps from saved probability artifacts.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--tpu-artifact-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--threshold-min", type=float, default=0.45)
    parser.add_argument("--threshold-max", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    return parser.parse_args()


def _load_probability_artifacts(directory: Path, suffix: str) -> dict[str, pd.DataFrame]:
    artifacts: dict[str, pd.DataFrame] = {}
    if not directory.exists():
        return artifacts
    for path in directory.glob(f"*_{suffix}.parquet"):
        model_name = path.name[: -len(f"_{suffix}.parquet")]
        artifacts[model_name] = read_dataframe_parquet(path)
    return artifacts


def _write_submission(ids: pd.Series, probabilities: np.ndarray, threshold: float, output_path: Path) -> None:
    submission = pd.DataFrame({"ID": ids, "Target": (probabilities >= threshold).astype(int)})
    submission.to_csv(output_path, index=False)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return (order + 1).astype(np.float64) / len(values)


def main() -> int:
    configure_logging()
    args = parse_args()
    output_dir = ensure_directory(args.output_dir or (args.baseline_dir / "posthoc_submission_sweeps"))
    _, test_frame, _, _ = load_feature_tables(args.feature_dir)
    ids = test_frame["ID"]

    baseline_summary = read_json(args.baseline_dir / "baseline_summary.json")
    baseline_prob_dir = args.baseline_dir / "models"
    baseline_test_probs = _load_probability_artifacts(baseline_prob_dir, "test_probabilities")
    ranked_baseline_names = [
        name
        for name, _ in sorted(
            baseline_summary["models"].items(),
            key=lambda item: float(item[1]["oof_f1"]),
            reverse=True,
        )[: args.top_k]
        if name in baseline_test_probs
    ]
    LOGGER.info("Generating posthoc sweeps from baseline models %s", ranked_baseline_names)

    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    written: list[str] = []

    for model_name in ranked_baseline_names:
        probabilities = baseline_test_probs[model_name]["probability"].to_numpy()
        for threshold in thresholds:
            output_path = output_dir / f"{model_name}_thr_{threshold:.2f}.csv"
            _write_submission(ids, probabilities, float(threshold), output_path)
            written.append(str(output_path))

    for left_name, right_name in combinations(ranked_baseline_names[:3], 2):
        left_probs = baseline_test_probs[left_name]["probability"].to_numpy()
        right_probs = baseline_test_probs[right_name]["probability"].to_numpy()
        for left_weight in (0.2, 0.4, 0.6, 0.8):
            right_weight = 1.0 - left_weight
            blend_name = f"blend_{left_name}_{left_weight:.1f}__{right_name}_{right_weight:.1f}"
            blend_probs = left_weight * left_probs + right_weight * right_probs
            for threshold in thresholds:
                output_path = output_dir / f"{blend_name}_thr_{threshold:.2f}.csv"
                _write_submission(ids, blend_probs, float(threshold), output_path)
                written.append(str(output_path))

        rank_name = f"rank_{left_name}__{right_name}"
        rank_probs = (_rank_normalize(left_probs) + _rank_normalize(right_probs)) / 2.0
        for threshold in thresholds:
            output_path = output_dir / f"{rank_name}_thr_{threshold:.2f}.csv"
            _write_submission(ids, rank_probs, float(threshold), output_path)
            written.append(str(output_path))

    if args.tpu_artifact_dir is not None and args.tpu_artifact_dir.exists():
        neural_prob_dir = args.tpu_artifact_dir / "predictions"
        neural_test_probs = _load_probability_artifacts(neural_prob_dir, "test_probabilities")
        neural_summary = read_json(args.tpu_artifact_dir / "tpu_summary.json")
        neural_names = [
            name
            for name, _ in sorted(
                neural_summary["models"].items(),
                key=lambda item: float(item[1]["oof_f1"]),
                reverse=True,
            )
            if not name.startswith("ensemble_") and name in neural_test_probs
        ][:2]
        for baseline_name in ranked_baseline_names[:2]:
            for neural_name in neural_names:
                baseline_probs = baseline_test_probs[baseline_name]["probability"].to_numpy()
                neural_probs = neural_test_probs[neural_name]["probability"].to_numpy()
                blend_name = f"cross_{baseline_name}_0.7__{neural_name}_0.3"
                blend_probs = 0.7 * baseline_probs + 0.3 * neural_probs
                for threshold in thresholds:
                    output_path = output_dir / f"{blend_name}_thr_{threshold:.2f}.csv"
                    _write_submission(ids, blend_probs, float(threshold), output_path)
                    written.append(str(output_path))

    manifest_path = output_dir / "posthoc_generation_manifest.json"
    write_json(
        manifest_path,
        {
            "feature_dir": str(args.feature_dir),
            "baseline_dir": str(args.baseline_dir),
            "tpu_artifact_dir": None if args.tpu_artifact_dir is None else str(args.tpu_artifact_dir),
            "output_dir": str(output_dir),
            "written_count": len(written),
            "paths": written,
        },
    )
    LOGGER.info("Generated %s posthoc submission files in %s", len(written), output_dir)
    LOGGER.info("Manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
