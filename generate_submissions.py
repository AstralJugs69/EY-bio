from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from frog_challenge.modeling import load_feature_tables
from frog_challenge.utils import configure_logging, ensure_directory, read_dataframe_parquet, read_json, write_json


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate extra submission sweeps from saved probability artifacts.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--tpu-artifact-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threshold-min", type=float, default=0.45)
    parser.add_argument("--threshold-max", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--max-candidates", type=int, default=10)
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


def _logit(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _prior_shift_adjust(
    probabilities: np.ndarray,
    source_prevalence: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    source_prevalence = float(np.clip(source_prevalence, 1e-4, 1.0 - 1e-4))
    target_prevalence = source_prevalence
    base_probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    adjusted = base_probabilities
    for _ in range(max_iter):
        numerator = base_probabilities * (target_prevalence / source_prevalence)
        denominator = numerator + (1.0 - base_probabilities) * ((1.0 - target_prevalence) / (1.0 - source_prevalence))
        adjusted = np.clip(numerator / denominator, 1e-6, 1.0 - 1e-6)
        updated_prevalence = float(adjusted.mean())
        if abs(updated_prevalence - target_prevalence) <= tol:
            target_prevalence = updated_prevalence
            break
        target_prevalence = float(np.clip(updated_prevalence, 1e-4, 1.0 - 1e-4))

    return adjusted, target_prevalence


def _fit_sigmoid_calibrator(train_probabilities: np.ndarray, y_train: np.ndarray):
    if len(np.unique(y_train)) < 2 or np.allclose(train_probabilities, train_probabilities[0]):
        return lambda values: np.asarray(values, dtype=np.float64)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(_logit(train_probabilities).reshape(-1, 1), y_train)
    return lambda values: model.predict_proba(_logit(values).reshape(-1, 1))[:, 1]


def _fit_isotonic_calibrator(train_probabilities: np.ndarray, y_train: np.ndarray):
    if len(np.unique(y_train)) < 2 or np.allclose(train_probabilities, train_probabilities[0]):
        return lambda values: np.asarray(values, dtype=np.float64)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(train_probabilities, y_train)
    return lambda values: model.transform(np.asarray(values, dtype=np.float64))


def _fit_beta_calibrator(train_probabilities: np.ndarray, y_train: np.ndarray):
    if len(np.unique(y_train)) < 2 or np.allclose(train_probabilities, train_probabilities[0]):
        return lambda values: np.asarray(values, dtype=np.float64)
    clipped = np.clip(np.asarray(train_probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    features = np.column_stack([np.log(clipped), -np.log1p(-clipped)])
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(features, y_train)

    def transform(values: np.ndarray) -> np.ndarray:
        clipped_values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        transform_features = np.column_stack([np.log(clipped_values), -np.log1p(-clipped_values)])
        return model.predict_proba(transform_features)[:, 1]

    return transform


def _cross_fit_calibration(
    train_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    y_true: np.ndarray,
    groups: pd.Series,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    calibrated_oof = np.zeros_like(train_probabilities, dtype=np.float64)

    for train_index, valid_index in splitter.split(train_probabilities.reshape(-1, 1), y_true, groups=groups):
        if method == "sigmoid":
            calibrator = _fit_sigmoid_calibrator(train_probabilities[train_index], y_true[train_index])
        elif method == "isotonic":
            calibrator = _fit_isotonic_calibrator(train_probabilities[train_index], y_true[train_index])
        elif method == "beta":
            calibrator = _fit_beta_calibrator(train_probabilities[train_index], y_true[train_index])
        else:
            raise ValueError(f"Unsupported calibration method: {method}")
        calibrated_oof[valid_index] = calibrator(train_probabilities[valid_index])

    if method == "sigmoid":
        full_calibrator = _fit_sigmoid_calibrator(train_probabilities, y_true)
    elif method == "isotonic":
        full_calibrator = _fit_isotonic_calibrator(train_probabilities, y_true)
    else:
        full_calibrator = _fit_beta_calibrator(train_probabilities, y_true)
    calibrated_test = full_calibrator(test_probabilities)
    return calibrated_oof, np.asarray(calibrated_test, dtype=np.float64)


def _evaluate_candidate(
    name: str,
    oof_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    focus_thresholds: np.ndarray,
) -> dict[str, object]:
    all_scores = np.array([f1_score(y_true, (oof_probabilities >= threshold).astype(int)) for threshold in thresholds], dtype=np.float64)
    focus_scores = np.array(
        [f1_score(y_true, (oof_probabilities >= threshold).astype(int)) for threshold in focus_thresholds],
        dtype=np.float64,
    )
    best_index = int(np.argmax(all_scores))
    focus_best_index = int(np.argmax(focus_scores))
    return {
        "name": name,
        "oof_probabilities": np.asarray(oof_probabilities, dtype=np.float64),
        "test_probabilities": np.asarray(test_probabilities, dtype=np.float64),
        "best_threshold": float(thresholds[best_index]),
        "best_oof_f1": float(all_scores[best_index]),
        "focus_best_threshold": float(focus_thresholds[focus_best_index]),
        "focus_best_oof_f1": float(focus_scores[focus_best_index]),
        "focus_mean_oof_f1": float(np.mean(focus_scores)),
        "focus_min_oof_f1": float(np.min(focus_scores)),
    }


def _ranking_key(candidate: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(candidate["focus_best_oof_f1"]),
        float(candidate["focus_mean_oof_f1"]),
        float(candidate["best_oof_f1"]),
    )


def main() -> int:
    configure_logging()
    args = parse_args()
    output_dir = ensure_directory(args.output_dir or (args.baseline_dir / "posthoc_submission_sweeps"))
    train_frame, test_frame, _, _ = load_feature_tables(args.feature_dir)
    ids = test_frame["ID"]
    y_true = train_frame["Occurrence Status"].to_numpy()
    groups = train_frame["spatial_group"]
    source_prevalence = float(train_frame["Occurrence Status"].mean())

    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    focus_thresholds = thresholds[(thresholds >= 0.55) & (thresholds <= 0.81)]
    if focus_thresholds.size == 0:
        focus_thresholds = thresholds

    baseline_summary = read_json(args.baseline_dir / "baseline_summary.json")
    baseline_prob_dir = args.baseline_dir / "models"
    baseline_oof_probs = _load_probability_artifacts(baseline_prob_dir, "oof")
    baseline_test_probs = _load_probability_artifacts(baseline_prob_dir, "test_probabilities")
    ranked_baseline_names = [
        name
        for name, _ in sorted(
            baseline_summary["models"].items(),
            key=lambda item: float(item[1]["oof_f1"]),
            reverse=True,
        )[: args.top_k]
        if name in baseline_oof_probs and name in baseline_test_probs
    ]
    LOGGER.info("Generating posthoc optimizations from baseline models %s", ranked_baseline_names)

    candidates: dict[str, dict[str, object]] = {}
    for model_name in ranked_baseline_names:
        oof_probabilities = baseline_oof_probs[model_name]["oof_probability"].to_numpy()
        test_probabilities = baseline_test_probs[model_name]["probability"].to_numpy()
        candidates[model_name] = _evaluate_candidate(
            model_name,
            oof_probabilities,
            test_probabilities,
            y_true,
            thresholds,
            focus_thresholds,
        )

    raw_calibration_sources = [name for name in ranked_baseline_names[:4] if "_cal_" not in name]
    for model_name in raw_calibration_sources:
        base_oof = baseline_oof_probs[model_name]["oof_probability"].to_numpy()
        base_test = baseline_test_probs[model_name]["probability"].to_numpy()
        for method in ("beta", "sigmoid", "isotonic"):
            calibrated_oof, calibrated_test = _cross_fit_calibration(base_oof, base_test, y_true, groups, method)
            calibrated_name = f"{model_name}_post_{method}"
            candidates[calibrated_name] = _evaluate_candidate(
                calibrated_name,
                calibrated_oof,
                calibrated_test,
                y_true,
                thresholds,
                focus_thresholds,
            )

    seed_candidates = sorted(candidates.values(), key=_ranking_key, reverse=True)[:6]
    seed_candidate_names = [str(candidate["name"]) for candidate in seed_candidates]
    LOGGER.info("Top posthoc candidate seeds %s", seed_candidate_names)

    for left_name, right_name in combinations(seed_candidate_names[:4], 2):
        left_candidate = candidates[left_name]
        right_candidate = candidates[right_name]
        left_oof = np.asarray(left_candidate["oof_probabilities"])
        right_oof = np.asarray(right_candidate["oof_probabilities"])
        left_test = np.asarray(left_candidate["test_probabilities"])
        right_test = np.asarray(right_candidate["test_probabilities"])

        best_blend: dict[str, object] | None = None
        for left_weight in np.arange(0.1, 1.0, 0.1):
            right_weight = 1.0 - left_weight
            blend_name = f"blend_{left_name}_{left_weight:.1f}__{right_name}_{right_weight:.1f}"
            blend_candidate = _evaluate_candidate(
                blend_name,
                left_weight * left_oof + right_weight * right_oof,
                left_weight * left_test + right_weight * right_test,
                y_true,
                thresholds,
                focus_thresholds,
            )
            if best_blend is None or _ranking_key(blend_candidate) > _ranking_key(best_blend):
                best_blend = blend_candidate

        if best_blend is not None:
            candidates[str(best_blend["name"])] = best_blend

        rank_name = f"rank_{left_name}__{right_name}"
        rank_candidate = _evaluate_candidate(
            rank_name,
            (_rank_normalize(left_oof) + _rank_normalize(right_oof)) / 2.0,
            (_rank_normalize(left_test) + _rank_normalize(right_test)) / 2.0,
            y_true,
            thresholds,
            focus_thresholds,
        )
        candidates[rank_name] = rank_candidate

    for candidate_name in seed_candidate_names[:3]:
        candidate = candidates[candidate_name]
        adjusted_test, estimated_prevalence = _prior_shift_adjust(np.asarray(candidate["test_probabilities"]), source_prevalence)
        adjusted_name = f"prior_shift_{candidate_name}_p_{estimated_prevalence:.3f}"
        candidates[adjusted_name] = {
            **candidate,
            "name": adjusted_name,
            "test_probabilities": adjusted_test,
            "estimated_target_prevalence": estimated_prevalence,
        }

    if args.tpu_artifact_dir is not None and args.tpu_artifact_dir.exists():
        neural_prob_dir = args.tpu_artifact_dir / "predictions"
        neural_oof_probs = _load_probability_artifacts(neural_prob_dir, "oof")
        neural_test_probs = _load_probability_artifacts(neural_prob_dir, "test_probabilities")
        neural_summary = read_json(args.tpu_artifact_dir / "tpu_summary.json")
        neural_names = [
            name
            for name, _ in sorted(
                neural_summary["models"].items(),
                key=lambda item: float(item[1]["oof_f1"]),
                reverse=True,
            )
            if not name.startswith("ensemble_") and name in neural_oof_probs and name in neural_test_probs
        ][:2]
        top_baseline_for_cross = sorted(candidates.values(), key=_ranking_key, reverse=True)[:2]
        for baseline_candidate in top_baseline_for_cross:
            baseline_oof = np.asarray(baseline_candidate["oof_probabilities"])
            baseline_test = np.asarray(baseline_candidate["test_probabilities"])
            for neural_name in neural_names:
                neural_oof = neural_oof_probs[neural_name]["oof_probability"].to_numpy()
                neural_test = neural_test_probs[neural_name]["probability"].to_numpy()
                cross_name = f"cross_{baseline_candidate['name']}_0.8__{neural_name}_0.2"
                candidates[cross_name] = _evaluate_candidate(
                    cross_name,
                    0.8 * baseline_oof + 0.2 * neural_oof,
                    0.8 * baseline_test + 0.2 * neural_test,
                    y_true,
                    thresholds,
                    focus_thresholds,
                )

    ranked_candidates = sorted(candidates.values(), key=_ranking_key, reverse=True)
    selected_candidates = ranked_candidates[: args.max_candidates]
    written: list[str] = []
    recommendations: list[dict[str, object]] = []

    for candidate in selected_candidates:
        candidate_name = str(candidate["name"])
        test_probabilities = np.asarray(candidate["test_probabilities"])
        for threshold in thresholds:
            output_path = output_dir / f"{candidate_name}_thr_{threshold:.2f}.csv"
            _write_submission(ids, test_probabilities, float(threshold), output_path)
            written.append(str(output_path))

        focus_center = float(candidate["focus_best_threshold"])
        recommended_thresholds = [
            threshold
            for threshold in thresholds
            if abs(float(threshold) - focus_center) <= 0.04 + 1e-9
        ]
        recommendations.append(
            {
                "name": candidate_name,
                "best_oof_f1": float(candidate["best_oof_f1"]),
                "best_threshold": float(candidate["best_threshold"]),
                "focus_best_oof_f1": float(candidate["focus_best_oof_f1"]),
                "focus_best_threshold": focus_center,
                "focus_mean_oof_f1": float(candidate["focus_mean_oof_f1"]),
                "recommended_paths": [
                    str(output_dir / f"{candidate_name}_thr_{float(threshold):.2f}.csv") for threshold in recommended_thresholds
                ],
            }
        )

    manifest_path = output_dir / "posthoc_generation_manifest.json"
    write_json(
        manifest_path,
        {
            "feature_dir": str(args.feature_dir),
            "baseline_dir": str(args.baseline_dir),
            "tpu_artifact_dir": None if args.tpu_artifact_dir is None else str(args.tpu_artifact_dir),
            "output_dir": str(output_dir),
            "thresholds": [float(threshold) for threshold in thresholds],
            "focus_thresholds": [float(threshold) for threshold in focus_thresholds],
            "selected_candidate_count": len(selected_candidates),
            "written_count": len(written),
            "candidate_rankings": recommendations,
            "paths": written,
        },
    )
    LOGGER.info("Generated %s posthoc submission files in %s", len(written), output_dir)
    LOGGER.info("Manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
