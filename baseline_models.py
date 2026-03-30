from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from frog_challenge.bootstrap import ensure_feature_artifacts
from frog_challenge.config import ModelConfig
from frog_challenge.modeling import (
    ModelRunArtifacts,
    load_feature_tables,
    optimize_threshold,
    predict_probabilities,
    run_baseline_suite,
)
from frog_challenge.tpu import predict_saved_tpu_ensemble
from frog_challenge.utils import configure_logging, ensure_directory, read_dataframe_parquet, read_json, write_json

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPU baseline models for the EY frog challenge.")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--feature-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--tpu-artifact-dir", type=Path, default=None)
    parser.add_argument("--pseudo-absence-cache-dir", type=Path, default=None)
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


def _load_probability_artifacts(directory: Path, suffix: str) -> dict[str, pd.DataFrame]:
    artifacts: dict[str, pd.DataFrame] = {}
    if not directory.exists():
        return artifacts
    for path in directory.glob(f"*_{suffix}.parquet"):
        model_name = path.name[: -len(f"_{suffix}.parquet")]
        artifacts[model_name] = read_dataframe_parquet(path)
    return artifacts


def _stacking_factories(random_state: int) -> dict[str, Pipeline]:
    return {
        "combined_stack_logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=random_state, class_weight="balanced")),
            ]
        ),
        "combined_stack_histgb": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.03,
                        max_iter=500,
                        max_leaf_nodes=31,
                        min_samples_leaf=8,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _run_combined_stack(
    feature_dir: Path,
    baseline_dir: Path,
    neural_dir: Path,
) -> dict[str, object] | None:
    train_frame, test_frame, _, _ = load_feature_tables(feature_dir)
    baseline_prob_dir = baseline_dir / "models"
    neural_prob_dir = neural_dir / "predictions"
    baseline_oof = _load_probability_artifacts(baseline_prob_dir, "oof")
    baseline_test = _load_probability_artifacts(baseline_prob_dir, "test_probabilities")
    neural_oof = _load_probability_artifacts(neural_prob_dir, "oof")
    neural_test = _load_probability_artifacts(neural_prob_dir, "test_probabilities")

    candidate_names = []
    for model_name, frame in baseline_oof.items():
        if model_name in baseline_test:
            candidate_names.append(("baseline", model_name))
    for model_name, frame in neural_oof.items():
        if model_name in neural_test:
            candidate_names.append(("neural", model_name))

    if len(candidate_names) < 2:
        return None

    baseline_summary = read_json(baseline_dir / "baseline_summary.json")
    top_baseline_names = [
        name
        for name, _ in sorted(
            baseline_summary["models"].items(),
            key=lambda item: float(item[1]["oof_f1"]),
            reverse=True,
        )[:4]
        if name in baseline_oof and name in baseline_test
    ]
    neural_summary = read_json(neural_dir / "tpu_summary.json")
    top_neural_names = [
        name
        for name, metrics in sorted(
            neural_summary["models"].items(),
            key=lambda item: float(item[1]["oof_f1"]),
            reverse=True,
        )
        if not name.startswith("ensemble_") and name in neural_oof and name in neural_test
    ][:3]
    meta_feature_names = top_baseline_names + top_neural_names
    if len(meta_feature_names) < 2:
        return None

    meta_train = pd.DataFrame(
        {
            "ID": train_frame["ID"],
            "Occurrence Status": train_frame["Occurrence Status"],
            "spatial_group": train_frame["spatial_group"],
        }
    )
    meta_test = pd.DataFrame({"ID": test_frame["ID"]})
    for name in top_baseline_names:
        meta_train[name] = baseline_oof[name]["oof_probability"].to_numpy()
        meta_test[name] = baseline_test[name]["probability"].to_numpy()
    for name in top_neural_names:
        meta_train[name] = neural_oof[name]["oof_probability"].to_numpy()
        meta_test[name] = neural_test[name]["probability"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = meta_train["spatial_group"]
    y = meta_train["Occurrence Status"].to_numpy()
    best_result: dict[str, object] | None = None

    for model_name, estimator in _stacking_factories(42).items():
        oof_probabilities = np.zeros(meta_train.shape[0], dtype=np.float64)
        test_fold_probabilities: list[np.ndarray] = []
        for train_index, valid_index in splitter.split(meta_train[meta_feature_names], y, groups=groups):
            fitted = estimator
            fitted.fit(meta_train.iloc[train_index][meta_feature_names], y[train_index])
            oof_probabilities[valid_index] = predict_probabilities(fitted, meta_train.iloc[valid_index][meta_feature_names])
            test_fold_probabilities.append(predict_probabilities(fitted, meta_test[meta_feature_names]))
        threshold, oof_f1 = optimize_threshold(y, oof_probabilities, 0.10, 0.90, 0.01)
        result = {
            "model_name": model_name,
            "oof_f1": float(oof_f1),
            "threshold": float(threshold),
            "oof_probabilities": oof_probabilities,
            "test_probabilities": np.mean(np.vstack(test_fold_probabilities), axis=0),
            "meta_feature_names": meta_feature_names,
        }
        if best_result is None or float(result["oof_f1"]) > float(best_result["oof_f1"]):
            best_result = result
    return best_result


def _write_threshold_sweep_submissions(
    ids: pd.Series,
    probabilities: np.ndarray,
    center_threshold: float,
    output_dir: Path,
    prefix: str,
    radius: float = 0.12,
    step: float = 0.02,
) -> list[str]:
    sweep_dir = ensure_directory(output_dir / "submission_sweeps")
    local_thresholds = np.arange(
        max(0.05, center_threshold - radius),
        min(0.95, center_threshold + radius) + step,
        step,
    )
    public_sweep_dir = ensure_directory(output_dir / "public_threshold_sweeps")
    public_thresholds = np.arange(0.45, 0.85 + 0.02, 0.02)
    written: list[str] = []
    for threshold in local_thresholds:
        sweep_path = sweep_dir / f"{prefix}_thr_{threshold:.2f}.csv"
        _write_final_submission(ids, probabilities, float(threshold), sweep_path)
        written.append(str(sweep_path))
    for threshold in public_thresholds:
        sweep_path = public_sweep_dir / f"{prefix}_thr_{threshold:.2f}.csv"
        _write_final_submission(ids, probabilities, float(threshold), sweep_path)
        written.append(str(sweep_path))
    return written


def finalize_submission(
    feature_dir: Path,
    output_dir: Path,
    tpu_artifact_dir: Path | None = None,
    baseline_artifacts: ModelRunArtifacts | None = None,
) -> dict[str, object]:
    output_dir = ensure_directory(output_dir)
    LOGGER.info(
        "Finalizing submission | feature_dir=%s | output_dir=%s | tpu_artifact_dir=%s",
        feature_dir,
        output_dir,
        tpu_artifact_dir,
    )
    if baseline_artifacts is None:
        if (output_dir / "baseline_summary.json").exists():
            baseline_artifacts = _load_baseline_artifacts(output_dir)
        else:
            baseline_artifacts = run_baseline_suite(ModelConfig(feature_dir=feature_dir, output_dir=output_dir))

    _, test_frame, _, _ = load_feature_tables(feature_dir)
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
        combined_stack = _run_combined_stack(feature_dir, output_dir, tpu_artifact_dir)
        if combined_stack is not None and float(combined_stack["oof_f1"]) > max(
            float(baseline_summary["best_oof_f1"]),
            float(tpu_summary["best_oof_f1"]),
        ):
            LOGGER.info(
                "Combined stack beats both baseline and neural models | combined_oof_f1=%.4f | baseline_oof_f1=%.4f | neural_oof_f1=%.4f",
                float(combined_stack["oof_f1"]),
                float(baseline_summary["best_oof_f1"]),
                float(tpu_summary["best_oof_f1"]),
            )
            _write_final_submission(
                test_frame["ID"],
                np.asarray(combined_stack["test_probabilities"]),
                float(combined_stack["threshold"]),
                final_submission_path,
            )
            sweep_paths = _write_threshold_sweep_submissions(
                ids=test_frame["ID"],
                probabilities=np.asarray(combined_stack["test_probabilities"]),
                center_threshold=float(combined_stack["threshold"]),
                output_dir=output_dir,
                prefix=combined_stack["model_name"],
            )
            final_choice = {
                "source": "combined_stack",
                "summary_path": str(tpu_artifact_dir / "tpu_summary.json"),
                "submission_path": str(final_submission_path),
                "best_model_name": combined_stack["model_name"],
                "best_oof_f1": float(combined_stack["oof_f1"]),
                "best_threshold": float(combined_stack["threshold"]),
                "meta_feature_names": combined_stack["meta_feature_names"],
                "threshold_sweep_paths": sweep_paths,
            }
        elif float(tpu_summary["best_oof_f1"]) > float(baseline_summary["best_oof_f1"]):
            LOGGER.info(
                "TPU artifacts beat baseline locally | baseline_oof_f1=%.4f | tpu_oof_f1=%.4f",
                float(baseline_summary["best_oof_f1"]),
                float(tpu_summary["best_oof_f1"]),
            )
            probabilities, threshold, tpu_summary = predict_saved_tpu_ensemble(tpu_artifact_dir, test_frame)
            _write_final_submission(test_frame["ID"], probabilities, threshold, final_submission_path)
            sweep_paths = _write_threshold_sweep_submissions(
                ids=test_frame["ID"],
                probabilities=np.asarray(probabilities),
                center_threshold=float(threshold),
                output_dir=output_dir,
                prefix=str(tpu_summary["best_model_name"]),
            )
            final_choice = {
                "source": "tpu",
                "summary_path": str(tpu_artifact_dir / "tpu_summary.json"),
                "submission_path": str(final_submission_path),
                "best_model_name": tpu_summary["best_model_name"],
                "best_oof_f1": float(tpu_summary["best_oof_f1"]),
                "best_threshold": float(tpu_summary["best_threshold"]),
                "threshold_sweep_paths": sweep_paths,
            }
        else:
            LOGGER.info(
                "Keeping baseline submission | baseline_oof_f1=%.4f | tpu_oof_f1=%.4f",
                float(baseline_summary["best_oof_f1"]),
                float(tpu_summary["best_oof_f1"]),
            )

    if final_choice["source"] == "baseline":
        best_model_name = str(final_choice["best_model_name"])
        probability_path = output_dir / "models" / f"{best_model_name}_test_probabilities.parquet"
        if probability_path.exists():
            probabilities = read_dataframe_parquet(probability_path)["probability"].to_numpy()
            final_choice["threshold_sweep_paths"] = _write_threshold_sweep_submissions(
                ids=test_frame["ID"],
                probabilities=np.asarray(probabilities),
                center_threshold=float(final_choice["best_threshold"]),
                output_dir=output_dir,
                prefix=best_model_name,
            )

    write_json(output_dir / "final_selection.json", final_choice)
    LOGGER.info("Final selection written | choice=%s", final_choice)
    return final_choice


def main() -> int:
    configure_logging()
    args = parse_args()
    LOGGER.info(
        "Baseline entrypoint starting | data_root=%s | feature_dir=%s | output_dir=%s | tpu_artifact_dir=%s",
        args.data_root,
        args.feature_dir,
        args.output_dir,
        args.tpu_artifact_dir,
    )
    ensure_feature_artifacts(args.feature_dir, args.data_root, pseudo_absence_cache_dir=args.pseudo_absence_cache_dir)
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
    LOGGER.info("Baseline entrypoint finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
