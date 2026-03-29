from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .config import ModelConfig
from .utils import ensure_directory, log_step, read_dataframe_parquet, read_json, write_dataframe_parquet, write_json


EstimatorFactory = Callable[[], object]
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelRunArtifacts:
    summary_path: Path
    submission_path: Path
    best_model_name: str
    best_threshold: float
    best_oof_f1: float


def load_feature_tables(feature_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    LOGGER.info("Loading feature tables from %s", feature_dir)
    train_features = read_dataframe_parquet(feature_dir / "train_features.parquet")
    test_features = read_dataframe_parquet(feature_dir / "test_features.parquet")
    manifest = read_json(feature_dir / "feature_manifest.json")
    LOGGER.info(
        "Loaded feature tables | train_rows=%s | test_rows=%s | feature_count=%s",
        train_features.shape[0],
        test_features.shape[0],
        len(manifest.get("feature_columns", [])),
    )
    return train_features, test_features, manifest


def select_feature_columns(frame: pd.DataFrame, protected_columns: tuple[str, ...]) -> list[str]:
    protected = set(protected_columns)
    return [column for column in frame.columns if column not in protected]


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, float]:
    best_threshold = 0.50
    best_score = -1.0
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def predict_probabilities(estimator: object, features: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError(f"Estimator {type(estimator)!r} does not expose probabilities.")


def _logistic_factory(random_state: int) -> EstimatorFactory:
    def factory() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    return factory


def _hist_gradient_factory(random_state: int) -> EstimatorFactory:
    def factory() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=6,
                        max_iter=500,
                        min_samples_leaf=20,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    return factory


def _extra_trees_factory(random_state: int) -> EstimatorFactory:
    def factory() -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    return factory


def _catboost_factory(random_state: int) -> EstimatorFactory | None:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None

    def factory() -> object:
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            iterations=800,
            depth=6,
            learning_rate=0.03,
            random_seed=random_state,
            auto_class_weights="Balanced",
            verbose=False,
        )

    return factory


def _xgboost_factory(random_state: int) -> EstimatorFactory | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    def factory() -> object:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=900,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1.5,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )

    return factory


def get_baseline_model_factories(random_state: int) -> dict[str, EstimatorFactory]:
    factories: dict[str, EstimatorFactory] = {
        "logistic_regression": _logistic_factory(random_state),
        "hist_gradient_boosting": _hist_gradient_factory(random_state),
        "extra_trees": _extra_trees_factory(random_state),
    }
    catboost_factory = _catboost_factory(random_state)
    if catboost_factory is not None:
        factories["catboost"] = catboost_factory
    xgboost_factory = _xgboost_factory(random_state)
    if xgboost_factory is not None:
        factories["xgboost"] = xgboost_factory
    return factories


def reproduce_random_split_benchmark(
    train_frame: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> dict[str, float | list[str]]:
    benchmark_features = [column for column in ("srad_median", "vap_median") if column in train_frame.columns]
    if len(benchmark_features) != 2:
        raise ValueError(
            "Expected benchmark feature columns 'srad_median' and 'vap_median'. "
            "Run feature generation before baseline modeling."
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_frame[benchmark_features],
        train_frame[target_column].to_numpy(),
        test_size=0.30,
        stratify=train_frame[target_column].to_numpy(),
        random_state=random_state,
    )
    pipeline = _logistic_factory(random_state)()
    LOGGER.info("Running random-split benchmark reproduction on features %s", benchmark_features)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)
    score = f1_score(y_valid, predictions)
    LOGGER.info("Benchmark reproduction finished | random_split_f1=%.4f", score)
    return {"benchmark_features": benchmark_features, "random_split_f1": float(score)}


def _run_cross_validated_model(
    model_name: str,
    estimator_factory: EstimatorFactory,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    splitter: StratifiedGroupKFold,
    groups: pd.Series,
    config: ModelConfig,
) -> dict[str, object]:
    oof_probabilities = np.zeros(train_frame.shape[0], dtype=np.float64)
    test_fold_probabilities: list[np.ndarray] = []
    fold_rows: list[dict[str, float | int]] = []

    X = train_frame[feature_columns]
    y = train_frame[target_column].to_numpy()
    X_test = test_frame[feature_columns]

    for fold_index, (train_index, valid_index) in enumerate(splitter.split(X, y, groups=groups), start=1):
        LOGGER.info(
            "Model %s | fold %s/%s | train_rows=%s | valid_rows=%s",
            model_name,
            fold_index,
            config.n_splits,
            len(train_index),
            len(valid_index),
        )
        estimator = estimator_factory()
        X_train_fold = X.iloc[train_index]
        y_train_fold = y[train_index]
        X_valid_fold = X.iloc[valid_index]
        y_valid_fold = y[valid_index]

        if model_name == "catboost":
            estimator.fit(X_train_fold, y_train_fold, eval_set=(X_valid_fold, y_valid_fold), use_best_model=True)
        else:
            estimator.fit(X_train_fold, y_train_fold)

        valid_probabilities = predict_probabilities(estimator, X_valid_fold)
        test_probabilities = predict_probabilities(estimator, X_test)
        oof_probabilities[valid_index] = valid_probabilities
        test_fold_probabilities.append(test_probabilities)
        fold_rows.append(
            {
                "fold": fold_index,
                "f1_at_0_50": float(f1_score(y_valid_fold, (valid_probabilities >= 0.50).astype(int))),
            }
        )
        LOGGER.info(
            "Model %s | fold %s complete | f1_at_0_50=%.4f",
            model_name,
            fold_index,
            fold_rows[-1]["f1_at_0_50"],
        )

    threshold, oof_f1 = optimize_threshold(
        y_true=y,
        probabilities=oof_probabilities,
        threshold_min=config.optimize_threshold_min,
        threshold_max=config.optimize_threshold_max,
        threshold_step=config.optimize_threshold_step,
    )
    mean_test_probability = np.mean(np.vstack(test_fold_probabilities), axis=0)
    LOGGER.info(
        "Model %s complete | optimized_threshold=%.2f | oof_f1=%.4f",
        model_name,
        threshold,
        oof_f1,
    )

    return {
        "model_name": model_name,
        "oof_probabilities": oof_probabilities,
        "test_probabilities": mean_test_probability,
        "threshold": threshold,
        "oof_f1": oof_f1,
        "fold_rows": fold_rows,
    }


def _candidate_record(
    name: str,
    probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    y_true: np.ndarray,
    config: ModelConfig,
) -> dict[str, object]:
    threshold, oof_f1 = optimize_threshold(
        y_true=y_true,
        probabilities=probabilities,
        threshold_min=config.optimize_threshold_min,
        threshold_max=config.optimize_threshold_max,
        threshold_step=config.optimize_threshold_step,
    )
    return {
        "model_name": name,
        "oof_probabilities": probabilities,
        "test_probabilities": test_probabilities,
        "threshold": threshold,
        "oof_f1": oof_f1,
        "fold_rows": [],
    }


def _best_weighted_ensemble_candidates(
    model_outputs: dict[str, dict[str, object]],
    y_true: np.ndarray,
    config: ModelConfig,
) -> dict[str, dict[str, object]]:
    weighted_candidates: dict[str, dict[str, object]] = {}
    ranked = sorted(model_outputs.values(), key=lambda item: float(item["oof_f1"]), reverse=True)[:4]
    if len(ranked) < 2:
        return weighted_candidates

    best_pair_candidate: dict[str, object] | None = None
    best_pair_name: str | None = None
    for left, right in combinations(ranked, 2):
        left_name = str(left["model_name"])
        right_name = str(right["model_name"])
        left_oof = np.asarray(left["oof_probabilities"])
        right_oof = np.asarray(right["oof_probabilities"])
        left_test = np.asarray(left["test_probabilities"])
        right_test = np.asarray(right["test_probabilities"])

        for left_weight in np.arange(0.1, 1.0, 0.1):
            right_weight = 1.0 - left_weight
            candidate_name = f"weighted_{left_name}_{left_weight:.1f}__{right_name}_{right_weight:.1f}"
            candidate = _candidate_record(
                candidate_name,
                left_weight * left_oof + right_weight * right_oof,
                left_weight * left_test + right_weight * right_test,
                y_true,
                config,
            )
            if best_pair_candidate is None or float(candidate["oof_f1"]) > float(best_pair_candidate["oof_f1"]):
                best_pair_candidate = candidate
                best_pair_name = candidate_name

    if best_pair_candidate is not None and best_pair_name is not None:
        weighted_candidates[best_pair_name] = best_pair_candidate
        LOGGER.info(
            "Best weighted pair ensemble | name=%s | oof_f1=%.4f",
            best_pair_name,
            float(best_pair_candidate["oof_f1"]),
        )

    if len(ranked) >= 3:
        best_triplet_candidate: dict[str, object] | None = None
        best_triplet_name: str | None = None
        for first, second, third in combinations(ranked, 3):
            first_name = str(first["model_name"])
            second_name = str(second["model_name"])
            third_name = str(third["model_name"])
            first_oof = np.asarray(first["oof_probabilities"])
            second_oof = np.asarray(second["oof_probabilities"])
            third_oof = np.asarray(third["oof_probabilities"])
            first_test = np.asarray(first["test_probabilities"])
            second_test = np.asarray(second["test_probabilities"])
            third_test = np.asarray(third["test_probabilities"])

            for first_weight in np.arange(0.2, 0.8, 0.2):
                for second_weight in np.arange(0.2, 0.8, 0.2):
                    third_weight = 1.0 - first_weight - second_weight
                    if third_weight <= 0:
                        continue
                    candidate_name = (
                        f"weighted_{first_name}_{first_weight:.1f}__"
                        f"{second_name}_{second_weight:.1f}__"
                        f"{third_name}_{third_weight:.1f}"
                    )
                    candidate = _candidate_record(
                        candidate_name,
                        first_weight * first_oof + second_weight * second_oof + third_weight * third_oof,
                        first_weight * first_test + second_weight * second_test + third_weight * third_test,
                        y_true,
                        config,
                    )
                    if best_triplet_candidate is None or float(candidate["oof_f1"]) > float(best_triplet_candidate["oof_f1"]):
                        best_triplet_candidate = candidate
                        best_triplet_name = candidate_name

        if best_triplet_candidate is not None and best_triplet_name is not None:
            weighted_candidates[best_triplet_name] = best_triplet_candidate
            LOGGER.info(
                "Best weighted triplet ensemble | name=%s | oof_f1=%.4f",
                best_triplet_name,
                float(best_triplet_candidate["oof_f1"]),
            )

    return weighted_candidates


def run_baseline_suite(config: ModelConfig) -> ModelRunArtifacts:
    output_dir = ensure_directory(config.output_dir)
    train_frame, test_frame, manifest = load_feature_tables(config.feature_dir)
    feature_columns = select_feature_columns(train_frame, config.protected_columns)
    LOGGER.info(
        "Starting CPU baseline suite | output_dir=%s | feature_count=%s | n_splits=%s",
        output_dir,
        len(feature_columns),
        config.n_splits,
    )
    benchmark = reproduce_random_split_benchmark(train_frame, config.target_column, config.random_state)

    splitter = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    groups = train_frame["spatial_group"]
    model_outputs: dict[str, dict[str, object]] = {}
    y_true = train_frame[config.target_column].to_numpy()

    for model_name, estimator_factory in get_baseline_model_factories(config.random_state).items():
        with log_step(LOGGER, f"Train baseline model {model_name}"):
            model_outputs[model_name] = _run_cross_validated_model(
                model_name=model_name,
                estimator_factory=estimator_factory,
                train_frame=train_frame,
                test_frame=test_frame,
                feature_columns=feature_columns,
                target_column=config.target_column,
                splitter=splitter,
                groups=groups,
                config=config,
            )

    sorted_models = sorted(model_outputs.values(), key=lambda item: item["oof_f1"], reverse=True)
    candidate_outputs = dict(model_outputs)
    for ensemble_size in (2, 3):
        if len(sorted_models) >= ensemble_size:
            members = sorted_models[:ensemble_size]
            member_names = [str(member["model_name"]) for member in members]
            ensemble_name = "ensemble_" + "__".join(member_names)
            ensemble_oof = np.mean(np.vstack([member["oof_probabilities"] for member in members]), axis=0)
            ensemble_test = np.mean(np.vstack([member["test_probabilities"] for member in members]), axis=0)
            candidate_outputs[ensemble_name] = _candidate_record(
                ensemble_name,
                ensemble_oof,
                ensemble_test,
                y_true,
                config,
            )

    candidate_outputs.update(_best_weighted_ensemble_candidates(model_outputs, y_true, config))

    best_name, best_output = max(candidate_outputs.items(), key=lambda item: float(item[1]["oof_f1"]))
    LOGGER.info(
        "Best baseline candidate selected | name=%s | oof_f1=%.4f | threshold=%.2f",
        best_name,
        best_output["oof_f1"],
        best_output["threshold"],
    )

    model_dir = ensure_directory(output_dir / "models")
    for name, output in model_outputs.items():
        oof_frame = pd.DataFrame(
            {
                config.id_column: train_frame[config.id_column],
                config.target_column: train_frame[config.target_column],
                "oof_probability": output["oof_probabilities"],
            }
        )
        write_dataframe_parquet(model_dir / f"{name}_oof.parquet", oof_frame)
        test_probability_frame = pd.DataFrame(
            {
                config.id_column: test_frame[config.id_column],
                "probability": output["test_probabilities"],
            }
        )
        write_dataframe_parquet(model_dir / f"{name}_test_probabilities.parquet", test_probability_frame)
        write_json(
            model_dir / f"{name}_metrics.json",
            {
                "threshold": output["threshold"],
                "oof_f1": output["oof_f1"],
                "fold_rows": output["fold_rows"],
            },
        )

    best_submission = pd.DataFrame(
        {
            config.id_column: test_frame[config.id_column],
            "Target": (np.asarray(best_output["test_probabilities"]) >= float(best_output["threshold"])).astype(int),
        }
    )
    submission_path = output_dir / "submission.csv"
    best_submission.to_csv(submission_path, index=False)

    summary = {
        "benchmark": benchmark,
        "feature_columns": feature_columns,
        "feature_manifest": manifest,
        "models": {
            name: {
                "threshold": float(output["threshold"]),
                "oof_f1": float(output["oof_f1"]),
                "fold_rows": output["fold_rows"],
            }
            for name, output in candidate_outputs.items()
        },
        "best_model_name": best_name,
        "selected_model_names": best_name.replace("ensemble_", "").split("__") if best_name.startswith("ensemble_") else [best_name],
        "best_threshold": float(best_output["threshold"]),
        "best_oof_f1": float(best_output["oof_f1"]),
    }

    summary_path = output_dir / "baseline_summary.json"
    write_json(summary_path, summary)
    LOGGER.info(
        "Baseline suite finished | summary=%s | submission=%s",
        summary_path,
        submission_path,
    )
    return ModelRunArtifacts(
        summary_path=summary_path,
        submission_path=submission_path,
        best_model_name=best_name,
        best_threshold=float(best_output["threshold"]),
        best_oof_f1=float(best_output["oof_f1"]),
    )
