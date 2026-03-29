from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .config import TPUConfig
from .modeling import load_feature_tables, optimize_threshold, select_feature_columns
from .utils import ensure_directory, log_step, read_json, write_dataframe_parquet, write_json

LOGGER = logging.getLogger(__name__)

@dataclass(slots=True)
class GPURunArtifacts:
    summary_path: Path
    best_model_name: str
    best_threshold: float
    best_oof_f1: float


TPURunArtifacts = GPURunArtifacts


def create_gpu_strategy():
    import tensorflow as tf

    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    LOGGER.info(
        "Visible GPU devices | physical=%s | logical=%s | names=%s",
        len(physical_gpus),
        len(logical_gpus),
        [device.name for device in physical_gpus],
    )

    for device in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as exc:
            LOGGER.warning("Could not enable memory growth for %s | %s: %s", device.name, type(exc).__name__, exc)

    if physical_gpus:
        strategy = tf.distribute.MirroredStrategy()
        return tf, strategy, "GPU", {
            "physical_gpu_count": len(physical_gpus),
            "logical_gpu_count": len(tf.config.list_logical_devices("GPU")),
            "replicas": strategy.num_replicas_in_sync,
        }

    strategy = tf.distribute.get_strategy()
    return tf, strategy, "CPU", {"physical_gpu_count": 0, "logical_gpu_count": 0, "replicas": 1}


def _dense_mlp(tf, input_dim: int, hidden_units: tuple[int, ...], dropout: float, learning_rate: float):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for units in hidden_units:
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dense_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _residual_mlp(tf, input_dim: int, hidden_units: tuple[int, ...], dropout: float, learning_rate: float):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(hidden_units[0])(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for units in hidden_units:
        shortcut = x
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units)(shortcut)
        y = layers.Dense(units)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Dropout(dropout)(y)
        y = layers.Dense(units)(y)
        y = layers.BatchNormalization()(y)
        x = layers.Add()([shortcut, y])
        x = layers.Activation("relu")(x)

    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="residual_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _model_builder(tf, architecture: str, input_dim: int, config: TPUConfig):
    if architecture == "dense_mlp":
        return _dense_mlp(tf, input_dim, config.dense_hidden_units, config.dropout, config.learning_rate)
    if architecture == "residual_mlp":
        return _residual_mlp(tf, input_dim, config.residual_hidden_units, config.dropout, config.learning_rate)
    raise ValueError(f"Unsupported architecture: {architecture}")


def _preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _fit_single_architecture(
    tf,
    strategy,
    architecture: str,
    config: TPUConfig,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    groups = train_frame["spatial_group"]
    splitter = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    y = train_frame[config.target_column].to_numpy()
    X = train_frame[feature_columns]
    X_test = test_frame[feature_columns]

    oof_probabilities = np.zeros(train_frame.shape[0], dtype=np.float64)
    test_fold_probabilities: list[np.ndarray] = []
    fold_rows: list[dict[str, float | int]] = []
    model_dir = ensure_directory(config.output_dir / "models" / architecture)
    preprocessor_dir = ensure_directory(config.output_dir / "preprocessors" / architecture)

    for fold_index, (train_index, valid_index) in enumerate(splitter.split(X, y, groups=groups), start=1):
        LOGGER.info(
            "Neural architecture %s | fold %s/%s | train_rows=%s | valid_rows=%s",
            architecture,
            fold_index,
            config.n_splits,
            len(train_index),
            len(valid_index),
        )
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config.random_state + fold_index)

        effective_batch_size = int(config.batch_size * max(1, strategy.num_replicas_in_sync))
        fold_preprocessor = _preprocessor()
        X_train_fold = fold_preprocessor.fit_transform(X.iloc[train_index]).astype("float32")
        X_valid_fold = fold_preprocessor.transform(X.iloc[valid_index]).astype("float32")
        X_test_fold = fold_preprocessor.transform(X_test).astype("float32")
        y_train_fold = y[train_index].astype("float32")
        y_valid_fold = y[valid_index].astype("float32")

        classes = np.unique(y_train_fold.astype(int))
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_fold.astype(int))
        class_weight = {int(label): float(weight) for label, weight in zip(classes, class_weights)}

        with strategy.scope():
            model = _model_builder(tf, architecture, input_dim=X_train_fold.shape[1], config=config)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(3, config.patience // 3),
                min_lr=1e-5,
            ),
        ]

        history = model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_valid_fold, y_valid_fold),
            epochs=config.epochs,
            batch_size=effective_batch_size,
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weight,
        )

        valid_probabilities = model.predict(X_valid_fold, batch_size=effective_batch_size, verbose=0).reshape(-1)
        test_probabilities = model.predict(X_test_fold, batch_size=effective_batch_size, verbose=0).reshape(-1)
        oof_probabilities[valid_index] = valid_probabilities
        test_fold_probabilities.append(test_probabilities)
        fold_rows.append(
            {
                "fold": fold_index,
                "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                "f1_at_0_50": float(f1_score(y_valid_fold.astype(int), (valid_probabilities >= 0.50).astype(int))),
            }
        )
        LOGGER.info(
            "Neural architecture %s | fold %s complete | best_epoch=%s | batch_size=%s | f1_at_0_50=%.4f",
            architecture,
            fold_index,
            fold_rows[-1]["best_epoch"],
            effective_batch_size,
            fold_rows[-1]["f1_at_0_50"],
        )

        if config.save_fold_models:
            model.save(model_dir / f"fold_{fold_index}.keras")
            joblib.dump(fold_preprocessor, preprocessor_dir / f"fold_{fold_index}.joblib")

    mean_test_probability = np.mean(np.vstack(test_fold_probabilities), axis=0)
    threshold, oof_f1 = optimize_threshold(
        y_true=y.astype(int),
        probabilities=oof_probabilities,
        threshold_min=config.optimize_threshold_min,
        threshold_max=config.optimize_threshold_max,
        threshold_step=config.optimize_threshold_step,
    )
    LOGGER.info(
        "Neural architecture %s complete | optimized_threshold=%.2f | oof_f1=%.4f",
        architecture,
        threshold,
        oof_f1,
    )

    return {
        "architecture": architecture,
        "oof_probabilities": oof_probabilities,
        "test_probabilities": mean_test_probability,
        "threshold": threshold,
        "oof_f1": oof_f1,
        "fold_rows": fold_rows,
    }


def _candidate_record(name: str, probabilities: np.ndarray, test_probabilities: np.ndarray, y_true: np.ndarray, config: TPUConfig) -> dict[str, object]:
    threshold, oof_f1 = optimize_threshold(
        y_true=y_true,
        probabilities=probabilities,
        threshold_min=config.optimize_threshold_min,
        threshold_max=config.optimize_threshold_max,
        threshold_step=config.optimize_threshold_step,
    )
    return {
        "architecture": name,
        "oof_probabilities": probabilities,
        "test_probabilities": test_probabilities,
        "threshold": threshold,
        "oof_f1": oof_f1,
        "fold_rows": [],
    }


def run_gpu_suite(config: TPUConfig) -> GPURunArtifacts:
    output_dir = ensure_directory(config.output_dir)
    train_frame, test_frame, _, _ = load_feature_tables(config.feature_dir)
    feature_columns = select_feature_columns(train_frame, config.protected_columns)
    y_true = train_frame[config.target_column].to_numpy().astype(int)

    tf, strategy, accelerator, accelerator_details = create_gpu_strategy()
    if accelerator != "GPU" and config.require_tpu:
        raise RuntimeError(
            "GPU stage was requested, but no GPU was initialized. "
            f"Visible devices: {accelerator_details}"
        )
    LOGGER.info(
        "Starting GPU suite | accelerator=%s | accelerator_details=%s | output_dir=%s | feature_count=%s | architectures=%s",
        accelerator,
        accelerator_details,
        output_dir,
        len(feature_columns),
        list(config.architectures),
    )
    architecture_outputs: dict[str, dict[str, object]] = {}
    for architecture in config.architectures:
        with log_step(LOGGER, f"Train neural architecture {architecture}"):
            architecture_outputs[architecture] = _fit_single_architecture(
                tf=tf,
                strategy=strategy,
                architecture=architecture,
                config=config,
                train_frame=train_frame,
                test_frame=test_frame,
                feature_columns=feature_columns,
            )

    candidates = dict(architecture_outputs)
    if len(architecture_outputs) >= 2:
        sorted_outputs = sorted(architecture_outputs.values(), key=lambda item: item["oof_f1"], reverse=True)
        best_pair = sorted_outputs[:2]
        ensemble_name = "ensemble_" + "__".join(str(item["architecture"]) for item in best_pair)
        ensemble_oof = np.mean(np.vstack([item["oof_probabilities"] for item in best_pair]), axis=0)
        ensemble_test = np.mean(np.vstack([item["test_probabilities"] for item in best_pair]), axis=0)
        candidates[ensemble_name] = _candidate_record(ensemble_name, ensemble_oof, ensemble_test, y_true, config)

    best_name, best_output = max(candidates.items(), key=lambda item: float(item[1]["oof_f1"]))
    LOGGER.info(
        "Best neural candidate selected | name=%s | oof_f1=%.4f | threshold=%.2f",
        best_name,
        best_output["oof_f1"],
        best_output["threshold"],
    )

    predictions_dir = ensure_directory(output_dir / "predictions")
    for name, output in candidates.items():
        write_dataframe_parquet(
            predictions_dir / f"{name}_oof.parquet",
            pd.DataFrame(
                {
                    config.id_column: train_frame[config.id_column],
                    config.target_column: train_frame[config.target_column],
                    "oof_probability": output["oof_probabilities"],
                }
            ),
        )
        write_dataframe_parquet(
            predictions_dir / f"{name}_test_probabilities.parquet",
            pd.DataFrame(
                {
                    config.id_column: test_frame[config.id_column],
                    "probability": output["test_probabilities"],
                }
            ),
        )
        write_json(
            predictions_dir / f"{name}_metrics.json",
            {
                "threshold": float(output["threshold"]),
                "oof_f1": float(output["oof_f1"]),
                "fold_rows": output["fold_rows"],
            },
        )

    summary_path = output_dir / "tpu_summary.json"
    write_json(
        summary_path,
        {
            "accelerator": accelerator,
            "feature_columns": feature_columns,
            "models": {
                name: {
                    "threshold": float(output["threshold"]),
                    "oof_f1": float(output["oof_f1"]),
                    "fold_rows": output["fold_rows"],
                }
                for name, output in candidates.items()
            },
            "best_model_name": best_name,
            "selected_model_names": best_name.replace("ensemble_", "").split("__") if best_name.startswith("ensemble_") else [best_name],
            "best_threshold": float(best_output["threshold"]),
            "best_oof_f1": float(best_output["oof_f1"]),
        },
    )
    LOGGER.info("GPU suite finished | summary=%s", summary_path)

    return GPURunArtifacts(
        summary_path=summary_path,
        best_model_name=best_name,
        best_threshold=float(best_output["threshold"]),
        best_oof_f1=float(best_output["oof_f1"]),
    )


def run_tpu_suite(config: TPUConfig) -> TPURunArtifacts:
    return run_gpu_suite(config)


def predict_saved_tpu_ensemble(
    tpu_artifact_dir: Path,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, float, dict[str, object]]:
    import tensorflow as tf

    summary = read_json(tpu_artifact_dir / "tpu_summary.json")
    model_names = summary["selected_model_names"]
    feature_columns = summary["feature_columns"]
    all_model_probabilities: list[np.ndarray] = []

    for model_name in model_names:
        model_dir = tpu_artifact_dir / "models" / model_name
        preprocessor_dir = tpu_artifact_dir / "preprocessors" / model_name
        fold_probabilities: list[np.ndarray] = []

        model_paths = sorted(model_dir.glob("fold_*.keras"))
        for model_path in model_paths:
            fold_suffix = model_path.stem.split("_")[-1]
            preprocessor_path = preprocessor_dir / f"fold_{fold_suffix}.joblib"
            preprocessor = joblib.load(preprocessor_path)
            model = tf.keras.models.load_model(model_path)
            transformed = preprocessor.transform(test_frame[feature_columns]).astype("float32")
            fold_probabilities.append(model.predict(transformed, verbose=0).reshape(-1))

        if not fold_probabilities:
            raise FileNotFoundError(f"No saved fold models found for {model_name} in {model_dir}")
        all_model_probabilities.append(np.mean(np.vstack(fold_probabilities), axis=0))

    probabilities = np.mean(np.vstack(all_model_probabilities), axis=0)
    threshold = float(summary["best_threshold"])
    return probabilities, threshold, summary
