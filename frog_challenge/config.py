from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TERRACLIMATE_VARIABLES: tuple[str, ...] = (
    "tmax",
    "tmin",
    "vap",
    "ppt",
    "srad",
    "ws",
    "aet",
    "pet",
    "q",
    "def",
    "soil",
    "swe",
    "pdsi",
    "vpd",
)

SEASONS: tuple[str, ...] = ("DJF", "MAM", "JJA", "SON")


@dataclass(slots=True)
class FeatureBuildConfig:
    train_path: Path
    test_path: Path
    output_dir: Path
    start_date: str = "2017-11-01"
    end_date: str = "2019-11-01"
    min_lon: float = 139.94
    min_lat: float = -39.74
    max_lon: float = 151.48
    max_lat: float = -30.92
    variables: tuple[str, ...] = TERRACLIMATE_VARIABLES
    last_n_months: int = 6
    neighborhood_sizes: tuple[int, ...] = (3, 5, 7)
    quantiles: tuple[float, float] = (0.10, 0.90)
    spatial_group_size_degrees: float = 0.5


@dataclass(slots=True)
class ModelConfig:
    feature_dir: Path
    output_dir: Path
    target_column: str = "Occurrence Status"
    id_column: str = "ID"
    random_state: int = 42
    n_splits: int = 5
    spatial_group_size_degrees: float = 0.5
    optimize_threshold_min: float = 0.10
    optimize_threshold_max: float = 0.90
    optimize_threshold_step: float = 0.01
    negative_bagging_bags: int = 5
    negative_bagging_models: tuple[str, ...] = ("extra_trees", "xgboost")
    negative_bagging_negative_fraction: float = 1.0
    stacking_models: tuple[str, ...] = ("logistic_regression", "hist_gradient_boosting")
    stacking_top_k: int = 5
    protected_columns: tuple[str, ...] = (
        "ID",
        "Latitude",
        "Longitude",
        "Occurrence Status",
        "spatial_group",
    )


@dataclass(slots=True)
class TPUConfig:
    feature_dir: Path
    output_dir: Path
    target_column: str = "Occurrence Status"
    id_column: str = "ID"
    random_state: int = 42
    n_splits: int = 5
    spatial_group_size_degrees: float = 0.5
    batch_size: int = 256
    epochs: int = 100
    patience: int = 12
    learning_rate: float = 1e-3
    dropout: float = 0.20
    dense_hidden_units: tuple[int, ...] = (256, 128, 64)
    residual_hidden_units: tuple[int, ...] = (256, 128, 64)
    optimize_threshold_min: float = 0.10
    optimize_threshold_max: float = 0.90
    optimize_threshold_step: float = 0.01
    architectures: tuple[str, ...] = ("dense_mlp", "residual_mlp")
    save_fold_models: bool = True
    require_tpu: bool = True
    protected_columns: tuple[str, ...] = (
        "ID",
        "Latitude",
        "Longitude",
        "Occurrence Status",
        "spatial_group",
    )
