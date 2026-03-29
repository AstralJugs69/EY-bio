from __future__ import annotations

import logging
from pathlib import Path

from .config import FeatureBuildConfig
from .features import FEATURE_SCHEMA_VERSION, build_feature_artifacts
from .utils import read_json

LOGGER = logging.getLogger(__name__)


def feature_artifacts_exist(feature_dir: Path) -> bool:
    train_path = feature_dir / "train_features.parquet"
    test_path = feature_dir / "test_features.parquet"
    pseudo_absence_path = feature_dir / "pseudo_absence_candidates.parquet"
    manifest_path = feature_dir / "feature_manifest.json"
    if not (train_path.exists() and test_path.exists() and pseudo_absence_path.exists() and manifest_path.exists()):
        return False

    try:
        manifest = read_json(manifest_path)
    except Exception:
        return False
    return int(manifest.get("feature_schema_version", 0)) >= FEATURE_SCHEMA_VERSION


def ensure_feature_artifacts(
    feature_dir: Path,
    data_root: Path,
    train_path: Path | None = None,
    test_path: Path | None = None,
) -> Path:
    if feature_artifacts_exist(feature_dir):
        LOGGER.info("Using existing feature artifacts in %s", feature_dir)
        return feature_dir

    resolved_train = train_path or (data_root / "Training_Data.csv")
    resolved_test = test_path or (data_root / "Test.csv")
    if not resolved_train.exists():
        raise FileNotFoundError(f"Training data not found: {resolved_train}")
    if not resolved_test.exists():
        raise FileNotFoundError(f"Test data not found: {resolved_test}")

    LOGGER.info(
        "Feature artifacts missing; building them now from %s and %s into %s",
        resolved_train,
        resolved_test,
        feature_dir,
    )
    build_feature_artifacts(
        FeatureBuildConfig(
            train_path=resolved_train,
            test_path=resolved_test,
            output_dir=feature_dir,
        )
    )
    LOGGER.info("Feature artifacts created in %s", feature_dir)
    return feature_dir
