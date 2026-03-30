from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from frog_challenge.config import FeatureBuildConfig
from frog_challenge.features import build_feature_artifacts
from frog_challenge.utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TerraClimate features for the EY frog challenge.")
    parser.add_argument("--train-path", type=Path, default=Path("Training_Data.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("Test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--pseudo-absence-cache-dir", type=Path, default=None)
    parser.add_argument("--start-date", default="2017-11-01")
    parser.add_argument("--end-date", default="2019-11-01")
    parser.add_argument("--min-lon", type=float, default=139.94)
    parser.add_argument("--min-lat", type=float, default=-39.74)
    parser.add_argument("--max-lon", type=float, default=151.48)
    parser.add_argument("--max-lat", type=float, default=-30.92)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    LOGGER.info(
        "Feature build entrypoint starting | train=%s | test=%s | output_dir=%s",
        args.train_path,
        args.test_path,
        args.output_dir,
    )
    config = FeatureBuildConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        pseudo_absence_cache_dir=args.pseudo_absence_cache_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        min_lon=args.min_lon,
        min_lat=args.min_lat,
        max_lon=args.max_lon,
        max_lat=args.max_lat,
    )
    artifacts = build_feature_artifacts(config)
    LOGGER.info("Feature build complete | artifacts=%s", artifacts.to_dict())
    print(json.dumps(artifacts.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
