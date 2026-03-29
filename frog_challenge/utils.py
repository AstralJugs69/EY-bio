from __future__ import annotations

import logging
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dataframe_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_dataframe_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def coerce_native(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("adlfs").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)


@contextmanager
def log_step(logger: logging.Logger, label: str, **details: Any):
    suffix = f" | details={json.dumps(details, default=str, sort_keys=True)}" if details else ""
    logger.info("START %s%s", label, suffix)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        logger.info("DONE %s | elapsed=%.1fs", label, time.perf_counter() - started_at)
