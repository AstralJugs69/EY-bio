from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .config import FeatureBuildConfig, SEASONS
from .utils import coerce_native, ensure_directory, write_dataframe_parquet, write_json

if TYPE_CHECKING:
    import xarray as xr


@dataclass(slots=True)
class FeatureArtifacts:
    train_features_path: Path
    test_features_path: Path
    manifest_path: Path
    feature_columns: list[str]
    dropped_feature_columns: list[str]
    coordinate_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train_features_path": str(self.train_features_path),
            "test_features_path": str(self.test_features_path),
            "manifest_path": str(self.manifest_path),
            "feature_columns": self.feature_columns,
            "dropped_feature_columns": self.dropped_feature_columns,
            "coordinate_rows": self.coordinate_rows,
        }


def build_spatial_groups(
    frame: pd.DataFrame,
    lat_column: str = "Latitude",
    lon_column: str = "Longitude",
    block_size_degrees: float = 0.5,
) -> pd.Series:
    lat_bins = np.floor(frame[lat_column].to_numpy() / block_size_degrees).astype(int)
    lon_bins = np.floor(frame[lon_column].to_numpy() / block_size_degrees).astype(int)
    return pd.Series([f"{lat_bin}_{lon_bin}" for lat_bin, lon_bin in zip(lat_bins, lon_bins)], index=frame.index)


def load_challenge_frames(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def open_terraclimate_dataset(config: FeatureBuildConfig) -> xr.Dataset:
    import planetary_computer
    import pystac_client
    import xarray as xr

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]
    open_kwargs = asset.extra_fields["xarray:open_kwargs"]

    dataset = xr.open_dataset(asset.href, **open_kwargs)
    dataset = dataset.drop_vars("crs", errors="ignore")
    dataset = dataset.sel(time=slice(config.start_date, config.end_date))
    dataset = dataset[list(config.variables)]

    mask_lon = (dataset.lon >= config.min_lon) & (dataset.lon <= config.max_lon)
    mask_lat = (dataset.lat >= config.min_lat) & (dataset.lat <= config.max_lat)
    dataset = dataset.where(mask_lon & mask_lat, drop=True)
    return dataset


def _linear_slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan

    x = np.arange(values.shape[0], dtype=np.float64)[mask]
    y = values[mask].astype(np.float64)
    x_centered = x - x.mean()
    denominator = np.square(x_centered).sum()
    if denominator == 0:
        return 0.0
    return float((x_centered * (y - y.mean())).sum() / denominator)


def _quantile_name(quantile: float) -> str:
    return f"p{int(round(quantile * 100)):02d}"


def _seasonal_mean(variable: xr.DataArray, season: str) -> xr.DataArray:
    import xarray as xr

    grouped = variable.groupby("time.season").mean(dim="time", skipna=True)
    if "season" in grouped.coords and season in set(grouped["season"].values.tolist()):
        return grouped.sel(season=season, drop=True)
    return xr.full_like(variable.isel(time=0, drop=True), np.nan, dtype=np.float64)


def compute_feature_dataset(dataset: xr.Dataset, config: FeatureBuildConfig) -> xr.Dataset:
    import xarray as xr

    feature_arrays: dict[str, xr.DataArray] = {}

    for variable_name in config.variables:
        variable = dataset[variable_name]
        feature_arrays[f"{variable_name}_mean"] = variable.mean(dim="time", skipna=True)
        feature_arrays[f"{variable_name}_median"] = variable.median(dim="time", skipna=True)
        feature_arrays[f"{variable_name}_std"] = variable.std(dim="time", skipna=True)
        feature_arrays[f"{variable_name}_min"] = variable.min(dim="time", skipna=True)
        feature_arrays[f"{variable_name}_max"] = variable.max(dim="time", skipna=True)

        quantiles = variable.quantile(config.quantiles, dim="time", skipna=True)
        for quantile in config.quantiles:
            quantile_name = _quantile_name(quantile)
            feature_arrays[f"{variable_name}_{quantile_name}"] = quantiles.sel(quantile=quantile, drop=True)

        feature_arrays[f"{variable_name}_trend"] = xr.apply_ufunc(
            _linear_slope,
            variable,
            input_core_dims=[["time"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
        )

        last_n = variable.isel(time=slice(-config.last_n_months, None))
        feature_arrays[f"{variable_name}_last{config.last_n_months}_mean"] = last_n.mean(dim="time", skipna=True)

        for season in SEASONS:
            feature_arrays[f"{variable_name}_{season.lower()}_mean"] = _seasonal_mean(variable, season)

        median_surface = feature_arrays[f"{variable_name}_median"]
        rolling = median_surface.rolling(
            lat=config.neighborhood_size,
            lon=config.neighborhood_size,
            center=True,
            min_periods=1,
        )
        feature_arrays[f"{variable_name}_local_mean"] = rolling.mean()
        feature_arrays[f"{variable_name}_local_std"] = rolling.std()

    return xr.Dataset(feature_arrays)


def sample_feature_dataset(feature_dataset: xr.Dataset, points_frame: pd.DataFrame) -> pd.DataFrame:
    import xarray as xr

    lat_indexer = xr.DataArray(points_frame["Latitude"].to_numpy(), dims="point")
    lon_indexer = xr.DataArray(points_frame["Longitude"].to_numpy(), dims="point")
    sampled = feature_dataset.sel(lat=lat_indexer, lon=lon_indexer, method="nearest").load()
    sampled_columns = {
        name: np.asarray(data_array.values).reshape(-1)
        for name, data_array in sampled.data_vars.items()
    }
    sampled_frame = pd.DataFrame(sampled_columns, index=points_frame.index)
    return sampled_frame.applymap(coerce_native)


def _drop_all_null_feature_columns(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    protected_columns: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    protected = set(protected_columns)
    feature_columns = [column for column in train_features.columns if column not in protected]
    dropped = [
        column
        for column in feature_columns
        if train_features[column].isna().all() and test_features[column].isna().all()
    ]
    if not dropped:
        return train_features, test_features, dropped

    return train_features.drop(columns=dropped), test_features.drop(columns=dropped), dropped


def build_feature_artifacts(config: FeatureBuildConfig) -> FeatureArtifacts:
    output_dir = ensure_directory(config.output_dir)
    train_df, test_df = load_challenge_frames(config.train_path, config.test_path)

    coordinate_frame = (
        pd.concat(
            [
                train_df[["Latitude", "Longitude"]],
                test_df[["Latitude", "Longitude"]],
            ],
            axis=0,
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    dataset = open_terraclimate_dataset(config)
    feature_dataset = compute_feature_dataset(dataset, config)
    sampled_coordinates = pd.concat(
        [coordinate_frame, sample_feature_dataset(feature_dataset, coordinate_frame)],
        axis=1,
    )

    train_features = train_df.merge(
        sampled_coordinates,
        on=["Latitude", "Longitude"],
        how="left",
        validate="m:1",
    )
    test_features = test_df.merge(
        sampled_coordinates,
        on=["Latitude", "Longitude"],
        how="left",
        validate="m:1",
    )

    train_features["spatial_group"] = build_spatial_groups(
        train_features,
        block_size_degrees=config.spatial_group_size_degrees,
    )
    test_features["spatial_group"] = build_spatial_groups(
        test_features,
        block_size_degrees=config.spatial_group_size_degrees,
    )

    train_features, test_features, dropped_columns = _drop_all_null_feature_columns(
        train_features,
        test_features,
        protected_columns=("ID", "Latitude", "Longitude", "Occurrence Status", "spatial_group"),
    )
    feature_columns = [
        column
        for column in train_features.columns
        if column not in {"ID", "Latitude", "Longitude", "Occurrence Status", "spatial_group"}
    ]

    train_path = output_dir / "train_features.parquet"
    test_path = output_dir / "test_features.parquet"
    manifest_path = output_dir / "feature_manifest.json"

    write_dataframe_parquet(train_path, train_features)
    write_dataframe_parquet(test_path, test_features)
    write_json(
        manifest_path,
        {
            "train_rows": int(train_features.shape[0]),
            "test_rows": int(test_features.shape[0]),
            "coordinate_rows": int(coordinate_frame.shape[0]),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "bounds": {
                "min_lon": config.min_lon,
                "min_lat": config.min_lat,
                "max_lon": config.max_lon,
                "max_lat": config.max_lat,
            },
            "variables": list(config.variables),
            "feature_columns": feature_columns,
            "aggregations": [
                "mean",
                "median",
                "std",
                "min",
                "max",
                _quantile_name(config.quantiles[0]),
                _quantile_name(config.quantiles[1]),
                "trend",
                f"last{config.last_n_months}_mean",
                "djf_mean",
                "mam_mean",
                "jja_mean",
                "son_mean",
                "local_mean",
                "local_std",
            ],
            "dropped_all_null_columns": dropped_columns,
            "spatial_group_size_degrees": config.spatial_group_size_degrees,
        },
    )

    return FeatureArtifacts(
        train_features_path=train_path,
        test_features_path=test_path,
        manifest_path=manifest_path,
        feature_columns=feature_columns,
        dropped_feature_columns=dropped_columns,
        coordinate_rows=int(coordinate_frame.shape[0]),
    )
