from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
import warnings

import numpy as np
import pandas as pd

from .config import FeatureBuildConfig, SEASONS
from .utils import coerce_native, ensure_directory, log_step, read_dataframe_parquet, write_dataframe_parquet, write_json

if TYPE_CHECKING:
    import xarray as xr

LOGGER = logging.getLogger(__name__)
FEATURE_SCHEMA_VERSION = 3


@dataclass(slots=True)
class FeatureArtifacts:
    train_features_path: Path
    test_features_path: Path
    pseudo_absence_candidates_path: Path
    manifest_path: Path
    feature_columns: list[str]
    dropped_feature_columns: list[str]
    coordinate_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train_features_path": str(self.train_features_path),
            "test_features_path": str(self.test_features_path),
            "pseudo_absence_candidates_path": str(self.pseudo_absence_candidates_path),
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
    LOGGER.info("Loading challenge data | train=%s | test=%s", train_path, test_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    LOGGER.info("Loaded challenge data | train_rows=%s | test_rows=%s", train_df.shape[0], test_df.shape[0])
    return train_df, test_df


def open_terraclimate_dataset(config: FeatureBuildConfig) -> xr.Dataset:
    import planetary_computer
    import pystac_client
    import xarray as xr

    if importlib.util.find_spec("dask") is None:
        raise RuntimeError(
            "TerraClimate feature extraction requires dask, but it is not installed. "
            "Install the Kaggle requirements from requirements-kaggle.txt and rerun."
        )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]
    open_kwargs = asset.extra_fields["xarray:open_kwargs"]

    with log_step(
        LOGGER,
        "Open TerraClimate dataset",
        start_date=config.start_date,
        end_date=config.end_date,
        min_lon=config.min_lon,
        min_lat=config.min_lat,
        max_lon=config.max_lon,
        max_lat=config.max_lat,
    ):
        try:
            dataset = xr.open_dataset(asset.href, **open_kwargs)
        except ValueError as exc:
            if "unrecognized chunk manager dask" in str(exc):
                raise RuntimeError(
                    "xarray attempted to use dask chunking, but dask is unavailable in the runtime. "
                    "Install the dependencies from requirements-kaggle.txt and rerun the notebook."
                ) from exc
            raise
        dataset = dataset.drop_vars("crs", errors="ignore")
        dataset = dataset.sel(time=slice(config.start_date, config.end_date))
        dataset = dataset[list(config.variables)]
        # Quantile and trend calculations treat time as a core dimension.
        # TerraClimate only spans 25 monthly steps for this challenge window,
        # so forcing a single chunk on time is cheap and avoids dask core-dim errors.
        dataset = dataset.chunk({"time": -1})

        mask_lon = (dataset.lon >= config.min_lon) & (dataset.lon <= config.max_lon)
        mask_lat = (dataset.lat >= config.min_lat) & (dataset.lat <= config.max_lat)
        dataset = dataset.where(mask_lon & mask_lat, drop=True)
    LOGGER.info("Opened TerraClimate dataset | sizes=%s | chunks=%s", dict(dataset.sizes), dataset.chunksizes)
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


def _pseudo_absence_cache_key(config: FeatureBuildConfig) -> str:
    payload = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "bounds": [config.min_lon, config.min_lat, config.max_lon, config.max_lat],
        "variables": list(config.variables),
        "last_n_months": config.last_n_months,
        "neighborhood_sizes": list(config.neighborhood_sizes),
        "quantiles": list(config.quantiles),
        "raw_monthly_variables": list(config.raw_monthly_variables),
        "extreme_quantile_low": config.extreme_quantile_low,
        "extreme_quantile_high": config.extreme_quantile_high,
        "pseudo_absence_grid_stride": config.pseudo_absence_grid_stride,
        "pseudo_absence_max_candidates": config.pseudo_absence_max_candidates,
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _seasonal_mean(variable: xr.DataArray, season: str) -> xr.DataArray:
    import xarray as xr

    grouped = variable.groupby("time.season").mean(dim="time", skipna=True)
    if "season" in grouped.coords and season in set(grouped["season"].values.tolist()):
        return grouped.sel(season=season, drop=True)
    return xr.full_like(variable.isel(time=0, drop=True), np.nan, dtype=np.float64)


def _safe_ratio(numerator: xr.DataArray, denominator: xr.DataArray, epsilon: float = 1e-6) -> xr.DataArray:
    import xarray as xr

    return xr.where(np.abs(denominator) > epsilon, numerator / denominator, np.nan)


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for flag in np.asarray(values, dtype=bool):
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def compute_feature_dataset(dataset: xr.Dataset, config: FeatureBuildConfig) -> xr.Dataset:
    import xarray as xr

    feature_arrays: dict[str, xr.DataArray] = {}
    q_low = float(config.extreme_quantile_low)
    q_high = float(config.extreme_quantile_high)

    for index, variable_name in enumerate(config.variables, start=1):
        LOGGER.info(
            "Computing feature grids for variable %s (%s/%s)",
            variable_name,
            index,
            len(config.variables),
        )
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
        seasonal_stack = xr.concat(
            [feature_arrays[f"{variable_name}_{season.lower()}_mean"] for season in SEASONS],
            dim="seasonal_stat",
        )
        feature_arrays[f"{variable_name}_seasonal_amp"] = seasonal_stack.max(axis=0) - seasonal_stack.min(axis=0)

        median_surface = feature_arrays[f"{variable_name}_median"]
        for neighborhood_size in config.neighborhood_sizes:
            rolling = median_surface.rolling(
                lat=neighborhood_size,
                lon=neighborhood_size,
                center=True,
                min_periods=1,
            )
            local_mean = rolling.mean()
            local_std = rolling.std()
            local_min = rolling.min()
            local_max = rolling.max()
            suffix = f"w{neighborhood_size}"
            feature_arrays[f"{variable_name}_local_mean_{suffix}"] = local_mean
            feature_arrays[f"{variable_name}_local_std_{suffix}"] = local_std
            feature_arrays[f"{variable_name}_local_range_{suffix}"] = local_max - local_min
            feature_arrays[f"{variable_name}_local_contrast_{suffix}"] = median_surface - local_mean
        feature_arrays[f"{variable_name}_cv"] = _safe_ratio(
            feature_arrays[f"{variable_name}_std"],
            np.abs(feature_arrays[f"{variable_name}_mean"]) + 1e-6,
        )

    # Bioclim-style proxies derived from TerraClimate monthly series.
    diurnal_range = dataset["tmax"] - dataset["tmin"]
    feature_arrays["bio_mean_diurnal_range"] = diurnal_range.mean(dim="time", skipna=True)
    feature_arrays["bio_temp_annual_range"] = dataset["tmax"].max(dim="time", skipna=True) - dataset["tmin"].min(
        dim="time", skipna=True
    )
    feature_arrays["bio_isothermality_proxy"] = _safe_ratio(
        feature_arrays["bio_mean_diurnal_range"],
        feature_arrays["bio_temp_annual_range"],
    )

    ppt_rolling3_sum = dataset["ppt"].rolling(time=3, min_periods=3).sum()
    tmax_rolling3_mean = dataset["tmax"].rolling(time=3, min_periods=3).mean()
    tmin_rolling3_mean = dataset["tmin"].rolling(time=3, min_periods=3).mean()
    soil_rolling3_mean = dataset["soil"].rolling(time=3, min_periods=3).mean()
    water_balance = dataset["ppt"] - dataset["pet"]

    feature_arrays["bio_annual_ppt_total"] = dataset["ppt"].sum(dim="time", skipna=True)
    feature_arrays["bio_wettest_quarter_ppt"] = ppt_rolling3_sum.max(dim="time", skipna=True)
    feature_arrays["bio_driest_quarter_ppt"] = ppt_rolling3_sum.min(dim="time", skipna=True)
    feature_arrays["bio_warmest_quarter_tmax"] = tmax_rolling3_mean.max(dim="time", skipna=True)
    feature_arrays["bio_coldest_quarter_tmin"] = tmin_rolling3_mean.min(dim="time", skipna=True)
    feature_arrays["bio_wettest_quarter_soil"] = soil_rolling3_mean.max(dim="time", skipna=True)
    feature_arrays["bio_driest_quarter_soil"] = soil_rolling3_mean.min(dim="time", skipna=True)
    feature_arrays["bio_water_balance_total"] = water_balance.sum(dim="time", skipna=True)
    feature_arrays["bio_water_balance_last6"] = water_balance.isel(time=slice(-config.last_n_months, None)).sum(
        dim="time", skipna=True
    )

    ppt_low_threshold = dataset["ppt"].quantile(q_low, dim="time", skipna=True).drop_vars("quantile", errors="ignore")
    ppt_high_threshold = dataset["ppt"].quantile(q_high, dim="time", skipna=True).drop_vars("quantile", errors="ignore")
    soil_low_threshold = dataset["soil"].quantile(q_low, dim="time", skipna=True).drop_vars("quantile", errors="ignore")
    tmax_high_threshold = dataset["tmax"].quantile(q_high, dim="time", skipna=True).drop_vars("quantile", errors="ignore")
    vpd_high_threshold = dataset["vpd"].quantile(q_high, dim="time", skipna=True).drop_vars("quantile", errors="ignore")

    dry_month_mask = dataset["ppt"] <= ppt_low_threshold
    wet_month_mask = dataset["ppt"] >= ppt_high_threshold
    low_soil_mask = dataset["soil"] <= soil_low_threshold
    hot_month_mask = dataset["tmax"] >= tmax_high_threshold
    high_vpd_mask = dataset["vpd"] >= vpd_high_threshold
    water_deficit_mask = water_balance < 0
    pdsi_drought_mask = dataset["pdsi"] < 0
    hot_dry_mask = hot_month_mask & water_deficit_mask

    feature_arrays["bio_ppt_dry_month_count"] = dry_month_mask.sum(dim="time")
    feature_arrays["bio_ppt_wet_month_count"] = wet_month_mask.sum(dim="time")
    feature_arrays["bio_soil_low_month_count"] = low_soil_mask.sum(dim="time")
    feature_arrays["bio_tmax_hot_month_count"] = hot_month_mask.sum(dim="time")
    feature_arrays["bio_vpd_high_month_count"] = high_vpd_mask.sum(dim="time")
    feature_arrays["bio_water_deficit_month_count"] = water_deficit_mask.sum(dim="time")
    feature_arrays["bio_water_surplus_month_count"] = (water_balance > 0).sum(dim="time")
    feature_arrays["bio_pdsi_drought_month_count"] = pdsi_drought_mask.sum(dim="time")
    feature_arrays["bio_hot_dry_month_count"] = hot_dry_mask.sum(dim="time")

    feature_arrays["bio_ppt_longest_dry_spell"] = xr.apply_ufunc(
        _longest_true_run,
        dry_month_mask,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int16],
    )
    feature_arrays["bio_water_deficit_longest_spell"] = xr.apply_ufunc(
        _longest_true_run,
        water_deficit_mask,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int16],
    )
    feature_arrays["bio_pdsi_drought_longest_spell"] = xr.apply_ufunc(
        _longest_true_run,
        pdsi_drought_mask,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int16],
    )
    feature_arrays["bio_hot_dry_longest_spell"] = xr.apply_ufunc(
        _longest_true_run,
        hot_dry_mask,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int16],
    )

    for variable_name in ("ppt", "soil", "vpd", "def"):
        variable = dataset[variable_name]
        recent_mean = variable.isel(time=slice(-config.last_n_months, None)).mean(dim="time", skipna=True)
        overall_mean = variable.mean(dim="time", skipna=True)
        overall_std = variable.std(dim="time", skipna=True)
        feature_arrays[f"bio_{variable_name}_recent{config.last_n_months}_anomaly"] = _safe_ratio(
            recent_mean - overall_mean,
            overall_std + 1e-6,
        )

    water_balance_mean = water_balance.mean(dim="time", skipna=True)
    water_balance_std = water_balance.std(dim="time", skipna=True)
    water_balance_recent_mean = water_balance.isel(time=slice(-config.last_n_months, None)).mean(dim="time", skipna=True)
    feature_arrays[f"bio_water_balance_recent{config.last_n_months}_anomaly"] = _safe_ratio(
        water_balance_recent_mean - water_balance_mean,
        water_balance_std + 1e-6,
    )

    feature_dataset = xr.Dataset(feature_arrays)
    LOGGER.info("Completed feature grid generation | feature_count=%s", len(feature_dataset.data_vars))
    return feature_dataset


def sample_feature_dataset(feature_dataset: xr.Dataset, points_frame: pd.DataFrame) -> pd.DataFrame:
    import xarray as xr

    LOGGER.info("Sampling feature grids at %s point locations", points_frame.shape[0])
    lat_indexer = xr.DataArray(points_frame["Latitude"].to_numpy(), dims="point")
    lon_indexer = xr.DataArray(points_frame["Longitude"].to_numpy(), dims="point")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        sampled = feature_dataset.sel(lat=lat_indexer, lon=lon_indexer, method="nearest").load()
    sampled_columns = {
        name: np.asarray(data_array.values).reshape(-1)
        for name, data_array in sampled.data_vars.items()
    }
    sampled_frame = pd.DataFrame(sampled_columns, index=points_frame.index)
    return sampled_frame.apply(lambda column: column.map(coerce_native))


def sample_pseudo_absence_candidates(
    feature_dataset: xr.Dataset,
    config: FeatureBuildConfig,
) -> pd.DataFrame:
    stride = max(1, int(config.pseudo_absence_grid_stride))
    LOGGER.info(
        "Sampling pseudo-absence candidate grid | stride=%s | max_candidates=%s",
        stride,
        config.pseudo_absence_max_candidates,
    )
    candidate_variables = [
        name
        for name in feature_dataset.data_vars
        if name.startswith("bio_")
        or name.endswith("_median")
        or name.endswith("_seasonal_amp")
        or name.endswith("_cv")
        or name.endswith(f"_last{config.last_n_months}_mean")
    ]
    candidate_grid = feature_dataset[candidate_variables].isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        candidate_frame = candidate_grid.to_dataframe().reset_index()
    candidate_frame = candidate_frame.dropna(how="all", subset=candidate_variables)
    candidate_frame = candidate_frame.rename(columns={"lat": "Latitude", "lon": "Longitude"})
    if candidate_frame.shape[0] > config.pseudo_absence_max_candidates:
        candidate_frame = candidate_frame.sample(
            n=config.pseudo_absence_max_candidates,
            random_state=42,
        ).reset_index(drop=True)
    LOGGER.info("Prepared pseudo-absence candidates | rows=%s", candidate_frame.shape[0])
    return candidate_frame.apply(lambda column: column.map(coerce_native))


def get_or_build_pseudo_absence_candidates(
    feature_dataset: xr.Dataset,
    config: FeatureBuildConfig,
) -> pd.DataFrame:
    cache_dir = config.pseudo_absence_cache_dir
    if cache_dir is None:
        return sample_pseudo_absence_candidates(feature_dataset, config)

    cache_dir = ensure_directory(cache_dir)
    cache_key = _pseudo_absence_cache_key(config)
    cache_path = cache_dir / f"pseudo_absence_candidates_{cache_key}.parquet"
    if cache_path.exists():
        LOGGER.info("Using cached pseudo-absence candidates from %s", cache_path)
        return read_dataframe_parquet(cache_path)

    LOGGER.info("Pseudo-absence cache miss | cache_path=%s", cache_path)
    candidate_frame = sample_pseudo_absence_candidates(feature_dataset, config)
    write_dataframe_parquet(cache_path, candidate_frame)
    LOGGER.info("Cached pseudo-absence candidates at %s", cache_path)
    return candidate_frame


def sample_monthly_sequence_features(
    dataset: xr.Dataset,
    points_frame: pd.DataFrame,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    import xarray as xr

    selected_variables = [variable for variable in variables if variable in dataset.data_vars]
    if not selected_variables:
        return pd.DataFrame(index=points_frame.index)

    LOGGER.info("Sampling raw monthly sequence features | variables=%s | point_rows=%s", selected_variables, points_frame.shape[0])
    lat_indexer = xr.DataArray(points_frame["Latitude"].to_numpy(), dims="point")
    lon_indexer = xr.DataArray(points_frame["Longitude"].to_numpy(), dims="point")
    sampled = dataset[selected_variables].sel(lat=lat_indexer, lon=lon_indexer, method="nearest").load()
    time_labels = pd.to_datetime(sampled["time"].values).strftime("%Y_%m").tolist()

    columns: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        for variable in selected_variables:
            values = np.asarray(sampled[variable].transpose("point", "time").values)
            for time_index, time_label in enumerate(time_labels):
                columns[f"{variable}_{time_label}"] = values[:, time_index]
            columns[f"{variable}_delta_start_end"] = values[:, -1] - values[:, 0]
            columns[f"{variable}_recent3_mean"] = np.nanmean(values[:, -3:], axis=1)
            columns[f"{variable}_recent6_std"] = np.nanstd(values[:, -6:], axis=1)

    monthly_frame = pd.DataFrame(columns, index=points_frame.index)
    return monthly_frame.apply(lambda column: column.map(coerce_native))


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
    LOGGER.info("Building feature artifacts into %s", output_dir)
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
    LOGGER.info("Prepared unique coordinate set | rows=%s", coordinate_frame.shape[0])

    dataset = open_terraclimate_dataset(config)
    try:
        with log_step(LOGGER, "Compute TerraClimate feature dataset"):
            feature_dataset = compute_feature_dataset(dataset, config)
        try:
            with log_step(LOGGER, "Sample features at coordinates", coordinate_rows=coordinate_frame.shape[0]):
                sampled_coordinates = pd.concat(
                    [coordinate_frame, sample_feature_dataset(feature_dataset, coordinate_frame)],
                    axis=1,
                )
            with log_step(LOGGER, "Sample raw monthly sequence features", variables=list(config.raw_monthly_variables)):
                sampled_coordinates = pd.concat(
                    [sampled_coordinates, sample_monthly_sequence_features(dataset, coordinate_frame, config.raw_monthly_variables)],
                    axis=1,
                )
            with log_step(LOGGER, "Sample pseudo-absence candidates"):
                pseudo_absence_candidates = get_or_build_pseudo_absence_candidates(feature_dataset, config)
        finally:
            feature_dataset.close()
    finally:
        dataset.close()

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
    LOGGER.info(
        "Prepared feature tables | train_rows=%s | test_rows=%s | feature_count=%s | dropped_null_columns=%s",
        train_features.shape[0],
        test_features.shape[0],
        len(feature_columns),
        len(dropped_columns),
    )

    train_path = output_dir / "train_features.parquet"
    test_path = output_dir / "test_features.parquet"
    pseudo_absence_candidates_path = output_dir / "pseudo_absence_candidates.parquet"
    manifest_path = output_dir / "feature_manifest.json"

    with log_step(LOGGER, "Write feature artifacts", output_dir=output_dir):
        write_dataframe_parquet(train_path, train_features)
        write_dataframe_parquet(test_path, test_features)
        write_dataframe_parquet(pseudo_absence_candidates_path, pseudo_absence_candidates)
        write_json(
            manifest_path,
            {
                "train_rows": int(train_features.shape[0]),
                "test_rows": int(test_features.shape[0]),
                "pseudo_absence_candidate_rows": int(pseudo_absence_candidates.shape[0]),
                "coordinate_rows": int(coordinate_frame.shape[0]),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
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
                "seasonal_amp",
                "cv",
                "local_mean_w3",
                "local_std_w3",
                "local_range_w3",
                "local_contrast_w3",
                "local_mean_w5",
                "local_std_w5",
                "local_range_w5",
                "local_contrast_w5",
                "local_mean_w7",
                "local_std_w7",
                "local_range_w7",
                "local_contrast_w7",
                "bio_mean_diurnal_range",
                "bio_temp_annual_range",
                "bio_isothermality_proxy",
                "bio_annual_ppt_total",
                "bio_wettest_quarter_ppt",
                "bio_driest_quarter_ppt",
                "bio_warmest_quarter_tmax",
                "bio_coldest_quarter_tmin",
                "bio_wettest_quarter_soil",
                "bio_driest_quarter_soil",
                "bio_water_balance_total",
                "bio_water_balance_last6",
                "extreme_month_counts",
                "longest_event_spells",
                f"recent{config.last_n_months}_anomaly",
                "compound_hot_dry_counts",
            ],
            "dropped_all_null_columns": dropped_columns,
            "spatial_group_size_degrees": config.spatial_group_size_degrees,
            },
        )
    LOGGER.info(
        "Feature artifacts written | train=%s | test=%s | pseudo_absence_candidates=%s | manifest=%s",
        train_path,
        test_path,
        pseudo_absence_candidates_path,
        manifest_path,
    )

    return FeatureArtifacts(
        train_features_path=train_path,
        test_features_path=test_path,
        pseudo_absence_candidates_path=pseudo_absence_candidates_path,
        manifest_path=manifest_path,
        feature_columns=feature_columns,
        dropped_feature_columns=dropped_columns,
        coordinate_rows=int(coordinate_frame.shape[0]),
    )
