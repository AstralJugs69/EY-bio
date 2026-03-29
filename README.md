# EY Frog Challenge Pipeline

This repo contains a Kaggle-ready implementation for the EY Biodiversity frog challenge. The challenge CSV files are already committed here, so the Kaggle notebooks can clone the repo and bootstrap themselves without separate data uploads.

## What is here

- `kaggle_run.py`: unified stage dispatcher used by the launcher notebook
- `feature_build.py`: TerraClimate feature extraction entrypoint
- `baseline_models.py`: CPU baseline training plus final submission selection
- `tpu_train.py`: TPU neural training entrypoint
- `frog_challenge/`: shared feature, modeling, TPU, and bootstrap code
- `notebooks/`: self-bootstrapping Kaggle notebooks

## Easiest Kaggle start

Use one of these notebooks directly on Kaggle with internet enabled:

- `notebooks/02_baseline_models_kaggle.ipynb`
  - CPU runtime
  - Clones the repo
  - Builds features if needed from repo-local CSVs
  - Trains CPU baselines
  - Writes `artifacts/baselines/final_submission.csv`
- `notebooks/03_tpu_train_kaggle.ipynb`
  - TPU runtime
  - Clones the repo
  - Builds features if needed from repo-local CSVs
  - Trains TPU neural models
  - Runs CPU baselines in the same session
  - Writes `artifacts/baselines/final_submission.csv`

If you want a generic dispatcher notebook instead, use `notebooks/00_kaggle_github_launcher.ipynb`.

## Artifact layout

- `artifacts/features/`
  - `train_features.parquet`
  - `test_features.parquet`
  - `feature_manifest.json`
- `artifacts/baselines/`
  - `baseline_summary.json`
  - `final_selection.json`
  - `final_submission.csv`
  - `models/*.parquet`
- `artifacts/tpu/`
  - `tpu_summary.json`
  - `models/<architecture>/fold_*.keras`
  - `preprocessors/<architecture>/fold_*.joblib`
  - `predictions/*.parquet`

## Notes

- Latitude and longitude are used only for TerraClimate lookup and spatial grouping.
- The modeling code excludes `ID`, `Latitude`, `Longitude`, and `spatial_group` from the learned feature matrix.
- CPU baselines remain the control path. TPU is only used to accelerate neural training experiments.
