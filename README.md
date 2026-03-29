# EY Frog Challenge Pipeline

This repo contains a Kaggle-ready implementation for the EY Biodiversity frog challenge. The challenge CSV files are already committed here, and Kaggle is now driven through a dedicated bootstrap script so the notebooks stay very thin.

## What is here

- `kaggle_bootstrap.py`: Kaggle bootstrap runner that clones the repo, installs requirements, and dispatches the selected stage
- `kaggle_run.py`: unified stage dispatcher used after bootstrap
- `feature_build.py`: TerraClimate feature extraction entrypoint
- `baseline_models.py`: CPU baseline training plus final submission selection
- `tpu_train.py`: TPU neural training entrypoint
- `frog_challenge/`: shared feature, modeling, TPU, and bootstrap code
- `notebooks/`: self-bootstrapping Kaggle notebooks

## Easiest Kaggle start

Use one of these notebooks directly on Kaggle with internet enabled. Each notebook is now a two-code-cell flow: one config cell and one bootstrap-run cell.

- `notebooks/02_baseline_models_kaggle.ipynb`
  - CPU runtime
  - Downloads and runs `kaggle_bootstrap.py`
  - Builds features if needed from repo-local CSVs
  - Trains CPU baselines
  - Writes `artifacts/baselines/final_submission.csv`
- `notebooks/03_tpu_train_kaggle.ipynb`
  - TPU runtime
  - Downloads and runs `kaggle_bootstrap.py`
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
- Pipeline logs are timestamped and include stage, fold, artifact, and model-selection reporting so Kaggle output is easy to follow.
