# EY Frog Challenge Pipeline

This repo contains a Kaggle-ready implementation for the EY Biodiversity frog challenge. The challenge CSV files are already committed here, and Kaggle is driven through a dedicated bootstrap script so the notebook stays very thin.

## What is here

- `kaggle_bootstrap.py`: Kaggle bootstrap runner that clones the repo, installs requirements, and dispatches the selected stage
- `kaggle_run.py`: unified stage dispatcher used after bootstrap
- `feature_build.py`: TerraClimate feature extraction entrypoint
- `baseline_models.py`: CPU baseline training plus final submission selection
- `tpu_train.py`: TPU neural training entrypoint
- `frog_challenge/`: shared feature, modeling, TPU, and bootstrap code
- `notebooks/run_on_kaggle.ipynb`: the single Kaggle notebook you run

## Easiest Kaggle start

Use `notebooks/run_on_kaggle.ipynb` directly on Kaggle with internet enabled. It is a two-code-cell flow:

- Cell 1: config
- Cell 2: download and run `kaggle_bootstrap.py`

The default stage is `tpu`, which does the full run:

- builds features if needed from repo-local CSVs
- trains TPU neural models
- runs CPU baselines in the same session
- writes `artifacts/baselines/final_submission.csv`

If you want a CPU-only run, change `STAGE` to `baseline` in the first cell.

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
- If the last log line says `Installing requirements ...`, that is progress, not a failure. The bootstrap script is still in the dependency installation phase until a real traceback appears.
