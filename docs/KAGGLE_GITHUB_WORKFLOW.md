# Kaggle + GitHub Workflow

## Current lowest-friction path

The repo now contains the challenge CSV files and a dedicated Kaggle bootstrap script. The normal Kaggle flow is:

1. Open `notebooks/run_on_kaggle.ipynb` in Kaggle.
2. Run all cells.
3. Let the notebook download `kaggle_bootstrap.py`, which then clones the repo, installs dependencies, uses the repo-local CSV files, and writes artifacts to `/kaggle/working/artifacts`.

No separate Kaggle dataset is required for the source training and test data.

## Notebook behavior

`notebooks/run_on_kaggle.ipynb` is now the only Kaggle notebook in the repo.

- Default `STAGE` is `tpu`
- That full run builds features, trains TPU models, runs CPU baselines, compares both, and writes `artifacts/baselines/final_submission.csv`
- If you want a lighter run, change `STAGE` to `baseline`

## How bootstrap works

The notebook:

1. Downloads `kaggle_bootstrap.py` from GitHub
2. Runs the bootstrap script with notebook-provided config
3. Lets the bootstrap script clone `AstralJugs69/EY-bio` into `/kaggle/working/ey-frog-repo`
4. Installs `requirements-kaggle.txt`
5. Uses `Training_Data.csv` and `Test.csv` from the cloned repo
6. Writes outputs to `/kaggle/working/artifacts`

## Artifact layout

- `artifacts/features/`
  - feature parquet files and manifest
- `artifacts/baselines/`
  - baseline summary
  - final selection summary
  - `final_submission.csv`
- `artifacts/tpu/`
  - TPU model summaries
  - saved fold models
  - saved preprocessors

## Logging

The codebase now emits timestamped progress logs for:

- bootstrap start and repo checkout
- dependency installation
- feature reuse vs rebuild
- TerraClimate loading and feature generation
- fold-by-fold CPU model training
- fold-by-fold TPU model training
- threshold selection and final submission choice

## Practical note

Kaggle sessions are ephemeral. If you want artifacts from one run available in another session, save the notebook output or publish `/kaggle/working/artifacts` as a Kaggle dataset version.
