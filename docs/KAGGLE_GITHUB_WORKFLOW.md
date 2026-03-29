# Kaggle + GitHub Workflow

## Current lowest-friction path

The repo now contains the challenge CSV files and self-bootstrapping Kaggle notebooks. The normal Kaggle flow is:

1. Open a notebook from this repo in Kaggle.
2. Run all cells.
3. Let the notebook clone the repo, install dependencies, use the repo-local CSV files, and write artifacts to `/kaggle/working/artifacts`.

No separate Kaggle dataset is required for the source training and test data.

## Recommended notebooks

- `notebooks/02_baseline_models_kaggle.ipynb`
  - Run on CPU
  - Produces a valid baseline submission in one run
- `notebooks/03_tpu_train_kaggle.ipynb`
  - Run on TPU
  - Builds features, trains TPU models, runs CPU baselines, compares both, and writes the final submission in one run
- `notebooks/00_kaggle_github_launcher.ipynb`
  - Generic dispatcher if you want to select `feature`, `baseline`, `tpu`, or `finalize`

## How bootstrap works

Each notebook:

1. Clones `AstralJugs69/EY-bio` into `/kaggle/working/ey-frog-repo`
2. Installs `requirements-kaggle.txt`
3. Uses `Training_Data.csv` and `Test.csv` from the cloned repo
4. Writes outputs to `/kaggle/working/artifacts`

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

## Practical note

Kaggle sessions are ephemeral. If you want artifacts from one run available in another session, save the notebook output or publish `/kaggle/working/artifacts` as a Kaggle dataset version.
