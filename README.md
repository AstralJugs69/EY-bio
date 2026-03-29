# EY Frog Challenge Pipeline

This workspace now includes a Kaggle-ready implementation for the EY Biodiversity frog challenge:

- `kaggle_run.py`: unified stage dispatcher for Kaggle launcher notebooks.
- `feature_build.py`: CPU notebook/script entrypoint for TerraClimate feature extraction.
- `baseline_models.py`: CPU notebook/script entrypoint for spatial cross-validation baselines and optional CPU-side final selection against TPU artifacts.
- `tpu_train.py`: TPU notebook/script entrypoint for neural tabular training only.
- `frog_challenge/`: shared reusable code for feature engineering, baseline modeling, and TPU training.
- `notebooks/`: one GitHub launcher notebook plus three thin stage-specific notebooks.

## Recommended Kaggle workflow

1. Push this repo to GitHub.
2. Copy [notebooks/00_kaggle_github_launcher.ipynb](<C:/dev/data/dataset/ZIndi/EY bio/notebooks/00_kaggle_github_launcher.ipynb>) into Kaggle once.
3. Set `GITHUB_REPO`, `GITHUB_REF`, and `STAGE` in the launcher notebook.
4. Run the launcher notebook. It clones the latest repo ref into `/kaggle/working`, installs `requirements-kaggle.txt`, and executes `kaggle_run.py`.
5. For private repos, store a read-only `GITHUB_TOKEN` in Kaggle Secrets.
6. Use `main` while iterating, then pin `GITHUB_REF` to a commit SHA when you want a reproducible run.

The detailed workflow is documented in [docs/KAGGLE_GITHUB_WORKFLOW.md](<C:/dev/data/dataset/ZIndi/EY bio/docs/KAGGLE_GITHUB_WORKFLOW.md>).

## Manual stage notebooks

If you still want separate notebooks per stage, the existing stage notebooks remain available:

- `notebooks/01_feature_build_kaggle.ipynb`
- `notebooks/02_baseline_models_kaggle.ipynb`
- `notebooks/03_tpu_train_kaggle.ipynb`

## Artifact layout

- `artifacts/features/`
  - `train_features.parquet`
  - `test_features.parquet`
  - `feature_manifest.json`
- `artifacts/baselines/`
  - `baseline_summary.json`
  - `submission.csv`
  - `models/*.parquet`
- `artifacts/tpu/`
  - `tpu_summary.json`
  - `models/<architecture>/fold_*.keras`
  - `preprocessors/<architecture>/fold_*.joblib`
  - `predictions/*.parquet`

## Notes

- Latitude and longitude are used only for TerraClimate lookup and spatial grouping.
- The feature tables keep coordinate columns for traceability, but the modeling code excludes them from the model matrix.
- CPU baselines remain the control path. TPU models are only for faster neural training experiments.
