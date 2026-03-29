# Kaggle + GitHub Workflow

## Recommended setup

Use one stable Kaggle launcher notebook that does four things every run:

1. Pull the latest code from GitHub into `/kaggle/working`.
2. Install `requirements-kaggle.txt` from the cloned repo.
3. Dispatch to a single stage runner inside the repo.
4. Write artifacts to `/kaggle/working/artifacts`.

This removes the need to re-upload notebooks or Python files to Kaggle whenever the code changes.

## One-time GitHub setup

1. Create the GitHub repo.
2. Initialize git locally in this workspace.
3. Add the remote and push.
4. If the repo will be private, create a read-only GitHub token and save it in Kaggle Secrets as `GITHUB_TOKEN`.

## Day-to-day workflow

1. Change code locally.
2. Commit and push to GitHub.
3. Open the Kaggle launcher notebook.
4. Set `GITHUB_REF` to `main` for fast iteration, or to a commit SHA when you need reproducibility.
5. Run the launcher notebook with the right `STAGE` and accelerator:
   - `feature`: CPU, internet enabled
   - `baseline`: CPU, with `FEATURE_DIR` pointing to a mounted features dataset if this is a fresh session
   - `tpu`: TPU, with `FEATURE_DIR` pointing to a mounted features dataset if this is a fresh session
   - `finalize`: CPU, with `FEATURE_DIR` and `TPU_DIR` pointing to mounted artifact datasets if this is a fresh session

## Why this is the lowest-friction path

- The launcher notebook is stable and rarely changes.
- The real code always lives in GitHub, not inside a Kaggle notebook.
- Every Kaggle run can pull the latest branch tip or a pinned commit.
- You only maintain one notebook on Kaggle instead of repeatedly syncing code files by hand.

## Artifact handoff

- `feature` writes parquet feature tables to `artifacts/features/`
- `baseline` writes CPU summaries and submissions to `artifacts/baselines/`
- `tpu` writes saved fold models and preprocessors to `artifacts/tpu/`
- `finalize` compares CPU and TPU outputs and writes `artifacts/baselines/final_submission.csv`

## Practical note

Kaggle sessions are ephemeral. If you want artifacts from one run available in another notebook, save the notebook version or publish the output directory as a Kaggle dataset version.
