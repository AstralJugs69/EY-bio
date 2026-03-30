from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("kaggle_bootstrap")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the EY frog challenge pipeline inside Kaggle.")
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--github-ref", default="main")
    parser.add_argument("--stage", choices=("feature", "baseline", "gpu", "tpu", "finalize", "submissions"), required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path("/kaggle/working/ey-frog-repo"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=Path("/kaggle/working/artifacts"))
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--feature-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--tpu-dir", type=Path, default=None)
    parser.add_argument("--tpu-artifact-dir", type=Path, default=None)
    parser.add_argument("--pseudo-absence-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def get_github_token() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if token:
            return token
    except Exception:
        pass
    return os.getenv("GITHUB_TOKEN")


def github_remote_url(repo: str, token: str | None) -> str:
    if token:
        return f"https://{token}:x-oauth-basic@github.com/{repo}.git"
    return f"https://github.com/{repo}.git"


def clone_repo(repo_dir: Path, github_repo: str, github_ref: str, token: str | None) -> str:
    if repo_dir.exists():
        LOGGER.info("Removing existing repo directory %s", repo_dir)
        shutil.rmtree(repo_dir)

    repo_dir.mkdir(parents=True, exist_ok=True)
    remote_url = github_remote_url(github_repo, token)

    LOGGER.info("Cloning repo | repo=%s | ref=%s | repo_dir=%s", github_repo, github_ref, repo_dir)
    subprocess.run(["git", "init", str(repo_dir)], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "remote", "add", "origin", remote_url], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", github_ref], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "-B", "kaggle-run", "FETCH_HEAD"], check=True)
    commit_sha = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    LOGGER.info("Repo ready | commit=%s", commit_sha)
    return commit_sha


def install_requirements(repo_dir: Path) -> None:
    requirements_path = repo_dir / "requirements-kaggle.txt"
    LOGGER.info("Installing requirements from %s", requirements_path)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements_path)], check=True)
    LOGGER.info("Requirements installation complete")


def run_stage(args: argparse.Namespace) -> None:
    repo_dir = args.repo_dir
    data_root = args.data_root or repo_dir
    command = [
        sys.executable,
        str(repo_dir / ("generate_submissions.py" if args.stage == "submissions" else "kaggle_run.py")),
    ]
    if args.stage == "submissions":
        command.extend(
            [
                "--feature-dir",
                str(args.feature_dir),
                "--baseline-dir",
                str(args.baseline_dir),
            ]
        )
        if args.tpu_artifact_dir is not None:
            command.extend(["--tpu-artifact-dir", str(args.tpu_artifact_dir)])
        elif args.tpu_dir is not None:
            command.extend(["--tpu-artifact-dir", str(args.tpu_dir)])
    else:
        command.extend(
            [
                "--stage",
                args.stage,
                "--data-root",
                str(data_root),
                "--feature-dir",
                str(args.feature_dir),
                "--baseline-dir",
                str(args.baseline_dir),
                "--tpu-dir",
                str(args.tpu_dir),
                "--pseudo-absence-cache-dir",
                str(args.pseudo_absence_cache_dir),
                "--batch-size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
            ]
        )

    LOGGER.info("Dispatching stage runner | command=%s", command)
    subprocess.run(command, check=True)


def report_artifacts(args: argparse.Namespace) -> None:
    candidate_paths = [
        args.feature_dir / "feature_manifest.json",
        args.baseline_dir / "baseline_summary.json",
        args.baseline_dir / "final_selection.json",
        args.baseline_dir / "final_submission.csv",
        args.tpu_dir / "tpu_summary.json",
    ]
    for path in candidate_paths:
        if path.exists():
            LOGGER.info("Artifact available | path=%s", path)


def main() -> int:
    configure_logging()
    args = parse_args()
    artifact_root = args.artifact_root
    run_label = args.run_label or f"{args.stage}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_root = artifact_root / "runs" / run_label
    args.feature_dir = args.feature_dir or (run_root / "features")
    args.baseline_dir = args.baseline_dir or (run_root / "baselines")
    args.tpu_dir = args.tpu_dir or (run_root / "gpu")
    args.pseudo_absence_cache_dir = args.pseudo_absence_cache_dir or (artifact_root / "_cache" / "pseudo_absence")
    LOGGER.info(
        "Bootstrap starting | repo=%s | ref=%s | stage=%s | repo_dir=%s | run_label=%s | run_root=%s",
        args.github_repo,
        args.github_ref,
        args.stage,
        args.repo_dir,
        run_label,
        run_root,
    )
    token = get_github_token()
    if token:
        LOGGER.info("Using authenticated GitHub access")
    else:
        LOGGER.info("Using unauthenticated GitHub access")

    clone_repo(args.repo_dir, args.github_repo, args.github_ref, token)
    install_requirements(args.repo_dir)
    run_stage(args)
    report_artifacts(args)
    LOGGER.info("Bootstrap complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
