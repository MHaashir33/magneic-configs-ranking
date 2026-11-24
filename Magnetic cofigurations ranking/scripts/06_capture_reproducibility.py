from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASET_REL = Path("data/database_snapshot/magnetic-orderings-database-snapshot.part1.json.gz")
DATASET_URL = (
    "https://raw.githubusercontent.com/ncfrey/magnetic-topological-materials/"
    "master/data/database_snapshot/magnetic-orderings-database-snapshot.part1.json.gz"
)
ALIGNN_REPO_URL = "https://github.com/atomgptlab/alignn.git"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _alignn_provenance(alignn_repo: Path | None, supplied_commit: str | None) -> tuple[str, str]:
    if supplied_commit:
        commit = supplied_commit.strip()
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
            raise ValueError("--alignn-commit must be a 7-40 character hexadecimal Git SHA.")
        return ALIGNN_REPO_URL, commit

    if alignn_repo is not None:
        commit = _git_commit(alignn_repo)
        if not commit:
            raise RuntimeError(f"Could not obtain a Git commit from {alignn_repo}.")
        return ALIGNN_REPO_URL, commit

    # pip records the resolved VCS revision in the installed distribution's PEP 610 file.
    try:
        dist = importlib.metadata.distribution("alignn")
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            vcs = direct_url.get("vcs_info", {})
            commit = vcs.get("commit_id")
            url = direct_url.get("url", ALIGNN_REPO_URL)
            if commit:
                return url, commit
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass

    # Last fallback: inspect pip freeze for a VCS reference with a resolved SHA.
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    for line in freeze.splitlines():
        if "alignn" in line.lower() and "git+" in line and "@" in line:
            match = re.search(r"@([0-9a-fA-F]{7,40})(?:#|$)", line)
            if match:
                return ALIGNN_REPO_URL, match.group(1)

    raise RuntimeError(
        "The installed ALIGNN package did not expose its resolved Git commit. "
        "Pass --alignn-repo PATH to a local ALIGNN clone, or --alignn-commit SHA."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a paper-run environment lock and dataset SHA-256 manifest."
    )
    parser.add_argument(
        "--alignn-repo", type=Path, default=None,
        help="Optional local clone of atomgptlab/alignn used for the run.",
    )
    parser.add_argument(
        "--alignn-commit", type=str, default=None,
        help="Explicit ALIGNN commit SHA, if the installed package has no VCS metadata.",
    )
    args = parser.parse_args()

    dataset_path = ROOT / DATASET_REL
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {dataset_path}. Run scripts/00_download_part1.py first."
        )

    alignn_url, alignn_commit = _alignn_provenance(
        args.alignn_repo.resolve() if args.alignn_repo else None,
        args.alignn_commit,
    )
    dataset_hash = sha256_file(dataset_path)

    pip_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    ).strip()
    project_commit = _git_commit(ROOT) or "not-in-a-git-checkout"
    try:
        git_status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        project_worktree_clean = not bool(git_status.strip())
    except (OSError, subprocess.CalledProcessError):
        project_worktree_clean = None

    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "cpu-only"
        cudnn_version = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None
        )
        gpu_names = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        torch_version = "not-importable"
        cuda_version = "unknown"
        cudnn_version = None
        gpu_names = []

    metadata_lines = [
        "# Reproducibility lock generated by scripts/06_capture_reproducibility.py",
        f"# Captured_UTC: {datetime.now(timezone.utc).isoformat()}",
        f"# Project_Git_commit: {project_commit}",
        f"# Project_worktree_clean_before_capture: {project_worktree_clean}",
        f"# Python: {sys.version.replace(chr(10), ' ')}",
        f"# Platform: {platform.platform()}",
        f"# PyTorch: {torch_version}",
        f"# CUDA_runtime_used_by_PyTorch: {cuda_version}",
        f"# cuDNN: {cudnn_version}",
        f"# GPUs: {', '.join(gpu_names) if gpu_names else 'none-detected'}",
        f"# ALIGNN_repository: {alignn_url}",
        f"# ALIGNN_commit: {alignn_commit}",
        f"# Dataset_file: {DATASET_REL.as_posix()}",
        f"# Dataset_SHA256: {dataset_hash}",
        "# The package pins below are the output of: python -m pip freeze --all",
        "",
    ]
    lock_path = ROOT / "environment-lock.txt"
    lock_path.write_text("\n".join(metadata_lines) + pip_freeze + "\n", encoding="utf-8")

    manifest = {
        "source_repository": "https://github.com/ncfrey/magnetic-topological-materials",
        "source_url": DATASET_URL,
        "dataset_file": DATASET_REL.as_posix(),
        "sha256": dataset_hash,
        "record_count": None,
        "alignn_repository": alignn_url,
        "alignn_commit": alignn_commit,
        "project_git_commit": project_commit,
        "project_worktree_clean_before_capture": project_worktree_clean,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pytorch": torch_version,
        "pytorch_cuda_runtime": cuda_version,
        "cudnn": cudnn_version,
        "gpu_names": gpu_names,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Validate the gzip JSON and record its entry count without modifying the source file.
    import gzip
    with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("The part1 snapshot is not a JSON list.")
    manifest["record_count"] = len(payload)

    manifest_path = ROOT / "data" / "processed" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Conventional checksum file, stored beside the unchanged source archive.
    checksum_path = ROOT / "data" / "database_snapshot" / "SHA256SUMS.txt"
    checksum_path.write_text(
        f"{dataset_hash}  {DATASET_REL.name}\n", encoding="utf-8"
    )

    # Pin future installs to the exact ALIGNN revision captured in this run.
    req_path = ROOT / "requirements-alignn.txt"
    if req_path.is_file():
        lines = req_path.read_text(encoding="utf-8").splitlines()
        pinned = f"git+{alignn_url}@{alignn_commit}"
        replaced = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("git+https://github.com/atomgptlab/alignn"):
                new_lines.append(pinned)
                replaced = True
            else:
                new_lines.append(line)
        if replaced:
            req_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")

    print("ALIGNN commit:", alignn_commit)
    print("Dataset SHA256:", dataset_hash)
    print("Dataset records:", len(payload))
    print("Wrote:", lock_path)
    print("Wrote:", manifest_path)
    print("Wrote:", checksum_path)
    print("Pinned:", req_path)


if __name__ == "__main__":
    main()
