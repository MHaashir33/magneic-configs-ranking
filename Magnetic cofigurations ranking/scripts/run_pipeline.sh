#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. The ALIGNN/GPU environment must already be installed.
python scripts/00_download_part1.py
python scripts/06_capture_reproducibility.py
python scripts/01_prepare_data.py
python scripts/02_build_graphs.py
python scripts/03_train_models.py --epochs 50
python scripts/04_screen_candidates.py
python scripts/05_run_benchmarks.py

echo "Pipeline completed. See results/ and models/."
