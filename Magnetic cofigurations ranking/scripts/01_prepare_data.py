from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from altermagnet_screening.data_pipeline import prepare_project_data  # noqa: E402

if __name__ == "__main__":
    prepare_project_data(ROOT, strict_counts=True)
