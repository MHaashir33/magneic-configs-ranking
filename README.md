# Reproducing ALIGNN magnetic-ordering benchmarks and symmetry screening

## Project overview

This project evaluates whether a spin-aware ALIGNN model can reduce the number of density-functional-theory (DFT) magnetic-ordering calculations needed to identify the lowest-energy magnetic configuration of a parent crystal.

The main benchmark compares two models:

- **Structure-only ALIGNN baseline:** uses the chemical/structural atom features.
- **Spin-aware ALIGNN:** uses the same structural features augmented with the initial magnetic moment on each site.

For each test parent structure, the models predict the energy-above-ground-state target (`delta_E`) for its magnetic configurations. The configurations are ranked by predicted energy, and the Top-1, Top-2, Top-3, Top-5 and Top-10 selections are evaluated against the DFT ground state. The analysis reports ground-state recovery, estimated DFT savings, energy regret, ground-state energy gaps, pairwise ranking and related diagnostics.

The application stage then uses the spin-aware Top-5 workflow and applies two symmetry filters:

1. **Preliminary magnetic-symmetry screen:** compensated magnetic moment and broken PT symmetry.
2. **Strict spin-space-symmetry screen:** retain only preliminary candidates that pass the `spinspg` spin-space-symmetry test.

The benchmark script reports the number of unique Task IDs at both stages, the number and percentage removed by the strict filter, a Task-ID audit table, and the final screened-candidate table.

**Scientific scope:** the DFT-saving values are retrospective estimates based on known DFT labels; they do not mean those calculations were actually avoided in the original dataset. The symmetry-screened materials are computational candidates, not experimentally confirmed or DFT-validated altermagnets. Further magnetic-ground-state and spin-resolved electronic-structure calculations are required for validation.

## Data and model sources

- **Dataset repository:** <https://github.com/ncfrey/magnetic-topological-materials>
- **Dataset file used:** `magnetic-orderings-database-snapshot.part1.json.gz`
- **ALIGNN repository:** <https://github.com/atomgptlab/alignn>
- **ALIGNN model class:** `ALIGNNAtomWisePure` from `alignn.models.alignn_atomwise_pure`

Only dataset **part 1** is downloaded and used. Part 2 is not concatenated with it or used in training. The source gzip is downloaded unchanged; the committed SHA-256 checksum is used to verify it.

## Repository contents

The release should contain the following project files (generated outputs appear after running the workflow):

```text
repository-root/
├── README.md
├── BENCHMARKING.md
├── NOTEBOOK_AUDIT.md
├── CITATION.md
├── DEPENDENCY_NOTES.md
├── environment-lock.txt
├── requirements.txt
├── requirements-alignn.txt
├── pyproject.toml
├── notebooks/
│   └── reproduce_results.ipynb
├── data/
│   ├── database_snapshot/
│   │   └── SHA256SUMS.txt
│   └── processed/
│       └── dataset_manifest.json
├── models/                         # generated model checkpoints; may be hosted separately
├── scripts/
│   ├── 00_download_part1.py
│   ├── 01_prepare_data.py
│   ├── 02_build_graphs.py
│   ├── 03_train_models.py
│   ├── 04_screen_candidates.py
│   ├── 05_run_benchmarks.py
│   ├── 06_capture_reproducibility.py
│   ├── 07_generate_figures.py
│   └── run_pipeline.sh
├── src/altermagnet_screening/
│   ├── data_pipeline.py
│   ├── model_pipeline.py
│   ├── benchmarking.py
│   ├── screening.py
│   └── symmetry.py
├── tests/
└── results/                        # selected tables, metrics and figures
```

The exact optional documentation/notebook files may vary by release, but all scripts, source modules, dependencies and provenance files required by the commands below must be present.

## Reproduce the published release

Run commands from the repository root. The published Git tag and committed provenance files should be used rather than the moving `main` branch.

### 1. Clone the repository and select the release

Replace the URL and tag with the values published by the project maintainers:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
git checkout v1.0.0
```

`git checkout v1.0.0` selects the exact commit associated with that release tag. Git may report a *detached HEAD* state; this is expected for reproducing a fixed release.

### 2. Create a Python environment

Python 3.10 is recommended.

**Linux/macOS:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

**Windows PowerShell:**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install the recorded software environment

For the same platform and backend as the published run, install the committed package lock and pinned ALIGNN implementation:

```bash
python -m pip install -r environment-lock.txt
python -m pip install -r requirements-alignn.txt
```

The `environment-lock.txt` file records the Python package versions captured for the release. `requirements-alignn.txt` pins the ALIGNN source to its recorded Git commit. The package lock does not itself install the operating-system GPU driver or CUDA/ROCm system runtime; those must be compatible with the installed PyTorch build.

For a GPU, install/use the PyTorch build matching the machine before running the workflow:

- **NVIDIA:** use a compatible PyTorch CUDA build and NVIDIA driver.
- **AMD:** use a supported PyTorch ROCm build on a supported Linux distribution and GPU. ROCm does not support every AMD GPU.
- **CPU:** supported, but training will generally be slower.

PyTorch installation selector: <https://pytorch.org/get-started/locally/>. AMD ROCm installation and compatibility: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>.

Check the detected backend:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('ROCm/HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. Download and verify dataset part 1

```bash
python scripts/00_download_part1.py
```

The file is saved to:

```text
data/database_snapshot/magnetic-orderings-database-snapshot.part1.json.gz
```

The downloader uses `data/database_snapshot/SHA256SUMS.txt` from the release to verify the downloaded bytes. It downloads part 1 only. If the checksum does not match, stop and check the dataset source rather than proceeding with different data.

### 5. Prepare the dataset and parent-level split

```bash
python scripts/01_prepare_data.py
```

This applies the notebook-derived data processing, duplicate handling, target preparation and parent-level train/validation/test split. With the published input snapshot and preprocessing, the expected split is:

| Split | Parent structures | Magnetic configurations |
|---|---:|---:|
| Train | 2,070 | 18,010 |
| Validation | 259 | 2,246 |
| Test | 259 | 2,203 |
| **Total** | **2,588** | **22,459** |

The processed tables and split manifest are generated under `data/processed/`.

### 6. Build ALIGNN graph caches

```bash
python scripts/02_build_graphs.py
```

This builds the train, validation and test graph caches using the project configuration and the `pure_torch` graph path. The caches are generated locally and do not need to be committed to Git.

### 7. Train both ALIGNN models

```bash
python scripts/03_train_models.py --epochs 50
```

The script trains the structure-only baseline and spin-aware model for at least 50 epochs and selects the best checkpoint using validation MAE. The published training setup uses the same ALIGNN model family, scalar energy output, MSE loss, AdamW optimiser and learning-rate scheduler specified in the project code.

Expected outputs include:

```text
models/alignn_baseline_best.pt
models/alignn_spin_aware_best.pt
results/baseline_training_history.csv
results/spin_aware_training_history.csv
results/training_metrics.json
results/test_predictions_baseline.csv
results/test_predictions.csv
```

The precise best epoch and test metrics can differ with platform-level numerical behaviour. Use the outputs generated by the tagged workflow; do not substitute predictions or metrics from older notebook checkpoints.

### 8. Run preliminary and strict candidate screening

```bash
python scripts/04_screen_candidates.py
```

This applies the spin-aware Top-5 application workflow, then performs the preliminary compensated/PT-broken magnetic-symmetry analysis and the strict `spinspg` spin-space-symmetry test. It generates the intermediate screening tables needed by the benchmark step, including:

```text
results/new_preliminary_candidates.csv
results/new_candidates_spin_space_test.csv
```

### 9. Run model benchmarks and the screening funnel

```bash
python scripts/05_run_benchmarks.py
```

This compares the structure-only and spin-aware models and creates the principal benchmark tables under `results/`, including:

- `model_test_regression_metrics.csv`
- `structure_only_TopK_DFT_benchmark.csv`
- `spin_aware_TopK_DFT_benchmark.csv`
- `all_test_ground_state_TopK_DFT_benchmark.csv`
- `TopK_structure_only_vs_spin_aware_comparison.csv`
- `parent_Top1_energy_regret.csv`
- `Top1_energy_regret_summary.csv`
- `ground_state_energy_gaps.csv`
- `ground_state_gap_summary.csv`
- `Top1_accuracy_by_ground_state_gap.csv`
- `spin_aware_metrics_by_ordering_changed.csv`
- `pairwise_ranking_accuracy.csv`
- `pairwise_ranking_accuracy_by_energy_separation.csv`
- `top5_screen_table_all_test_parents.csv`

It also reports and exports the symmetry-screening funnel:

| Output file | Description |
|---|---|
| `screening_funnel_counts.csv` | Number of unique Task IDs passing preliminary screening and strict spin-space screening, plus removed/retained percentages |
| `screening_funnel_task_ids.csv` | Every preliminary Task ID, its strict-pass status, and available material/symmetry metadata |
| `final_screened_candidates.csv` | Final Task IDs that pass the strict spin-space-symmetry criterion, with available formula, ranking, space-group and symmetry details |

The screening counts are calculated from the current run's screening tables. They are not hard-coded. No historical 22-candidate list or 55-known-AM recovery reference is used by the current benchmark script.

### 10. Generate publication-quality figures (optional)

```bash
python -m pip install matplotlib
python scripts/07_generate_figures.py
```

Figures are saved under `results/figures/` in PNG (600 DPI), PDF and SVG formats by default. They include training histories, test MAE/RMSE comparisons, Top-K recovery, DFT savings, energy regret, energy-gap and pairwise-ranking diagnostics, predicted-versus-DFT scatter plots, the screening funnel, and final-candidate summaries. A figure is skipped if its required result table or columns are not present.

To generate only selected formats:

```bash
python scripts/07_generate_figures.py --formats png,pdf
```

### 11. Run the tests

```bash
python -m pip install pytest
PYTHONPATH=src python -m pytest -q
```

The unit tests check selected data-processing and project functions. They do not replace the complete training, screening and benchmark workflow.

## Reproducibility notes

- Use the tagged release, not an unpinned branch of the project or ALIGNN.
- Keep `environment-lock.txt`, `requirements-alignn.txt`, `data/database_snapshot/SHA256SUMS.txt` and `data/processed/dataset_manifest.json` with the release.
- The raw compressed dataset is downloaded by the script and verified against the committed SHA-256 checksum; it is not necessary to commit the large gzip file.
- Model checkpoints, processed pickle files and graph/LMDB caches are generated artifacts. Their availability depends on the release policy; the workflow can regenerate them.
- `environment-lock.txt` records Python package versions, but bitwise-identical training is not guaranteed across different GPU models, drivers, CUDA/ROCm versions or hardware backends.
- The final candidate table is a computational screening result. It is not proof of experimental synthesis or a confirmed altermagnetic electronic structure; DFT magnetic-ground-state comparisons and spin-resolved band calculations are still required.

## Citation

Please cite the source magnetic-orderings dataset, the ALIGNN implementation, `spinspg`, and this repository's exact release tag/commit. See [`CITATION.md`](CITATION.md) for the project-specific citation details. Report the dataset SHA-256 and ALIGNN commit recorded in the release provenance files.

