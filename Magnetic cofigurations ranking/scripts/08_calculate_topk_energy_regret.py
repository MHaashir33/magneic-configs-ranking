#!/usr/bin/env python3
"""Calculate Top-K DFT energy-regret comparison for both models.

Expected inputs in <project>/results/:
  - test_predictions_baseline.csv  (structure-only baseline)
  - test_predictions.csv           (spin-aware model)

Required columns in each CSV:
  task_id, parent_structure_id, energy_per_atom, delta_E, pred_energy

Only TopK_energy_regret_comparison.csv is written. It contains mean and
median regret across parent structures for Top-K = 1, 2, 3, 5, and 10.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

K_VALUES = (1, 2, 3, 5, 10)
REQUIRED_COLUMNS = {
    "task_id",
    "parent_structure_id",
    "energy_per_atom",
    "delta_E",
    "pred_energy",
}


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(
        path,
        dtype={"task_id": "string", "parent_structure_id": "string"},
    )

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df.copy()
    df["task_id"] = df["task_id"].str.strip()
    df["parent_structure_id"] = df["parent_structure_id"].str.strip()

    if df["task_id"].isna().any() or (df["task_id"] == "").any():
        raise ValueError(f"{path.name} contains missing task_id values.")
    if df["parent_structure_id"].isna().any() or (df["parent_structure_id"] == "").any():
        raise ValueError(f"{path.name} contains missing parent_structure_id values.")
    if df["task_id"].duplicated().any():
        examples = df.loc[df["task_id"].duplicated(), "task_id"].head().tolist()
        raise ValueError(f"{path.name} contains duplicate task_id values, e.g. {examples}")

    for col in ("energy_per_atom", "delta_E", "pred_energy"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name} contains non-finite values in {col}.")

    df["model"] = model_name
    return df


def validate_common_test_data(baseline: pd.DataFrame, spin: pd.DataFrame) -> None:
    base_ids = set(baseline["task_id"])
    spin_ids = set(spin["task_id"])
    if base_ids != spin_ids:
        missing_spin = sorted(base_ids - spin_ids)[:10]
        missing_base = sorted(spin_ids - base_ids)[:10]
        raise ValueError(
            "The two model files do not contain the same test task IDs. "
            f"Missing from spin-aware={missing_spin}; missing from baseline={missing_base}"
        )

    b = baseline.set_index("task_id").sort_index()
    s = spin.set_index("task_id").sort_index()

    if not b["parent_structure_id"].equals(s["parent_structure_id"]):
        raise ValueError("Parent IDs do not match between the two prediction files.")

    # These are DFT reference values, so they must be identical for both models.
    for col in ("energy_per_atom", "delta_E"):
        b_values = b[col].to_numpy(dtype=float)
        s_values = s[col].to_numpy(dtype=float)
        if not np.allclose(b_values, s_values, rtol=1e-9, atol=1e-9):
            max_diff = float(np.max(np.abs(b_values - s_values)))
            raise ValueError(
                f"DFT reference column '{col}' differs between model files "
                f"(maximum absolute difference {max_diff:.3e})."
            )


def calculate_model_regrets(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict] = []

    for parent_id, group in df.groupby("parent_structure_id", sort=True):
        # The model ranks configurations by its predicted energy (lower is better).
        ranked = group.sort_values(["pred_energy", "task_id"], kind="mergesort")

        # True ground state is determined using the actual DFT energy.
        true_ground_idx = group["energy_per_atom"].idxmin()
        true_ground = group.loc[true_ground_idx]
        true_ground_energy = float(true_ground["energy_per_atom"])
        true_ground_task = str(true_ground["task_id"])

        for k in K_VALUES:
            selected = ranked.head(k)
            selected_best_idx = selected["energy_per_atom"].idxmin()
            selected_best = selected.loc[selected_best_idx]

            raw_regret = float(selected_best["energy_per_atom"] - true_ground_energy)
            # Remove only tiny negative floating-point noise; larger negatives signal bad data.
            if raw_regret < -1e-7:
                raise ValueError(
                    f"Negative regret ({raw_regret}) for parent {parent_id}, model {model_name}, Top-{k}."
                )
            regret = max(0.0, raw_regret)

            # Cross-check using the repository's delta_E (= energy above ground state per atom).
            delta_regret = float(selected["delta_E"].min() - group["delta_E"].min())

            rows.append(
                {
                    "model": model_name,
                    "parent_structure_id": str(parent_id),
                    "formula": str(group["formula"].iloc[0]) if "formula" in group.columns else "",
                    "Top_K": k,
                    "n_configurations_for_parent": int(len(group)),
                    "n_selected": int(len(selected)),
                    "true_ground_state_task_id": true_ground_task,
                    "best_selected_task_id": str(selected_best["task_id"]),
                    "true_ground_state_DFT_energy_eV_atom": true_ground_energy,
                    "best_selected_DFT_energy_eV_atom": float(selected_best["energy_per_atom"]),
                    "energy_regret_eV_atom": regret,
                    "energy_regret_meV_atom": regret * 1000.0,
                    "delta_E_based_regret_eV_atom_check": max(0.0, delta_regret),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing test_predictions_baseline.csv and test_predictions.csv",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()

    baseline = load_predictions(
        results_dir / "test_predictions_baseline.csv", "structure_only"
    )
    spin = load_predictions(results_dir / "test_predictions.csv", "spin_aware")
    validate_common_test_data(baseline, spin)

    all_parent_results = pd.concat(
        [
            calculate_model_regrets(baseline, "structure_only"),
            calculate_model_regrets(spin, "spin_aware"),
        ],
        ignore_index=True,
    )

    summary = (
        all_parent_results.groupby(["model", "Top_K"], as_index=False)
        .agg(
            n_parents=("parent_structure_id", "nunique"),
            mean_regret_eV_atom=("energy_regret_eV_atom", "mean"),
            median_regret_eV_atom=("energy_regret_eV_atom", "median"),
            std_regret_eV_atom=("energy_regret_eV_atom", "std"),
            mean_regret_meV_atom=("energy_regret_meV_atom", "mean"),
            median_regret_meV_atom=("energy_regret_meV_atom", "median"),
        )
        .sort_values(["Top_K", "model"])
        .reset_index(drop=True)
    )

    comparison = summary.pivot(
        index="Top_K",
        columns="model",
        values=["mean_regret_eV_atom", "median_regret_eV_atom", "mean_regret_meV_atom", "median_regret_meV_atom"],
    )
    comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]
    comparison = comparison.reset_index()

    # Save only the requested side-by-side comparison CSV.
    comparison_path = results_dir / "TopK_energy_regret_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    display_table = comparison[
        [
            "Top_K",
            "mean_regret_meV_atom_structure_only",
            "mean_regret_meV_atom_spin_aware",
            "median_regret_meV_atom_structure_only",
            "median_regret_meV_atom_spin_aware",
        ]
    ].copy()
    display_table = display_table.rename(
        columns={
            "Top_K": "Top-K",
            "mean_regret_meV_atom_structure_only": "Baseline mean (meV/atom)",
            "mean_regret_meV_atom_spin_aware": "Spin-aware mean (meV/atom)",
            "median_regret_meV_atom_structure_only": "Baseline median (meV/atom)",
            "median_regret_meV_atom_spin_aware": "Spin-aware median (meV/atom)",
        }
    )

    print("\nMean and median energy regret by Top-K (meV/atom):")
    print(display_table.round(6).to_string(index=False))
    print("\nSaved comparison file:")
    print(comparison_path)


if __name__ == "__main__":
    main()
