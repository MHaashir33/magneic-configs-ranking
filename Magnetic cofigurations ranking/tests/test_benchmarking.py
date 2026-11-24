from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from altermagnet_screening.benchmarking import run_benchmarks


def test_benchmark_pipeline_smoke(tmp_path: Path):
    rows = []
    for parent, base_energy, prefix in [
        ("parent-a", -5.0, "a"),
        ("parent-b", -4.0, "b"),
    ]:
        for index, offset in enumerate((0.0, 0.01, 0.03)):
            rows.append(
                {
                    "task_id": f"{prefix}{index}",
                    "parent_structure_id": parent,
                    "formula": "FeO",
                    "delta_E": offset,
                    "energy_per_atom": base_energy + offset,
                    "ordering": ("AFM", "FM", "FiM")[index],
                    "ordering_changed": bool(index == 2),
                    "split": "test",
                }
            )

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    with (processed / "graph_df.pkl").open("wb") as handle:
        pickle.dump(pd.DataFrame(rows), handle)

    results = tmp_path / "results"
    results.mkdir()
    test_ids = [row["task_id"] for row in rows]
    # The true DFT ground state is rank 1 for parent-a and rank 2 for parent-b.
    baseline_pred = [0.0, 0.02, 0.01, 0.02, 0.0, 0.01]
    spin_pred = [0.0, 0.01, 0.02, 0.01, 0.0, 0.02]
    for filename, values in [
        ("test_predictions_baseline.csv", baseline_pred),
        ("test_predictions.csv", spin_pred),
    ]:
        pd.DataFrame(
            {
                "task_id": test_ids,
                "parent_structure_id": [r["parent_structure_id"] for r in rows],
                "pred_energy": values,
            }
        ).to_csv(results / filename, index=False)

    pd.DataFrame(
        {
            "task_id": ["a0"],
            "AM": [True],
            "P_preserved": [True],
        }
    ).to_csv(results / "test_true_ground_state_symmetry.csv", index=False)

    summary = run_benchmarks(tmp_path)
    assert summary["test_parents"] == 2
    assert summary["reference_AM_ground_states"] == 1
    assert (results / "all_test_ground_state_TopK_DFT_benchmark.csv").is_file()
    assert (results / "Top1_energy_regret_summary.csv").is_file()
    assert (results / "AM_DFT_saving_benchmark.csv").is_file()
