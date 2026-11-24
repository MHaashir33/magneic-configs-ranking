from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .symmetry import (
    get_PTR_symmetry,
    screen_altermagnet_corrected,
    spin_space_am_test,
)


def _ground_states_by_parent(test_df: pd.DataFrame) -> pd.DataFrame:
    # Match the notebook: lowest delta_E row per test parent.
    return (
        test_df[
            test_df.groupby("parent_structure_id")["delta_E"]
            .transform("min") == test_df["delta_E"]
        ]
        .drop_duplicates(subset=["parent_structure_id"])
        .copy()
    )


def run_screening(project_root: Path) -> dict:
    processed = project_root / "data" / "processed"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with (processed / "graph_df.pkl").open("rb") as handle:
        graph_df = pickle.load(handle)

    dataset_path = (
        project_root
        / "data"
        / "database_snapshot"
        / "magnetic-orderings-database-snapshot.part1.json.gz"
    )
    from .data_pipeline import load_part1_json_gz
    records = load_part1_json_gz(dataset_path)
    record_by_task = {str(record["task_id"]): record for record in records}

    prediction_path = results_dir / "test_predictions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"{prediction_path} not found. Run scripts/03_train_models.py first."
        )
    predictions = pd.read_csv(prediction_path)
    test_df = (
        graph_df[graph_df["split"] == "test"]
        .copy()
        .reset_index(drop=True)
    )

    predictions["task_id"] = predictions["task_id"].astype(str)
    test_df["task_id"] = test_df["task_id"].astype(str)
    if predictions["task_id"].tolist() != test_df["task_id"].tolist():
        raise AssertionError(
            "Prediction task_id order differs from the parent-level test split."
        )
    test_df["pred_energy"] = predictions["pred_energy"].to_numpy(dtype=float)
    test_df[
        [
            "task_id",
            "parent_structure_id",
            "formula",
            "delta_E",
            "energy_per_atom",
            "pred_energy",
        ]
    ].to_csv(results_dir / "test_predictions_aligned.csv", index=False)

    # The test ground-state reference set used by the notebook (55 AM states).
    test_ground_states = _ground_states_by_parent(test_df)
    known_rows = []
    for task_id in test_ground_states["task_id"].astype(str):
        record = record_by_task[task_id]

        # Match notebook Phase 6 exactly: first apply the corrected
        # compensated + PT-broken screen (which uses the first crystallographic
        # inversion operation), then run the rigorous P/T/PT operation test only
        # on those preliminary states. The final 55-state reference retains
        # preliminary candidates for which the rigorous PT test is also broken.
        corrected = screen_altermagnet_corrected(
            record,
            symprec=1e-2,
            angle_tolerance=5,
            net_tol=1e-4,
            coord_tol=0.01,
            mag_tol=0.05,
        )
        if corrected["AM_candidate"]:
            ptr = get_PTR_symmetry(record, symprec=0.01)
        else:
            ptr = {
                "spacegroup": None,
                "P_preserved": None,
                "T_preserved": None,
                "PT_preserved": None,
            }

        known_am = bool(
            corrected["AM_candidate"]
            and ptr["PT_preserved"] is False
        )
        known_rows.append(
            {
                "task_id": task_id,
                "net_moment": corrected["net_moment"],
                "compensated": corrected["compensated"],
                "first_pass_AM_candidate": corrected["AM_candidate"],
                **ptr,
                "AM": known_am,
            }
        )
    actual_am_df = pd.DataFrame(known_rows)
    actual_am_df.to_csv(
        results_dir / "test_true_ground_state_symmetry.csv", index=False
    )
    known_am_ids = set(
        actual_am_df.loc[actual_am_df["AM"], "task_id"].astype(str)
    )

    # ML ranking within each parent -> Top-5 pool.
    ranked_test = test_df.sort_values(
        ["parent_structure_id", "pred_energy"]
    ).copy()
    ranked_test["ML_rank"] = (
        ranked_test.groupby("parent_structure_id").cumcount() + 1
    )
    top5_candidates = ranked_test[
        ranked_test["ML_rank"] <= 5
    ].copy()
    top5_candidates[
        [
            "task_id",
            "parent_structure_id",
            "formula",
            "delta_E",
            "energy_per_atom",
            "pred_energy",
            "ML_rank",
        ]
    ].to_csv(
        results_dir / "top5_ml_candidates.csv", index=False
    )

    # Attach the original final DFT structure/moments and values by task_id.
    dft_rows = []
    missing_tasks = []
    for _, row in top5_candidates.iterrows():
        task_id = str(row["task_id"])
        record = record_by_task.get(task_id)
        if record is None:
            missing_tasks.append(task_id)
            continue
        mag_data = record.get("magmoms", {})
        dft_rows.append(
            {
                "task_id": task_id,
                "dft_structure": record.get("structure"),
                "final_magmoms": mag_data.get("vasp"),
                "total_magnetization": record.get("total_magnetization"),
                "dft_ordering": record.get("ordering"),
                "dft_energy_per_atom": record.get("energy_per_atom"),
                "dft_energy_above_ground": record.get(
                    "energy_above_ground_state_per_atom"
                ),
                "symmetry": record.get("symmetry"),
                "symmetry_changed": record.get("symmetry_changed"),
            }
        )
    if missing_tasks:
        raise KeyError(f"Missing DFT records for task_ids: {missing_tasks[:20]}")
    dft_info = pd.DataFrame(dft_rows)

    top5_full = top5_candidates.merge(
        dft_info, on="task_id", how="left"
    )
    top5_dft_selected = (
        top5_full.sort_values(
            ["parent_structure_id", "dft_energy_per_atom"]
        )
        .groupby("parent_structure_id", as_index=False)
        .first()
    )
    top5_dft_selected.to_csv(
        results_dir / "top5_dft_selected_states.csv", index=False
    )

    # Apply the notebook's final consistent compensated + PT-broken screen.
    screen_rows = []
    for _, row in top5_dft_selected.iterrows():
        record = record_by_task[str(row["task_id"])]
        try:
            screen_result = screen_altermagnet_corrected(
                record,
                symprec=1e-2,
                angle_tolerance=5,
                net_tol=1e-4,
                coord_tol=0.01,
                mag_tol=0.05,
            )
        except Exception as exc:
            screen_result = {
                "task_id": str(row["task_id"]),
                "formula": row["formula"],
                "ordering": row["dft_ordering"],
                "net_moment": np.nan,
                "compensated": False,
                "spacegroup": None,
                "inversion_found": None,
                "PT_preserved": None,
                "AM_candidate": False,
                "error": str(exc),
            }
        screen_rows.append(screen_result)

    am_screen_df = pd.DataFrame(screen_rows)
    am_screen_df["task_id"] = am_screen_df["task_id"].astype(str)
    top5_consistent = top5_dft_selected.merge(
        am_screen_df,
        on="task_id",
        how="left",
        suffixes=("", "_screen"),
    )

    top5_consistent["known_AM_ground"] = (
        top5_consistent["task_id"].astype(str).isin(known_am_ids)
    )
    top5_consistent["P_preserved"] = top5_consistent["inversion_found"]
    top5_consistent["AM_type"] = np.select(
        [
            top5_consistent["AM_candidate"]
            & top5_consistent["P_preserved"].fillna(False),
            top5_consistent["AM_candidate"]
            & (~top5_consistent["P_preserved"].fillna(False)),
        ],
        ["S-type", "S/A-type"],
        default="Not AM",
    )

    am_candidates = top5_consistent[
        top5_consistent["AM_candidate"].fillna(False)
    ].sort_values(["AM_type", "ML_rank"]).reset_index(drop=True)
    preliminary_columns = [
        "task_id", "parent_structure_id", "formula", "ML_rank",
        "pred_energy", "delta_E", "dft_energy_per_atom",
        "dft_ordering", "net_moment", "compensated",
        "spacegroup", "inversion_found", "PT_preserved",
        "AM_candidate", "AM_type", "known_AM_ground",
    ]
    am_candidates[preliminary_columns].to_csv(
        results_dir / "top5_preliminary_am_candidates.csv", index=False
    )

    new_preliminary = am_candidates[
        ~am_candidates["known_AM_ground"]
    ].copy()
    new_preliminary[preliminary_columns].to_csv(
        results_dir / "new_preliminary_candidates.csv", index=False
    )

    # Stronger spin-space test, same routine and tolerances as notebook.
    new_spin_rows = []
    for _, row in new_preliminary.iterrows():
        task_id = str(row["task_id"])
        record = record_by_task[task_id]
        base = {
            "task_id": int(task_id),
            "formula": row["formula"],
            "ML_rank": int(row["ML_rank"]),
            "AM_type_preliminary": row["AM_type"],
            "dft_ordering": row["dft_ordering"],
            "spacegroup": row["spacegroup"],
        }
        try:
            result = spin_space_am_test(
                record,
                symprec=1e-2,
                mag_symprec=5e-2,
                spin_tol=0.05,
            )
            base.update(
                {
                    "n_spin_ops": result["n_spin_space_operations"],
                    "n_spin_flip_ops": result["n_spin_flip_rotation_ops"],
                    "spin_space_AM": result["spin_space_AM_candidate"],
                }
            )
        except Exception as exc:
            base.update(
                {
                    "n_spin_ops": np.nan,
                    "n_spin_flip_ops": np.nan,
                    "spin_space_AM": False,
                    "error": str(exc),
                }
            )
        new_spin_rows.append(base)

    new_spin_df = pd.DataFrame(
        new_spin_rows,
        columns=[
            "task_id",
            "formula",
            "ML_rank",
            "AM_type_preliminary",
            "dft_ordering",
            "spacegroup",
            "n_spin_ops",
            "n_spin_flip_ops",
            "spin_space_AM",
            "error",
        ],
    )
    new_spin_df.to_csv(
        results_dir / "new_candidates_spin_space_test.csv", index=False
    )

    stronger = (
        new_spin_df[new_spin_df["spin_space_AM"]]
        .sort_values(["AM_type_preliminary", "ML_rank"])
        .reset_index(drop=True)
    )

    # Parent-wise DFT gap, identical to the notebook's second-lowest minus lowest delta_E.
    sorted_energy = test_df[
        ["parent_structure_id", "task_id", "formula", "delta_E"]
    ].sort_values(["parent_structure_id", "delta_E"])
    gap_values = (
        sorted_energy.groupby("parent_structure_id")["delta_E"]
        .apply(
            lambda values: (
                float(values.iloc[1] - values.iloc[0])
                if len(values) >= 2 else np.nan
            )
        )
        .rename("ground_gap_eV_atom")
        .reset_index()
    )
    gap_values["ground_gap_meV_atom"] = (
        gap_values["ground_gap_eV_atom"] * 1000
    )
    candidate_parent_map = (
        top5_candidates[["task_id", "parent_structure_id"]]
        .drop_duplicates("task_id")
    )
    candidate_parent_map["task_id"] = candidate_parent_map["task_id"].astype(str)

    stronger_with_parent = stronger.copy()
    stronger_with_parent["task_id"] = (
        stronger_with_parent["task_id"].astype(str)
    )
    stronger_with_parent = stronger_with_parent.merge(
        candidate_parent_map, on="task_id", how="left"
    )
    final_candidate_table = (
        stronger_with_parent.merge(
            gap_values, on="parent_structure_id", how="left"
        )
        .sort_values("ground_gap_meV_atom", ascending=False)
        .reset_index(drop=True)
    )

    final_columns = [
        "task_id",
        "formula",
        "AM_type_preliminary",
        "ML_rank",
        "spacegroup",
        "n_spin_ops",
        "n_spin_flip_ops",
        "ground_gap_meV_atom",
    ]
    final_candidate_table[final_columns].to_csv(
        results_dir / "final_candidate_table.csv", index=False
    )
    final_candidate_table.to_csv(
        results_dir / "screened_new_candidates.csv", index=False
    )

    summary = {
        "test_configurations": int(len(test_df)),
        "test_parents": int(test_df["parent_structure_id"].nunique()),
        "test_true_ground_states": int(len(test_ground_states)),
        "known_test_altermagnetic_ground_states": int(len(known_am_ids)),
        "top5_selected_states": int(len(top5_candidates)),
        "one_dft_selected_state_per_parent": int(len(top5_dft_selected)),
        "preliminary_am_candidates_in_top5": int(len(am_candidates)),
        "new_preliminary_candidates": int(len(new_preliminary)),
        "new_candidates_passing_spin_space_test": int(
            new_spin_df["spin_space_AM"].sum()
        ) if len(new_spin_df) else 0,
        "note": (
            "Counts depend on the newly trained 50-epoch model and package "
            "environment; the historical 22-candidate notebook output is "
            "stored separately under results/reference_outputs."
        ),
    }
    (results_dir / "screening_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Final screened candidate table:", results_dir / "final_candidate_table.csv")
    return summary
