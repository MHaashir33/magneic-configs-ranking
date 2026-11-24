"""Notebook-faithful preparation of the part-1 magnetic-ordering snapshot."""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PART1_NAME = "magnetic-orderings-database-snapshot.part1.json.gz"
EXPECTED_RAW_RECORDS = 23155
EXPECTED_PARENT_COUNTS = {"train": 2070, "val": 259, "test": 259}
EXPECTED_SAMPLE_COUNTS = {"train": 18010, "val": 2246, "test": 2203}


def load_part1_json_gz(path: str | Path) -> list[dict[str, Any]]:
    """Load only the supplied part-1 gzip JSON file; never inspect part 2."""
    path = Path(path)
    if path.name != PART1_NAME:
        raise ValueError(
            f"This project is intentionally restricted to {PART1_NAME}; got {path.name}."
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("The part-1 snapshot must contain a JSON list.")
    return records


def structure_signature(parent_structure: dict[str, Any]) -> str:
    """Reproduce the notebook's parent signature (six-decimal lattice/sites)."""
    sites = []
    for site in parent_structure["sites"]:
        element = site["species"][0]["element"]
        abc = [round(float(value), 6) for value in site["abc"]]
        sites.append({"element": element, "abc": abc})
    sites.sort(key=lambda item: (item["element"], *item["abc"]))
    payload = {
        "lattice": [
            [round(float(value), 6) for value in row]
            for row in parent_structure["lattice"]["matrix"]
        ],
        "sites": sites,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_initial_magmoms(record: dict[str, Any]) -> list[float]:
    """Read initial site moments from input.structure.site.properties.magmom."""
    sites = record["input"]["structure"]["sites"]
    moments = []
    for site in sites:
        value = site.get("properties", {}).get("magmom", 0.0)
        moments.append(float(0.0 if value is None else value))
    return moments


def input_magnetic_signature(record: dict[str, Any]) -> str:
    """Reproduce notebook signature of INPUT lattice, ordered sites and exact magmoms."""
    structure = record["input"]["structure"]
    sites = []
    for site in structure["sites"]:
        element = site["species"][0]["element"]
        abc = site["abc"]
        moment = site.get("properties", {}).get("magmom", 0.0)
        moment = 0.0 if moment is None else float(moment)
        sites.append(
            {
                "element": element,
                "abc": [round(float(value), 6) for value in abc],
                "magmom": round(moment, 6),
            }
        )
    payload = {
        "lattice": [
            [round(float(value), 6) for value in row]
            for row in structure["lattice"]["matrix"]
        ],
        "sites": sites,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_clean_dataset(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the notebook's V1 table and collapse duplicate initial states."""
    rows = []
    for record in records:
        rows.append(
            {
                "task_id": record.get("task_id"),
                "formula": record.get("formula_pretty", record.get("formula")),
                "parent_structure_id": structure_signature(record["parent_structure"]),
                "ordering": record.get("ordering"),
                "ordering_changed": bool(record.get("ordering_changed", False)),
                "energy_per_atom": float(record["energy_per_atom"]),
                "delta_E": float(record["energy_above_ground_state_per_atom"]),
                "initial_magmoms": extract_initial_magmoms(record),
                "input_magnetic_signature": input_magnetic_signature(record),
            }
        )
    raw_df = pd.DataFrame(rows)
    group_columns = ["parent_structure_id", "input_magnetic_signature"]
    # Same pandas aggregation policy as notebook cell 80.
    clean = (
        raw_df.groupby(group_columns, as_index=False)
        .agg(
            formula=("formula", "first"),
            task_id=("task_id", "first"),
            initial_magmoms=("initial_magmoms", "first"),
            energy_per_atom=("energy_per_atom", "mean"),
            delta_E=("delta_E", "mean"),
            ordering=("ordering", "first"),
            ordering_changed=("ordering_changed", "max"),
            n_dft_records=("task_id", "count"),
        )
    )
    return clean


def parent_level_split(graph_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Reproduce train_test_split calls from notebook cell 92."""
    parents = graph_df["parent_structure_id"].dropna().unique().tolist()
    train_parents, temporary_parents = train_test_split(
        parents, test_size=0.20, random_state=42
    )
    val_parents, test_parents = train_test_split(
        temporary_parents, test_size=0.50, random_state=42
    )
    parent_sets = {
        "train": set(train_parents),
        "val": set(val_parents),
        "test": set(test_parents),
    }
    output = graph_df.copy()
    output["split"] = ""
    for split_name, parent_set in parent_sets.items():
        output.loc[output["parent_structure_id"].isin(parent_set), "split"] = split_name
    if (output["split"] == "").any():
        raise RuntimeError("At least one parent was not assigned to a split.")
    manifest = {name: sorted(str(parent) for parent in values) for name, values in parent_sets.items()}
    return output, manifest


def build_graph_dataframe(
    clean_dataset: pd.DataFrame, records: list[dict[str, Any]]
) -> pd.DataFrame:
    """Reconstruct graph_df using each representative row's INPUT structure."""
    record_lookup = {str(record["task_id"]): record for record in records}
    rows = []
    for row in clean_dataset.to_dict(orient="records"):
        task_id = str(row["task_id"])
        if task_id not in record_lookup:
            raise KeyError(f"Representative task_id {task_id} absent from part 1.")
        record = record_lookup[task_id]
        input_data = record["input"]
        rows.append(
            {
                "task_id": row["task_id"],
                "parent_structure_id": row["parent_structure_id"],
                "formula": row["formula"],
                "input_structure": input_data["structure"],
                "initial_magmoms": row["initial_magmoms"],
                "delta_E": float(row["delta_E"]),
                "energy_per_atom": float(row["energy_per_atom"]),
                "ordering": row["ordering"],
                "ordering_changed": bool(row["ordering_changed"]),
                "n_dft_records": int(row["n_dft_records"]),
                "input_ordering": input_data.get("ordering"),
                "origin": input_data.get("origin"),
                "input_index": input_data.get("input_index"),
                "index": input_data.get("index"),
            }
        )
    return pd.DataFrame(rows)


def prepare_project_data(project_root: str | Path, strict_counts: bool = True) -> dict[str, Any]:
    """Prepare all intermediate tables and the fixed parent-level split."""
    project_root = Path(project_root).resolve()
    source_path = project_root / "data" / "database_snapshot" / PART1_NAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing {source_path}. Run scripts/00_download_part1.py first."
        )
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = load_part1_json_gz(source_path)
    if strict_counts and len(records) != EXPECTED_RAW_RECORDS:
        raise RuntimeError(
            f"Part 1 record count changed: expected {EXPECTED_RAW_RECORDS}, got {len(records)}. "
            "Check the source revision and dataset SHA before continuing."
        )
    clean = build_clean_dataset(records)
    graph_df = build_graph_dataframe(clean, records)
    graph_df, split_manifest = parent_level_split(graph_df)

    if strict_counts:
        parent_counts = graph_df.groupby("split")["parent_structure_id"].nunique().to_dict()
        sample_counts = graph_df["split"].value_counts().to_dict()
        if parent_counts != EXPECTED_PARENT_COUNTS or sample_counts != EXPECTED_SAMPLE_COUNTS:
            raise RuntimeError(
                "Prepared split sizes differ from the supplied notebook. "
                f"Parent counts={parent_counts}; sample counts={sample_counts}. "
                "Do not silently publish a changed split."
            )

    clean.to_pickle(processed_dir / "magnetic_dataset_v1.pkl")
    clean.to_csv(processed_dir / "magnetic_dataset_v1.csv", index=False)
    graph_df.to_pickle(processed_dir / "graph_df.pkl")
    graph_export = graph_df.drop(columns=["input_structure", "initial_magmoms"]).copy()
    graph_export.to_csv(processed_dir / "graph_df_metadata.csv", index=False)

    split_payload = {
        "seed": 42,
        "test_size_first_split": 0.20,
        "test_size_second_split": 0.50,
        "parents": split_manifest,
        "sample_counts": graph_df["split"].value_counts().to_dict(),
        "parent_counts": graph_df.groupby("split")["parent_structure_id"].nunique().to_dict(),
    }
    (processed_dir / "split_manifest.json").write_text(
        json.dumps(split_payload, indent=2) + "\n", encoding="utf-8"
    )

    # Keep the notebook's headerless two-column id_prop convention for auditability.
    for split_name in ("train", "val", "test"):
        part = graph_df.loc[graph_df["split"] == split_name]
        pd.DataFrame(
            [[f"{task_id}.cif", float(target)] for task_id, target in zip(part["task_id"], part["delta_E"])]
        ).to_csv(processed_dir / f"id_prop_{split_name}.csv", index=False, header=False)

    summary = {
        "raw_records": len(records),
        "unique_ml_states": len(clean),
        "unique_parent_structures": int(graph_df["parent_structure_id"].nunique()),
        "train_samples": int((graph_df["split"] == "train").sum()),
        "val_samples": int((graph_df["split"] == "val").sum()),
        "test_samples": int((graph_df["split"] == "test").sum()),
        "train_parents": int(graph_df.loc[graph_df["split"] == "train", "parent_structure_id"].nunique()),
        "val_parents": int(graph_df.loc[graph_df["split"] == "val", "parent_structure_id"].nunique()),
        "test_parents": int(graph_df.loc[graph_df["split"] == "test", "parent_structure_id"].nunique()),
        "dataset_file": PART1_NAME,
    }
    (processed_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary
