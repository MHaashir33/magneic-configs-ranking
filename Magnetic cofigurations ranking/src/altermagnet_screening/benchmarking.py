from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

K_VALUES = (1, 2, 3, 5, 10)
REGRET_THRESHOLDS = (0.001, 0.002, 0.005, 0.010, 0.020)
GAP_BINS = (0.0, 0.001, 0.005, 0.010, np.inf)
GAP_LABELS = ("<1 meV", "1–5 meV", "5–10 meV", ">10 meV")


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert CSV booleans safely, including strings such as 'False'."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "1.0": True,
        "yes": True,
        "y": True,
        "pass": True,
        "passed": True,
        "allowed": True,
        "broken": True,
        "compensated": True,
        "false": False,
        "f": False,
        "0": False,
        "0.0": False,
        "no": False,
        "n": False,
        "fail": False,
        "failed": False,
        "not allowed": False,
        "preserved": False,
        "uncompensated": False,
        "none": False,
        "nan": False,
        "": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(False)


def _load_test_frame(project_root: Path) -> pd.DataFrame:
    graph_path = project_root / "data" / "processed" / "graph_df.pkl"
    if not graph_path.is_file():
        raise FileNotFoundError(
            f"{graph_path} is missing. Run scripts/01_prepare_data.py first."
        )
    with graph_path.open("rb") as handle:
        graph_df = pickle.load(handle)

    test_df = graph_df.loc[graph_df["split"] == "test"].copy().reset_index(drop=True)
    required = {
        "task_id", "parent_structure_id", "formula", "delta_E",
        "energy_per_atom", "ordering_changed",
    }
    missing = sorted(required - set(test_df.columns))
    if missing:
        raise KeyError(f"graph_df.pkl is missing required test columns: {missing}")

    test_df["task_id"] = test_df["task_id"].astype(str)
    test_df["parent_structure_id"] = test_df["parent_structure_id"].astype(str)
    if test_df["task_id"].duplicated().any():
        raise ValueError("The test split contains duplicate task_id values.")
    return test_df


def _attach_predictions(
    test_df: pd.DataFrame,
    prediction_path: Path,
    model_name: str,
) -> pd.DataFrame:
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"{prediction_path} is missing. Re-run scripts/03_train_models.py."
        )
    pred = pd.read_csv(prediction_path)
    required = {"task_id", "pred_energy"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise KeyError(f"{prediction_path.name} lacks columns: {missing}")

    pred["task_id"] = pred["task_id"].astype(str)
    if pred["task_id"].duplicated().any():
        raise ValueError(f"Duplicate task_id values in {prediction_path.name}.")

    expected = set(test_df["task_id"])
    observed = set(pred["task_id"])
    if expected != observed:
        absent = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise AssertionError(
            f"{model_name} prediction IDs do not match the test split. "
            f"Missing={absent}; extra={extra}."
        )

    if "parent_structure_id" in pred.columns:
        pred["parent_structure_id"] = pred["parent_structure_id"].astype(str)
        expected_parent = test_df.set_index("task_id")["parent_structure_id"]
        observed_parent = pred.set_index("task_id")["parent_structure_id"]
        mismatch = (expected_parent.sort_index() != observed_parent.sort_index())
        if mismatch.any():
            raise AssertionError(
                f"{model_name} prediction parent IDs do not align with graph_df.pkl."
            )

    result = test_df.merge(
        pred[["task_id", "pred_energy"]],
        on="task_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if result["pred_energy"].isna().any():
        raise ValueError(f"Missing {model_name} predictions after task_id merge.")
    result["pred_energy"] = result["pred_energy"].astype(float)
    return result


def _rank_configurations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank predicted energies within each parent and locate its DFT ground state."""
    ranked = df.sort_values(
        ["parent_structure_id", "pred_energy"], kind="mergesort"
    ).copy()
    ranked["ML_rank"] = ranked.groupby("parent_structure_id").cumcount() + 1

    # The notebook's Phase 7 uses energy_per_atom to identify the DFT minimum.
    actual = (
        df.loc[df.groupby("parent_structure_id")["energy_per_atom"].idxmin(),
               ["parent_structure_id", "task_id", "delta_E", "energy_per_atom"]]
        .rename(columns={"task_id": "true_ground_task"})
        .copy()
    )
    actual["parent_structure_id"] = actual["parent_structure_id"].astype(str)

    rank_of_truth = ranked.loc[
        ranked["task_id"].isin(actual["true_ground_task"]),
        ["parent_structure_id", "task_id", "ML_rank"],
    ].rename(columns={"task_id": "true_ground_task", "ML_rank": "true_ground_rank"})

    parent_ranks = actual.merge(
        rank_of_truth,
        on=["parent_structure_id", "true_ground_task"],
        how="left",
        validate="one_to_one",
    )
    counts = df.groupby("parent_structure_id").size().rename("n_configs")
    parent_ranks = parent_ranks.merge(counts, on="parent_structure_id", how="left")
    if parent_ranks["true_ground_rank"].isna().any():
        raise AssertionError("A true DFT ground state could not be assigned an ML rank.")
    parent_ranks["true_ground_rank"] = parent_ranks["true_ground_rank"].astype(int)
    return ranked, parent_ranks


def _regression_metrics(df: pd.DataFrame, model_name: str) -> dict:
    error = df["pred_energy"].to_numpy(dtype=float) - df["delta_E"].to_numpy(dtype=float)
    return {
        "model": model_name,
        "n_test_configurations": int(len(df)),
        "MAE_eV_atom": float(np.mean(np.abs(error))),
        "RMSE_eV_atom": float(np.sqrt(np.mean(error ** 2))),
    }


def _topk_benchmark(parent_ranks: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    total_parents = len(parent_ranks)
    full_dft_per_parent = float(parent_ranks["n_configs"].mean())
    for k in K_VALUES:
        recovered = int((parent_ranks["true_ground_rank"] <= k).sum())
        selected_dft_per_parent = float(np.minimum(parent_ranks["n_configs"], k).mean())
        rows.append({
            "model": model_name,
            "Top_K": k,
            "ground_states_recovered": recovered,
            "total_parents": total_parents,
            "ground_state_recovery_pct": 100.0 * recovered / total_parents,
            "avg_DFT_calculations_per_parent": selected_dft_per_parent,
            "avg_full_DFT_calculations_per_parent": full_dft_per_parent,
            "DFT_saving_pct": 100.0 * (1.0 - selected_dft_per_parent / full_dft_per_parent),
        })
    return pd.DataFrame(rows)


def _top1_regret(df: pd.DataFrame, ranked: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, dict]:
    selected = (
        ranked.loc[ranked["ML_rank"] == 1,
                   ["parent_structure_id", "task_id", "delta_E", "pred_energy"]]
        .rename(columns={
            "task_id": "ML_selected_task",
            "delta_E": "selected_true_delta_E",
            "pred_energy": "selected_predicted_delta_E",
        })
        .copy()
    )
    dft_min = df.groupby("parent_structure_id")["delta_E"].min().rename("true_min_delta_E")
    selected = selected.merge(dft_min, on="parent_structure_id", how="left")
    # In the notebook delta_E is already an energy above the DFT ground state,
    # so selected_true_delta_E is its regret. The subtraction is retained as
    # a cross-check in case the input table contains a non-zero reference offset.
    selected["notebook_regret_eV_atom"] = selected["selected_true_delta_E"]
    selected["energy_regret_eV_atom"] = (
        selected["selected_true_delta_E"] - selected["true_min_delta_E"]
    )
    selected.insert(0, "model", model_name)

    values = selected["notebook_regret_eV_atom"].to_numpy(dtype=float)
    summary = {
        "model": model_name,
        "mean_regret_eV_atom": float(np.mean(values)),
        "median_regret_eV_atom": float(np.median(values)),
        "max_regret_eV_atom": float(np.max(values)),
    }
    for threshold in REGRET_THRESHOLDS:
        summary[f"fraction_regret_below_{threshold:.3f}_eV_atom_pct"] = (
            100.0 * float(np.mean(values < threshold))
        )
    return selected, summary


def _ground_state_gap_tables(
    df: pd.DataFrame, ranked: pd.DataFrame, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_best = (
        ranked.loc[ranked["ML_rank"] == 1,
                   ["parent_structure_id", "task_id"]]
        .set_index("parent_structure_id")["task_id"]
    )
    for parent, group in df.groupby("parent_structure_id", sort=False):
        energies = np.sort(group["delta_E"].to_numpy(dtype=float))
        if len(energies) < 2:
            continue
        actual_task = group.loc[group["delta_E"].idxmin(), "task_id"]
        rows.append({
            "model": model_name,
            "parent_structure_id": parent,
            "ground_to_first_excited_gap_eV_atom": float(energies[1] - energies[0]),
            "top1_correct": bool(pred_best.loc[parent] == actual_task),
        })
    gap_df = pd.DataFrame(rows)
    if gap_df.empty:
        return gap_df, pd.DataFrame()

    gap_df["gap_group"] = pd.cut(
        gap_df["ground_to_first_excited_gap_eV_atom"],
        bins=GAP_BINS,
        labels=GAP_LABELS,
        right=False,
    )
    grouped = (
        gap_df.groupby("gap_group", observed=False)
        .agg(parents=("top1_correct", "size"), top1_accuracy=("top1_correct", "mean"))
        .reset_index()
    )
    grouped.insert(0, "model", model_name)
    grouped["top1_accuracy_pct"] = grouped["top1_accuracy"] * 100.0
    return gap_df, grouped


def _ground_state_gap_summary(gap_df: pd.DataFrame, model_name: str) -> dict:
    """Notebook Phase 4.1: summarize the DFT gap between the two lowest states."""
    gaps = gap_df["ground_to_first_excited_gap_eV_atom"].to_numpy(dtype=float)
    summary = {
        "model": model_name,
        "n_parents": int(len(gaps)),
        "mean_gap_eV_atom": float(np.mean(gaps)),
        "median_gap_eV_atom": float(np.median(gaps)),
    }
    for threshold in REGRET_THRESHOLDS:
        summary[f"fraction_gap_below_{threshold:.3f}_eV_atom_pct"] = (
            100.0 * float(np.mean(gaps < threshold))
        )
    return summary


def _ordering_change_metrics(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for flag in (False, True):
        subset = df.loc[_as_bool(df["ordering_changed"]) == flag].copy()
        if subset.empty:
            continue
        error = subset["pred_energy"].to_numpy(float) - subset["delta_E"].to_numpy(float)
        correct = 0
        total = 0
        for _, group in subset.groupby("parent_structure_id"):
            actual_task = group.loc[group["delta_E"].idxmin(), "task_id"]
            predicted_task = group.loc[group["pred_energy"].idxmin(), "task_id"]
            correct += int(actual_task == predicted_task)
            total += 1
        rows.append({
            "model": model_name,
            "ordering_changed": flag,
            "n_configurations": int(len(subset)),
            "MAE_eV_atom": float(np.mean(np.abs(error))),
            "RMSE_eV_atom": float(np.sqrt(np.mean(error ** 2))),
            "ground_state_correct": correct,
            "parents_in_group": total,
            "ground_state_accuracy_pct": 100.0 * correct / total if total else np.nan,
        })
    return pd.DataFrame(rows)


def _pairwise_ranking(df: pd.DataFrame, model_name: str) -> tuple[dict, pd.DataFrame]:
    totals = {label: {"correct": 0, "total": 0} for label in GAP_LABELS}
    overall_correct = 0
    overall_total = 0
    for _, group in df.groupby("parent_structure_id"):
        true_energy = group["delta_E"].to_numpy(dtype=float)
        pred_energy = group["pred_energy"].to_numpy(dtype=float)
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                true_diff = true_energy[i] - true_energy[j]
                if true_diff == 0:
                    continue
                pred_diff = pred_energy[i] - pred_energy[j]
                is_correct = bool(true_diff * pred_diff > 0)
                overall_total += 1
                overall_correct += int(is_correct)
                separation = abs(true_diff)
                bin_index = int(np.digitize(separation, GAP_BINS[1:-1], right=False))
                label = GAP_LABELS[bin_index]
                totals[label]["total"] += 1
                totals[label]["correct"] += int(is_correct)

    summary = {
        "model": model_name,
        "correct_pairs": overall_correct,
        "total_non_tied_pairs": overall_total,
        "pairwise_accuracy_pct": (
            100.0 * overall_correct / overall_total if overall_total else np.nan
        ),
    }
    by_gap = []
    for label in GAP_LABELS:
        correct = totals[label]["correct"]
        total = totals[label]["total"]
        by_gap.append({
            "model": model_name,
            "energy_separation": label,
            "correct_pairs": correct,
            "total_pairs": total,
            "pairwise_accuracy_pct": 100.0 * correct / total if total else np.nan,
        })
    return summary, pd.DataFrame(by_gap)


# Candidate-screening funnel inputs. These are stage outputs from the current
# screening run; historical notebook candidate lists are deliberately excluded.
_DEFAULT_COMBINED_SCREEN_TABLES = (
    "results/symmetry_screening_results.csv",
    "results/magnetic_symmetry_screening.csv",
    "results/combined_symmetry_screen.csv",
)
_DEFAULT_PRELIMINARY_TABLES = (
    # Exact filenames produced by scripts/04_screen_candidates.py.
    "results/new_preliminary_candidates.csv",
    "results/preliminary_symmetry_candidates.csv",
    "results/preliminary_magnetic_symmetry_candidates.csv",
    "results/compensated_pt_broken_candidates.csv",
    "results/preliminary_screening_candidates.csv",
    "results/preliminary_candidates.csv",
)
_DEFAULT_STRICT_TABLES = (
    # Full 33-row spin-space test output; filter using its spin_space_AM flag.
    "results/new_candidates_spin_space_test.csv",
    "results/strict_spin_space_symmetry_candidates.csv",
    "results/strict_spin_space_candidates.csv",
    "results/spin_space_symmetry_candidates.csv",
    "results/screened_new_candidates.csv",
    "results/final_candidate_table.csv",
)

_COMPENSATION_COLUMNS = (
    "compensated", "is_compensated",
    "net_moment_compensated", "moment_compensated",
)
_PT_BROKEN_COLUMNS = (
    "PT_broken", "pt_broken", "is_pt_broken", "PT_symmetry_broken",
    "pt_symmetry_broken",
)
_PT_PRESERVED_COLUMNS = (
    "PT_preserved", "pt_preserved", "preserves_PT", "preserve_PT",
)
_STRICT_PASS_COLUMNS = (
    "spin_space_AM", "spin_space_AM_pass", "spin_space_AM_candidate",
    "spin_space_symmetry_pass", "spin_space_symmetry_passed",
    "strict_spin_space_pass", "strict_spin_space_symmetry_pass",
    "strict_spin_space_symmetry_passed", "spin_space_symmetry_satisfied",
    "spinspg_pass", "spinspg_passed", "spinspg_AM",
    "is_spin_space_symmetry_allowed", "strict_AM_candidate",
    "is_altermagnetic", "altermagnetic",
)
_CANDIDATE_ID_COLUMNS = (
    # The requested screening-funnel unit is a task ID, not a parent row.
    "task_id", "taskid", "task id",
    "parent_structure_id", "parent_id", "material_id", "mp_id",
    "structure_id", "jid", "formula",
)


def _first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column, case-insensitively."""
    lookup = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _is_boolean_like(series: pd.Series) -> bool:
    """Avoid interpreting arbitrary numeric/string measurements as booleans."""
    if pd.api.types.is_bool_dtype(series):
        return True
    allowed = {
        "true", "t", "1", "1.0", "yes", "y", "pass", "passed", "allowed",
        "broken", "compensated", "false", "f", "0", "0.0", "no", "n",
        "fail", "failed", "not allowed", "preserved", "uncompensated", "none", "nan", "",
    }
    values = series.dropna().astype(str).str.strip().str.lower()
    return bool(len(values)) and values.isin(allowed).all()


def _filter_preliminary_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply compensated AND PT-broken filters when raw flags are present.

    If the supplied file is already a preliminary-stage candidate list, the
    absent flags are treated as already applied. Any flag that is present is
    still checked, so a raw table can also be supplied.
    """
    filtered = df.copy()
    applied: list[str] = []

    compensation_col = _first_column(filtered, _COMPENSATION_COLUMNS)
    if compensation_col is not None and _is_boolean_like(filtered[compensation_col]):
        filtered = filtered.loc[_as_bool(filtered[compensation_col])].copy()
        applied.append(f"{compensation_col}=True")

    pt_broken_col = _first_column(filtered, _PT_BROKEN_COLUMNS)
    pt_preserved_col = _first_column(filtered, _PT_PRESERVED_COLUMNS)
    if pt_broken_col is not None and _is_boolean_like(filtered[pt_broken_col]):
        filtered = filtered.loc[_as_bool(filtered[pt_broken_col])].copy()
        applied.append(f"{pt_broken_col}=True")
    elif pt_preserved_col is not None and _is_boolean_like(filtered[pt_preserved_col]):
        filtered = filtered.loc[~_as_bool(filtered[pt_preserved_col])].copy()
        applied.append(f"{pt_preserved_col}=False")

    if not applied:
        applied.append("rows assumed to be pre-filtered by compensated + PT-broken criteria")
    elif compensation_col is None or (pt_broken_col is None and pt_preserved_col is None):
        missing = []
        if compensation_col is None or not _is_boolean_like(filtered[compensation_col]):
            missing.append("boolean compensation flag")
        if (
            (pt_broken_col is None or not _is_boolean_like(filtered[pt_broken_col]))
            and (pt_preserved_col is None or not _is_boolean_like(filtered[pt_preserved_col]))
        ):
            missing.append("boolean PT-broken/PT-preserved flag")
        applied.append("missing " + " and ".join(missing) + "; assumed already applied in source file")

    return filtered, applied


def _filter_strict_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Apply a strict spin-space pass flag if present; otherwise trust stage output."""
    strict_col = _first_column(df, _STRICT_PASS_COLUMNS)
    if strict_col is None or not _is_boolean_like(df[strict_col]):
        return df.copy(), "rows assumed to have passed strict spin-space symmetry analysis"
    return df.loc[_as_bool(df[strict_col])].copy(), f"{strict_col}=True"


def _candidate_identity(df: pd.DataFrame, stage: str) -> tuple[set[str], str]:
    """Return unique material identifiers, preferring parent/material IDs."""
    identity_col = _first_column(df, _CANDIDATE_ID_COLUMNS)
    if identity_col is None:
        # A candidate table should identify materials. Do not silently count
        # arbitrary rows as distinct materials when no identifier is supplied.
        raise KeyError(
            f"The {stage} screening table has no material identifier. Expected one of: "
            f"{', '.join(_CANDIDATE_ID_COLUMNS)}."
        )
    values = df[identity_col].dropna().astype(str).str.strip()
    values = values.loc[values.ne("")]
    return set(values.unique()), identity_col


def _resolve_input_path(project_root: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _first_existing(project_root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        candidate = project_root / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def _canonical_task_table(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Return one row per Task ID, normalising CSV integer/string IDs."""
    task_col = _first_column(df, ("task_id", "taskid", "task id"))
    if task_col is None:
        raise KeyError(
            f"The {stage} table must contain a Task ID column named task_id. "
            f"Available columns: {list(df.columns)}"
        )
    out = df.copy()
    out["task_id"] = out[task_col].astype("string").str.strip()
    out = out.loc[out["task_id"].notna() & out["task_id"].ne("")].copy()
    # IDs commonly arrive from CSV as floats (e.g. 1234.0); normalise those.
    out["task_id"] = out["task_id"].str.replace(r"\.0$", "", regex=True)
    if out["task_id"].duplicated().any():
        duplicate_ids = out.loc[out["task_id"].duplicated(), "task_id"].unique().tolist()[:10]
        raise ValueError(
            f"The {stage} table contains duplicate Task IDs: {duplicate_ids}. "
            "Resolve duplicates before counting screened candidates."
        )
    if task_col != "task_id":
        out = out.drop(columns=[task_col])
    return out.reset_index(drop=True)


def _screening_funnel_benchmark(
    project_root: Path,
    results_dir: Path,
    preliminary_table_path: Path | None = None,
    strict_table_path: Path | None = None,
    combined_table_path: Path | None = None,
) -> dict | None:
    """Count preliminary and strict-stage Task IDs and export final candidates.

    Default inputs are the outputs produced by the project's screening script:
      * results/new_preliminary_candidates.csv (compensated + PT-broken)
      * results/new_candidates_spin_space_test.csv (all preliminary candidates
        annotated with the spin_space_AM strict-pass flag)

    The earlier notebook's 22 candidates and the 55 known-AM reference set are
    never loaded or used here.
    """
    if combined_table_path is not None and (preliminary_table_path or strict_table_path):
        raise ValueError(
            "Use either --symmetry-screen-table OR the pair of "
            "--preliminary-screen-table and --strict-screen-table, not both."
        )

    if combined_table_path is not None:
        combined_path = _resolve_input_path(project_root, combined_table_path)
        if not combined_path.is_file():
            raise FileNotFoundError(f"Symmetry screening table not found: {combined_path}")
        combined = pd.read_csv(combined_path)
        preliminary_df, preliminary_filters = _filter_preliminary_candidates(combined)
        strict_df, strict_filter = _filter_strict_candidates(preliminary_df)
        if strict_filter.startswith("rows assumed"):
            raise KeyError(
                f"Combined symmetry table {combined_path.name} needs a strict spin-space "
                f"pass column (one of: {', '.join(_STRICT_PASS_COLUMNS)}), or provide "
                "separate preliminary and strict-stage CSV files."
            )
        preliminary_source = strict_source = combined_path
        preliminary_criterion = " AND ".join(preliminary_filters)
        strict_criterion = f"preliminary filter AND {strict_filter}"
    else:
        preliminary_path = _resolve_input_path(project_root, preliminary_table_path)
        strict_path = _resolve_input_path(project_root, strict_table_path)

        if (preliminary_path is None) != (strict_path is None):
            raise ValueError(
                "Provide both --preliminary-screen-table and --strict-screen-table."
            )

        if preliminary_path is None:
            combined_path = _first_existing(project_root, _DEFAULT_COMBINED_SCREEN_TABLES)
            if combined_path is not None:
                return _screening_funnel_benchmark(
                    project_root, results_dir, combined_table_path=combined_path
                )
            preliminary_path = _first_existing(project_root, _DEFAULT_PRELIMINARY_TABLES)
            strict_path = _first_existing(project_root, _DEFAULT_STRICT_TABLES)

        if preliminary_path is None or strict_path is None:
            for name in (
                "screening_funnel_counts.csv",
                "screening_funnel_task_ids.csv",
                "final_screened_candidates.csv",
            ):
                stale_path = results_dir / name
                if stale_path.exists():
                    stale_path.unlink()
            print(
                "\nScreening funnel not calculated: expected the current screening outputs "
                "results/new_preliminary_candidates.csv and "
                "results/new_candidates_spin_space_test.csv."
            )
            return None

        if not preliminary_path.is_file():
            raise FileNotFoundError(f"Preliminary screening table not found: {preliminary_path}")
        if not strict_path.is_file():
            raise FileNotFoundError(f"Strict spin-space screening table not found: {strict_path}")

        preliminary_raw = pd.read_csv(preliminary_path)
        strict_raw = pd.read_csv(strict_path)
        preliminary_df, preliminary_filters = _filter_preliminary_candidates(
            preliminary_raw
        )
        # The strict-stage file is a copy of the preliminary list plus strict
        # spin-space results. Apply preliminary flags if included, then strict flag.
        strict_df, _ = _filter_preliminary_candidates(strict_raw)
        strict_df, strict_pass_filter = _filter_strict_candidates(strict_df)
        preliminary_source, strict_source = preliminary_path, strict_path
        preliminary_criterion = " AND ".join(preliminary_filters)
        strict_criterion = strict_pass_filter

    preliminary_df = _canonical_task_table(preliminary_df, "preliminary magnetic-symmetry")
    strict_df = _canonical_task_table(strict_df, "strict spin-space symmetry")
    preliminary_ids = set(preliminary_df["task_id"])
    strict_ids = set(strict_df["task_id"])

    outside = strict_ids - preliminary_ids
    if outside:
        examples = sorted(outside)[:10]
        raise ValueError(
            "Strict spin-space Task IDs are not a subset of preliminary Task IDs "
            f"({len(outside)} unmatched; examples: {examples}). Check that both files "
            "come from the same screening run."
        )

    n_preliminary = len(preliminary_ids)
    n_strict = len(strict_ids)
    n_removed = n_preliminary - n_strict
    reduction_pct = (100.0 * n_removed / n_preliminary) if n_preliminary else 0.0
    retention_pct = (100.0 * n_strict / n_preliminary) if n_preliminary else 0.0

    # Export a row-level audit trail showing which Task IDs passed each stage.
    task_status = pd.DataFrame({"task_id": sorted(preliminary_ids)})
    task_status["preliminary_compensated_PT_broken"] = True
    task_status["strict_spin_space_pass"] = task_status["task_id"].isin(strict_ids)
    task_status["screening_outcome"] = np.where(
        task_status["strict_spin_space_pass"],
        "Final screened candidate",
        "Removed by strict spin-space symmetry",
    )
    for source_df in (preliminary_df, strict_df):
        detail_cols = [
            c for c in (
                "formula", "parent_structure_id", "ML_rank", "AM_type_preliminary",
                "AM_type", "compensated", "net_moment", "PT_preserved",
                "spacegroup", "n_spin_ops", "n_spin_flip_ops", "spin_space_AM",
                "ground_gap_meV_atom", "ground_gap_eV_atom", "dft_ordering",
            ) if c in source_df.columns and c not in task_status.columns
        ]
        if detail_cols:
            lookup = source_df[["task_id", *detail_cols]].drop_duplicates("task_id")
            task_status = task_status.merge(
                lookup, on="task_id", how="left", validate="one_to_one"
            )

    task_status.to_csv(results_dir / "screening_funnel_task_ids.csv", index=False)

    # Prefer the screening pipeline's richer final output, but verify that it
    # exactly matches the Task IDs that passed spin-space symmetry.
    final_candidates = None
    final_source = None
    for rel in (
        "results/screened_new_candidates.csv",
        "results/final_candidate_table.csv",
    ):
        candidate_path = project_root / rel
        if not candidate_path.is_file():
            continue
        candidate_df = _canonical_task_table(
            pd.read_csv(candidate_path), "final screened candidate"
        )
        candidate_ids = set(candidate_df["task_id"])
        if candidate_ids == strict_ids:
            final_candidates = candidate_df
            final_source = candidate_path
            break

    if final_candidates is None:
        # Reconstruct the final table directly from the strict-stage rows and
        # attach useful preliminary information not present in that file.
        final_candidates = strict_df.copy()
        add_cols = [
            c for c in preliminary_df.columns
            if c != "task_id" and c not in final_candidates.columns
        ]
        if add_cols:
            final_candidates = final_candidates.merge(
                preliminary_df[["task_id", *add_cols]],
                on="task_id", how="left", validate="one_to_one"
            )
        final_source = strict_source

    final_candidates = final_candidates.sort_values(
        [c for c in ("ML_rank", "task_id") if c in final_candidates.columns],
        kind="mergesort",
    ).reset_index(drop=True)
    final_candidates.to_csv(results_dir / "final_screened_candidates.csv", index=False)

    funnel = pd.DataFrame([
        {
            "screening_stage": "Preliminary magnetic symmetry",
            "criterion": "compensated AND PT broken",
            "task_id_count": n_preliminary,
            "task_ids": ";".join(sorted(preliminary_ids)),
            "task_ids_removed_at_stage": 0,
            "reduction_from_preliminary_pct": 0.0 if n_preliminary else np.nan,
            "retention_from_preliminary_pct": 100.0 if n_preliminary else 0.0,
            "candidate_rows": int(len(preliminary_df)),
            "source_file": str(preliminary_source),
            "applied_filter": preliminary_criterion,
        },
        {
            "screening_stage": "Strict spin-space symmetry",
            "criterion": "preliminary candidates passing strict spin-space symmetry",
            "task_id_count": n_strict,
            "task_ids": ";".join(sorted(strict_ids)),
            "task_ids_removed_at_stage": n_removed,
            "reduction_from_preliminary_pct": reduction_pct,
            "retention_from_preliminary_pct": retention_pct,
            "candidate_rows": int(len(strict_df)),
            "source_file": str(strict_source),
            "applied_filter": strict_criterion,
        },
    ])
    funnel.to_csv(results_dir / "screening_funnel_counts.csv", index=False)

    summary = {
        "preliminary_task_id_count": n_preliminary,
        "strict_spin_space_task_id_count": n_strict,
        "removed_task_id_count": n_removed,
        "reduction_from_preliminary_pct": reduction_pct,
        "final_screened_candidates": str(results_dir / "final_screened_candidates.csv"),
        "final_screened_candidates_source": str(final_source),
        "task_id_audit_table": str(results_dir / "screening_funnel_task_ids.csv"),
        "count_table": str(results_dir / "screening_funnel_counts.csv"),
    }
    print("\nScreening funnel (unique Task IDs):")
    print(f"Preliminary (compensated + PT broken): {n_preliminary}")
    print(f"After strict spin-space symmetry:       {n_strict}")
    print(f"Removed by strict analysis:             {n_removed} ({reduction_pct:.2f}%)")
    print("\nFinal screened candidates:")
    display_cols = [
        c for c in (
            "task_id", "formula", "AM_type_preliminary", "AM_type", "ML_rank",
            "spacegroup", "n_spin_ops", "n_spin_flip_ops", "ground_gap_meV_atom",
        ) if c in final_candidates.columns
    ]
    print(final_candidates[display_cols].to_string(index=False))
    print("\nSaved screening outputs:")
    print(f"  Counts:       {results_dir / 'screening_funnel_counts.csv'}")
    print(f"  Task ID list: {results_dir / 'screening_funnel_task_ids.csv'}")
    print(f"  Final table:  {results_dir / 'final_screened_candidates.csv'}")
    return summary


def _write_unlabelled_top5_table(spin_ranked: pd.DataFrame, results_dir: Path) -> None:
    """Export Top-5 options per test parent without any historical AM labels."""
    columns = [
        column for column in (
            "parent_structure_id", "task_id", "formula", "ordering",
            "ML_rank", "pred_energy", "delta_E",
        ) if column in spin_ranked.columns
    ]
    top5 = spin_ranked.loc[spin_ranked["ML_rank"] <= 5, columns].copy()
    top5.sort_values(["parent_structure_id", "ML_rank"]).to_csv(
        results_dir / "top5_screen_table_all_test_parents.csv", index=False
    )

def run_benchmarks(
    project_root: Path,
    preliminary_screen_table_path: Path | None = None,
    strict_screen_table_path: Path | None = None,
    symmetry_screen_table_path: Path | None = None,
) -> dict:
    """Run model/DFT benchmarks and count the current symmetry-screening funnel."""
    project_root = Path(project_root).resolve()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Remove outputs from the former AM-reference recovery workflow. The current
    # benchmark deliberately does not use the notebook's historical 55-state
    # reference or its earlier 22 screened materials.
    deprecated_outputs = (
        "AM_DFT_saving_benchmark.csv",
        "TopK_AM_recovery_and_DFT_saving.csv",
        "TopK_all_test_vs_AM_comparison.csv",
        "AM_ground_state_rankings.csv",
        "AM_subtype_TopK_recovery.csv",
        "top5_recovered_AM_ground_states.csv",
        "AM_ground_states_missed_by_Top5.csv",
        "AM_ground_states_rescued_by_Top10.csv",
        "AM_ground_states_still_missed_by_Top10.csv",
        "Top10_missed_AM_error_analysis.csv",
    )
    for filename in deprecated_outputs:
        stale_path = results_dir / filename
        if stale_path.exists():
            stale_path.unlink()

    test_df = _load_test_frame(project_root)
    model_frames = {
        "structure_only": _attach_predictions(
            test_df, results_dir / "test_predictions_baseline.csv", "structure-only"
        ),
        "spin_aware": _attach_predictions(
            test_df, results_dir / "test_predictions.csv", "spin-aware"
        ),
    }

    regression_rows = []
    topk_frames = []
    regret_frames = []
    regret_summaries = []
    rank_frames = {}
    gap_frames = []
    gap_group_frames = []
    order_frames = []
    pair_summaries = []
    pair_gap_frames = []
    ground_gap_summaries = []

    for model_name, df in model_frames.items():
        regression_rows.append(_regression_metrics(df, model_name))
        ranked, parent_ranks = _rank_configurations(df)
        rank_frames[model_name] = ranked
        parent_ranks.insert(0, "model", model_name)
        parent_ranks.to_csv(
            results_dir / f"parent_ground_state_ranks_{model_name}.csv", index=False
        )
        topk_frames.append(_topk_benchmark(parent_ranks, model_name))
        regret_df, regret_summary = _top1_regret(df, ranked, model_name)
        regret_frames.append(regret_df)
        regret_summaries.append(regret_summary)
        # The original notebook's Phase 4.4-4.6 diagnostics are for the
        # spin-aware model. The structure-only model is the Top-K/regret control.
        if model_name == "spin_aware":
            gap_df, gap_group_df = _ground_state_gap_tables(df, ranked, model_name)
            gap_frames.append(gap_df)
            if not gap_group_df.empty:
                gap_group_frames.append(gap_group_df)
            ground_gap_summaries.append(
                _ground_state_gap_summary(gap_df, model_name)
            )
            order_frames.append(_ordering_change_metrics(df, model_name))
            pair_summary, pair_gap_df = _pairwise_ranking(df, model_name)
            pair_summaries.append(pair_summary)
            pair_gap_frames.append(pair_gap_df)

    pd.DataFrame(regression_rows).to_csv(
        results_dir / "model_test_regression_metrics.csv", index=False
    )
    all_topk = pd.concat(topk_frames, ignore_index=True)
    all_topk.to_csv(results_dir / "all_test_ground_state_TopK_DFT_benchmark.csv", index=False)

    # Also export one easy-to-read table per model. Both models use the same
    # Top-K evaluation rule and therefore the same DFT workload at each K;
    # their ground-state recovery rates can differ.
    for model_name in ("structure_only", "spin_aware"):
        model_table = all_topk.loc[all_topk["model"] == model_name].copy()
        model_table.to_csv(
            results_dir / f"{model_name}_TopK_DFT_benchmark.csv", index=False
        )

    comparison = all_topk.pivot(
        index="Top_K",
        columns="model",
        values=[
            "ground_state_recovery_pct",
            "avg_DFT_calculations_per_parent",
            "avg_full_DFT_calculations_per_parent",
            "DFT_saving_pct",
        ],
    )
    comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]
    comparison = comparison.reset_index()
    comparison["spin_aware_minus_structure_only_recovery_pp"] = (
        comparison["ground_state_recovery_pct_spin_aware"]
        - comparison["ground_state_recovery_pct_structure_only"]
    )
    comparison["spin_aware_minus_structure_only_DFT_saving_pp"] = (
        comparison["DFT_saving_pct_spin_aware"]
        - comparison["DFT_saving_pct_structure_only"]
    )
    comparison.to_csv(
        results_dir / "TopK_structure_only_vs_spin_aware_comparison.csv", index=False
    )
    pd.concat(regret_frames, ignore_index=True).to_csv(
        results_dir / "parent_Top1_energy_regret.csv", index=False
    )
    pd.DataFrame(regret_summaries).to_csv(
        results_dir / "Top1_energy_regret_summary.csv", index=False
    )
    if gap_frames:
        pd.concat(gap_frames, ignore_index=True).to_csv(
            results_dir / "ground_state_energy_gaps.csv", index=False
        )
    if ground_gap_summaries:
        pd.DataFrame(ground_gap_summaries).to_csv(
            results_dir / "ground_state_gap_summary.csv", index=False
        )
    if gap_group_frames:
        pd.concat(gap_group_frames, ignore_index=True).to_csv(
            results_dir / "Top1_accuracy_by_ground_state_gap.csv", index=False
        )
    if order_frames:
        pd.concat(order_frames, ignore_index=True).to_csv(
            results_dir / "spin_aware_metrics_by_ordering_changed.csv", index=False
        )
    if pair_summaries:
        pd.DataFrame(pair_summaries).to_csv(
            results_dir / "pairwise_ranking_accuracy.csv", index=False
        )
    if pair_gap_frames:
        pd.concat(pair_gap_frames, ignore_index=True).to_csv(
            results_dir / "pairwise_ranking_accuracy_by_energy_separation.csv", index=False
        )

    spin_ranked = rank_frames["spin_aware"]
    _write_unlabelled_top5_table(spin_ranked, results_dir)
    screening_funnel = _screening_funnel_benchmark(
        project_root,
        results_dir,
        preliminary_table_path=preliminary_screen_table_path,
        strict_table_path=strict_screen_table_path,
        combined_table_path=symmetry_screen_table_path,
    )

    summary = {
        "test_parents": int(test_df["parent_structure_id"].nunique()),
        "test_configurations": int(len(test_df)),
        "benchmark_models": ["structure_only", "spin_aware"],
        "advanced_phase4_diagnostics": "spin_aware only (matches source notebook)",
        "screening_funnel": screening_funnel,
        "outputs_directory": str(results_dir),
    }
    print("Benchmark summary:")
    print(pd.DataFrame(regression_rows).to_string(index=False))
    for model_name, label in (
        ("structure_only", "Structure-only baseline"),
        ("spin_aware", "Spin-aware model"),
    ):
        print(f"\nTop-K DFT benchmark ({label}):")
        print(
            all_topk.loc[all_topk["model"] == model_name]
            .drop(columns="model").to_string(index=False)
        )
    print("\nSaved benchmark tables under:", results_dir)
    return summary
