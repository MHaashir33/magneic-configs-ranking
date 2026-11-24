#!/usr/bin/env python3
"""Generate publication-quality figures from the ALIGNN altermagnet project outputs.

Place this file at ``scripts/07_generate_figures.py`` and run from the repository
root with ``python scripts/07_generate_figures.py``.

The script only reads existing CSV/JSON outputs. It does not retrain models,
change predictions, or modify benchmark tables. Missing optional inputs are
reported and their figures are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "matplotlib is required to generate figures. Install it with:\n"
        "  python -m pip install matplotlib"
    ) from exc


# Consistent model colors across all figures.
COLORS = {
    "structure_only": "#2878B5",
    "spin_aware": "#E07A28",
    "neutral": "#555555",
    "accent": "#4C956C",
}
MODEL_LABELS = {
    "structure_only": "Structure-only baseline",
    "spin_aware": "Spin-aware ALIGNN",
}


def configure_plot_style() -> None:
    """Set a clean, print-friendly Matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_csv(results_dir: Path, filename: str, required: Iterable[str] = ()) -> pd.DataFrame | None:
    path = results_dir / filename
    if not path.is_file():
        print(f"[SKIP] {filename}: not found")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[SKIP] {filename}: could not read ({exc})")
        return None
    missing = sorted(set(required) - set(df.columns))
    if missing:
        print(f"[SKIP] {filename}: missing columns {missing}")
        return None
    return df


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=600 if fmt == "png" else None)
        saved.append(path)
    plt.close(fig)
    print(f"[OK] {stem}: " + ", ".join(p.name for p in saved))
    return saved


def model_color(model: str) -> str:
    return COLORS.get(str(model), COLORS["neutral"])


def pretty_model(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model).replace("_", " ").title())


def plot_training_history(results: Path, out: Path, formats: list[str], metric: str) -> bool:
    files = {
        "structure_only": "baseline_training_history.csv",
        "spin_aware": "spin_aware_training_history.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for model, filename in files.items():
        df = read_csv(results, filename, ["epoch", f"train_{metric}", f"val_{metric}"])
        if df is not None and not df.empty:
            frames[model] = df
    if not frames:
        return False

    fig, axes = plt.subplots(1, len(frames), figsize=(6.2 * len(frames), 4.4), squeeze=False)
    for ax, (model, df) in zip(axes[0], frames.items()):
        x = pd.to_numeric(df["epoch"], errors="coerce")
        train = pd.to_numeric(df[f"train_{metric}"], errors="coerce") * 1000.0
        val = pd.to_numeric(df[f"val_{metric}"], errors="coerce") * 1000.0
        ax.plot(x, train, color=COLORS[model], linestyle="--", label="Training")
        ax.plot(x, val, color=COLORS[model], linestyle="-", label="Validation")
        if "best_checkpoint" in df.columns:
            best_mask = df["best_checkpoint"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            if best_mask.any():
                best_row = df.loc[best_mask].iloc[-1]
            else:
                best_row = df.loc[pd.to_numeric(df[f"val_{metric}"], errors="coerce").idxmin()]
        else:
            best_row = df.loc[pd.to_numeric(df[f"val_{metric}"], errors="coerce").idxmin()]
        best_epoch = float(best_row["epoch"])
        best_value = float(best_row[f"val_{metric}"]) * 1000.0
        ax.scatter([best_epoch], [best_value], color=COLORS[model], edgecolor="black", zorder=5, s=42,
                   label=f"Best validation (epoch {int(best_epoch)})")
        ax.axvline(best_epoch, color=COLORS[model], alpha=0.35, linestyle=":", linewidth=1.2)
        ax.set_title(pretty_model(model))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{metric.upper()} (meV/atom)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.legend(frameon=False)
    fig.suptitle(f"ALIGNN training history: {metric.upper()}", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out, f"training_history_{metric}", formats)
    return True


def plot_test_error_metrics(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "model_test_regression_metrics.csv",
                  ["model", "MAE_eV_atom", "RMSE_eV_atom"])
    if df is None or df.empty:
        return False
    models = [m for m in ("structure_only", "spin_aware") if m in set(df["model"].astype(str))]
    if not models:
        models = df["model"].astype(str).tolist()
    df = df.set_index("model").loc[models]
    x = np.arange(len(models))
    width = 0.34
    mae = df["MAE_eV_atom"].to_numpy(float) * 1000.0
    rmse = df["RMSE_eV_atom"].to_numpy(float) * 1000.0
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b1 = ax.bar(x - width / 2, mae, width, label="MAE", color="#2878B5")
    b2 = ax.bar(x + width / 2, rmse, width, label="RMSE", color="#E07A28")
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x, [pretty_model(m) for m in models])
    ax.set_ylabel("Test error (meV/atom)")
    ax.set_title("Test-set energy prediction error")
    ax.legend(frameon=False)
    ax.margins(y=0.18)
    fig.tight_layout()
    save_figure(fig, out, "test_set_mae_rmse_comparison", formats)
    return True


def plot_topk_recovery(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "all_test_ground_state_TopK_DFT_benchmark.csv",
                  ["model", "Top_K", "ground_state_recovery_pct"])
    if df is None or df.empty:
        return False
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plotted = False
    for model in ("structure_only", "spin_aware"):
        sub = df.loc[df["model"].astype(str) == model].sort_values("Top_K")
        if sub.empty:
            continue
        ax.plot(sub["Top_K"], sub["ground_state_recovery_pct"], marker="o",
                color=COLORS[model], label=pretty_model(model))
        plotted = True
    if not plotted:
        return False
    ax.set_xticks(sorted(df["Top_K"].dropna().unique()))
    ax.set_xlabel("Number of configurations selected (Top-K)")
    ax.set_ylabel("Ground-state recovery (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Top-K ground-state recovery")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "topk_ground_state_recovery", formats)
    return True


def plot_dft_savings(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "all_test_ground_state_TopK_DFT_benchmark.csv",
                  ["Top_K", "DFT_saving_pct"])
    if df is None or df.empty:
        # Support a single-model / comparison table from older benchmark exports.
        df = read_csv(results, "TopK_structure_only_vs_spin_aware_comparison.csv", ["Top_K"])
        if df is None:
            return False
        savings_cols = [c for c in df.columns if "DFT_saving_pct" in c]
        if not savings_cols:
            print("[SKIP] DFT-savings figure: no DFT_saving_pct column found")
            return False
        # Both models use the same selected DFT workload at a given K; use one curve.
        y = pd.to_numeric(df[savings_cols[0]], errors="coerce")
    else:
        # DFT workload is the same for both models at each K; don't plot duplicate curves.
        grouped = df.groupby("Top_K", as_index=False)["DFT_saving_pct"].mean()
        df = grouped
        y = pd.to_numeric(df["DFT_saving_pct"], errors="coerce")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(df["Top_K"], y, marker="s", color=COLORS["accent"], label="DFT calculations avoided")
    ax.set_xticks(sorted(pd.to_numeric(df["Top_K"], errors="coerce").dropna().unique()))
    ax.set_xlabel("Number of configurations selected (Top-K)")
    ax.set_ylabel("Estimated DFT saving (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Estimated DFT workload reduction")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "topk_estimated_dft_savings", formats)
    return True


def plot_top1_regret(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "parent_Top1_energy_regret.csv", ["model"])
    if df is None or df.empty:
        return False
    value_col = next((c for c in ("notebook_regret_eV_atom", "energy_regret_eV_atom") if c in df.columns), None)
    if value_col is None:
        print("[SKIP] energy-regret figure: no compatible regret column found")
        return False
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    arrays, labels, colors = [], [], []
    for model in ("structure_only", "spin_aware"):
        sub = pd.to_numeric(df.loc[df["model"].astype(str) == model, value_col], errors="coerce").dropna()
        if not sub.empty:
            arrays.append(sub.to_numpy() * 1000.0)
            labels.append(pretty_model(model))
            colors.append(COLORS[model])
    if not arrays:
        return False
    bp = ax.boxplot(arrays, patch_artist=True, showfliers=False)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("Top-1 energy regret (meV/atom; symlog scale)")
    ax.set_title("Distribution of Top-1 energy regret")
    ax.text(0.99, 0.98, "Outliers hidden for readability", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=COLORS["neutral"])
    fig.tight_layout()
    save_figure(fig, out, "top1_energy_regret_distribution", formats)
    return True


def plot_gap_accuracy(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "Top1_accuracy_by_ground_state_gap.csv", ["gap_group", "top1_accuracy_pct"])
    if df is None or df.empty:
        return False
    if "model" in df.columns:
        df = df.loc[df["model"].astype(str) == "spin_aware"]
    if df.empty:
        return False
    order = ["<1 meV", "1–5 meV", "5–10 meV", ">10 meV"]
    df = df.copy()
    df["_order"] = df["gap_group"].astype(str).map({v: i for i, v in enumerate(order)}).fillna(99)
    df = df.sort_values("_order")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.bar(df["gap_group"].astype(str), df["top1_accuracy_pct"], color=COLORS["spin_aware"], alpha=0.85)
    if "parents" in df.columns:
        labels = [f"{v:.1f}%\n(n={int(n)})" for v, n in zip(df["top1_accuracy_pct"], df["parents"])]
    else:
        labels = [f"{v:.1f}%" for v in df["top1_accuracy_pct"]]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_xlabel("DFT energy gap between ground and first excited state")
    ax.set_ylabel("Top-1 ground-state accuracy (%)")
    ax.set_ylim(0, max(100, float(df["top1_accuracy_pct"].max()) * 1.2))
    ax.set_title("Top-1 accuracy versus ground-state energy gap")
    fig.tight_layout()
    save_figure(fig, out, "top1_accuracy_by_ground_state_gap", formats)
    return True


def plot_pairwise_accuracy(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "pairwise_ranking_accuracy_by_energy_separation.csv",
                  ["energy_separation", "pairwise_accuracy_pct"])
    if df is None or df.empty:
        return False
    order = ["<1 meV", "1–5 meV", "5–10 meV", ">10 meV"]
    df = df.copy()
    df["_order"] = df["energy_separation"].astype(str).map({v: i for i, v in enumerate(order)}).fillna(99)
    df = df.sort_values("_order")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars = ax.bar(df["energy_separation"].astype(str), df["pairwise_accuracy_pct"], color=COLORS["spin_aware"], alpha=0.85)
    if "total_pairs" in df.columns:
        labels = [f"{v:.1f}%\n(n={int(n)})" for v, n in zip(df["pairwise_accuracy_pct"], df["total_pairs"])]
    else:
        labels = [f"{v:.1f}%" for v in df["pairwise_accuracy_pct"]]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_xlabel("True DFT energy separation between configuration pairs")
    ax.set_ylabel("Pairwise ranking accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pairwise energy-ranking accuracy by energy separation")
    fig.tight_layout()
    save_figure(fig, out, "pairwise_ranking_accuracy_by_energy_gap", formats)
    return True


def plot_ordering_changed(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "spin_aware_metrics_by_ordering_changed.csv",
                  ["ordering_changed", "MAE_eV_atom", "ground_state_accuracy_pct"])
    if df is None or df.empty:
        return False
    df = df.copy()
    df["condition"] = df["ordering_changed"].astype(str).str.lower().map({"true": "Ordering changed", "false": "Ordering unchanged", "1": "Ordering changed", "0": "Ordering unchanged"})
    df = df.dropna(subset=["condition"])
    if df.empty:
        return False
    order = ["Ordering unchanged", "Ordering changed"]
    df["_order"] = df["condition"].map({v: i for i, v in enumerate(order)})
    df = df.sort_values("_order")
    x = np.arange(len(df))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(7.4, 4.8))
    b1 = ax1.bar(x - width / 2, df["MAE_eV_atom"] * 1000.0, width, color="#2878B5", label="MAE")
    ax1.set_ylabel("MAE (meV/atom)")
    ax1.set_xticks(x, df["condition"])
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    b2 = ax2.bar(x + width / 2, df["ground_state_accuracy_pct"], width, color="#E07A28", label="Ground-state accuracy")
    ax2.set_ylabel("Ground-state accuracy (%)")
    ax2.set_ylim(0, 100)
    ax1.set_title("Spin-aware performance by ordering-change status")
    ax1.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
    ax2.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=8)
    ax1.legend([b1, b2], ["MAE (meV/atom)", "Ground-state accuracy (%)"], frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out, "spin_aware_metrics_by_ordering_change", formats)
    return True


def plot_prediction_scatter(results: Path, out: Path, formats: list[str]) -> bool:
    sources = {
        "structure_only": "test_predictions_baseline.csv",
        "spin_aware": "test_predictions.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for model, filename in sources.items():
        df = read_csv(results, filename, ["delta_E", "pred_energy"])
        if df is not None and not df.empty:
            clean = df[["delta_E", "pred_energy"]].apply(pd.to_numeric, errors="coerce").dropna()
            if not clean.empty:
                frames[model] = clean
    if not frames:
        return False
    fig, axes = plt.subplots(1, len(frames), figsize=(6.2 * len(frames), 5.2), squeeze=False)
    all_values = np.concatenate([np.r_[d["delta_E"].to_numpy(), d["pred_energy"].to_numpy()] for d in frames.values()]) * 1000.0
    low, high = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    pad = max((high - low) * 0.04, 1.0)
    low, high = low - pad, high + pad
    for ax, (model, df) in zip(axes[0], frames.items()):
        actual = df["delta_E"].to_numpy(float) * 1000.0
        predicted = df["pred_energy"].to_numpy(float) * 1000.0
        ax.scatter(actual, predicted, s=14, alpha=0.35, color=COLORS[model], edgecolors="none", rasterized=True)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.2, label="Ideal: predicted = DFT")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("DFT reference ΔE (meV/atom)")
        ax.set_ylabel("Predicted ΔE (meV/atom)")
        ax.set_title(pretty_model(model))
        mae = float(np.mean(np.abs(predicted - actual)))
        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        ax.text(0.04, 0.96, f"MAE = {mae:.2f} meV/atom\nRMSE = {rmse:.2f} meV/atom",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    fig.suptitle("Predicted versus DFT reference energies", y=1.01, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out, "predicted_vs_dft_energy_scatter", formats)
    return True


def _screening_counts(results: Path) -> tuple[int, int] | None:
    counts_path = results / "screening_funnel_counts.csv"
    if counts_path.is_file():
        try:
            df = pd.read_csv(counts_path)
            if {"screening_stage", "task_id_count"}.issubset(df.columns):
                preliminary = df.loc[df["screening_stage"].astype(str).str.contains("preliminary", case=False), "task_id_count"]
                strict = df.loc[df["screening_stage"].astype(str).str.contains("strict|spin.space", case=False, regex=True), "task_id_count"]
                if not preliminary.empty and not strict.empty:
                    return int(preliminary.iloc[0]), int(strict.iloc[0])
        except Exception as exc:
            print(f"[WARN] Could not parse screening_funnel_counts.csv: {exc}")
    summary_path = results / "screening_summary.json"
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            p = data.get("new_preliminary_candidates")
            s = data.get("new_candidates_passing_spin_space_test")
            if p is not None and s is not None:
                return int(p), int(s)
        except Exception as exc:
            print(f"[WARN] Could not parse screening_summary.json: {exc}")
    # Last-resort reconstruction from task-ID output tables.
    preliminary_path = results / "new_preliminary_candidates.csv"
    strict_path = results / "new_candidates_spin_space_test.csv"
    if preliminary_path.is_file() and strict_path.is_file():
        p_df, s_df = pd.read_csv(preliminary_path), pd.read_csv(strict_path)
        if "task_id" in p_df and "task_id" in s_df:
            if "spin_space_AM" in s_df:
                strict_ids = s_df.loc[s_df["spin_space_AM"].astype(str).str.lower().isin({"true", "1", "yes"}), "task_id"]
            else:
                strict_ids = s_df["task_id"]
            return int(p_df["task_id"].astype(str).nunique()), int(strict_ids.astype(str).nunique())
    return None


def plot_screening_funnel(results: Path, out: Path, formats: list[str]) -> bool:
    counts = _screening_counts(results)
    if counts is None:
        print("[SKIP] screening funnel: no screening_funnel_counts.csv, screening_summary.json, or compatible screening CSVs found")
        return False
    preliminary, strict = counts
    removed = preliminary - strict
    reduction_pct = 100.0 * removed / preliminary if preliminary else 0.0
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    labels = ["Preliminary magnetic\nsymmetry screen", "Strict spin-space\nsymmetry screen"]
    bars = ax.bar(labels, [preliminary, strict], color=["#2878B5", "#4C956C"], width=0.55)
    ax.bar_label(bars, labels=[str(preliminary), str(strict)], padding=4, fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of unique Task IDs")
    ax.set_title("Candidate reduction through symmetry screening")
    ax.set_ylim(0, max(preliminary, strict, 1) * 1.28)
    ax.text(0.5, 0.91, f"Removed by strict analysis: {removed} ({reduction_pct:.1f}%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.8"})
    fig.tight_layout()
    save_figure(fig, out, "screening_funnel_task_id_counts", formats)
    return True


def plot_final_candidate_subtypes(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "final_screened_candidates.csv", ["task_id"])
    if df is None:
        df = read_csv(results, "final_candidate_table.csv", ["task_id"])
    if df is None or df.empty:
        return False
    subtype_col = next((c for c in ("AM_type_preliminary", "AM_type") if c in df.columns), None)
    if subtype_col is None:
        print("[SKIP] final-candidate subtype figure: no AM_type_preliminary or AM_type column")
        return False
    counts = df.assign(_subtype=df[subtype_col].fillna("Unclassified").astype(str)).groupby("_subtype")["task_id"].nunique()
    counts = counts.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(counts.index, counts.values, color=["#2878B5", "#E07A28", "#888888"][:len(counts)])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Number of unique final candidate Task IDs")
    ax.set_title(f"Final screened candidates by preliminary subtype (n={df['task_id'].astype(str).nunique()})")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(counts.values) * 1.25 if len(counts) else 1)
    fig.tight_layout()
    save_figure(fig, out, "final_screened_candidates_by_subtype", formats)
    return True


def plot_candidate_gap_distribution(results: Path, out: Path, formats: list[str]) -> bool:
    df = read_csv(results, "final_screened_candidates.csv", ["task_id"])
    if df is None:
        df = read_csv(results, "final_candidate_table.csv", ["task_id"])
    if df is None or df.empty:
        return False
    gap_col = next((c for c in ("ground_gap_meV_atom", "ground_gap_eV_atom") if c in df.columns), None)
    if gap_col is None:
        print("[SKIP] final-candidate gap figure: no ground_gap_meV_atom or ground_gap_eV_atom column")
        return False
    gaps = pd.to_numeric(df[gap_col], errors="coerce").dropna()
    if gaps.empty:
        return False
    if gap_col == "ground_gap_eV_atom":
        gaps = gaps * 1000.0
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.hist(gaps, bins="auto", color=COLORS["accent"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("DFT ground-state gap (meV/atom)")
    ax.set_ylabel("Number of final candidates")
    ax.set_title("Ground-state energy-gap distribution of final candidates")
    fig.tight_layout()
    save_figure(fig, out, "final_candidate_ground_state_gap_distribution", formats)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from project results.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Results directory (default: <repository root>/results).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Figure output directory (default: <results-dir>/figures).")
    parser.add_argument("--formats", default="png,pdf,svg",
                        help="Comma-separated formats to save (default: png,pdf,svg).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = (args.results_dir or (project_root / "results")).resolve()
    output_dir = (args.output_dir or (results_dir / "figures")).resolve()
    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    allowed = {"png", "pdf", "svg"}
    invalid = sorted(set(formats) - allowed)
    if invalid or not formats:
        parser.error(f"Unsupported or empty format list: {invalid}. Allowed: {sorted(allowed)}")
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    configure_plot_style()
    print(f"Reading result tables from: {results_dir}")
    print(f"Saving figures to: {output_dir}")

    generators = [
        ("Training MAE", lambda: plot_training_history(results_dir, output_dir, formats, "mae")),
        ("Training RMSE", lambda: plot_training_history(results_dir, output_dir, formats, "rmse")),
        ("Test MAE/RMSE", lambda: plot_test_error_metrics(results_dir, output_dir, formats)),
        ("Top-K recovery", lambda: plot_topk_recovery(results_dir, output_dir, formats)),
        ("DFT savings", lambda: plot_dft_savings(results_dir, output_dir, formats)),
        ("Energy regret", lambda: plot_top1_regret(results_dir, output_dir, formats)),
        ("Gap versus accuracy", lambda: plot_gap_accuracy(results_dir, output_dir, formats)),
        ("Pairwise ranking", lambda: plot_pairwise_accuracy(results_dir, output_dir, formats)),
        ("Ordering-change diagnostics", lambda: plot_ordering_changed(results_dir, output_dir, formats)),
        ("Predicted versus DFT scatter", lambda: plot_prediction_scatter(results_dir, output_dir, formats)),
        ("Screening funnel", lambda: plot_screening_funnel(results_dir, output_dir, formats)),
        ("Final-candidate subtypes", lambda: plot_final_candidate_subtypes(results_dir, output_dir, formats)),
        ("Final-candidate energy gaps", lambda: plot_candidate_gap_distribution(results_dir, output_dir, formats)),
    ]
    made = 0
    for label, function in generators:
        try:
            made += int(bool(function()))
        except Exception as exc:  # keep one unavailable/odd table from blocking all other figures
            print(f"[WARN] {label} figure failed: {type(exc).__name__}: {exc}")

    print(f"\nFinished: generated {made} figure(s).")
    if made:
        print(f"Figure directory: {output_dir}")
        print("Each figure is saved as: " + ", ".join(formats))
    else:
        print("No figures were generated. Check the required results CSV files above.")


if __name__ == "__main__":
    main()
