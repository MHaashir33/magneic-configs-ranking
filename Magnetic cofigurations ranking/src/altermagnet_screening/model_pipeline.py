from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from alignn.pure_lmdb_dataset import (
    PureTorchLMDBDataset,
    batch_torch_graph_pairs,
)
from alignn.models.alignn_atomwise_pure import (
    ALIGNNAtomWisePure,
    ALIGNNAtomWisePureConfig,
)


class SpinAwareDataset(Dataset):
    """Notebook wrapper adding the per-site initial magnetic moment."""

    def __init__(self, lmdb_dataset, dataframe: pd.DataFrame):
        self.lmdb_dataset = lmdb_dataset
        self.df = dataframe.reset_index(drop=True).copy()
        if len(self.lmdb_dataset) != len(self.df):
            raise AssertionError(
                "LMDB order/length does not match the split dataframe."
            )

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, idx):
        g, lg, lattice, label = self.lmdb_dataset[idx]
        magmoms = torch.tensor(
            self.df.iloc[idx]["initial_magmoms"],
            dtype=torch.float32,
        ).reshape(-1, 1)
        if magmoms.shape[0] != g.num_nodes:
            raise ValueError(
                f"Index {idx}: {g.num_nodes} atoms but "
                f"{magmoms.shape[0]} initial moments."
            )
        g.ndata["magmom"] = magmoms
        return g, lg, lattice, label


def collate_alignn(batch):
    g, lg = batch_torch_graph_pairs(
        [(item[0], item[1]) for item in batch]
    )
    targets = torch.stack(
        [torch.as_tensor(item[3], dtype=torch.float32) for item in batch]
    )
    return g, lg, targets


def collate_spin_alignn(batch):
    g, lg = batch_torch_graph_pairs(
        [(item[0], item[1]) for item in batch]
    )
    # Same 92 chemical features + 1 initial magnetic moment used in the notebook.
    g.ndata["atom_features"] = torch.cat(
        [g.ndata["atom_features"], g.ndata["magmom"]],
        dim=1,
    )
    targets = torch.stack(
        [torch.as_tensor(item[3], dtype=torch.float32) for item in batch]
    )
    return g, lg, targets


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def model_config(atom_input_features: int):
    return ALIGNNAtomWisePureConfig(
        name="alignn_atomwise_pure",
        alignn_layers=2,
        gcn_layers=2,
        atom_input_features=atom_input_features,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=64,
        output_features=1,
        calculate_gradient=False,
        atomwise_output_features=0,
        graphwise_weight=1.0,
        gradwise_weight=0.0,
        stresswise_weight=0.0,
        atomwise_weight=0.0,
        energy_mult_natoms=False,
        use_penalty=False,
        classification=False,
    )


def _run_epoch(model, loader, device, optimizer=None, grad_clip_norm=5.0):
    training = optimizer is not None
    model.train(training)
    mse = nn.MSELoss(reduction="sum")
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for g, lg, target in loader:
            g = g.to(device)
            lg = lg.to(device)
            target = target.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            output = model((g, lg, None))["out"].reshape(-1)
            loss_sum = mse(output, target)

            if training:
                (loss_sum / target.numel()).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
                optimizer.step()

            n = int(target.numel())
            total_squared_error += float(loss_sum.detach().item())
            total_absolute_error += float(
                torch.abs(output.detach() - target).sum().item()
            )
            total_count += n

    if total_count == 0:
        raise RuntimeError("Encountered an empty dataloader.")
    mse_value = total_squared_error / total_count
    return {
        "mse": mse_value,
        "mae": total_absolute_error / total_count,
        "rmse": mse_value ** 0.5,
        "n": total_count,
    }


def _train_one_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    checkpoint_path: Path,
    history_path: Path,
    seed: int = 42,
):
    if epochs < 50:
        raise ValueError("The requested epoch count must be at least 50.")

    seed_everything(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_mae = float("inf")
    history = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, device, optimizer=optimizer,
            grad_clip_norm=5.0,
        )
        val_metrics = _run_epoch(
            model, val_loader, device, optimizer=None,
            grad_clip_norm=5.0,
        )
        scheduler.step(val_metrics["mae"])

        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": best_val_mae,
                    "model_name": "alignn_atomwise_pure",
                    "seed": seed,
                    "epochs_requested": epochs,
                },
                checkpoint_path,
            )

        row = {
            "epoch": epoch,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_checkpoint": bool(is_best),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MAE: {train_metrics['mae']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.6f} | "
            f"Val RMSE: {val_metrics['rmse']:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            f"{'  <-- BEST' if is_best else ''}"
        )

    pd.DataFrame(history).to_csv(history_path, index=False)
    return {
        "best_validation_mae": best_val_mae,
        "best_checkpoint": str(checkpoint_path),
        "best_epoch": min(history, key=lambda x: x["val_mae"])["epoch"],
        "history_csv": str(history_path),
    }


def _load_split_datasets(project_root: Path):
    from alignn.pure_lmdb_dataset import PureTorchLMDBDataset

    processed = project_root / "data" / "processed"
    import pickle
    with (processed / "graph_df.pkl").open("rb") as handle:
        graph_df = pickle.load(handle)

    cache = project_root / "data" / "graph_cache"
    base_datasets = {}
    spin_datasets = {}
    split_frames = {}

    for split in ["train", "val", "test"]:
        frame = (
            graph_df[graph_df["split"] == split]
            .copy()
            .reset_index(drop=True)
        )
        lmdb_ds = PureTorchLMDBDataset(
            lmdb_path=str(cache / split),
            line_graph=True,
        )
        if len(lmdb_ds) != len(frame):
            raise AssertionError(
                f"{split}: LMDB has {len(lmdb_ds)} entries but dataframe "
                f"has {len(frame)}."
            )
        base_datasets[split] = lmdb_ds
        spin_datasets[split] = SpinAwareDataset(lmdb_ds, frame)
        split_frames[split] = frame

    return graph_df, base_datasets, spin_datasets, split_frames


def train_models(project_root: Path, epochs: int = 50, batch_size: int = 32):
    """Train the notebook's structure-only control and spin-aware ALIGNN."""
    if epochs < 50:
        raise ValueError("epochs must be >= 50.")

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    graph_df, base_ds, spin_ds, frames = _load_split_datasets(project_root)

    train_loader = DataLoader(
        base_ds["train"], batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_alignn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        base_ds["val"], batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_alignn,
        pin_memory=torch.cuda.is_available(),
    )
    spin_train_loader = DataLoader(
        spin_ds["train"], batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_spin_alignn,
        pin_memory=torch.cuda.is_available(),
    )
    spin_val_loader = DataLoader(
        spin_ds["val"], batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_spin_alignn,
        pin_memory=torch.cuda.is_available(),
    )

    # Feature dimensions are checked against the notebook's 92/93 inputs.
    sample_g, _, _, _ = base_ds["train"][0]
    chemical_dim = int(sample_g.ndata["atom_features"].shape[1])
    if chemical_dim != 92:
        raise AssertionError(
            f"Expected 92 CGCNN chemical features, found {chemical_dim}."
        )

    models_dir = project_root / "models"
    results_dir = project_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\nTraining structure-only baseline (92 input features)")
    seed_everything(42)
    baseline_model = ALIGNNAtomWisePure(model_config(92)).to(device)
    baseline_result = _train_one_model(
        baseline_model, train_loader, val_loader, device, epochs,
        models_dir / "alignn_baseline_best.pt",
        results_dir / "baseline_training_history.csv",
        seed=42,
    )

    print("\nTraining spin-aware model (92 chemical + 1 magnetic feature)")
    seed_everything(42)
    spin_model = ALIGNNAtomWisePure(model_config(93)).to(device)
    spin_result = _train_one_model(
        spin_model, spin_train_loader, spin_val_loader, device, epochs,
        models_dir / "alignn_spin_aware_best.pt",
        results_dir / "spin_aware_training_history.csv",
        seed=42,
    )

    # Load each best checkpoint and evaluate exactly once on the held-out test split.
    test_base_loader = DataLoader(
        base_ds["test"], batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_alignn,
        pin_memory=torch.cuda.is_available(),
    )
    test_spin_loader = DataLoader(
        spin_ds["test"], batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_spin_alignn,
        pin_memory=torch.cuda.is_available(),
    )

    baseline_model = ALIGNNAtomWisePure(model_config(92)).to(device)
    baseline_checkpoint = torch.load(
        models_dir / "alignn_baseline_best.pt", map_location=device
    )
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
    baseline_test = _run_epoch(
        baseline_model, test_base_loader, device, optimizer=None
    )

    # Export structure-only baseline predictions as well as its test metrics.
    # The original notebook compares both models in Phase 4, so the baseline
    # predictions are required by the reproducible benchmark scripts.
    baseline_model.eval()
    baseline_predictions = []
    with torch.no_grad():
        for g, lg, target in test_base_loader:
            g = g.to(device)
            lg = lg.to(device)
            prediction = baseline_model((g, lg, None))["out"].reshape(-1)
            baseline_predictions.extend(
                prediction.detach().cpu().numpy().tolist()
            )

    test_frame = frames["test"][
        [
            "task_id",
            "parent_structure_id",
            "formula",
            "delta_E",
            "energy_per_atom",
            "ordering",
            "ordering_changed",
        ]
    ].copy().reset_index(drop=True)

    if len(baseline_predictions) != len(test_frame):
        raise AssertionError(
            "Structure-only predictions do not align with test rows."
        )
    baseline_frame = test_frame.copy()
    baseline_frame["pred_energy"] = np.asarray(
        baseline_predictions, dtype=float
    )
    baseline_frame.to_csv(
        results_dir / "test_predictions_baseline.csv", index=False
    )

    spin_model = ALIGNNAtomWisePure(model_config(93)).to(device)
    spin_checkpoint = torch.load(
        models_dir / "alignn_spin_aware_best.pt", map_location=device
    )
    spin_model.load_state_dict(spin_checkpoint["model_state_dict"])
    spin_test = _run_epoch(
        spin_model, test_spin_loader, device, optimizer=None
    )

    spin_model.eval()
    spin_predictions = []
    with torch.no_grad():
        for g, lg, target in test_spin_loader:
            g = g.to(device)
            lg = lg.to(device)
            prediction = spin_model((g, lg, None))["out"].reshape(-1)
            spin_predictions.extend(
                prediction.detach().cpu().numpy().tolist()
            )

    if len(spin_predictions) != len(test_frame):
        raise AssertionError("Spin-aware predictions do not align with test rows.")
    test_frame["pred_energy"] = np.asarray(spin_predictions, dtype=float)
    test_frame.to_csv(results_dir / "test_predictions.csv", index=False)

    metrics = {
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "seed": 42,
        "device": str(device),
        "baseline": {
            **baseline_result,
            "test_mae_eV_atom": baseline_test["mae"],
            "test_rmse_eV_atom": baseline_test["rmse"],
        },
        "spin_aware": {
            **spin_result,
            "test_mae_eV_atom": spin_test["mae"],
            "test_rmse_eV_atom": spin_test["rmse"],
        },
    }
    (results_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\nTest results:")
    print("Baseline MAE/RMSE:", baseline_test["mae"], baseline_test["rmse"])
    print("Spin-aware MAE/RMSE:", spin_test["mae"], spin_test["rmse"])
    print("Predictions:", results_dir / "test_predictions.csv")
    return metrics
