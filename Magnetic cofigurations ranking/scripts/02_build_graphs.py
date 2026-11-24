from __future__ import annotations
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def make_alignn_record(row):
    structure = row["input_structure"]
    return {
        "jid": str(row["task_id"]),
        "atoms": {
            "lattice_mat": structure["lattice"]["matrix"],
            "coords": [site["abc"] for site in structure["sites"]],
            "elements": [site["species"][0]["element"] for site in structure["sites"]],
            "props": [site.get("properties", {}) for site in structure["sites"]],
            "cartesian": False,
        },
        "target": float(row["delta_E"]),
    }


def main():
    try:
        from alignn.pure_lmdb_dataset import get_torch_dataset, PureTorchLMDBDataset
    except ImportError as exc:
        raise SystemExit(
            "Could not import alignn.pure_lmdb_dataset. Install the pinned ALIGNN fork "
            "from requirements-alignn.txt before building graph caches."
        ) from exc

    processed = ROOT / "data" / "processed"
    with (processed / "graph_df.pkl").open("rb") as handle:
        graph_df = pickle.load(handle)
    cache_root = ROOT / "data" / "graph_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        frame = graph_df.loc[graph_df["split"] == split].reset_index(drop=True)
        records = [make_alignn_record(row) for row in frame.to_dict(orient="records")]
        if not records:
            raise RuntimeError(f"No {split} records were prepared.")
        lmdb_path = cache_root / split
        # Rebuild cleanly: LMDB cannot safely be re-created over a stale cache.
        if lmdb_path.is_dir():
            shutil.rmtree(lmdb_path)
        elif lmdb_path.exists():
            lmdb_path.unlink()
        print(f"Building {split}: {len(records)} configurations -> {lmdb_path}")
        get_torch_dataset(
            dataset=records,
            id_tag="jid",
            target="target",
            neighbor_strategy="pure_torch",
            atom_features="cgcnn",
            line_graph=True,
            cutoff=8.0,
            max_neighbors=12,
            output_dir=str(cache_root),
            tmp_name=str(lmdb_path),
            read_existing=False,
        )
        check = PureTorchLMDBDataset(lmdb_path=str(lmdb_path), line_graph=True)
        if len(check) != len(frame):
            raise AssertionError(
                f"{split} cache length {len(check)} != dataframe length {len(frame)}"
            )
        print(f"Verified {split}: {len(check)} graph pairs")
    print("All three pure-PyTorch ALIGNN graph caches are ready.")


if __name__ == "__main__":
    main()
