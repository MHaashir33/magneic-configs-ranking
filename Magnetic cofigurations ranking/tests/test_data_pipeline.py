from __future__ import annotations

from altermagnet_screening.data_pipeline import (
    build_clean_dataset,
    input_magnetic_signature,
    parent_level_split,
    structure_signature,
)


def _site(element, abc, magmom):
    return {
        "species": [{"element": element}],
        "abc": abc,
        "properties": {"magmom": magmom},
    }


def _record(task_id, parent_lattice=None, input_lattice=None, moments=(1.0, -1.0), energy=0.0):
    lattice = parent_lattice or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    in_lattice = input_lattice or lattice
    sites = [_site("Fe", [0.0, 0.0, 0.0], moments[0]), _site("O", [0.5, 0.5, 0.5], moments[1])]
    return {
        "task_id": task_id,
        "formula_pretty": "FeO",
        "parent_structure": {"lattice": {"matrix": lattice}, "sites": sites},
        "input": {"structure": {"lattice": {"matrix": in_lattice}, "sites": sites}},
        "energy_per_atom": -1.0 + energy,
        "energy_above_ground_state_per_atom": energy,
        "ordering": "AFM",
        "ordering_changed": False,
    }


def test_parent_signature_ignores_site_order():
    a = {"lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "sites": [
        _site("Fe", [0, 0, 0], 1), _site("O", [0.5, 0.5, 0.5], -1)
    ]}
    b = {"lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "sites": [
        _site("O", [0.5, 0.5, 0.5], -1), _site("Fe", [0, 0, 0], 1)
    ]}
    assert structure_signature(a) == structure_signature(b)


def test_input_signature_includes_initial_moments():
    rec_a = _record(1, moments=(1.0, -1.0))
    rec_b = _record(2, moments=(-1.0, 1.0))
    assert input_magnetic_signature(rec_a) != input_magnetic_signature(rec_b)


def test_duplicate_initial_states_are_collapsed():
    records = [_record(1, energy=0.0), _record(2, energy=0.02)]
    clean = build_clean_dataset(records)
    assert len(clean) == 1
    assert clean.iloc[0]["n_dft_records"] == 2
    assert abs(clean.iloc[0]["delta_E"] - 0.01) < 1e-12


def test_parent_split_has_no_overlap():
    import pandas as pd
    df = pd.DataFrame({"parent_structure_id": [f"p{i}" for i in range(100)] * 2})
    split_df, manifest = parent_level_split(df)
    sets = [set(manifest[key]) for key in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set(split_df["split"]) == {"train", "val", "test"}
