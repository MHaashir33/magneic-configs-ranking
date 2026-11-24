"""Magnetic P/T/PT and spin-space tests copied from the source notebook logic."""
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def check_PT_symmetry(structure, magmoms, op, coord_tol=0.01, mag_tol=0.05):
    """Test whether a crystallographic operation plus time reversal preserves moments."""
    for i, site in enumerate(structure):
        transformed = op.operate(site.frac_coords) % 1.0
        best_j = None
        best_dist = np.inf
        for j, target in enumerate(structure):
            diff = transformed - target.frac_coords
            diff -= np.round(diff)
            distance = np.linalg.norm(diff)
            if distance < best_dist:
                best_dist, best_j = distance, j
        if best_dist > coord_tol:
            return False
        if abs(float(magmoms[best_j]) + float(magmoms[i])) > mag_tol:
            return False
    return True


def magnetic_operation_preserves(
    structure, magmoms, op, time_reversal=+1, coord_tol=0.01, mag_tol=0.05
):
    """Notebook's site matching test for P, T and PT operations."""
    for i, site in enumerate(structure):
        transformed = op.operate(site.frac_coords) % 1.0
        best_j = None
        best_dist = np.inf
        for j, target in enumerate(structure):
            diff = transformed - target.frac_coords
            diff -= np.round(diff)
            distance = np.linalg.norm(diff)
            if distance < best_dist:
                best_dist, best_j = distance, j
        if best_dist > coord_tol:
            return False
        expected = float(time_reversal) * float(magmoms[i])
        if abs(float(magmoms[best_j]) - expected) > mag_tol:
            return False
    return True


def get_PTR_symmetry(record, symprec=0.01):
    """Determine whether P, T and PT preserve a collinear DFT magnetic state."""
    structure = Structure.from_dict(record["structure"])
    magmoms = np.asarray(record["magmoms"]["vasp"], dtype=float)
    if len(magmoms) != len(structure):
        raise ValueError("Number of VASP magnetic moments does not match structure sites.")
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=5)
    operations = sga.get_symmetry_operations(cartesian=False)
    p_found = False
    pt_found = False
    t_found = False
    identity = None
    for operation in operations:
        rotation = np.asarray(operation.rotation_matrix, dtype=float)
        ordinary = magnetic_operation_preserves(
            structure, magmoms, operation, time_reversal=+1
        )
        with_time_reversal = magnetic_operation_preserves(
            structure, magmoms, operation, time_reversal=-1
        )
        if (
            np.allclose(rotation, np.eye(3), atol=1e-5)
            and np.allclose(operation.translation_vector, [0, 0, 0], atol=1e-5)
        ):
            identity = operation
        if np.allclose(rotation, -np.eye(3), atol=1e-5):
            if ordinary:
                p_found = True
            if with_time_reversal:
                pt_found = True
    if identity is not None:
        t_found = magnetic_operation_preserves(
            structure, magmoms, identity, time_reversal=-1
        )
    return {
        "spacegroup": sga.get_space_group_symbol(),
        "P_preserved": bool(p_found),
        "T_preserved": bool(t_found),
        "PT_preserved": bool(pt_found),
    }


def screen_altermagnet_corrected(
    record, symprec=1e-2, angle_tolerance=5, net_tol=1e-4,
    coord_tol=0.01, mag_tol=0.05,
):
    """Notebook first-pass criterion: compensated state and broken PT."""
    structure = Structure.from_dict(record["structure"])
    magmoms = np.asarray(record["magmoms"]["vasp"], dtype=float)
    if len(magmoms) != len(structure):
        raise ValueError("Number of VASP magnetic moments does not match structure sites.")
    net_moment = abs(float(magmoms.sum()))
    compensated = net_moment <= net_tol
    if not compensated:
        return {
            "task_id": str(record["task_id"]),
            "formula": record.get("formula_pretty", record.get("formula")),
            "ordering": record.get("ordering"),
            "net_moment": net_moment,
            "compensated": False,
            "spacegroup": None,
            "inversion_found": False,
            "PT_preserved": None,
            "AM_candidate": False,
        }
    sga = SpacegroupAnalyzer(
        structure, symprec=symprec, angle_tolerance=angle_tolerance
    )
    operations = sga.get_symmetry_operations(cartesian=False)
    inversion_found = False
    pt_preserved = False
    # Deliberately retain the notebook's first-inversion-operation behaviour.
    for operation in operations:
        rotation = np.asarray(operation.rotation_matrix, dtype=float)
        if np.allclose(rotation, -np.eye(3), atol=1e-5):
            inversion_found = True
            if check_PT_symmetry(
                structure, magmoms, operation,
                coord_tol=coord_tol, mag_tol=mag_tol,
            ):
                pt_preserved = True
            break
    if not inversion_found:
        pt_preserved = False
    return {
        "task_id": str(record["task_id"]),
        "formula": record.get("formula_pretty", record.get("formula")),
        "ordering": record.get("ordering"),
        "net_moment": net_moment,
        "compensated": compensated,
        "spacegroup": sga.get_space_group_symbol(),
        "inversion_found": inversion_found,
        "PT_preserved": pt_preserved,
        "AM_candidate": bool(compensated and not pt_preserved),
    }


def spin_space_am_test(record, symprec=1e-2, mag_symprec=5e-2, spin_tol=0.05):
    """Apply the notebook's spinspg test to collinear VASP moments."""
    try:
        from spinspg import get_spin_symmetry
    except ImportError as exc:
        raise ImportError(
            "spinspg is required for the final spin-space symmetry stage. "
            "Install it with `python -m pip install spinspg`."
        ) from exc

    structure = Structure.from_dict(record["structure"])
    raw_moments = np.asarray(record["magmoms"]["vasp"], dtype=float)
    if len(raw_moments) != len(structure):
        raise ValueError("Number of VASP magnetic moments does not match structure sites.")
    moments = raw_moments.copy()
    moments[np.abs(moments) < spin_tol] = 0.0
    moments_3d = np.zeros((len(moments), 3), dtype=float)
    moments_3d[:, 2] = moments

    sog, rotations, translations, spin_rotations = get_spin_symmetry(
        np.asarray(structure.lattice.matrix, dtype=float),
        np.asarray(structure.frac_coords, dtype=float),
        np.asarray(structure.atomic_numbers, dtype=int),
        moments_3d,
        symprec=symprec,
        mag_symprec=mag_symprec,
        angle_tolerance=5,
    )
    axis = np.array([0.0, 0.0, 1.0])
    spin_flip_ops = []
    for index, (rotation, translation, spin_rotation) in enumerate(
        zip(np.asarray(rotations), np.asarray(translations), np.asarray(spin_rotations))
    ):
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        spin_rotation = np.asarray(spin_rotation, dtype=float)
        flips_spin = np.allclose(spin_rotation @ axis, -axis, atol=1e-5)
        if not flips_spin:
            continue
        pure_translation = np.allclose(rotation, np.eye(3), atol=1e-5)
        pure_inversion = np.allclose(rotation, -np.eye(3), atol=1e-5)
        if pure_translation or pure_inversion:
            continue
        spin_flip_ops.append(
            {
                "operation_index": index,
                "R": rotation,
                "translation": translation,
                "W": spin_rotation,
                "det_R": float(np.linalg.det(rotation)),
            }
        )
    return {
        "spin_group": str(sog),
        "n_spin_space_operations": len(rotations),
        "n_spin_flip_rotation_ops": len(spin_flip_ops),
        "spin_flip_ops": spin_flip_ops,
        "spin_space_AM_candidate": len(spin_flip_ops) > 0,
        "cleaned_magmoms": moments,
    }
