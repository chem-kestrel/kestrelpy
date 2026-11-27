"""MIT License

KestrelPy
Copyright (c) 2026
George Nunn, Benjamin S. Grace, Martin A. Bates and Paul H. Walton.
Department of Chemistry, University of York, United Kingdom.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS  OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN  AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION  WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from kestrelpy.builder import Builder
from kestrelpy.molecule import Atom

FeCl4 = dedent(
    """
    5
    symmetry c1
    Fe       0.000000000      0.000000000      0.000000000
    Cl       1.714642820      0.000000000      1.212435566
    Cl      -1.714642820      0.000000000      1.212435566
    Cl      -0.000000001      1.714642820     -1.212435565
    Cl       0.000000000     -1.714642819     -1.212435565
    """
).strip()


# Same as FeCl4.xyz but translated by (2.0, -3.0, 4.0)
FeCl4_translated = dedent(
    """
    5
    symmetry c1
    Fe       2.000000000     -3.000000000      4.000000000
    Cl       3.714642820     -3.000000000      5.212435566
    Cl       0.285357180     -3.000000000      5.212435566
    Cl       1.999999999     -1.285357180      2.787564435
    Cl       2.000000000     -4.714642819      2.787564435
    """
).strip()


def test_builder() -> None:
    builder = Builder(contents=FeCl4, metal_index=0)
    assert builder._atoms == [
        Atom(
            index=0,
            element="Fe",
            coordinates=(0.0, 0.0, 0.0),
        ),
        Atom(
            index=1,
            element="Cl",
            coordinates=(1.714642820, 0.000000000, 1.212435566),
        ),
        Atom(
            index=2,
            element="Cl",
            coordinates=(-1.714642820, 0.000000000, 1.212435566),
        ),
        Atom(
            index=3,
            element="Cl",
            coordinates=(-0.000000001, 1.714642820, -1.212435565),
        ),
        Atom(
            index=4,
            element="Cl",
            coordinates=(0.0, -1.714642819, -1.212435565),
        ),
    ]
    assert builder._metal_index == 0
    assert builder._metal_atom == builder._atoms[0]


def test_adding_ligand_with_duplicate_name_raises_error() -> None:
    builder = Builder(contents=FeCl4, metal_index=0)
    builder.add_ligand("Cl", builder.assign_atom(coordination_index=1))
    with pytest.raises(ValueError):
        builder.add_ligand("Cl", builder.assign_atom(coordination_index=1))


@pytest.mark.parametrize(("contents"), (FeCl4, FeCl4_translated))
def test_build_molecule_from_file(contents: str, tmp_path: Path) -> None:
    file_name = tmp_path / "molecule.xyz"
    with open(file_name, "w") as f:
        f.write(contents)

    builder = Builder.from_xyz_file(file=file_name, metal_index=0)
    assert builder.file_name == "molecule.xyz"
    builder.add_ligand(
        "Cl",
        builder.assign_atom(coordination_index=1, orientation_x_index=2),
        builder.assign_atom(coordination_index=2, orientation_x_index=3),
        builder.assign_atom(coordination_index=3, orientation_x_index=None),
        builder.assign_atom(coordination_index=4, orientation_x_index=1),
    )
    molecule = builder.get_molecule()
    assert molecule.molecular_geometry.file_name == "molecule.xyz"
    assert molecule.molecular_geometry.atoms == tuple(builder._atoms)
    assert molecule.molecular_geometry.xyz_text == builder._contents

    expected_theta_phi_psi = [
        (54.736, 0.000, 180.000),
        (54.736, 180.0, -60.000),
        (125.264, 90.000, 0.000),
        (125.264, -90.000, 120.000),
    ]

    actual_theta_phi_psi = [
        tuple([atom.theta, atom.phi, atom.psi])
        for atom in molecule.get_ligand("Cl").atoms
    ]

    assert np.array_equal(np.round(actual_theta_phi_psi, 3), expected_theta_phi_psi)

    molecule = builder.get_molecule()
    assert len(molecule.ligands) == 1
    assert len(molecule.get_ligand("Cl").atoms) == 4
