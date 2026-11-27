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

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import numpy as np

from kestrelpy.builder import Builder
from kestrelpy.molecule import Assignment, Molecule, load_molecule, save_molecule

RESOURCES_PATH = Path(__file__).parent / "resources"


def _get_model_molecule() -> Molecule:
    builder = Builder(
        contents=dedent(
            """
            5
            symmetry c1
            Fe       0.000000000      0.000000000      0.000000000
            Cl       1.714642820      0.000000000      1.212435566
            Cl      -1.714642820      0.000000000      1.212435566
            Cl      -0.000000001      1.714642820     -1.212435565
            Cl       0.000000000     -1.714642819     -1.212435565
            """
        ),
        metal_index=0,
    )
    builder.add_ligand(
        "Cl",
        builder.assign_atom(coordination_index=1, orientation_x_index=2),
        builder.assign_atom(coordination_index=2, orientation_x_index=3),
        builder.assign_atom(coordination_index=3, orientation_x_index=4),
        builder.assign_atom(coordination_index=4, orientation_x_index=1),
    )
    return builder.get_molecule()


def test_serialising_molecule(tmp_path: Path) -> None:
    save_molecule(tmp_path / "molecule.json", _get_model_molecule())
    with open(tmp_path / "molecule.json", "r") as f:
        actual_json = json.load(f)
    with open(RESOURCES_PATH / "serialised_molecule.json", "r") as f:
        expected_json = json.load(f)
    assert actual_json == expected_json


def test_loading_molecule_from_file() -> None:
    molecule = load_molecule(RESOURCES_PATH / "serialised_molecule.json")
    assert molecule == _get_model_molecule()


def test_assignment_gives_correct_local_frame() -> None:
    assignment = Assignment(
        theta=90.0,
        phi=90.0,
        psi=90.0,
        coordinating_atom=Mock(),
        local_x_orientation=None,
    )
    assert np.allclose(
        assignment.local_frame,
        np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )
