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

import numpy as np

from kestrelpy.calculations.ds_mixing import calculate_ds_mixing_matrix
from kestrelpy.calculations.rotation_matrix import rotate_ligand_field_matrix
from kestrelpy.molecule import Molecule
from kestrelpy.parameters import LigandParameters


def get_ligand_field_matrices(
    molecule: Molecule,
    ligand_field_parameters: list[LigandParameters],
) -> tuple[np.ndarray, np.ndarray]:
    ligand_field_matrix = np.zeros((5, 5))
    ds_mixing_parameters: list[tuple[float, float, float]] = []
    for ligand_ps in ligand_field_parameters:
        ligand = molecule.get_ligand(ligand_ps.name)  # TODO: cache this
        for atom in ligand.atoms:
            ligand_field_matrix += rotate_ligand_field_matrix(
                ligand_field_matrix=ligand_ps.ligand_field_matrix,
                rotation_matrix=atom.ligand_field_rotation_matrix,
            )
            if abs(ligand_ps.eds) > 1.0e-5:
                ds_mixing_parameters.append(
                    (ligand_ps.eds, np.radians(atom.theta), np.radians(atom.phi))
                )
    if ds_mixing_parameters:
        ds_stabilisation_matrix = calculate_ds_mixing_matrix(*ds_mixing_parameters)
    else:
        ds_stabilisation_matrix = np.zeros((5, 5))
    return ligand_field_matrix, ds_stabilisation_matrix
