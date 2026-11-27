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

from dataclasses import fields

import numpy as np
from tabulate import tabulate

from kestrelpy.calculations.ligand_field_matrix import (
    DOrbitalBasis,
    LigandField,
    MultipoleExpansion,
)


def format_ligand_field_matrix(ligand_field_matrix: np.ndarray) -> str:
    d_orbital_basis = [f"{DOrbitalBasis(i).name:^10s}" for i in range(5)]
    table_rows: list[list[str]] = []
    for i in range(5):
        table_rows.append(
            [
                d_orbital_basis[i],
                *[value for value in ligand_field_matrix[i, :]],
            ]
        )
    return tabulate(
        tabular_data=table_rows,
        headers=["  <Di|V|Dj>  ", *d_orbital_basis],
        stralign="center",
        numalign="center",
        floatfmt=("", " .1f", " .1f", " .1f", " .1f", " .1f"),
    )


def format_ligand_field_eigenfunctions(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> str:
    return tabulate(
        tabular_data=[
            [eigenvalues[i] - eigenvalues[0], *eigenvectors[:, i]] for i in range(5)
        ],
        headers=list(
            map(
                lambda x: f"{x:^10s}",
                ["Energy (cm-1)", *[DOrbitalBasis(i).name for i in range(5)]],
            )
        ),
        stralign="center",
        numalign="center",
        floatfmt=(" .1f", "+.3f", "+.3f", "+.3f", "+.3f", "+.3f"),
    )


def format_multipole_expansion(multipole_expansion: MultipoleExpansion) -> str:
    rows: list[tuple[str, str, str]] = []
    for f in fields(multipole_expansion):
        value: complex = getattr(multipole_expansion, f.name)
        rows.append((f.name, f"{value.real: .1f}", f"{value.imag: .1f}"))
    return tabulate(tabular_data=rows, headers=["C_k_q", "Real", "Imag"])


def format_ligand_field(ligand_field: LigandField) -> str:
    return "\n".join(
        [
            "5x5 one-electron ligand-field matrix:",
            format_ligand_field_matrix(ligand_field.ligand_field_matrix),
            f"\nOne-electron ligand-field eigenfunctions (E0 = {ligand_field.zero_point_energy:.1f} cm-1):",
            format_ligand_field_eigenfunctions(
                ligand_field.eigenvalues, ligand_field.eigenvectors
            ),
            "\nSpherical Harmonic expansion of the ligand-field potential:",
            f"{format_multipole_expansion(ligand_field.multipole_expansion)}",
        ]
    )
