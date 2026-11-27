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

from kestrelpy.calculations.many_electron_matrices import NonZeroMatrixElements
from kestrelpy.calculations.map_eigenvectors_to_operator_matrix import (
    map_eigenvectors_to_operator_matrix,
)
from kestrelpy.calculations.parameters import OrbitalReductionFactors


def get_muxyz_matrices(
    eigenvectors: np.ndarray,
    matrices: NonZeroMatrixElements,
    reduction_factors: OrbitalReductionFactors,
) -> np.ndarray:
    """Calculate the mu_x, mu_y, and mu_z matrix elements acting between eigenfunctions.

    Args:
        eigenvectors (np.ndarray): Normalised column vectors.
        matrices (NonZeroMatrixElements): The indices and values of Lxyz and 2Sxyz matrix elements.
        reduction_factors (OrbitalReductionFactors): The orbital reduction factors.

    Returns:
        np.ndarray: Shape(3, basis_size, basis_size)
            The indexes of 0, 1, and 2 in the first dimension represent the mu_x, mu_y, and mu_z
            matrix elements. The magnetic moment operator matrices are complex quantities.
    """
    basis_size = eigenvectors.shape[0]
    mu_matrices = np.zeros((3, basis_size, basis_size), dtype=complex)
    for index, operators in enumerate(
        (("LX", "SX", "KX"), ("LY", "SY", "KY"), ("LZ", "SZ", "KZ"))
    ):
        orbital, spin, reduction_factor = operators
        reduction_factor = getattr(reduction_factors, reduction_factor)

        indices = np.vstack((matrices.indices[orbital], matrices.indices[spin]))
        rows, columns = indices[:, 0], indices[:, 1]
        values = np.hstack(
            (matrices.values[orbital] * reduction_factor, matrices.values[spin])
        )
        operator_matrix = np.zeros((basis_size, basis_size), dtype=complex)
        np.add.at(operator_matrix, (rows, columns), values)

        mu_matrices[index, :, :] = map_eigenvectors_to_operator_matrix(
            eigenvectors=eigenvectors, operator_matrix=operator_matrix
        )

    # mu_y matrices are multiplied by sqrt(-1.0)
    mu_matrices[1, :, :] *= complex(0.0, 1.0)

    return mu_matrices
