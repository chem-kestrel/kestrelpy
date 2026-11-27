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


def get_centered_matrix(matrix: np.ndarray) -> np.ndarray:
    """Form a centered matrix A' from the matrix A.

    A' = A - I(tr(A) / n);      where n is the size of the matrix.
    """
    if matrix.shape != (5, 5):
        raise ValueError(
            f"Matrix must have a shape of (5, 5), instead got {matrix.shape=}"
        )
    return matrix - np.diag([np.trace(matrix) / 5] * 5)


def compare_ligand_field_matrices(
    calculated_matrix: np.ndarray,
    reference_matrix: np.ndarray,
) -> float:
    """Get the root-mean-square between two 5x5 ligand-field matrix.

    5x5 ligand-field matrices are symmetric so we only need to compare the upper
    / lower triangle.

    Args:
        calculated_matrix: The 5x5 ligand-field matrix generated from Kestrel
            calculations.
        reference_matrix: The 5x5 ligand-field matrix to fit to.

    Returns:
        The root-mean-square difference in the unique 15 matrix elements.
    """
    diff = calculated_matrix[np.triu_indices(5)] - reference_matrix[np.triu_indices(5)]
    rms = float(np.sqrt(np.sum(diff**2) / 15))
    return rms
