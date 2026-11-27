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

from dataclasses import dataclass

import numpy as np

from kestrelpy.calculations.rotation_matrix import ensure_right_handed_axis


@dataclass(frozen=True)
class MatrixDiagonalisation:
    values: np.ndarray
    vectors: np.ndarray
    reference_value: float

    def get_relative_eigenvalues(self) -> np.ndarray:
        return self.values - self.reference_value


def solve_matrix(matrix: np.ndarray) -> MatrixDiagonalisation:
    """Take a generic, symmetric, n x n matrix and get the eigenvalues and eigenvectors.

    Args:
        matrix: The input matrix.

    Returns:
        MatrixDiagonalisation: A struct containing the eigenvalues and
            eigenvectors of the diagonalisation.
    """
    values, vectors = np.linalg.eigh(matrix)
    if np.shape(vectors) == (3, 3):
        vectors = ensure_right_handed_axis(vectors=vectors)

    reference_value = values[0]
    return MatrixDiagonalisation(
        values=values, vectors=vectors, reference_value=reference_value
    )
