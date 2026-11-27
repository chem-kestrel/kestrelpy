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

from kestrelpy.comparison.ligand_field_matrix import (
    compare_ligand_field_matrices,
    get_centered_matrix,
)


def test_get_centered_matrix() -> None:
    original_matrix = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ]
    )
    expected_centered_matrix = np.array(
        # Subtract `12` (0.2 * (0+6+12+18+24)) for each diagonal entry!
        [
            [-12, 1, 2, 3, 4],
            [5, -6, 7, 8, 9],
            [10, 11, 0, 13, 14],
            [15, 16, 17, 6, 19],
            [20, 21, 22, 23, 12],
        ]
    )
    assert np.allclose(
        get_centered_matrix(matrix=original_matrix),
        expected_centered_matrix,
    )


def test_compare_ligand_field_matrices() -> None:
    matrix = np.ones((5, 5), dtype=float)
    assert (
        compare_ligand_field_matrices(calculated_matrix=matrix, reference_matrix=matrix)
        == 0.0
    )
