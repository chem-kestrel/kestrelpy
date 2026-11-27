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

from kestrelpy.calculations.solve_matrix import MatrixDiagonalisation, solve_matrix


def test_hermitian_matrix_diagonalisation() -> None:
    matrix = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 3.0)],
            [complex(2.0, -3.0), complex(5.0, 0.0)],
        ]
    )
    expected_output = MatrixDiagonalisation(
        values=np.array([-1.123105625617661, 7.1231056243823385]),
        vectors=np.array(
            [
                [complex(-0.86170507, 0.0), complex(-0.50740947, 0.0)],
                [complex(0.28146014, -0.4221902), complex(-0.47798797, 0.71698196)],
            ]
        ),
        reference_value=-1.123105625617661,
    )
    matrix_diagonalisation = solve_matrix(matrix)
    assert np.allclose(matrix_diagonalisation.values, expected_output.values)
    assert np.allclose(matrix_diagonalisation.vectors, expected_output.vectors)
    assert matrix_diagonalisation.reference_value, expected_output.reference_value
    assert np.allclose(
        matrix_diagonalisation.get_relative_eigenvalues(),
        np.array([0.0, 8.24621125]),
    )
