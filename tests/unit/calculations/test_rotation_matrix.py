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
import pytest

from kestrelpy.calculations.rotation_matrix import get_rotation_matrix


class TestLigandRotationMatrix:
    @staticmethod
    def get_test_angles() -> list[tuple[float, float, float]]:
        return [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (90.0, 90.0, 90.0),
        ]

    @pytest.mark.parametrize("tta,phi,psi", get_test_angles())
    def test_ligand_rotation_matrix_invariant_from_degrees_or_radians(
        self,
        tta: float,
        phi: float,
        psi: float,
    ) -> None:
        rotation_matrix_degrees = get_rotation_matrix(tta, phi, psi, degrees=True)
        radian_tta, radian_phi, radian_psi = np.radians([tta, phi, psi])
        rotation_matrix_radians = get_rotation_matrix(
            radian_tta, radian_phi, radian_psi, degrees=False
        )
        assert np.array_equal(rotation_matrix_degrees, rotation_matrix_radians)

    @pytest.mark.parametrize("tta,phi,psi", get_test_angles())
    def test_rotation_matrix_is_normalised(
        self, tta: float, phi: float, psi: float
    ) -> None:
        normalised_rotation_matrix = (
            get_rotation_matrix(tta, phi, psi, degrees=True) ** 2.0
        )
        expected_normalised_sum = np.ones(5, dtype=float)
        assert np.allclose(
            np.sum(normalised_rotation_matrix, axis=0),
            expected_normalised_sum,
            rtol=1e-05,
            atol=1e-08,
        )
        assert np.allclose(
            np.sum(normalised_rotation_matrix, axis=1),
            expected_normalised_sum,
            rtol=1e-05,
            atol=1e-08,
        )

    @pytest.mark.parametrize("tta,phi,psi", get_test_angles())
    def test_rotation_matrix_is_unitary(
        self, tta: float, phi: float, psi: float
    ) -> None:
        """
        Recall that a unitary matrix satisfies the relationship:
            U(U**-1) = I
        where `U` is the unitary matrix and `I` is the identity matrix

        For a real matrix, the inverse is equivalent to the transpose of the matrix.
        """
        rotation_matrix = get_rotation_matrix(tta, phi, psi, degrees=True)
        assert np.allclose(
            np.matmul(rotation_matrix, rotation_matrix.T),
            np.diag(np.ones(5, dtype=float)),
            rtol=1e-05,
            atol=1e-08,
        )
