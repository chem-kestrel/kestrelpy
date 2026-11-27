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

from textwrap import dedent
from unittest.mock import Mock

import numpy as np
import pytest

from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_calculations import (
    ASquaredTensor,
    calculate_a_values,
    calculate_fermi_contact,
    calculate_orbital_values,
)
from kestrelpy.parameters import HyperfineParameters


class TestATensorCalculations:
    """
    Test that an arbitrary A^2 tensor, simulating a Cu(II) complex,
    is correctly diagonalised to give its eigenvalues and eigenvectors.
    Calculation of the orbital and Fermi contact contributions is also tested
    """

    @pytest.fixture
    def diagonalisation_output(self) -> ASquaredTensor:
        a2_tensor = np.array(([2.93e-4, 0, 0], [0, 7.33e-5, 0], [0, 0, 1.833e-01]))
        parameters = HyperfineParameters(metal=Mock(), p=1168, kappa=Mock())
        return calculate_a_values(a2_tensor, parameters.p)

    def test_calculate_a_values(self, diagonalisation_output: ASquaredTensor) -> None:
        """Test that diagonalisation results in the expected A values and vectors"""
        expected_values = np.array([10, 20, 500])
        expected_vectors = np.array(([0, 1, 0], [1, 0, 0], [0, 0, 1]))

        assert np.allclose(diagonalisation_output.values, expected_values, rtol=1e-03)
        assert np.array_equal(
            np.abs(diagonalisation_output.vectors), np.abs(expected_vectors)
        )

    @pytest.mark.parametrize(
        "p, kappa, fermi",
        [(1168.0, 0.14, 163.52), (515.0, 0.52, 267.80), (768.0, -1.05, -806.40)],
    )
    def test_calculate_fermi_contact(
        self, p: float, kappa: float, fermi: float
    ) -> None:
        expected_fermi_contact = np.array([fermi, fermi, fermi])
        calculated_fermi_contact = calculate_fermi_contact(p=p, kappa=kappa)
        assert np.allclose(calculated_fermi_contact, expected_fermi_contact)

    def test_calculate_orbital_a_values(self) -> None:
        """Test orbital values are calculated correctly"""
        g_factors = GSquaredTensor(
            tensor=Mock(),
            values=np.array([2.02, 2.30, 2.13]),
            vectors=np.eye(3),
            metadata=Mock(),
        )
        expected_result = ASquaredTensor(
            tensor=Mock(),
            values=np.array([20.651, 347.691, 149.131]),
            vectors=np.eye(3),
        )
        result = calculate_orbital_values(
            p=1168, g_factors=g_factors, a_vectors=np.eye(3), units=""
        )
        assert np.allclose(result.values, expected_result.values, rtol=1e-03)

    def test_get_axyz(self, diagonalisation_output: ASquaredTensor) -> None:
        """Test that a set of A values are correctly projected in the xyz directions"""
        expected_result = np.array([20, 10, 500])

        result = diagonalisation_output.get_axyz()
        assert np.allclose(result, expected_result, rtol=1e-03)

    def test_a_squared_tensor_output(
        self, diagonalisation_output: ASquaredTensor
    ) -> None:
        """Test that a set of A values and eigenvectors are correctly printed"""
        expected_output = dedent(
            """
            +-------------+-------+-------+--------+
            |   A values  |     x |     y |      z |
            +=============+=======+=======+========+
            |        10.0 | 0.000 | 1.000 |  0.000 |
            |        20.0 | 1.000 | 0.000 |  0.000 |
            |       500.1 | 0.000 | 0.000 | -1.000 |
            +-------------+-------+-------+--------+
            """
        ).strip()

        assert diagonalisation_output.__str__() == expected_output
