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

from unittest.mock import Mock

import numpy as np
import pytest

from kestrelpy.calculations.magnetism.g_factors import (
    GSquaredTensor,
    _calculate_denominator_term,
    _calculate_tensor_element_numerator,
    _get_effective_spin,
)


class TestGSquaredTensor:
    def test_xyz_projection_of_g_factors(self) -> None:
        g_squared_tensor = GSquaredTensor(
            tensor=Mock(),
            values=np.array([1.0, 1.5, 2.0]),
            vectors=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            metadata=Mock(),
        )
        assert np.array_equal(g_squared_tensor.get_gxyz(), np.array([1.5, 2.0, 1.0]))

    @pytest.mark.parametrize(
        "values, expected_order",
        [
            (np.array([1.8, 1.9, 1.95]), np.array([1.95, 1.9, 1.8])),
            (np.array([2.05, 2.1, 2.3]), np.array([2.05, 2.1, 2.3])),
            (np.array([1.8, 1.8, 1.99]), np.array([1.8, 1.8, 1.99])),
            (np.array([2.02, 2.3, 2.3]), np.array([2.3, 2.3, 2.02])),
        ],
    )
    def test_get_g_factor_order(
        self, values: np.ndarray, expected_order: np.ndarray
    ) -> None:
        g_squared_tensor = GSquaredTensor(
            tensor=Mock(), values=values, vectors=np.eye(3), metadata=Mock()
        )
        g_squared_tensor.order_g_factors()
        assert np.array_equal(g_squared_tensor.values, expected_order)


@pytest.mark.parametrize(
    "number_of_states, expected_effective_spin",
    [
        (1, 0.0),
        (2, 0.5),
        (3, 1.0),
        (4, 1.5),
    ],
)
def test_get_effective_spin(
    number_of_states: int, expected_effective_spin: float
) -> None:
    effective_spin = _get_effective_spin(number_of_states=number_of_states)
    assert effective_spin == expected_effective_spin


@pytest.mark.parametrize(
    "effective_spin, expected_denominator_term",
    [
        (0.5, 0.5),
        (1.0, 2.0),
        (1.5, 5.0),
    ],
)
def test_calculate_denominator_term(
    effective_spin: float, expected_denominator_term: float
) -> None:
    denominator_term = _calculate_denominator_term(effective_spin=effective_spin)
    assert denominator_term == expected_denominator_term


def test_calculate_tensor_element_numerator() -> None:
    diagonal_elements = np.array([0.0, 1.0, 2.0], dtype=complex)
    mu_alpha = np.diag(diagonal_elements)  # Create a diagonal matrix
    mu_beta = np.identity(3, dtype=complex)

    tensor_element = _calculate_tensor_element_numerator(
        mu_alpha=mu_alpha, mu_beta=mu_beta
    )
    assert tensor_element == np.sum(diagonal_elements)
