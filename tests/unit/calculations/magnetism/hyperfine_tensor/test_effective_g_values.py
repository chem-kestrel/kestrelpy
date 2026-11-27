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

from kestrelpy.calculations.magnetism.effective_g_values import (
    calculate_effective_g_values,
    get_xyz_index,
)


class TestGEffectiveCalculation:
    """
    Test that the effective g factors are correctly calculated.
    Where g_effective values are given by:
        geff = (gx^2 * lx^2) + (gy^2 * ly^2) + (gz^2 * lz^2)
    where gx/y/z are g factors and lx/y/z angles between vectors
    """

    def test_calculate_effective_g_factors(self) -> None:  # FIX
        """Test that g effective values are calculated correctly"""
        vectors = np.array(([0, 0, 1], [1, 0, 0], [0, 1, 0]))
        g_values = np.array([2.01, 2.08, 2.32])
        expected_output = np.array([2.01, 2.08, 2.32])

        output = calculate_effective_g_values(vectors, g_values, vectors)
        assert np.array_equal(output, expected_output)


def test_get_xyz_index() -> None:
    """
    Test that the correct xyz projection is given and that, when two vectors
    lie between the x and y axes, a unique projection is returned
    """
    vectors_a = np.array(([0, 0, 1], [1, 0, 0], [0, 1, 0]))
    vectors_b = np.array(([0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]))
    expected_output_a = np.array([2, 0, 1])
    expected_output_b = np.array([0, 1, 2])

    output_a = get_xyz_index(vectors_a)
    output_b = get_xyz_index(vectors_b)
    assert np.array_equal(output_a, expected_output_a)
    assert np.array_equal(output_b, expected_output_b)
