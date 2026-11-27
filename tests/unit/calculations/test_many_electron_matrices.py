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
from pytest_benchmark.fixture import BenchmarkFixture

from kestrelpy.calculations.many_electron_matrices import get_non_zero_matrix_elements
from kestrelpy.calculations.operators import Operators


@pytest.mark.parametrize("d_electrons", list(range(1, 10)))
def test_matrix_elements_are_returned_for_every_number_of_d_electrons(
    d_electrons: int,
) -> None:
    get_non_zero_matrix_elements(d_electrons=d_electrons)


@pytest.mark.parametrize("d_electrons", list(range(1, 5)))
def test_multipole_and_soc_matrix_elements_have_correct_phase(d_electrons: int) -> None:
    matrix_elements_d_le_5 = get_non_zero_matrix_elements(d_electrons)

    hole_equivalent = 10 - d_electrons
    matrix_elements_d_gt_5 = get_non_zero_matrix_elements(hole_equivalent)

    for operator_set in (Operators.SphericalHarmonic, Operators.SpinOrbitCoupling):
        for operator in operator_set:
            assert np.allclose(
                matrix_elements_d_le_5.values[operator],
                matrix_elements_d_gt_5.values[operator] * -1.0,
            )

    for ext_operator_set in (
        Operators.InterelectronicRepulsion,
        Operators.SpinAngularMomentum,
        Operators.OrbitalAngularMomentum,
    ):
        for operator in ext_operator_set:
            assert np.allclose(
                matrix_elements_d_le_5.values[operator],
                matrix_elements_d_gt_5.values[operator],
            )


@pytest.mark.parametrize("d_electrons", list(range(1, 10)))
def test_benchmark_load_matrix_elements(
    d_electrons: int, benchmark: BenchmarkFixture
) -> None:
    benchmark(get_non_zero_matrix_elements, d_electrons)
