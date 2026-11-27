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

from kestrelpy.calculations.ds_mixing import calculate_ds_mixing_matrix


@pytest.mark.parametrize(
    ("ligand_parameters", "expected_matrix"),
    (
        pytest.param(
            [
                (1000.0, 0.0, 0.0),
                (1000.0, 180.0, 0.0),
                (1000.0, 90.0, 0.0),
                (1000.0, 90.0, 90.0),
                (1000.0, 90.0, 180.0),
                (1000.0, 90.0, 270.0),
            ],
            np.zeros((5, 5)),
            id="[Oh] octahedral complex, eds stabilisation cancels out",
        ),
        pytest.param(
            [
                (1000.0, 90.0, 0.0),
                (1000.0, 90.0, 90.0),
                (1000.0, 90.0, 180.0),
                (1000.0, 90.0, 270.0),
            ],
            np.diag([0.0, 0.0, -4000.0, 0.0, 0.0]),
            id="[D4h] square planar complex, eds should stabilise dz2 by -4.0 * eds",
        ),
        pytest.param(
            [
                (1000.0, 0.0, 0.0),
                (1000.0, 180.0, 0.0),
            ],
            np.diag([0.0, 0.0, -4000.0, 0.0, 0.0]),
            id="[Cinfv] linear complex, should stabilise -4.0 * eds",
        ),
    ),
)
def test_calculate_ds_mixing_matrix(
    ligand_parameters: list[tuple[float, float, float]],
    expected_matrix: np.ndarray,
) -> None:
    actual_matrix = calculate_ds_mixing_matrix(
        *[(x[0], np.radians(x[1]), np.radians(x[2])) for x in ligand_parameters]
    )
    assert np.allclose(actual_matrix, expected_matrix)
