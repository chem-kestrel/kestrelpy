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

from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.magnetism.simm import (
    calculate_qtm_energy_barrier,
    calculate_qtm_time,
)


@pytest.mark.parametrize(
    ("gx", "gy", "gz", "expected_time"),
    [
        pytest.param(
            0.3610e-04,
            0.4197e-04,
            11.8650,
            13.8,
        ),
        pytest.param(0.3498e-04, 0.3833e-04, 11.9056, 15.8),
    ],
)
def test_qtm_rate_calculation(
    gx: float, gy: float, gz: float, expected_time: float
) -> None:
    """
    Data taken from reference:
        Fang Lu, Wen-Xiao Guo, and Yi-Quan Zhang
        Inorganic Chemistry 2022 61 (1), 295-301
        DOI: 10.1021/acs.inorgchem.1c02858
    """
    time = calculate_qtm_time(gx=gx, gy=gy, gz=gz)
    assert abs(time - expected_time) < 1.0e-01


def test_calculation_of_effective_energy_barrier() -> None:
    """
    Data taken from reference:
        Fang Lu, Wen-Xiao Guo, and Yi-Quan Zhang
        Inorganic Chemistry 2022 61 (1), 295-301
        DOI: 10.1021/acs.inorgchem.1c02858
    """
    energies = np.array([0.0, 449.2, 917.2, 1407.0, 1559.1])
    g_values = [
        GSquaredTensor(
            tensor=Mock(),
            values=np.array(g_values),
            vectors=np.diag([1.0, 1.0, 1.0]),
            metadata=Mock(),
        )
        for g_values in (
            (0.1100e-06, 0.1230e-06, 11.983),
            (0.007, 0.007, 8.053),
            (0.005, 0.008, 4.137),
            (0.011, 0.061, 0.501),
            (6.715, 6.481, 1.660),
        )
    ]
    temperatures = np.array([10.0, 75.0, 300.0])

    output = calculate_qtm_energy_barrier(
        relative_energies=energies, g_tensors=g_values, temperatures=temperatures
    )
    # The reference quotes 1558.7 cm-1. We cannot get exact agreement
    # because of rounding errors for quotes energies & g-factors.
    assert abs(output.effective_barrier[0]) < 1.0
    assert abs(output.effective_barrier[1] - 449.2) < 1.0
    assert abs(output.effective_barrier[2] - 1558.7) < 1.0

    # The maximum Kramer's doublet contribution per temperature (second index).
    assert np.argmax(output.kramers_doublet_contribution[:, 0]) == 0
    assert np.argmax(output.kramers_doublet_contribution[:, 1]) == 1
    assert np.argmax(output.kramers_doublet_contribution[:, 2]) == 4

    # Check the kramer's doublet contribution is normalised in columns.
    assert np.allclose(np.sum(output.kramers_doublet_contribution, axis=0), np.ones(3))
