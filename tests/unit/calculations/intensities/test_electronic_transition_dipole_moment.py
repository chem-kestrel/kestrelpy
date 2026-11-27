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

from kestrelpy.calculations.intensities.electronic_transition_dipole_moment import (
    IntensityParameters,
    get_electronic_transition_dipole_moment_matrix_elements,
    get_electronic_transition_dipole_moment_multipole_expansion,
)
from kestrelpy.calculations.many_electron_matrices import get_non_zero_matrix_elements
from tests.unit.calculations.intensities.conftest import get_numpy_resource


@pytest.mark.parametrize(
    ("angles1", "angles2"),
    [
        pytest.param(
            [0.0, 0.0, 0.0], [180.0, 0.0, 0.0], id="Check +Z and -Z cancel out"
        ),
        pytest.param(
            [90.0, 0.0, 0.0], [270.0, 0.0, 0.0], id="Check +X and -X cancel out"
        ),
        pytest.param(
            [90.0, 0.0, 0.0],
            [90.0, 180.0, 180.0],
            id="Check +X and -X cancel out alternative",
        ),
        pytest.param(
            [90.0, 90.0, 0.0], [90.0, 270.0, 180.0], id="Check +Y and -Y cancel out"
        ),
        pytest.param(
            [90.0, 90.0, 180.0],
            [90.0, 270.0, 0.0],
            id="Check +Y and -Y cancel out alternative",
        ),
    ],
)
def test_diametrically_opposed_ligands_cancel_out_contributions(
    angles1: list[float], angles2: list[float]
) -> None:
    theta1, phi1, psi1 = list(np.radians(angles1))
    theta2, phi2, psi2 = list(np.radians(angles2))
    multipole = get_electronic_transition_dipole_moment_multipole_expansion(
        IntensityParameters(
            psigma=100.0,
            ppix=50.0,
            ppiy=50.0,
            fsigma=-100.0,
            fpix=-50.0,
            fpiy=-50.0,
            theta=theta1,
            phi=phi1,
            psi=psi1,
        ),
        IntensityParameters(
            psigma=100.0,
            ppix=50.0,
            ppiy=50.0,
            fsigma=-100.0,
            fpix=-50.0,
            fpiy=-50.0,
            theta=theta2,
            phi=phi2,
            psi=psi2,
        ),
    )
    for mp in [multipole.x, multipole.y, multipole.z]:
        for k in mp.__dict__.keys():
            value: complex = getattr(mp, k)
            assert abs(value.real) < 1.0e-10
            assert abs(value.imag) < 1.0e-10


def test_get_electronic_transition_dipole_moment() -> None:
    expected_output = get_numpy_resource("transition-dipole-moments.npz")["arr_0"]

    multipole = get_electronic_transition_dipole_moment_multipole_expansion(
        IntensityParameters(
            psigma=100.0,
            ppix=0.0,
            ppiy=0.0,
            fsigma=0.0,
            fpix=0.0,
            fpiy=0.0,
            theta=0.0,
            phi=0.0,
            psi=0.0,
        )
    )

    eigenvectors = get_numpy_resource("eigenvectors.npz")["arr_0"]
    eigenvectors = eigenvectors[0, :, :] + 1j * eigenvectors[1, :, :]

    output = get_electronic_transition_dipole_moment_matrix_elements(
        eigenvectors=eigenvectors,
        matrices=get_non_zero_matrix_elements(9),
        global_transition_dipole_multipole_expansion=multipole,
    )
    assert np.allclose(output, expected_output, atol=1.0e-2)
