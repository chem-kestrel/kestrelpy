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

from kestrelpy.calculations.spherical_polar_coordinates import (
    Coordinates,
    calculate_direction_cosine_matrix,
    calculate_psi,
    euler_angles_from_matrix,
)


@pytest.mark.parametrize(
    ["theta", "phi", "psi", "expected_matrix"],
    (
        (0.0, 0.0, 0.0, np.diag([1.0, 1.0, 1.0])),
        (
            # In the right-handed frame, we should have:
            # * x --> -x,
            # * y --> +z,
            # * z --> +y
            90.0,
            90.0,
            90.0,
            np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        ),
    ),
)
def test_direction_cosines(
    theta: float, phi: float, psi: float, expected_matrix: np.ndarray
) -> None:
    direction_cosine = calculate_direction_cosine_matrix(
        *list(np.radians([-theta, -phi, -psi]))
    )
    assert np.allclose(direction_cosine, expected_matrix, atol=1.0e-05)


@pytest.mark.parametrize(
    ["theta", "phi", "coordinates", "expected_psi"],
    (
        # Ligand in +Z position
        pytest.param(
            0.0,
            0.0,
            np.array([1.0, 0.0, 1.0]),
            0.0,
        ),
        pytest.param(
            0.0,
            0.0,
            np.array([0.0, 1.0, 1.0]),
            90.0,
        ),
        pytest.param(
            0.0,
            0.0,
            np.array([-1.0, 0.0, 1.0]),
            180.0,
        ),
        pytest.param(
            0.0,
            0.0,
            np.array([0.0, -1.0, 1.0]),
            -90.0,
        ),
        # Ligand in +X position
        pytest.param(
            90.0,
            0.0,
            np.array([1.0, 0.0, 1.0]),
            180.0,
        ),
        pytest.param(
            90.0,
            0.0,
            np.array([1.0, 0.0, -1.0]),
            0.0,
        ),
        pytest.param(
            90.0,
            0.0,
            np.array([-1.0, 0.0, -1.0]),
            0.0,
        ),
        pytest.param(
            90.0,
            0.0,
            np.array([-1.0, 0.0, 1.0]),
            180.0,
        ),
        # Ligand in -Z position
        pytest.param(
            180.0,
            0.0,
            np.array([1.0, 0.0, 1.0]),
            180.0,
        ),
        pytest.param(
            180.0,
            0.0,
            np.array([0.0, 1.0, 1.0]),
            90.0,
        ),
        pytest.param(
            180.0,
            0.0,
            np.array([-1.0, 0.0, 1.0]),
            0.0,
        ),
        pytest.param(
            180.0,
            0.0,
            np.array([0.0, -1.0, 1.0]),
            -90.0,
        ),
        # Test results of colinearity
        pytest.param(
            0.0,
            0.0,
            np.array([0.0, 0.0, 2.0]),
            0.0,
            id="x-orientation along the same vector as M-L vector in +Z position",
        ),
        pytest.param(
            0.0,
            0.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in +Z position",
        ),
        pytest.param(
            0.0,
            45.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in -Z position, with a twist",
        ),
        pytest.param(
            180.0,
            0.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in -Z position",
        ),
        pytest.param(
            180.0,
            45.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in -Z position, with a twist",
        ),
        pytest.param(
            90.0,
            0.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in +X position",
        ),
        pytest.param(
            90.0,
            90.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in +Y position",
        ),
        pytest.param(
            90.0,
            180.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in -X position",
        ),
        pytest.param(
            270.0,
            0.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in -X position, alternative 1",
        ),
        pytest.param(
            90.0,
            270.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in -Y position",
        ),
        pytest.param(
            270.0,
            90.0,
            np.array([0.0, 0.0, 0.0]),  # At the origin
            0.0,
            id="x-orientation at the origin whilst ligand in -Y position, alternative 1",
        ),
    ),
)
def test_calculate_psi(
    theta: float, phi: float, coordinates: Coordinates, expected_psi: float
) -> None:
    psi = calculate_psi(local_x_orientation=coordinates, theta=theta, phi=phi)
    assert abs(psi - expected_psi) < 1.0e-5


def test_euler_angles_from_matrix() -> None:
    """Test that the euler angles for a 3x3 matrix are correctly extracted."""
    matrix = np.array([[0.956, 0.294, 0], [-0.294, 0.956, 0], [0, 0, -1]])
    expected_angles = np.array(([17.094, 0.0, 0.0]))

    result = euler_angles_from_matrix(matrix)
    assert np.allclose(result, expected_angles, rtol=1e-4)
