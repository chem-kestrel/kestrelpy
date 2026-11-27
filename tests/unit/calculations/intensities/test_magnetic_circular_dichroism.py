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

from kestrelpy.calculations.intensities.magnetic_circular_dichroism import (
    _perform_parallel_integration_over_sphere,
    get_lebedev_grid,
)


def test_get_lebedev_grid() -> None:
    for (
        values,
        expected_values,
    ) in zip(
        get_lebedev_grid(quadrature_order=3),
        (
            (1.5707963267948966, 0.0, 0.0, 0.1666666666666667),
            (1.5707963267948966, 3.14159265358979, 0.0, 0.1666666666666667),
            (1.5707963267948966, 1.5707963267948966, 0.0, 0.1666666666666667),
            (1.5707963267948966, 4.71238898038469, 0.0, 0.1666666666666667),
            (0.0, 0.0, 0.0, 0.1666666666666667),
            (3.141592653589793, 0.0, 0.0, 0.1666666666666667),
        ),
    ):  # 6 points in an octahedron:
        assert np.allclose(values, expected_values, atol=1.0e-06)


def _sphere(x: float, y: float, z: float, w: float) -> float:
    """Dummy functions for testing integration over a sphere.

    NOTE: has to be a top-level function to be allowed for parallel-processing.
    """
    # x**2 + y**2 + z**2 will always be 1.0
    # Final integral should be == 4.0 * pi
    return (x**2 + y**2 + z**2) * w


def test_perform_integration_over_sphere() -> None:
    from scipy.integrate import lebedev_rule

    inputs = []
    xs, ws = lebedev_rule(11)  # Picked 11 as an example
    for (x, y, z), w in zip(xs.T, ws):
        inputs.append([x, y, z, w])

    results = _perform_parallel_integration_over_sphere(
        _sphere,
        inputs,
        1,
    )
    final_integral = sum(results)

    error = abs(final_integral - 4.0 * np.pi) / (4.0 * np.pi)
    assert error < 1.0e-14
