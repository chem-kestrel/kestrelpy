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
from scipy.optimize import linear_sum_assignment

from kestrelpy.configuration import get_logger

_LOGGER = get_logger(__name__)


def get_xyz_index(vectors: np.ndarray) -> np.ndarray:
    """Return the xyz index for a set of 3x3 vectors

    The function takes the largest squared component of each vector and
    determines its xyz projection. An array is returned where:
        index[0] returns the vector projecting in the x-direction,
        index[1] the y-direction, index[2] the z-direction

    If the index projects the same vector twice, the second instance is
    replaced to each vector is uniquely projected.

        Args:
            vectors (np.ndarray): Numpy array containing the vectors

        Returns:
            np.ndarray: 1D array containing the x,y and z index
    """

    squared_projection = np.abs(vectors) ** 2
    _, vector_for_axis = linear_sum_assignment(-squared_projection)

    return vector_for_axis


def calculate_effective_g_values(
    g_vectors: np.ndarray, g_values: np.ndarray, a_vectors: np.ndarray
) -> np.ndarray:
    """Calculate the effective g values at the position of each A value.

    The angles between the vectors of one point (i.e. an A value) and a g value
    are calculated. The geff values for the x,y and z directions are then calculated
    and returned.

    Args:
        g_vectors (np.ndarray): Array of g factor eigenvectors
        g_values (np.ndarray): Array of principal g values
        a_vectors (np.ndarray): Array containing the vectors for the hyperfine
            a values, which geff is to be calculated at.

    Returns:
        np.ndarray: g_eff values for the x, y and z directions"""
    g_vectors_norm = g_vectors / np.linalg.norm(g_vectors, axis=0)
    a_vectors_norm = a_vectors / np.linalg.norm(a_vectors, axis=0)

    g_effective = []

    for i in range(3):
        axis = a_vectors_norm[:, i]

        # cos(theta_i) = (axis . g_i)
        direction_cosines = np.array(
            [np.dot(axis, g_vectors_norm[:, j]) for j in range(3)]
        )

        # g_eff^2 = sum( (g_values_i^2 * cos(theta_i)^2 )
        g_eff_sq = np.abs(np.sum((g_values * direction_cosines) ** 2))
        g_effective.append(np.sqrt(g_eff_sq))

    return np.array(g_effective)
