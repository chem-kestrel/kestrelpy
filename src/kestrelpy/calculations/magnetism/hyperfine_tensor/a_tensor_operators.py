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

from kestrelpy.calculations.many_electron_matrices import NonZeroMatrixElements
from kestrelpy.calculations.map_eigenvectors_to_operator_matrix import (
    map_eigenvectors_to_operator_matrix,
)

# spherical harmonic coefficients
# angular factors are taken from J. R. Morton et al. (DOI: 10.1016/0022-2364(78)90284-6)
DZ2_COEFFICIENT = (
    (np.sqrt(2 * np.pi)) / (np.sqrt(5)) * 4 / 7
)  # Derived from coefficient for Y_0_2: 1/4 * sqrt(5 / pi)
Y_2_2_COEFFICIENT = (
    (2 * np.sqrt(3 * np.pi)) / (np.sqrt(5)) * 2 / 7
)  # Derived from coefficient for Y_2_2: 1/4 * sqrt(15 / 2*pi)
Y_2_1_COEFFICIENT = (
    (2 * np.sqrt(3 * np.pi)) / (np.sqrt(5)) * 2 / 7
)  # Derived from coefficient for Y_2_1: 1/2 * sqrt(15 / 2*pi)


def construct_operator_matrices(
    matrices: NonZeroMatrixElements,
    eigenvectors_no_soc: np.ndarray,
    metal: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    Construct the Y (spherical harmonic) and S (spin) matrices which are mapped
    onto the eigenvectors to give operator matrices for the dipolar contribution.

    Args:
        matrices (NonZeroMatrixElements): indices and values of basis matrix elements
        eigenvectors_no_soc (np.ndarray): eigenvectors without SOC contribution
        metal (str): 3d transition metal

    Returns:
        tuple: Containing two dictionaries:
            - Spin matrices (dict[str, np.ndarray])
            - Spherical harmonic matrices (dict[str, np.ndarray])
    """
    basis_size = eigenvectors_no_soc.shape[0]

    # Construct the operator matrices
    ops = {}
    for operator in (
        "SX",
        "SY",
        "SZ",
        "Y_2_2",
        "Y_2_1",
        "Y_2_0",
        "Y_2_n1",
        "Y_2_n2",
    ):
        indices = matrices.indices[operator]
        values = matrices.values[operator]
        rows, columns = indices[:, 0], indices[:, 1]

        operator_matrix = np.zeros((basis_size, basis_size), dtype=complex)
        np.add.at(operator_matrix, (rows, columns), values)

        ops[operator] = map_eigenvectors_to_operator_matrix(
            eigenvectors=eigenvectors_no_soc,
            operator_matrix=operator_matrix,
        )

    # Ti, Cr and Ni have negative nuclear g values, therefore the signs of the
    # spin matrices are flipped
    s_sign = -1 if metal.lower() in ("ti", "cr", "ni") else 1

    # Construct and store the spin, spherical harmonic and orbital matrices
    spin_matrices = {
        "x": s_sign * ops["SX"],
        "y": s_sign * ops["SY"] * 1j,
        "z": s_sign * ops["SZ"],
    }

    y_matrices = {
        "y_2_2": ops["Y_2_2"],
        "y_2_1": ops["Y_2_1"],
        "y_2_0": ops["Y_2_0"],
        "y_2_n1": ops["Y_2_n1"],
        "y_2_n2": ops["Y_2_n2"],
    }

    return (
        spin_matrices,
        y_matrices,
    )


def calculate_dipolar(
    s_mat: dict[str, np.ndarray], y_mat: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Calculates the dipolar contribution

    Args:
        s_mat (dict[str, np.ndarray]): Spin matrices.
        y_mat (dict[str, np.ndarray]): Spherical harmonic matrices.

    Returns:
        dict[str, np.ndarray]: Dipolar contribution to hyperfine coupling
    """
    dipolar_x = _calculate_dipolar_x(s_mat, y_mat)
    dipolar_y = _calculate_dipolar_y(s_mat, y_mat)
    dipolar_z = _calculate_dipolar_z(s_mat, y_mat)

    return {
        "x": dipolar_x,
        "y": dipolar_y,
        "z": dipolar_z,
    }


def _calculate_dipolar_x(
    s_mat: dict[str, np.ndarray], y_mat: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Calculates the dipolar contribution in x direction
    """
    dz2 = DZ2_COEFFICIENT * np.matmul(y_mat["y_2_0"], s_mat["x"])
    dx2y2 = Y_2_2_COEFFICIENT * np.matmul(
        (y_mat["y_2_2"] + y_mat["y_2_n2"]), s_mat["x"]
    )
    dxy = (
        Y_2_2_COEFFICIENT
        * 1j
        * np.matmul((y_mat["y_2_n2"] - y_mat["y_2_2"]), s_mat["y"])
    )
    dxz = Y_2_1_COEFFICIENT * np.matmul((y_mat["y_2_n1"] - y_mat["y_2_1"]), s_mat["z"])

    return -dz2 + dx2y2 + dxy + dxz


def _calculate_dipolar_y(
    s_mat: dict[str, np.ndarray], y_mat: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Calculates the dipolar contribution in y direction
    """
    dz2 = DZ2_COEFFICIENT * np.matmul(y_mat["y_2_0"], s_mat["y"])
    dx2y2 = Y_2_2_COEFFICIENT * np.matmul(
        (y_mat["y_2_2"] + y_mat["y_2_n2"]), s_mat["y"]
    )
    dxy = (
        Y_2_2_COEFFICIENT
        * 1j
        * np.matmul((y_mat["y_2_n2"] - y_mat["y_2_2"]), s_mat["x"])
    )
    dyz = (
        Y_2_1_COEFFICIENT
        * 1j
        * np.matmul((y_mat["y_2_n1"] + y_mat["y_2_1"]), s_mat["z"])
    )

    return -dz2 - dx2y2 + dxy + dyz


def _calculate_dipolar_z(
    s_mat: dict[str, np.ndarray], y_mat: dict[str, np.ndarray]
) -> np.ndarray:
    """
    Calculates the dipolar contribution in z direction
    """
    dz2 = 2.0 * DZ2_COEFFICIENT * np.matmul(y_mat["y_2_0"], s_mat["z"])
    dxz = Y_2_1_COEFFICIENT * np.matmul((y_mat["y_2_n1"] - y_mat["y_2_1"]), s_mat["x"])
    dyz = (
        Y_2_1_COEFFICIENT
        * 1j
        * np.matmul((y_mat["y_2_1"] + y_mat["y_2_n1"]), s_mat["y"])
    )

    return dz2 + dxz + dyz
