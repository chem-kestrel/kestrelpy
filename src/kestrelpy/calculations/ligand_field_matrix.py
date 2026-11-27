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

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment

from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.magnetism.hyperfine_tensor.hyperfine_ligand_field import (
    D_BASIS,
)
from kestrelpy.calculations.rotation_matrix import (
    ensure_right_handed_axis,
    get_rotation_matrix,
    rotate_ligand_field_matrix,
)
from kestrelpy.calculations.solve_matrix import solve_matrix
from kestrelpy.calculations.spherical_polar_coordinates import euler_angles_from_matrix
from kestrelpy.dataclasses import NumpyDataclass


class DOrbitalBasis(int, Enum):
    """Represents the standard d orbital basis order."""

    DXY = 0
    DYZ = 1
    DZ2 = 2
    DXZ = 3
    DX2Y2 = 4


@dataclass(frozen=True)
class MultipoleExpansion:
    """Class to contain the multipole expansion operators."""

    Y_2_2: complex
    Y_2_1: complex
    Y_2_0: complex
    Y_2_n1: complex
    Y_2_n2: complex
    Y_4_4: complex
    Y_4_3: complex
    Y_4_2: complex
    Y_4_1: complex
    Y_4_0: complex
    Y_4_n1: complex
    Y_4_n2: complex
    Y_4_n3: complex
    Y_4_n4: complex

    def __iter__(self) -> Iterator[complex]:
        return iter(self.__dict__.values())


@dataclass(eq=False)
class LigandField(NumpyDataclass):
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    zero_point_energy: float
    ligand_field_matrix: np.ndarray
    multipole_expansion: MultipoleExpansion

    def rotate_to_g_frame(self, g_factors: GSquaredTensor) -> None:
        """
        Class method to rotate a LF matrix into the g tensor/magnetic frame.
        The rotated LF matrix is then diagonalised giving a new set of eigenvectors.

            Args:
                g_factors: GSquaredTensor object containing g factors and eigenvectors

            Returns:
                LigandField with eigenvectors and LF matrix expressed in the g tensor frame
        """
        g_factors.order_g_factors()
        in_g_frame = rotate_lf_frame(
            ref_vectors=g_factors.vectors, lf_matrix=self.ligand_field_matrix
        )
        self.eigenvalues = in_g_frame.eigenvalues
        self.eigenvectors = in_g_frame.eigenvectors
        self.ligand_field_matrix = in_g_frame.ligand_field_matrix
        self.multipole_expansion = in_g_frame.multipole_expansion


def get_multipole_expansion(ligand_field_matrix: np.ndarray) -> MultipoleExpansion:
    """Return the multipole expansion of a 5x5 ligand field matrix.

    This method can be equally applied to the ligand field perturbation matrix
    (VLFT), representing the perturbation of the d orbital basis by a electrostatic
    field; or a transition dipole moment field for determining intensities.

    The equations relating the multipole expansion elements to the matrix
    elements of the 5x5 ligand field matrix in a d orbitals basis are given by
        >>> Gerloch, M.; McMeeking, R. F. Paramagnetic Properties of
            Unsymmetrical Transition-Metal Complexes. Journal of the
            Chemical Society, Dalton Transactions, 1975, 2443.
            https://doi.org/10.1039/dt9750002443.

    Args:
        ligand_field_matrix: 5x5 ligand field matrix with basis set ordering
            of dxy, dyz, dz2, dxz, dx2y2.

    Returns:
        MultipoleExpansion
    """
    dxy = DOrbitalBasis.DXY.value
    dyz = DOrbitalBasis.DYZ.value
    dz2 = DOrbitalBasis.DZ2.value
    dxz = DOrbitalBasis.DXZ.value
    dx2y2 = DOrbitalBasis.DX2Y2.value

    multipole_expansion = {
        "Y_2_2": complex(
            -np.sqrt(0.4 * np.pi)
            * (
                2.0 * ligand_field_matrix[dx2y2, dz2]
                + np.sqrt(0.75) * ligand_field_matrix[dyz, dyz]
                - np.sqrt(0.75) * ligand_field_matrix[dxz, dxz]
            ),
            -np.sqrt(0.4 * np.pi)
            * (
                -2.0 * ligand_field_matrix[dxy, dz2]
                + np.sqrt(3.0) * ligand_field_matrix[dxz, dyz]
            ),
        ),
        # A CORRECTION HAS BEEN MADE HERE WHEN COMPARED TO TABLE 9.2 OF
        # PG.381 OF MAGNETISM AND LIGAND FIELD ANALYSIS BY M.GERLOCH.
        # THE <YZ|V|XZ> IS ACTUALLY MEANT TO BE <YZ|V|XY>. SEE:
        # Paramagnetic Properties of Unsymmetrical Transition-metal Complexes
        # 1975.
        "Y_2_1": complex(
            -np.sqrt(0.4 * np.pi)
            * (
                np.sqrt(3.0) * ligand_field_matrix[dyz, dxy]
                + np.sqrt(3.0) * ligand_field_matrix[dxz, dx2y2]
                + ligand_field_matrix[dxz, dz2]
            ),
            np.sqrt(0.4 * np.pi)
            * (
                -np.sqrt(3.0) * ligand_field_matrix[dyz, dx2y2]
                + np.sqrt(3.0) * ligand_field_matrix[dxz, dxy]
                + ligand_field_matrix[dyz, dz2]
            ),
        ),
        "Y_2_0": complex(
            -np.sqrt(0.2 * np.pi)
            * (
                2.0 * ligand_field_matrix[dx2y2, dx2y2]
                + 2.0 * ligand_field_matrix[dxy, dxy]
                - ligand_field_matrix[dxz, dxz]
                - ligand_field_matrix[dyz, dyz]
                - 2.0 * ligand_field_matrix[dz2, dz2]
            ),
            0.0,
        ),
        "Y_2_n1": complex(0.0, 0.0),
        "Y_2_n2": complex(0.0, 0.0),
        "Y_4_4": complex(
            np.sqrt(0.7 * np.pi)
            * (ligand_field_matrix[dx2y2, dx2y2] - ligand_field_matrix[dxy, dxy]),
            -np.sqrt(2.8 * np.pi) * ligand_field_matrix[dx2y2, dxy],
        ),
        "Y_4_3": complex(
            np.sqrt(1.4 * np.pi)
            * (ligand_field_matrix[dyz, dxy] - ligand_field_matrix[dxz, dx2y2]),
            np.sqrt(1.4 * np.pi)
            * (ligand_field_matrix[dxz, dxy] + ligand_field_matrix[dyz, dx2y2]),
        ),
        # A CORRECTION HAS BEEN MADE HERE WHEN COMPARED TO TABLE 9.2 OF
        # PG.381 OF MAGNETISM AND LIGAND FIELD ANALYSIS BY M.GERLOCH.
        # THE 4<XZ|V|YZ> IS ACTUALLY MEANT TO BE 2<YZ|V|XY>. SEE:
        # Paramagnetic Properties of Unsymmetrical Transition-metal Complexes
        # 1975.
        "Y_4_2": complex(
            np.sqrt(0.4 * np.pi)
            * (
                np.sqrt(3.0) * ligand_field_matrix[dx2y2, dz2]
                + ligand_field_matrix[dxz, dxz]
                - ligand_field_matrix[dyz, dyz]
            ),
            -np.sqrt(0.4 * np.pi)
            * (
                np.sqrt(3.0) * ligand_field_matrix[dxy, dz2]
                + 2.0 * ligand_field_matrix[dxz, dyz]
            ),
        ),
        "Y_4_1": complex(
            np.sqrt(0.8 * np.pi)
            * (
                -np.sqrt(3.0) * ligand_field_matrix[dxz, dz2]
                + 0.5 * ligand_field_matrix[dyz, dxy]
                + 0.5 * ligand_field_matrix[dxz, dx2y2]
            ),
            np.sqrt(0.8 * np.pi)
            * (
                np.sqrt(3.0) * ligand_field_matrix[dyz, dz2]
                + 0.5 * ligand_field_matrix[dyz, dx2y2]
                - 0.5 * ligand_field_matrix[dxz, dxy]
            ),
        ),
        "Y_4_0": complex(
            np.sqrt(np.pi)
            * 0.2
            * (
                ligand_field_matrix[dx2y2, dx2y2]
                + ligand_field_matrix[dxy, dxy]
                - 4.0 * ligand_field_matrix[dxz, dxz]
                - 4.0 * ligand_field_matrix[dyz, dyz]
                + 6.0 * ligand_field_matrix[dz2, dz2]
            ),
            0.0,
        ),
        "Y_4_n1": complex(0.0, 0.0),
        "Y_4_n2": complex(0.0, 0.0),
        "Y_4_n3": complex(0.0, 0.0),
        "Y_4_n4": complex(0.0, 0.0),
    }

    multipole_expansion.update(
        **{
            "Y_2_n1": _get_inverse_multiple_expansion(
                multipole_expansion["Y_2_1"],
                -1,
            ),
            "Y_2_n2": _get_inverse_multiple_expansion(
                multipole_expansion["Y_2_2"],
                -2,
            ),
            "Y_4_n1": _get_inverse_multiple_expansion(
                multipole_expansion["Y_4_1"],
                -1,
            ),
            "Y_4_n2": _get_inverse_multiple_expansion(
                multipole_expansion["Y_4_2"],
                -2,
            ),
            "Y_4_n3": _get_inverse_multiple_expansion(
                multipole_expansion["Y_4_3"],
                -3,
            ),
            "Y_4_n4": _get_inverse_multiple_expansion(
                multipole_expansion["Y_4_4"],
                -4,
            ),
        },
    )

    return MultipoleExpansion(**multipole_expansion)


def _get_inverse_multiple_expansion(value: complex, q: int) -> complex:
    """Compute the inverse multipole expansion.

    For example, if you have the multipole expansion for Ck,q then the inverse
    computes the value for Ck,-q.

    Args:
        value: The numerical value of Ck,q.
        q: the projection of `k` in Ck,q.

    Returns:
        The value for Ck,-q.
    """
    return value.conjugate() * pow(-1.0, q)


def rotate_lf_frame(ref_vectors: np.ndarray, lf_matrix: np.ndarray) -> LigandField:
    """Rotate the Ligand Field matrix into the g tensor/magnetic frame.

    Performs a unitary transformation to orient the ligand field matrix
    in the same orientation as that of the g^2 tensor. The new LF
    matrix, V is given by the application of a rotation matrix, U, on the
    original LF matrix, M::

        V = U*M*U(T)

    This ensures the computed ligand field eigenfunctions correspond with the
    magnetic, rather than arbitrary cartesian, axes.

    Args:
        ref_vectors: Vectors describing frame the ligand field is rotated
            to, i.e. the g tensor eigenvectors for rotation to the magnetic frame.
        lf_matrix: 5x5 LF matrix

    Returns:
        LigandField: Object containing the rotated ligand field matrix along with
            its eigenvectors, eigenvalues and multipole expansion.
    """
    g_rotation_matrix = ensure_right_handed_axis(vectors=ref_vectors)

    angles = euler_angles_from_matrix(rotation_matrix=g_rotation_matrix)

    lf_rotation_matrix = get_rotation_matrix(
        tta=angles[1], phi=angles[0], psi=angles[2]
    )
    rotated_lf_matrix = rotate_ligand_field_matrix(
        ligand_field_matrix=lf_matrix,
        rotation_matrix=lf_rotation_matrix,
    )
    return get_ligand_field_parameters(rotated_lf_matrix)


def _lf_quadrupole_tensor(lf_matrix: np.ndarray) -> np.ndarray:
    """Computes the exact quadrupole tensor from the LF matrix."""
    lf = solve_matrix(matrix=lf_matrix)
    orbitals = np.tensordot(lf.vectors.T, D_BASIS, axes=1)  # Orbital as 3x3 matrix
    quadrupole = np.tensordot(lf.values, orbitals @ orbitals, axes=1)
    return quadrupole - np.trace(quadrupole) / 3.0 * np.eye(3)


def determine_diagonal_frame(
    ligand_field_matrix: np.ndarray,
    g_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the intrinsic LF axes and aligns the arbitrary g-vectors to them.
    If the g-vectors are rotated or calculated in this frame then the xyz
    eigenvectors correspond to the magnetic xyz directions.

    Args:
        ligand_field_matrix: 5x5 ligand field matrix
        g_vectors: 3x3 g-tensor eigenvectors in the cartesian frame

    Returns:
        diagonal_g_vectors: Right-handed, aligned g-tensor eigenvectors.
        diagonal_lf_vectors: Right-handed LF intrinsic eigenvectors.
        xyz_order: The permutation applied to the original g_vectors.
    """
    q_tensor = _lf_quadrupole_tensor(lf_matrix=ligand_field_matrix)
    q_result = solve_matrix(matrix=q_tensor)

    z_lf = int(np.argmax(np.abs(q_result.values)))
    sign = 1.0 if q_result.values[z_lf] >= 0 else -1.0
    xy_lf = sorted(
        (i for i in range(3) if i != z_lf), key=lambda i: sign * q_result.values[i]
    )

    diagonal_lf_vectors = ensure_right_handed_axis(
        vectors=q_result.vectors[:, xy_lf + [z_lf]]
    )
    overlaps = (diagonal_lf_vectors.T @ g_vectors) ** 2
    _, xyz_order = linear_sum_assignment(-np.abs(overlaps))
    diagonal_g_vectors = ensure_right_handed_axis(vectors=g_vectors[:, xyz_order])

    return diagonal_g_vectors, diagonal_lf_vectors, xyz_order


def get_ligand_field_parameters(ligand_field_matrix: np.ndarray) -> LigandField:
    """Take an input 5x5 symmetrical ligand field matrix and return the ligand field parameters."""
    multipole_expansion = get_multipole_expansion(
        ligand_field_matrix=ligand_field_matrix
    )
    solution = solve_matrix(matrix=ligand_field_matrix)
    return LigandField(
        eigenvalues=solution.values,
        eigenvectors=solution.vectors,
        zero_point_energy=solution.reference_value,
        ligand_field_matrix=ligand_field_matrix,
        multipole_expansion=multipole_expansion,
    )
