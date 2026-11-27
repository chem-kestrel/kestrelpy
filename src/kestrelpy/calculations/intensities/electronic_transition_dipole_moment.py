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

=========================================================================

Module for calculating the electronic transition dipole moment matrix elements."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from kestrelpy.calculations.ligand_field_matrix import (
    DOrbitalBasis,
    MultipoleExpansion,
    get_multipole_expansion,
)
from kestrelpy.calculations.many_electron_matrices import NonZeroMatrixElements
from kestrelpy.calculations.map_eigenvectors_to_operator_matrix import (
    map_eigenvectors_to_operator_matrix,
)
from kestrelpy.calculations.operators import Operators
from kestrelpy.calculations.rotation_matrix import (
    get_rotation_matrix,
    rotate_ligand_field_matrix,
)
from kestrelpy.calculations.spherical_polar_coordinates import (
    calculate_direction_cosine_matrix,
)


class IntensityParameters(NamedTuple):
    # P-transition dipole moment parameters
    psigma: float
    ppix: float
    ppiy: float

    # F-transition dipole moment parameters
    fsigma: float
    fpix: float
    fpiy: float

    # Angles, in radians
    theta: float
    phi: float
    psi: float


@dataclass
class GlobalTransitionDipoleMultipoleExpansion:
    """
    Represent the Transition dipole moment multipole expansions for the P + F contributions.

    The x, y, and z directions are the cartesian directions of the molecular frame.
    """

    x: MultipoleExpansion
    y: MultipoleExpansion
    z: MultipoleExpansion


@dataclass
class ETDMMatrixElements:
    """Electronic transition dipole moment matrix elements in the x, y, and z directions."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def update_matrix(
    matrix: np.ndarray, updates: list[tuple[tuple[DOrbitalBasis, DOrbitalBasis], float]]
) -> np.ndarray:
    for (i, j), value in updates:
        matrix[i, j] = value
        matrix[j, i] = value


def calculate_local_x_transition_dipole_moment_matrix(
    psigma: float, ppix: float, ppiy: float, fsigma: float, fpix: float, fpiy: float
) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.float32)
    update_matrix(
        matrix=matrix,
        updates=[
            (
                (DOrbitalBasis.DXZ, DOrbitalBasis.DZ2),
                np.sqrt(1.0 / 3.0) * (1.5 * psigma - ppix + 1.5 * fpix - fsigma),
            ),
            (
                (DOrbitalBasis.DXY, DOrbitalBasis.DYZ),
                ppiy - 0.25 * fpiy,
            ),
            (
                (DOrbitalBasis.DX2Y2, DOrbitalBasis.DXZ),
                ppix - 0.25 * fpix,
            ),
        ],
    )
    return matrix


def calculate_local_y_transition_dipole_moment_matrix(
    psigma: float, ppix: float, ppiy: float, fsigma: float, fpix: float, fpiy: float
) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.float32)
    update_matrix(
        matrix=matrix,
        updates=[
            (
                (DOrbitalBasis.DYZ, DOrbitalBasis.DZ2),
                np.sqrt(1.0 / 3.0) * (1.5 * psigma - ppiy + 1.5 * fpiy - fsigma),
            ),
            (
                (DOrbitalBasis.DXY, DOrbitalBasis.DXZ),
                ppix - 0.25 * fpix,
            ),
            (
                (DOrbitalBasis.DX2Y2, DOrbitalBasis.DYZ),
                -ppiy + 0.25 * fpiy,
            ),
        ],
    )
    return matrix


def calculate_local_z_transition_dipole_moment_matrix(
    psigma: float, ppix: float, ppiy: float, fsigma: float, fpix: float, fpiy: float
) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.float32)
    update_matrix(
        matrix=matrix,
        updates=[
            (
                (DOrbitalBasis.DZ2, DOrbitalBasis.DZ2),
                2.0 * (psigma + fsigma),
            ),
            (
                (DOrbitalBasis.DXZ, DOrbitalBasis.DXZ),
                2.0 * (ppix + fpix),
            ),
            (
                (DOrbitalBasis.DYZ, DOrbitalBasis.DYZ),
                2.0 * (ppiy + fpiy),
            ),
        ],
    )
    return matrix


def calculate_rotated_xyz_transition_dipole_moment_matrix(
    psigma: float,
    ppix: float,
    ppiy: float,
    fsigma: float,
    fpix: float,
    fpiy: float,
    theta: float,
    phi: float,
    psi: float,
) -> np.ndarray:
    direction_cosines = calculate_direction_cosine_matrix(
        theta=theta, phi=phi, psi=psi, degrees=False
    )
    rotation_matrix = get_rotation_matrix(tta=theta, phi=phi, psi=psi, degrees=False)

    local_x = calculate_local_x_transition_dipole_moment_matrix(
        psigma=psigma, ppix=ppix, ppiy=ppiy, fsigma=fsigma, fpix=fpix, fpiy=fpiy
    )
    local_y = calculate_local_y_transition_dipole_moment_matrix(
        psigma=psigma, ppix=ppix, ppiy=ppiy, fsigma=fsigma, fpix=fpix, fpiy=fpiy
    )
    local_z = calculate_local_z_transition_dipole_moment_matrix(
        psigma=psigma, ppix=ppix, ppiy=ppiy, fsigma=fsigma, fpix=fpix, fpiy=fpiy
    )

    global_transition_dipole_moment_matrix = np.zeros((3, 5, 5), dtype=float)
    for local_tdpmm, direction_cosine in zip(
        [local_x, local_y, local_z],
        # with direction cosines, the rows are the local frame x, y, and z
        # the column values indicate how they project on to the global axis
        [
            direction_cosines[0, :],
            direction_cosines[1, :],
            direction_cosines[2, :],
        ],
    ):
        rotated_matrix = rotate_ligand_field_matrix(
            ligand_field_matrix=local_tdpmm, rotation_matrix=rotation_matrix
        )
        for i in range(3):
            global_transition_dipole_moment_matrix[i, :, :] += (
                rotated_matrix * direction_cosine[i]
            )

    return global_transition_dipole_moment_matrix


def _get_global_multipole_expansion(
    global_tdpmm: np.ndarray,
) -> GlobalTransitionDipoleMultipoleExpansion:
    """Calculate the global x, y, and z transition dipole multiple expansion.

    Args:
        global_tdpmm: Stands for the 'global transition dipole moment matrix'. A NumPy array
            with shape (3, 5, 5), where the first index indicates x, y, and z 5x5 matrices
            respectively.
    """
    return GlobalTransitionDipoleMultipoleExpansion(
        x=get_multipole_expansion(global_tdpmm[0, :, :]),
        y=get_multipole_expansion(global_tdpmm[1, :, :]),
        z=get_multipole_expansion(global_tdpmm[2, :, :]),
    )


def get_electronic_transition_dipole_moment_multipole_expansion(
    *parameters: IntensityParameters,
) -> GlobalTransitionDipoleMultipoleExpansion:
    """
    Calculate the global electronic transition dipole multipole expansion from parameters.
    """
    total_tdm_matrix = np.sum(
        [
            calculate_rotated_xyz_transition_dipole_moment_matrix(*parameter)
            for parameter in parameters
        ],
        axis=0,  # Sum over the first index: x, y, and z.
    )
    return _get_global_multipole_expansion(global_tdpmm=total_tdm_matrix)


def get_electronic_transition_dipole_moment_matrix_elements(
    eigenvectors: np.ndarray,
    matrices: NonZeroMatrixElements,
    global_transition_dipole_multipole_expansion: GlobalTransitionDipoleMultipoleExpansion,
) -> np.ndarray:
    """Calculate the electronic transition dipole moment matrix elements.

    Args:
        eigenvectors: The eigenfunction eigenvectors.
        matrices: The basis matrices.
        global_transition_dipole_multipole_expansion: The global electronic transition
            dipole moment multipole expansion, representing the P and F polarisation.

    Returns:
        A (3 x basis_size x basis_size) complex matrix, whose matrix elements represent
        the electronic transition dipole moment operator between the eigenfunctions I
        and J: <I|r|J>. The first index represents the x, y, and z directions.
    """
    basis_size = eigenvectors.shape[0]
    etdpm = np.zeros((3, basis_size, basis_size), dtype=complex)  # 3 for x, y, and z.
    operator_matrix = np.zeros((basis_size, basis_size))

    for spherical_harmonic_operator in Operators.SphericalHarmonic:
        expansion_coefficients = np.array(
            [
                getattr(
                    global_transition_dipole_multipole_expansion.x,
                    spherical_harmonic_operator,
                ),
                getattr(
                    global_transition_dipole_multipole_expansion.y,
                    spherical_harmonic_operator,
                ),
                getattr(
                    global_transition_dipole_multipole_expansion.z,
                    spherical_harmonic_operator,
                ),
            ]
        )

        indices = matrices.indices[spherical_harmonic_operator]
        values = -matrices.values[spherical_harmonic_operator]

        operator_matrix[:, :] = 0.0
        np.add.at(operator_matrix, (indices[:, 0], indices[:, 1]), values)

        etdpm += (
            # np.newaxis needed for broadcasting:
            # [:, basis_size, basis_size] * [3, :, :] --> [3, basis_size, basis_size]
            map_eigenvectors_to_operator_matrix(
                eigenvectors=eigenvectors,
                operator_matrix=operator_matrix,
            )[np.newaxis, :, :]
            * expansion_coefficients[:, np.newaxis, np.newaxis]
        )
    return etdpm
