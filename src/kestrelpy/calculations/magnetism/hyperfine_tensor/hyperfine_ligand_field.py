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

from kestrelpy.calculations.many_electron_eigenfunctions.configuration_projection import (
    configuration_weighted_sum,
    get_level_configuration_projection,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ManyElectronEigenfunctions,
)
from kestrelpy.configuration import get_logger

_LOGGER = get_logger(__name__)

AZZ_COEFFICIENTS = np.array([-4, 2, 4, 2, -4]) / 7

KI_CONSTANTS = np.array([-1, 2, 1, 2, -1]) * ((-7 * np.sqrt(5)) / 10)

D_BASIS = np.array(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],  # xy
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],  # yz
        [[-1, 0, 0], [0, -1, 0], [0, 0, 2]],  # z2
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],  # xz
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],  # x2-y2
    ],
    dtype=np.float64,
)
D_BASIS /= np.linalg.norm(D_BASIS, axis=(1, 2), keepdims=True)

_GAUNT = np.einsum("kab,lbc,mca->klm", D_BASIS, D_BASIS, D_BASIS)
_M = _GAUNT.reshape(25, 5).T
RANK2_PROJECTOR = np.linalg.pinv(_M) @ _M


def calculate_4s_mixing_strengths(
    lf_matrix: np.ndarray, lf_vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the degree of 4s admixture into each d orbital.

    The degree of 4s admixture into each d-orbital can be calculated by multiplying
    the mixing of each 3d orbital with the 3d_z2 orbital by angular proportionality
    factors (ki) derived from Wigner 3-j symbols::

        <4s|H|3d_i> = ki * <3d_z2|H|3d_i>

    Where the matrix elements <4s|H|3d_i> are taken from the dz2 row of the rank2
    ligand-field matrix. The numerators are then normalised according to the
    compositions of each ligand-field orbital.

    Args:
        lf_matrix: 5x5 LF matrix aligned to its intrinsic frame.
        lf_vectors: 5x5 eigenvector matrix (columns are states).

    Returns:
        numerators: 1D array of length 5 containing state-specific numerators.
        quantum_weights: 1D array of length 5 containing normalised mixing weights.
    """
    rank2_projection = RANK2_PROJECTOR
    rank2_ligand_field = (rank2_projection @ lf_matrix.reshape(25)).reshape(5, 5)

    dz2_3d_couplings = rank2_ligand_field[2, :]
    basis_4s_coupling = KI_CONSTANTS * dz2_3d_couplings

    max_mixing_strength = float(np.sum(basis_4s_coupling**2))

    numerators = np.zeros(5)
    weights = np.zeros(5)

    for state_idx in range(5):
        mixing_vector = lf_vectors[:, state_idx]
        num = np.sum(mixing_vector * basis_4s_coupling)
        numerators[state_idx] = num

        if max_mixing_strength > 1e-6:
            weights[state_idx] = (num**2) / max_mixing_strength
        else:
            weights[state_idx] = 0.0

    return numerators, weights


def calculate_dipolar_z_sign(
    eigenfunctions: ManyElectronEigenfunctions,
    ligand_field_vectors: np.ndarray,
    start: int = 0,
    final: int = 1,
) -> float:
    """
    Calculate the sign of the dipolar value along the z direction.

    The calculated sign depends on the nature of each singly-occupied ligand-field
    orbitals. dx2-y2 and dxy orbitals give a negative sign, dxz/dyz and dz2 orbitals
    a positive sign. For each singly occupied ligand-field orbital the sign is
    therefore determined by multiplying its squared coefficients by the array::

        [-4/7, 2/7, 4/7, 2/7, -4/7] for dxy, dyz, dz2, dxz, dx2-y2 respectively.

    The resulting values are summed and weighted by the configuration projection
    with the sign of the sum returned.

    Args:
        eigenfunctions: ManyElectronEigenfunctions object
            containing the eigenfunctions in the absence of SOC and their configuration
            projection.
        ligand_field_vectors: 5x5 array of ligand-field eigenvectors.
        start: Start eigenfunction index the dipolar sign is calculated for
        final: Final eigenfunction index the dipolar sign is calculated for.

    Returns:
        Sign of Adip(z)
    """
    assert eigenfunctions is not None
    assert eigenfunctions.configuration_projection is not None
    level_configurations = get_level_configuration_projection(
        start=start,
        final=final,
        configuration_projection=eigenfunctions.configuration_projection,
    )
    eigenvectors = ligand_field_vectors

    orbital_azz = (np.abs(eigenvectors) ** 2).T @ AZZ_COEFFICIENTS
    azz_orbitals, _ = configuration_weighted_sum(
        level_configurations=level_configurations, values=orbital_azz
    )
    azz_relative = np.sum(azz_orbitals)
    return float(np.sign(azz_relative))
