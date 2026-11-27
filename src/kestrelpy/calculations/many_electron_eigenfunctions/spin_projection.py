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

Module for calculating the spin/spin-multiplicity projections of Eigenfunctions."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SpinProjections:
    spins: list[float]
    spin_multiplicities: list[int]
    # Rows are spins/spin-multiplicities, columns are eigenfunctions.
    # Values are the ratio contribution.
    projection: np.ndarray


def get_spin_projection_matrix(
    spin_vectors: np.ndarray, eigenvectors: np.ndarray
) -> np.ndarray:
    """Return the spin projections for columns of eigenvectors.

    Args:
        spin_vectors: M x N vectors, where M is the number of spin multiplicities
            and N is the size of the eigenvectors dimensions.
        eigenvectors: N x N, The eigenvectors of the diagonalisation of the effective
            Hamiltonian. Must have the same dimension as the basis set size.

    Returns:
        np.ndarray: The spin projections. The rows are for each of the possible
            spins/spin-multiplicities for the dn configurations. The columns are
            for each column in the eigenvectors matrix.
    """
    return np.dot(spin_vectors, eigenvectors * eigenvectors.conjugate()).real


def _get_spins_from_spin_multiplicities(spin_multiplicities: list[int]) -> list[float]:
    """Return the spins from the spin multiplicity.

    The spin (S) is related to the spin multiplicity (SM) with the formula:
        >>> SM = 2 * S + 1

    Hence,
        >>> S = (SM - 1)/2
    """
    return [(x - 1.0) / 2.0 for x in spin_multiplicities]


def get_spin_projections(
    spin_multiplicities: list[int],
    spin_vectors: np.ndarray,
    eigenvectors: np.ndarray,
) -> SpinProjections:
    """Return the spin projection class for a set of eigenvectors and the basis spin vectors.

    Args:
        spin_multiplicities: The possible spin_multiplicities (2*Spin+1) for the given dn
            electrons.
        spin_vectors: dim(spin_multiplicities) x N, where the entries are either 1 or 0
            indicating if that spin multiplicity corresponds to the basis function of that
            index.
        eigenvectors: N x N, the complex eigenvectors of the many-electron Hamiltonian.

    Returns:
        SpinProjections class, used for calculating the spin projection for eigenfunctions.
    """
    return SpinProjections(
        spins=_get_spins_from_spin_multiplicities(
            spin_multiplicities=spin_multiplicities
        ),
        spin_multiplicities=spin_multiplicities,
        projection=get_spin_projection_matrix(
            spin_vectors=spin_vectors,
            eigenvectors=eigenvectors,
        ),
    )


def get_eigenfunction_spin_multiplicity_projection(
    index: int,
    spin_projections: SpinProjections,
) -> dict[int, float]:
    """Calculate the spin multiplicity projections for a single eigenfunction.

    Args:
        index: The integer representing a single eigenfunction.

    Returns:
        dictionary representing the spin projection for the integer
            spin multiplicity.
    """
    summed_mapping = spin_projections.projection[:, index]
    return {
        spin_mult: summed_mapping[index]
        for index, spin_mult in enumerate(spin_projections.spin_multiplicities)
    }


def get_level_spin_multiplicity_projection(
    start: int,
    final: int,
    spin_projections: SpinProjections,
) -> dict[int, float]:
    """Calculate the spin multiplicity projections for an electronic level.

    An electronic level is a group of degenerate eigenfunctions.

    Args:
        start: The first index in the slice.
        final: The last index in the slice.

    Returns:
        dictionary representing the spin projection for the integer
            spin multiplicity.
    """
    summed_mapping = np.sum(spin_projections.projection[:, start:final], axis=1)
    return {
        spin_mult: summed_mapping[index]
        for index, spin_mult in enumerate(spin_projections.spin_multiplicities)
    }
