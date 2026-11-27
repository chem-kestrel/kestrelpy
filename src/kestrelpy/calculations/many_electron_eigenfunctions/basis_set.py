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

Module for calculating the Basis Set for the many-electron Hamiltonian.

The basis set for the matrices is a series of Russell-Saunders terms, expanded
as a set of coupled angular momentum numbers.
"""

from dataclasses import dataclass

import numpy as np

from kestrelpy.calculations.utils import (
    get_many_d_electron_basis_size,
    get_minimal_d_config,
)

RUSSELL_SAUNDERS_TERM_SYMBOLS = {
    1: ["2D"],
    2: ["1S", "1D", "1G", "3P", "3F"],
    3: ["2P", "2D1", "2D2", "2F", "2G", "2H", "4P", "4F"],
    4: [
        "1S1",
        "1S2",
        "1D1",
        "1D2",
        "1F",
        "1G1",
        "1G2",
        "1I",
        "3P1",
        "3P2",
        "3D",
        "3F1",
        "3F2",
        "3G",
        "3H",
        "5D",
    ],
    5: [
        "2S",
        "2P",
        "2D1",
        "2D2",
        "2D3",
        "2F1",
        "2F2",
        "2G1",
        "2G2",
        "2H",
        "2I",
        "4P",
        "4D",
        "4F",
        "4G",
        "6S",
    ],
}

ORBITAL_ANGULAR_MOMENTUM = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}

POSSIBLE_SPIN_MULTIPLICITIES = {
    1: [2],
    2: [1, 3],
    3: [2, 4],
    4: [1, 3, 5],
    5: [2, 4, 6],
}


class GenerateBasisError(Exception):
    """Raised when an issue is encountered creating the basis."""


@dataclass(frozen=True)
class CoupledBasisFunctions:
    """The angular momentum numbers for a coupled basis function."""

    spin_angular_momentum: float
    orbital_angular_momentum: float
    total_angular_momentum: float
    total_angular_momentum_projection: float


@dataclass(frozen=True)
class ManyElectronBasis:
    d_electrons: int
    russell_saunders_basis: list[str]
    spin_multiplicities: list[int]
    basis_size: int
    basis_functions: list[CoupledBasisFunctions]
    spin_vectors: np.ndarray


def _get_spin_vectors(
    spin_multiplicities: list[int], coupled_basis_functions: list[CoupledBasisFunctions]
) -> np.ndarray:
    """Return a masked array if the basis functions corresponds to the given spin multiplicity.

    Args:
        spin_multiplicities: List of spin multiplicities (2*spin + 1).
        coupled_basis_functions: The list of J, Mj, L, and S quantum numbers for the basis set.

    Returns:
        np.ndarray with a shape of (len(spin_multiplicities), len(coupled_basis_functions)).
            For example, if we have spin multiplicities 1 and 3 and the coupled basis functions
            with spin multiplicities of [1, 1, 3, 3, 3], then the output matrix will look like:
            >>> np.ndarray(
                [1, 1, 0, 0, 0],  <-- 1 meaning function with spin mult of `1`.
                [0, 0, 1, 1, 1],  <-- 1 meaning function with spin mult of `3`.
            )
    """
    return np.array(
        [
            np.array(
                [
                    1.0
                    if int(basis_function.spin_angular_momentum * 2 + 1)
                    == spin_multiplicity
                    else 0.0
                    for basis_function in coupled_basis_functions
                ]
            )
            for spin_multiplicity in spin_multiplicities
        ]
    )


def _get_russell_saunders_basis_set(
    russell_saunders_terms: list[str],
) -> list[CoupledBasisFunctions]:
    """Construct the Russell-Saunders basis set in terms of J, Mj, L, and S quantum numbers.

    Args:
        russell_saunders_terms: The list of string representations of Russell-Saunders terms
            such as `1S`, `1D`, `1G`, `3P`, `3F` etc.

    Returns:
        The coupled basis functions, each representing a basis function.
    """
    basis_functions = []
    for term in russell_saunders_terms:
        s_value = (int(term[0]) - 1) / 2
        l_value = float(ORBITAL_ANGULAR_MOMENTUM[term[1]])
        j_value = l_value + s_value
        j_values = np.arange(abs(l_value - s_value), 1 + (l_value + s_value))
        for j_value in j_values:
            mj_values = np.arange(-j_value, 1 + j_value)
            basis_functions.extend(
                [
                    CoupledBasisFunctions(
                        spin_angular_momentum=s_value,
                        orbital_angular_momentum=l_value,
                        total_angular_momentum=j_value,
                        total_angular_momentum_projection=mj_value,
                    )
                    for mj_value in mj_values
                ]
            )
    return basis_functions


def _validate_coupled_basis_functions(
    d_electrons: int, coupled_basis_functions: list[CoupledBasisFunctions]
) -> None:
    """Check that the number of coupled basis functions matches the expected size.

    Args:
        d_electrons: The d electron count.
        coupled_basis_functions: The generated list of coupled basis functions (quantum
            numbers J, Mj, L, and S)

    Raises:
        GenerateBasisError if the number of coupled basis functions does not match
            the expected number of coupled basis functions.
    """
    expected_length = get_many_d_electron_basis_size(d_electrons=d_electrons)
    if len(coupled_basis_functions) != expected_length:
        raise GenerateBasisError(
            "Length of basis functions does not match the predicted size. "
            f"{expected_length=}, {len(coupled_basis_functions)=}"
        )


def _get_spin_multiplicities(d_electrons: int) -> list[int]:
    """Return the spin multiplicities for the given d electron count."""
    return list(
        POSSIBLE_SPIN_MULTIPLICITIES[get_minimal_d_config(d_electrons=d_electrons)]
    )


def _get_russell_saunders_terms(d_electrons: int) -> list[str]:
    """Return the associated Russell-Saunders terms for the given d electron count."""
    return list(
        RUSSELL_SAUNDERS_TERM_SYMBOLS[get_minimal_d_config(d_electrons=d_electrons)]
    )


def get_many_electron_basis(d_electrons: int) -> ManyElectronBasis:
    """Return the many electron basis representation for the d electron count.

    Args:
        d_electrons: Number of d electrons.

    Returns:
        ManyElectronBasis
    """
    russell_saunders_terms = _get_russell_saunders_terms(d_electrons=d_electrons)
    basis_set = _get_russell_saunders_basis_set(
        russell_saunders_terms=russell_saunders_terms
    )
    _validate_coupled_basis_functions(
        d_electrons=d_electrons, coupled_basis_functions=basis_set
    )
    spin_multiplicities = _get_spin_multiplicities(d_electrons=d_electrons)
    spin_vectors = _get_spin_vectors(
        spin_multiplicities=spin_multiplicities, coupled_basis_functions=basis_set
    )
    return ManyElectronBasis(
        d_electrons=d_electrons,
        russell_saunders_basis=russell_saunders_terms,
        spin_multiplicities=spin_multiplicities,
        basis_size=len(basis_set),
        basis_functions=basis_set,
        spin_vectors=spin_vectors,
    )
