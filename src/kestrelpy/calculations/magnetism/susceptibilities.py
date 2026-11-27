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

import numpy as np

from kestrelpy.calculations.constants import (
    AVOGADROS,
    BOHR_MAGNETON,
    BOLTZMANN,
    JOULES_TO_WAVENUMBERS,
)
from kestrelpy.calculations.solve_matrix import solve_matrix

# For understanding units please read:
# Mugiraneza, S., Hallas, A.M. Tutorial: a beginner’s guide to interpreting magnetic susceptibility
# data with the Curie-Weiss law. Commun Phys 5, 95 (2022).
# https://doi.org/10.1038/s42005-022-00853-y

# Calculate molecular paramagnetic susceptibility in SI units: m3 mol-1 cm-1
# Should be approx: 3.27657e-5 m**3 mol**-1 cm**-1
# Introduction to Ligand Fields. New York (N.Y.): Interscience, 1966. APA: Figgis, B. N. (1966).
CONVERSION = (
    AVOGADROS
    * BOHR_MAGNETON
    * BOHR_MAGNETON
    * JOULES_TO_WAVENUMBERS
    * (4.0 * np.pi * 1e-7)
)

# Convert to CGS units: cm3 mol-1 cm-1
# Should be approx: 0.260741 cm**3 mol**-1 cm**-1
# Introduction to Ligand Fields. New York (N.Y.): Interscience, 1966. APA: Figgis, B. N. (1966).
CONVERSION /= 4.0 * np.pi * 1e-6


@dataclass
class ParamagneticSusceptibilityTensor:
    temperature: float
    tensor: np.ndarray
    values: np.ndarray
    vectors: np.ndarray


@dataclass
class ParamagneticSusceptibilityTensorMetadata:
    foz_index: int
    soz_index: int


@dataclass
class ParamagneticSusceptibilityTensors:
    temperatures: np.ndarray
    tensors: list[ParamagneticSusceptibilityTensor]
    metadata: ParamagneticSusceptibilityTensorMetadata


def calculate_paramagnetic_susceptibilities(
    muxyz: np.ndarray,
    energies: np.ndarray,
    temperatures: np.ndarray,
    first_order_zeeman_max_index: int,
    second_order_zeeman_max_index: int,
) -> ParamagneticSusceptibilityTensors:
    susceptibility_tensors = calculate_paramagnetic_susceptibility_tensors(
        muxyz=muxyz,
        energies=energies,
        temperatures=temperatures,
        first_order_zeeman_max_index=first_order_zeeman_max_index,
        second_order_zeeman_max_index=second_order_zeeman_max_index,
    )

    paramagnetic_susceptiblities: list[ParamagneticSusceptibilityTensor] = []
    for i in range(susceptibility_tensors.shape[0]):
        eigensolution = solve_matrix(matrix=susceptibility_tensors[i, :, :])
        paramagnetic_susceptiblities.append(
            ParamagneticSusceptibilityTensor(
                temperature=temperatures[i],
                tensor=susceptibility_tensors[i, :, :],
                values=eigensolution.values,
                vectors=eigensolution.vectors,
            )
        )

    return ParamagneticSusceptibilityTensors(
        temperatures=temperatures,
        tensors=paramagnetic_susceptiblities,
        metadata=ParamagneticSusceptibilityTensorMetadata(
            foz_index=first_order_zeeman_max_index,
            soz_index=second_order_zeeman_max_index,
        ),
    )


def _get_non_degenerate_energies(energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Determine which energies are not degenerate.

    Args:
        energies: Array of Relative energies in cm-1.

    Returns:
        delta_energies: N x N matrix of the energetic difference each {i,j} pair.

        non_degenerate_energies: N x N boolean of matrix, where `True` are an (Ei - Ej)
            > 1.0 cm-1.
    """
    energies_2d = energies[:, np.newaxis]
    # An N x N matrix of energies differences
    # For energies = [0, 0, 10, 10]
    # delta_energies = [
    #    [0, 0, -10, -10],
    #    [0, 0, -10, -10],
    #    [+10, +10, 0, 0],
    #    [+10, +10, 0, 0],
    # ]
    delta_energies = energies_2d - energies_2d.T
    non_degenerate_energies = np.abs(delta_energies) >= 1.0
    return delta_energies, non_degenerate_energies


def calculate_paramagnetic_susceptibility_tensors(
    muxyz: np.ndarray,
    energies: np.ndarray,
    temperatures: np.ndarray,
    first_order_zeeman_max_index: int,
    second_order_zeeman_max_index: int,
) -> np.ndarray:
    """Calculate the paramagnetic susceptibilities for a range of temperatures.

    Args:
        muxyz (np.ndarray): The magnetic moment operator matrices acting between
            eigenstates.
        energies (np.ndarray): The relative eigenvalue energies.
        temperatures (np.ndarray): The range of temperatures to compute paramagnetic
            susecptibilities for.
        first_order_zeeman_max_index (int): The number of thermally occupied states.
        second_order_zeeman_max_index (int): Truncate the number of thermally unoccupied
            states.

    Returns:
        np.ndarray: Shape(temperatures.size, 3, 3)
            Contain the 3x3 susceptibility tensor acting within a cartesian basis for
            each temperature.
    """
    fi = first_order_zeeman_max_index
    si = second_order_zeeman_max_index

    delta_energies, energies_not_zero = _get_non_degenerate_energies(energies[:si])

    # Calculate Boltzman quantities
    boltzmann = np.exp(-energies[:fi, np.newaxis] / (BOLTZMANN * temperatures))
    boltzmann /= boltzmann.sum(axis=0, keepdims=True)
    boltzmann = boltzmann.T

    number_of_temperatures = len(temperatures)
    van_vleck = np.zeros((number_of_temperatures, 3, 3), dtype=float)

    # Get relevant matrices for calculations.
    foz_matrices = np.copy(muxyz[:, :fi, :fi])
    foz_matrices[:, energies_not_zero[:fi, :fi]] = 0.0

    soz_matrices = np.copy(muxyz[:, :fi, :si])
    soz_normalized = np.divide(
        soz_matrices,
        delta_energies[:fi, :si],
        out=np.zeros_like(soz_matrices[:, :fi, :si]),
        where=energies_not_zero[:fi, :si],
    )

    boltzmann_temperature = (BOLTZMANN * temperatures)[:, np.newaxis]
    foz_diagonals = np.einsum(
        "kij,mji->kmi", foz_matrices, foz_matrices, dtype=complex, optimize=True
    )
    soz_diagonals = np.einsum(
        "kij,mij->kmi",
        soz_matrices,
        soz_normalized.conjugate(),
        dtype=complex,
        optimize=True,
    )

    # Indices for `alpha` and `beta` represent the upper triangle
    # of the 3x3 paramagnetic susceptibility tensor.
    for alpha, beta in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        # First-Order Zeeman contribution.
        # sum_i( sum_j[ <i|mua|j><j|mub|i> ] * exp{-Ei/kT} / kT )
        foz_diagonal = foz_diagonals[alpha, beta, :]
        foz_boltzman = foz_diagonal * boltzmann
        foz = np.sum(
            foz_boltzman / boltzmann_temperature,
            axis=1,
            dtype=complex,
        ).real

        # Second-Order Zeeman contribution.
        # sum_i( sum_j[ <i|mua|j><j|mub|i> + <j|mua|i><i|mub|j> / {Ei - Ej} ] * exp{-Ei / kT} )
        soz_diagonal = soz_diagonals[alpha, beta, :]
        soz_boltzman = soz_diagonal * boltzmann
        soz = np.sum(soz_boltzman, axis=1).real * 2.0

        # Paramagnetic susceptibility equation is symmetric.
        van_vleck[:, alpha, beta] = foz - soz
        van_vleck[:, beta, alpha] = foz - soz

    return van_vleck * CONVERSION
