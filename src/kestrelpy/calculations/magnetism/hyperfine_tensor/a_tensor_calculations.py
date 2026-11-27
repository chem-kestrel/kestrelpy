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

Calculate A-values in the ligand field framework using spin,
orbital and spherical harmonic operators.

Calculations are implemented from:
    A. Bencini, D. Gatteschi,
    A ligand field model for the hyperfine interaction in the EPR spectra of
    low-symmetry transition metal complexes,
    Journal of magnetic resonance,
    Volume 34, Issue 3,
    1979,
    Pages 653-656
    https://www.sciencedirect.com/science/article/pii/0022236479901495
"""

from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from kestrelpy.calculations.constants import G_E
from kestrelpy.calculations.magnetism.effective_g_values import (
    calculate_effective_g_values,
    get_xyz_index,
)
from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.solve_matrix import solve_matrix


@dataclass
class ASquaredTensor:
    tensor: np.ndarray
    values: np.ndarray
    vectors: np.ndarray
    units: str | None = ""

    def __str__(self) -> str:
        """Returns a table for a set of A values and their eigenvectors"""
        vectors = self.vectors.real + np.abs(self.vectors.imag)
        return tabulate(
            tabular_data=[
                [self.values[i], *np.real((vectors[:, i]))] for i in range(3)
            ],
            headers=[f"A values {self.units}", "x", "y", "z"],
            floatfmt=(".1f", ".3f", ".3f", ".3f"),
            tablefmt="outline",
        )

    @classmethod
    def zero_a_tensor(cls) -> "ASquaredTensor":
        """Return A tensor with NaN for systems which are EPR silent"""
        return cls(
            tensor=np.full((3, 3), np.nan),
            values=np.full(3, np.nan),
            vectors=np.full((3, 3), np.nan),
        )

    def get_axyz(self) -> np.ndarray:
        """projects A values along the x, y, and z directions.

        Obtain the xyz index of the eigenvectors. This then determines
        which eigenvalues correspond to Ax, Ay and Az

        Returns:
            np.ndarray: eigenvalues projected in x,y and z directions
        """
        index = get_xyz_index(self.vectors)

        return self.values[index]

    def convert_to_wavenumbers(self) -> None:
        """Convert A values from MHz to x10-4 cm-1 units"""
        if self.units != "(x10-4 cm-1)":
            self.values = self.values / 2.998
            self.units = "(x10-4 cm-1)"

    def convert_to_mhz(self) -> None:
        """Convert A values from x10-4 cm-1 to MHz units"""
        if self.units != "(MHz)":
            self.values = self.values * 2.998
            self.units = "(MHz)"


def calculate_fermi_contact(p: float, kappa: float) -> np.ndarray[float]:
    """
    Calculate the fermi contact interaction, AFC, using the following equation::

            AFC = P * kappa

    Args:
        p: Hyperfine scaling parameter (in MHz by default)
        kappa: Isotropic hyperfine parameter

    Returns:
        np.ndarray
    """
    fermi_contact = p * kappa
    fermi_contact = np.round(fermi_contact, 2)
    return np.array([fermi_contact, fermi_contact, fermi_contact])


def calculate_a_values(
    a2_tensor: np.ndarray, p: float, units: str | None = ""
) -> ASquaredTensor:
    """
    Function to calculate the A^2 tensor.

    The Sx, Sy and Sz operators are summed and the resulting 3x3 matrix
    is diagonalised. The resulting eigenvalues and eigenvectors are returned
    as an ASquaredTensor object.

    Args:
        a2_tensor: nxn matrix to be diagonalised
        p (float): For the hyperfine scaling parameter, P
        units (str): Units of P and hyperfine values

    Returns:
         ASquaredTensor: Contains diagonal eigenvalues, multiplied by the parameter
            p and their corresponding, eigenvectors, and the A^2 tensor.
    """
    result = solve_matrix(a2_tensor)

    return ASquaredTensor(
        tensor=a2_tensor,
        values=(np.sqrt(result.values)) * p,
        vectors=result.vectors,
        units=units,
    )


def calculate_orbital_values(
    p: float,
    g_factors: GSquaredTensor,
    a_vectors: np.ndarray,
    units: str,
) -> ASquaredTensor:
    """
    Calculate the orbital contribution to the hyperfine coupling (Aorb).

    Effective g values at the position of each A value and their difference
    from the free-electron g value (2.0023) calculated. Multiplication by P
    (g_diff * P) gives each orbital value.

    Args:
        p: Hyperfine scaling parameter
        g_factors: GSquaredTensor containing g values and vectors ordered as gx, gy and gz.
        a_vectors: A^2 tensor eigenvectors
        units: P value units

    Returns:
        ASquaredTensor: orbital values (with sign) and corresponding A tensor eigenvectors.
    """
    if not np.all(np.isfinite(g_factors.values)):
        return ASquaredTensor(
            tensor=np.zeros((3, 3)),
            values=np.array([0.0, 0.0, 0.0]),
            vectors=np.zeros((3, 3)),
            units=units,
        )

    g_effective = calculate_effective_g_values(
        g_vectors=np.eye(3),  # In g frame so g vectors set to identity
        g_values=g_factors.values,
        a_vectors=a_vectors,
    )

    g_diff = g_effective - G_E
    orbital_values = g_diff * p
    orbital_tensor = a_vectors @ np.diag(orbital_values) @ a_vectors.conj().T

    return ASquaredTensor(
        tensor=orbital_tensor,
        values=np.real(orbital_values),
        vectors=a_vectors,
        units=units,
    )


def calculate_a2tensor(
    ops: dict[str, np.ndarray], start: int, final: int, effective_spin: int
) -> np.ndarray:
    """
    Calculates the A^2 tensor for each contribution.

    The spin operators are summed over the spin multiplicity to
    yield a symmetrical, 3x3, matrix.

    Args:
        ops (dict): Dictionary of Sx,Sy,Sz matrices for the desired contribution
            (fermi contact, orbital, dipolar or total)
        start: First eigenfunction index
        final: Final eigenfunction index
        effective_spin: Spin of ground state

    Returns:
        np.ndarray: A^2 tensor as a 3x3 array
    """
    elements = [
        ops["x"][start : final + 1, start : final + 1],
        ops["y"][start : final + 1, start : final + 1],
        ops["z"][start : final + 1, start : final + 1],
    ]

    spin = (effective_spin - 1.0) / 2.0  # S = (N-1)/2
    numerator = _calculate_numerator_terms(elements)
    denominator = np.sum(np.arange(-spin, spin + 1) ** 2)
    return numerator / denominator


def _calculate_numerator_terms(operators: list[np.ndarray]) -> np.ndarray:
    """Calculate the A squared tensor numerator terms

    Args:
        operators: list[np.ndarray]: A tensor operator matrices acting between
            relevant eigenfunctions

    Returns:
        np.ndarray: 3x3 A^2 tensor numerator
    """
    a_squared_numerator = np.zeros((3, 3), dtype=complex)
    for alpha, beta in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        s_alpha = operators[alpha]
        s_beta = operators[beta]

        value = np.sum(np.diagonal(np.matmul(s_alpha, s_beta)))
        a_squared_numerator[alpha, beta] = value
        a_squared_numerator[beta, alpha] = value.conjugate()

    return a_squared_numerator
