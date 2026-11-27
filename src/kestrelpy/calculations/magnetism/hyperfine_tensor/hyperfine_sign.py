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

Module for calculating the sign of the hyperfine interaction,

The hyperfine coupling is calculated as the sum of 3 contributions:
Fermi contact, dipolar and orbital interactions.

The methodology is summarised as follows:
1.  Determine the fermi contact sign. In the absence of 4s admixture,
    This is negative for all 3d metals except Titanium, Chromium and Nickel,
    which give positive values due to their negative nuclear g values.
2.  Determine the orbital contribution sign. Orbital values in the x,y,z
    directions are compared with the effective g value in that direction.
    If g effective is less than ge (2.0023), the orbital values are negative,
    otherwise they're positive.
3.  Determine the sign of the dipolar contribution. As the dipolar tensor is
    traceless, only the sign of Adip(z) needs calculating.
        i.  This is determined by multiplying the eigenvectors of each singly-occupied
            orbital by angular factors for each d orbital. The sum is then weighted by
            the configuration projection to give the sign of Adip(z).
        ii. Since the dipolar tensor is traceless, the remaining signs can then
            be calculated by minimising their sum to 0.
    This calculated sign is flipped for metals with positive nuclear g values
    (i.e. Ti, Cr and Ni).

    EXAMPLE: for lsAA9 (|Az| = 466 MHz):
    fermi contact = -251 MHz
    orbital contribution: x = 73 MHz, y = 99 MHz, z = 326 Mhz
    dipolar contribution (magnitudes): |x| = 288 MHz, |y| = 245 MHz, |z| = 540 MHz

    Since the unpaired electron for LsAA9 is in a dx2-y2 orbital, the signs of the
    dipolar values are: x = 288 MHz, y = 245 MHz and z = -540 Mhz.

4.  The fermi contact, orbital and dipolar contributions in the x,y,z
    directions are then summed to give the absolute value of each A value
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

from kestrelpy.calculations.magnetism.effective_g_values import get_xyz_index
from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_calculations import (
    ASquaredTensor,
)
from kestrelpy.calculations.rotation_matrix import ensure_right_handed_axis
from kestrelpy.configuration import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class ASignResults:
    fermi: np.ndarray
    dipolar: np.ndarray
    orbital: np.ndarray
    total: np.ndarray
    total_vectors: np.ndarray
    units: str | None = ""

    def __str__(self) -> str:
        data = [
            ["Total", *[f"{val:.1f}" for val in self.total]],
            ["Fermi", *[f"{val:.1f}" for val in self.fermi]],
            ["Dipolar", *[f"{val:.1f}" for val in self.dipolar]],
            ["Orbital", *[f"{val:.1f}" for val in self.orbital]],
            ["---", "---", "---", "---"],
            ["Orientation"],
            ["x", *[f"{val:.3f}" for val in np.real((self.total_vectors[0]))]],
            ["y", *[f"{val:.3f}" for val in np.real((self.total_vectors[1]))]],
            ["z", *[f"{val:.3f}" for val in np.real((self.total_vectors[2]))]],
        ]

        return tabulate(
            tabular_data=data,
            headers=[
                f"Contribution {self.units}",
                "A1",
                "A2",
                "A3",
            ],
            stralign="center",
            tablefmt="outline",
        )

    @classmethod
    def zero_a_signs(cls) -> "ASignResults":
        """Return ATensorSign with NaN values for systems that are EPR silent"""
        return cls(
            fermi=np.full(3, np.nan),
            dipolar=np.full(3, np.nan),
            orbital=np.full(3, np.nan),
            total=np.full(3, np.nan),
            total_vectors=np.full((3, 3), np.nan),
        )

    def calculate_a_iso(self) -> float:
        """Calculate the value of Aiso where Aiso = (A1 + A2 + A3) / 3"""
        a_iso = np.sum(self.total) / 3
        a_iso = round(a_iso, 2)
        return float(a_iso)


@dataclass
class ATensorSign:
    orbital: ASquaredTensor
    dipolar: ASquaredTensor
    fermi_contact: np.ndarray
    metal: str
    g_factors: GSquaredTensor
    dipolar_z_sign: float | None = None

    def get_sign(self) -> ASignResults:
        """
        Determines the sign of the calculated A values

        Initially calculates the sign of the fermi contact, dipolar and
        orbital contributions. These are then summed to give the
        signs of the calculated A values, allowing comparison to DFT.

        Returns:
            ASignResults: ASignResults object containing the signs of the fermi contact,
                dipolar, orbital and total contributions
        """
        return calculate_a_value_signs(self)


def calculate_fermi_sign(fermi: np.ndarray, metal: str) -> np.ndarray:
    """
    Determines the sign of the Fermi contact contribution. Fo no 4s admixture,
    it is positive for Ti, Cr and Ni and negative for all other 3d metals.

    Args:
        fermi (np.ndarray): Fermi contact contribution to the hyperfine coupling
        metal (str): A 3d transition metal

    Returns:
        np.ndarray: Fermi contact values with their sign
    """
    # Fermi contact is positive for Ti, Cr and Ni, negative for other 3d metals
    if metal.lower() not in ("ti", "cr", "ni"):
        fermi = fermi * -1
    return fermi


def calculate_dipolar_sign(
    values: np.ndarray,
    vectors: np.ndarray,
    metal: str,
    dipolar_z_sign: float,
) -> np.ndarray:
    """Determine the sign and ordering of the dipolar hyperfine tensor components.

    Pairs dipolar values with their x,y,z directions by vector overlap and applies
    the calculated sign to the z value.

    As dip_x + dip_y + dip_z = 0, the signs of dip_x and dip_y can then be determined
    by the requirement that the dipolar values sum to zero. Thus, the component with
    the largest value must be of opposite sign to the two smaller components.
    The combination that sums to 0 is then returned.

    Args:
        values: 1x3 array of dipolar values
        vectors: 3x3 array of dipolar vectors expressed in the frame where
            x,y and z vectors correspond to the magnetic x, y and z directions.
        metal: Metal the hyperfine coupling is calculated for
        dipolar_z_sign: Sign of Adip(z)

    Returns:
        np.ndarray: dipolar contribution with signs in their original order.
    """
    _, col_indices = linear_sum_assignment(-np.abs(vectors))
    magnitudes = np.abs(values)[col_indices]
    max_dipolar_index = int(np.argmax(magnitudes))

    # Adip(z) sign determined by dipolar_z_sign. Adip(x/y) are determined by magnitudes.
    signs = np.empty(3, dtype=np.float64)
    signs[2] = dipolar_z_sign
    if max_dipolar_index == 2:
        signs[0] = dipolar_z_sign * -1
        signs[1] = dipolar_z_sign * -1
    elif max_dipolar_index == 0:
        signs[0] = dipolar_z_sign * -1
        signs[1] = dipolar_z_sign
    else:
        signs[0] = dipolar_z_sign
        signs[1] = dipolar_z_sign * -1

    signed_dipolar = signs * magnitudes

    residual = abs(signed_dipolar.sum())
    if residual > 0.01 * magnitudes.max():
        _LOGGER.warning("Dipolar signs do not sum to zero (residual %.3g)", residual)

    metal_sign = -1 if metal.lower() in ("ti", "cr", "ni") else 1
    dipolar = np.empty(3, dtype=np.float64)
    dipolar[col_indices] = signed_dipolar * metal_sign

    return dipolar


def calculate_a_value_signs(a_signs: ATensorSign) -> ASignResults:
    """
    Sums the signed dipolar, fermi contact, and orbital contributions
    to yield signed A values. The xyz projection for each value is taken
    to ensure that the correct values are summed together.

    Args:
        a_signs (ATensorSign): Object containing values for the fermi contact,
            dipolar and orbital contributions and g values; vectors for hyperfine values
            and g factors; parameters. Vectors should be relative to the magnetic frame.

    Returns:
        ASignResults: ASignResults object containing the signs of the fermi contact,
                dipolar, orbital and total contributions. Vectors are returned relative
                to the magnetic frame.
    """
    if a_signs.dipolar_z_sign is not None:
        dipolar_values = calculate_dipolar_sign(
            values=a_signs.dipolar.values,
            vectors=a_signs.dipolar.vectors,
            metal=a_signs.metal,
            dipolar_z_sign=a_signs.dipolar_z_sign,
        )
        order = get_xyz_index(vectors=a_signs.dipolar.vectors)
        a_vectors = ensure_right_handed_axis(vectors=a_signs.dipolar.vectors[:, order])

        dipolar_sign = dipolar_values[order]
        orbital_sign = a_signs.orbital.values[order]

    else:
        dipolar_sign = a_signs.dipolar.values
        orbital_sign = a_signs.orbital.values
        a_vectors = a_signs.dipolar.vectors

    return ASignResults(
        fermi=a_signs.fermi_contact,
        dipolar=dipolar_sign,
        orbital=orbital_sign,
        total=(dipolar_sign + orbital_sign + a_signs.fermi_contact),
        total_vectors=a_vectors,
        units=a_signs.dipolar.units,
    )


def _extract_sign(a_sign: np.ndarray) -> list[int]:
    """
    Extracts the signs of the A values

    Args:
        a_sign (np.ndarray): Sum of individual contributions with signs

    Returns:
         list[int]: Signs for each A value (in order: Ax, Ay, Az)
    """
    sign_x = int(np.sign(a_sign[0]))
    sign_y = int(np.sign(a_sign[1]))
    sign_z = int(np.sign(a_sign[2]))

    return [sign_x, sign_y, sign_z]
