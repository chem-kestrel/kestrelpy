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

Calculations to predict the 'relaxation time of quantum tunneling of magnetization
and the effective barrier of magnetic reversal for a Kramers single-ion magnet'.

Reference: Phys. Chem. Chem. Phys., 2020, 22, 9923-9933
"""

from dataclasses import dataclass

import numpy as np

from kestrelpy.calculations.constants import BOHR_MAGNETON, BOLTZMANN, PLANCKS
from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor


@dataclass
class QtmEnergyBarrier:
    effective_barrier: np.ndarray[float]
    kramers_doublet_contribution: np.ndarray[float]
    temperatures: np.ndarray[float]


def calculate_qtm_energy_barrier(
    relative_energies: np.ndarray[float],
    g_tensors: list[GSquaredTensor],
    temperatures: np.ndarray[float],
) -> QtmEnergyBarrier:
    """Calculate the effective barrier of magnetic reversal for Kramer's doublets.

    Args:
        relative_energies: The relative energies of each Kramer's doublet.
        g_tensors: The g**2 tensors for each Kramer's doublet.
        temperatures: The range of temperatures to calculate the barrier for.

    Returns
        QtmEnergyBarries: The results of the calculations.
    """
    number_of_levels = len(relative_energies)
    number_of_temperatures = len(temperatures)
    kd_contribution = np.zeros((number_of_levels, number_of_temperatures))

    # Define constants.
    kT = BOLTZMANN * temperatures
    Z = np.sum([np.exp(-energy / kT) for energy in relative_energies])

    effective_barrier: float = 0.0
    i: int = 0
    normalisation: float = 0.0
    for g_tensor, energy in zip(g_tensors, relative_energies):
        # Equation 10d and 10e from Phys. Chem. Chem. Phys., 2020, 22, 9923-9933.
        gx, gy, gz = g_tensor.get_gxyz()
        wqtm_eff = _calculate_g_factor_anisotropy(gx=gx, gy=gy, gz=gz)
        wqtm = wqtm_eff * np.exp(-energy / kT) / Z

        # Equation 11a (apply normalisation outside of loop).
        effective_barrier += wqtm * energy
        normalisation += wqtm

        kd_contribution[i, :] = wqtm
        i += 1

    # Normalise effective barrier as according to equation 11a.
    effective_barrier /= normalisation

    kd_contribution = _normalise_kramers_doublet_contribution_by_temperature(
        kramers_doublet_contribution=kd_contribution
    )

    return QtmEnergyBarrier(
        effective_barrier=effective_barrier,
        kramers_doublet_contribution=kd_contribution,
        temperatures=temperatures,
    )


def _normalise_kramers_doublet_contribution_by_temperature(
    kramers_doublet_contribution: np.ndarray,
) -> np.ndarray:
    normalisation_term = np.linalg.norm(kramers_doublet_contribution, axis=0)
    return (kramers_doublet_contribution / normalisation_term) ** 2.0


def calculate_qtm_time(
    gx: float, gy: float, gz: float, avg_internal_field: float = 20e-3
) -> float:
    """Computes the relaxation time from Quantum Tunnelling of Magnetisation.

    Args:
        gx, gy, gz: The projected EPR g-factors.
        avg_internal_field: The average magnetic field due to dipolar and hyperfine
            interactions. Default = 20 mT.

    Returns:
        rate: The relaxation time in seconds.

    Notes:
        (1) Phys. Chem. Chem, Phys., 2020, 22, 9923
    """
    constants = BOHR_MAGNETON / PLANCKS
    wqtm = (
        avg_internal_field
        * constants
        * _calculate_g_factor_anisotropy(gx=gx, gy=gy, gz=gz)
    )
    return float(0.5 / wqtm)


def _calculate_g_factor_anisotropy(gx: float, gy: float, gz: float) -> float:
    """Calculate the g-factor anisotropy factor.

    Args:
        gx, gy, gz: The projected EPR g-factors.

    Returns:
        float: The g-factor anisotropy.
    """
    gxy2 = (gx**2.0) + (gy**2.0)
    gz2 = gz**2.0
    return float(0.5 * gxy2 / np.sqrt(gxy2 + gz2))
