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
from typing import Protocol

import numpy as np

from kestrelpy.calculations.constants import (
    AVOGADROS,
    BOLTZMANN,
    CHARGE_OF_ELECTRON_ESU,
    MASS_OF_ELECTRON_KG,
    PLANCKS_CM,
    SPEED_OF_LIGHT_CM,
)

# Taken from:
#   >>> Creating UV/visible plots from the results of excited states calculations Gaussiancom.
#       Available at: https://gaussian.com/uvvisplot/ (Accessed: 07 August 2025).
# The collection of constants used to convert the electronic |transition-dipole-moment|**2
# (Debye**2) to have the units of cm.
# Should be approx 4.702e-7
CONVERT_TDM_DEBYE: float = (
    8.0
    * np.pi
    * np.pi
    * MASS_OF_ELECTRON_KG  # mass of electron (kg)
    * SPEED_OF_LIGHT_CM  # speed of light (cm s-1)
    / (
        3.0
        * PLANCKS_CM  # plancks constant (kg cm2 s-2)
        * (CHARGE_OF_ELECTRON_ESU**2.0)  # charge of electron (esu) --> (esu^2)
    )
)

# NOTE:
# >>> convert_tdm_to_molar_absorption_coefficient = (
#   CONVERT_FOSC_TO_ABSORPTION * CONVERT_TDM_DEBYE
# )
# Expression taken from J. A. Schellman, Chem. Rev., 1975, 75, 323–331.
# >>> expected_output = (
#     8 * np.pi * np.pi * np.pi
#     * AVOGADROS
#     / (
#         3
#         * np.sqrt(np.pi)
#         * 1000
#         * np.log(10)
#         * PLANCKS_CM * 1000  # convert to g cm2 s-1
#         * SPEED_OF_LIGHT_CM
#     )
# )
# assert convert_tdm_to_molar_absorption_coefficient == expected_output
CONVERT_FOSC_TO_ABSORPTION: float = 1.3062974e8  # fosc --> L mol-1 cm-1

# Expression taken from J. A. Schellman, Chem. Rev., 1975, 75, 323–331.
CONVERT_ROTARY_STRENGTH_TO_DIFFERENTIAL_ABSORPTION: float = (
    32
    * np.pi
    * np.pi
    * np.pi
    * AVOGADROS
    / (
        3.0
        * np.sqrt(np.pi)
        * 1000
        * np.log(10)
        * PLANCKS_CM
        * 1000  # erg s
        * SPEED_OF_LIGHT_CM
    )
)


@dataclass
class SpectroscopicTransition:
    """A spectroscopic transition.

    This represents the transition in the given spectrum including temperature-dependent
    terms (e.g. Boltzmann weighting).
    """

    energy: float  # wavenumbers (cm-1)
    intensity: float


@dataclass
class Spectrum:
    """
    Represents the spectroscopic energies and intensities including temperature-dependent terms
    (e.g. Boltzmann weighting).
    """

    temperature: float  # Kelvin
    magnetic_field: float  # Tesla
    transitions: list[SpectroscopicTransition]


def convolute_spectrum(
    spectrum: Spectrum,
    energy_range: np.ndarray[float],
    fwhm: float,
    summed: bool = False,
) -> np.ndarray:
    """Return the convoluted gaussian for a spectrum object.

    Args:
        spectrum: The spectrum object for Absorption, Circular Dichroism or Magnetic
            Circular Dichroism.
        energy_range: The x values of the spectrum in wavenumbers (cm-1).
        fwhm: The full-width half maximum of each gaussian band, in wavenumbers (cm-1).
        summed: Sum all of the gaussian contributions into a single convoluted spectrum.
            Default = False.

    Returns:
        The 'y' values for each transition band with shape `n_bands x n_xvals`. To get
            the summed spectrum call `np.sum(convoluted_spectrum, axis=0)`.
    """
    transition_energies = np.array([t.energy for t in spectrum.transitions])
    transition_intensities = np.array([t.intensity for t in spectrum.transitions])
    gaussians = calculate_gaussian(
        energy_range=energy_range,
        transition_energies=transition_energies,
        intensities=transition_intensities,
        fwhm=fwhm,
    )
    if summed:
        return np.sum(gaussians, axis=0)
    return gaussians


def calculate_oscillator_strength(
    transition_dipole_moment_squared: float,
    energy: float,
) -> float:
    """Calculate the dimensionless oscillator strength for the transition-dipole-moment (esu cm).

    Args:
        transition_dipole_moment_squared: |<i|r|j>|**2 in units of esu**2 cm**2
        energy: The transition energy, in units cm-1.

    Returns:
        The dimensionless oscillator strength.
    """
    return float(CONVERT_TDM_DEBYE * energy * transition_dipole_moment_squared)


def calculate_molar_absorption_coefficient(fosc: float) -> float:
    """Convert oscillator strength to molar absorption coefficient.

    NOTE: molar absorption coefficient has units of L mol-1 cm-1.
    """
    return float(fosc * CONVERT_FOSC_TO_ABSORPTION)


def calculate_gaussian(
    energy_range: np.ndarray,
    transition_energies: np.ndarray,
    intensities: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    """Calculate the Gaussian band shape from the oscillator strengths and transition energies.

    Args:
        energy_range: The "x" values of the graph, in wavenumbers (cm-1).
        transition_energies: The transition energies, in wavenumbers (cm-1).
        intensities: The intensities for each transition energy. This array must have
            the same length as `transition_energies`.
        fwhm: The Full-Width-Half-Maximum (FWHM) of the band, in wavenumbers (cm-1).
            conversion: The conversion factor.

    Returns:
        The "y" values of the Gaussian curve.

    References:
        1.  https://gaussian.com/uvvisplot/ (Accessed: 07 August 2025)

            NOTE: On the website, the gaussian [equation 1] is defined where sigma
            'is the half-width of the Gaussian band at epsilon = epsilon_max / e'.
            Hence, from https://mathworld.wolfram.com/GaussianFunction.html we
            need to sub 2*sigma**2 in equations (4)-(8) with sigma**2.
    """
    sigma = fwhm / 1.6651092223153954  # 2.0 * sqrt(ln(2))
    return (
        intensities[:, None]
        * np.exp(
            -(((energy_range[None, :] - transition_energies[:, None]) / sigma) ** 2.0)
        )
        / sigma
    )


def max_thermally_occupied_energy(
    energies: np.ndarray, temperature: float = 300.0, limit: float = 0.99
) -> int:
    boltzman = np.exp(-energies / (BOLTZMANN * temperature))
    boltzman /= np.sum(boltzman)
    total_proportion = 0.0
    for index, proportion in enumerate(boltzman):
        total_proportion += proportion
        if total_proportion >= limit:
            return index
    return index


def get_boltzmann_weightings(
    energies: np.ndarray, temperatures: np.ndarray
) -> np.ndarray:
    """Get the boltzmann weightings for the energies and temperatures.

    Args:
        energies: Relative energies, starting at 0 cm-1.
        temperatures: Temperatures in Kelvin.

    Returns:
        n_energies x n_temperatures floating array.
        Matrix element [i, j] would be the normalised np.exp( -E_i / [k * T_j] ).
    """
    boltzman = np.exp(
        -np.array(energies)[:, np.newaxis] / (BOLTZMANN * temperatures)[np.newaxis, :]
    )
    boltzman /= np.sum(boltzman, axis=0, keepdims=True)
    return boltzman


class IntensityResults(Protocol):
    """The intensities for the given transition types.

    This class does not contain spectroscopic information that are temperature dependent
    (e.g. Boltzmann weighting).
    """

    def get_spectrum(self, temperature: float) -> Spectrum:
        """Calculate the spectrum intensities.

        Args:
            temperature: The temperature (Kelvin) to compute the spectroscopic
                intensities for.

        Returns:
            The Spectrum object for that temperature.
        """
        ...
