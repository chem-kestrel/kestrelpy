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
from typing import Iterable, TypedDict

import numpy as np

from kestrelpy.calculations.constants import BOLTZMANN
from kestrelpy.calculations.intensities.spectroscopy import get_boltzmann_weightings
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    Eigenfunction,
    ManyElectronEigenfunctions,
)
from kestrelpy.configuration import get_logger
from kestrelpy.workflows.calculate_electronic_structure import Calculation

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class EigenfunctionIndices:
    """Class for storing the range of eigenfunction indices from `start`->`final`.

    A value of `start=2` and `final=4` would include eigenfunctions with the indices
    2, 3, and 4.
    """

    start: int
    final: int

    def __post_init__(self) -> None:
        """Validate the eigenfunction index values."""
        if self.start < 0:
            raise ValueError(
                f"The lowest eigenfunction index cannot be < 0, {self.start=}"
            )

        if self.final < self.start:
            raise ValueError(
                "The highest eigenfunction index cannot be < the lowest index, "
                f"{self.start=}, {self.final=}"
            )


class EPRLevelSOC(TypedDict):
    level_index: int
    start: int
    final: int
    multiplicity: int
    level_boltzmann: float
    spin_projection: dict[float, float]


class EPRLevelNoSOC(TypedDict):
    level_index: int
    start: int
    final: int
    multiplicity: int
    level_boltzmann: float
    multiplicity_level: int


def _get_number_eigenfunction_pairs(
    g_factors: Iterable[EigenfunctionIndices], a_values: Iterable[EigenfunctionIndices]
) -> int:
    """
    Checks that g_factor and A_value eigenfunction indices match as both are
    required to calculate the A value signs.
    If not, a warning is logged as the results will likely be unreliable.

    i.e. if there are 3 pairs for g_factors and 2 for A values, a warning
        would be displayed and the calculation carried out over 2 pairs.

    Args:
        g_factors: List of g_factor Eigenfunction indices
        a_values: List of A_value Eigenfunction indices

    Returns:
        int: The number Eigenfunction pairs to calculate the sign for
    """
    a_pairs = [(indices.start, indices.final) for indices in a_values]
    g_pairs = [(indices.start, indices.final) for indices in g_factors]
    if a_pairs == g_pairs:
        return len(a_pairs)
    else:
        _LOGGER.warning(
            f"Caution: Eigenfunction indices for g_factors ({g_pairs}) do not match those "
            f"for a_values ({a_pairs}) - calculation of A value signs may be unreliable"
        )
        return min(len(a_pairs), len(g_pairs))


def get_thermal_weighted_epr_levels(
    calculation: Calculation,
    temperature_k: float = 10.0,
    max_energy_cm: float = 5000.0,
) -> tuple[list[EPRLevelSOC], list[EPRLevelNoSOC]]:
    """Build SOC-level and no-SOC spin-manifold data for calculating
    thermal EPR properties.

    SOC populations are derived from eigenfunctions calculated in the presence
    of spin-orbit coupling. The boltzmann population and spin projection are
    determined to give the contributing weight of each eigenfunction.

    No-SOC populations are derived from eigenfunctions calculated in the
    absence of spin-orbital coupling. They are weighted by Boltzmann population
    to give the contribution of each electronic level.

    Args:
        calculation: Calculation object containing electronic eigenfunctions with
            and without spin-orbit coupling.
        temperature_k: Temperature Boltzmann populations are calculated for in K.
        max_energy_cm: Cut-off above which electronic levels are not considered in cm-1.
            By default, this is determined according to the temperature so any level
            with population below 1e-4 is ignored.

    Returns:
        list[EPRLevelSOC], list[EPRLevelNoSOC]

    """
    if max_energy_cm == 5000.0:
        max_energy_cm = -BOLTZMANN * temperature_k * np.log(1e-4)

    epr_levels_soc = build_thermal_soc_levels(
        eigenfunctions=calculation.eigenfunctions,
        temperature_k=temperature_k,
        max_energy_cm=max_energy_cm,
    )

    assert (
        calculation.eigenfunctions_no_soc is not None
    ), "No-SOC eigenfunctions missing."
    epr_levels_nosoc = build_thermal_nosoc_levels(
        eigenfunctions=calculation.eigenfunctions_no_soc,
        temperature_k=temperature_k,
        max_energy_cm=max_energy_cm,
    )

    return epr_levels_soc, epr_levels_nosoc


def _get_normalised_boltzmann_weights(
    states: list[Eigenfunction], temperature_k: float
) -> np.ndarray:
    """Helper function to determine normalised Boltzmann weights"""
    energies = np.array([s.energy for s in states])
    boltzmann_weights = get_boltzmann_weightings(
        energies=energies, temperatures=np.array([temperature_k])
    )
    return (boltzmann_weights / np.sum(boltzmann_weights)).flatten()


def build_thermal_soc_levels(
    eigenfunctions: ManyElectronEigenfunctions,
    temperature_k: float,
    max_energy_cm: float,
) -> list[EPRLevelSOC]:
    """
    Determine EPRLevelSOC for a set of electronic eigenfunctions (in the
    presence of SOC) at a specified temperature.

    Each electronic state is weighted by its spin projection and Boltzmann
    population which is returned as a dict.

    Args:
        eigenfunctions: ManyElectronEigenfunctions in the presence of SOC.
        temperature_k: Temperature Boltzmann populations are calculated at in K.
        max_energy_cm: Energy cut-off above which levels are neglected in cm-1.

    Returns:
        list[EPRLevelSOC] containing the indices of each level, its Boltzmann
            population, multiplicity and spin projection.
    """
    microstates = eigenfunctions.eigenfunctions
    levels = [lvl for lvl in eigenfunctions.levels if lvl.energy <= max_energy_cm]

    normalised_weights = _get_normalised_boltzmann_weights(
        states=microstates, temperature_k=temperature_k
    )
    microstate_weight = {
        s.index: float(w) for s, w in zip(microstates, normalised_weights)
    }

    epr_levels_soc: list[EPRLevelSOC] = []
    for level in levels:
        start = level.eigenfunctions[0].index
        final = level.eigenfunctions[-1].index
        projections = range(start, final + 1)
        level_weight = sum(microstate_weight[i] for i in projections)

        projection_totals: dict[float, float] = {}
        for i in projections:
            wi = microstate_weight[i]
            for key, val in microstates[i].spin_multiplicity_projection.items():
                projection_totals[key] = projection_totals.get(key, 0.0) + wi * val

        spin_projection = (
            {k: v / level_weight for k, v in projection_totals.items()}
            if level_weight > 0
            else {}
        )

        epr_levels_soc.append(
            {
                "level_index": level.index,
                "start": start,
                "final": final,
                "multiplicity": len(level.eigenfunctions),
                "level_boltzmann": level_weight,
                "spin_projection": spin_projection,
            }
        )
    return epr_levels_soc


def build_thermal_nosoc_levels(
    eigenfunctions: ManyElectronEigenfunctions,
    temperature_k: float,
    max_energy_cm: float,
) -> list[EPRLevelNoSOC]:
    """
    Determine EPRLevelNoSOC for a set of electronic eigenfunctions (in the
    absence of SOC) at a specified temperature.

    Each electronic state is weighted by its Boltzmann
    population which is returned as a dict.

    Args:
        eigenfunctions: ManyElectronEigenfunctions in the absence of SOC.
        temperature_k: Temperature Boltzmann populations are calculated at in K.
        max_energy_cm: Energy cut-off above which levels are neglected in cm-1.

    Returns:
        list[EPRLevelNoSOC] containing the indices of each level, its Boltzmann
            population and multiplicity.
    """
    microstates = eigenfunctions.eigenfunctions
    nosoc_levels = [lvl for lvl in eigenfunctions.levels if lvl.energy <= max_energy_cm]

    normalised_weights = _get_normalised_boltzmann_weights(
        states=microstates, temperature_k=temperature_k
    )

    # Loop through no-SOC levels
    epr_levels_nosoc: list[EPRLevelNoSOC] = []
    multiplicity_counts: dict[int, int] = {}
    for level in nosoc_levels:
        mult = level.dominant_multiplicity()

        for i in range(0, len(level.eigenfunctions), mult):
            chunk = level.eigenfunctions[i : i + mult]
            start, final = chunk[0].index, chunk[-1].index
            p_level = float(np.sum(normalised_weights[start : final + 1]))

            multiplicity_level = multiplicity_counts.get(mult, 0)
            multiplicity_counts[mult] = multiplicity_level + 1

            epr_levels_nosoc.append(
                {
                    "level_index": level.index,
                    "start": start,
                    "final": final,
                    "multiplicity": mult,
                    "level_boltzmann": p_level,
                    "multiplicity_level": multiplicity_level,
                }
            )

    return epr_levels_nosoc
