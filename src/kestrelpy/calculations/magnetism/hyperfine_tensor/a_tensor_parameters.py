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

METAL_FACTOR and R3_VALUES constants are required for calculating the value of
the parameter P used in hyperfine calculations. P represents the overall strength
of the hyperfine interaction.

Where:
METAL_FACTOR = 2x10-6 * gn * beta(e) * beta(n)
<R-3> = 1/(mean 3d radial distribution function ^3)
Are tabulated from:
    A. K. Koh, D. J. Miller
    Hyperfine coupling constants and atomic parameters for electron paramagnetic
    resonance data
    Atomic Data and Nuclear Data Tables
    Volume 33, Issue 2
    1985
    Pages 235-253
    https://www.sciencedirect.com/science/article/pii/0092640X85900038

S_FERMI values are constants describing the hyperfine interaction of a 4s
electron; an unpaired electron with 100% 4s character will exhibit
this Aiso (fermi contact) value. These constants are required for calculating
the contribution of 4s admixture to the parameter kappa. Kappa describes
the fermi contact (isotropic) hyperfine interaction which describes the
s-electron density at the nucleus. Values are taken from:
    A. K. Koh, D. J. Miller
    Hyperfine coupling constants and atomic parameters for electron paramagnetic
    resonance data
    Atomic Data and Nuclear Data Tables
    Volume 33, Issue 2
    1985
    Pages 235-253
    https://www.sciencedirect.com/science/article/pii/0092640X85900038

Kappa is calculated as the sum of three contributions: core-shell polarisation,
valence-shell polarisation and direct 4s admixture into 3d orbitals. Calculation
of kappa is based on unpublished work by B. Grace.
"""

import numpy as np

from kestrelpy.calculations.ligand_field_matrix import LigandField
from kestrelpy.calculations.magnetism.hyperfine_tensor.hyperfine_ligand_field import (
    calculate_4s_mixing_strengths,
)
from kestrelpy.calculations.many_electron_eigenfunctions.configuration_projection import (
    Configuration,
    configuration_weighted_sum,
    get_level_configuration_projection,
)
from kestrelpy.configuration import get_logger
from kestrelpy.molecule import Molecule
from kestrelpy.parameters import LigandParameters
from kestrelpy.workflows.calculate_electronic_structure import (
    Calculation,
    ElectronicStructureCalculation,
)

_LOGGER = get_logger(__name__)


S_FERMI = {
    "ti": -707.9,
    "v": 3868.6,
    "cr": -560.7,
    "mn": 4730.8,
    "fe": 447.4,
    "co": 3636.8,
    "ni": -1522.8,
    "cu": 3506.4,
}

P_PRIME = {
    "47ti": 30.07,
    "49ti": 30.08,
    "50v": 53.20,
    "51v": 140.36,
    "53cr": 30.15,
    "53mn": 137.13,
    "55mn": 131.63,
    "57fe": 17.17,
    "59co": 125.83,
    "60co": 72.32,
    "61ni": 47.71,
    "63/65cu": 144.50,
    "63cu": 141.4,
    "65cu": 151.45,
}

R3_VALUES = {
    "ti": {"0": 2.02, "1": 2.04, "2": 2.14, "3": 2.54, "-1": 1.63},
    "v": {"0": 2.64, "1": 2.66, "2": 2.76, "3": 3.22, "4": 3.67, "-1": 2.23},
    "cr": {
        "0": 2.89,
        "1": 2.97,
        "2": 3.46,
        "3": 3.96,
        "4": 4.48,
        "5": 4.98,
        "-1": 2.91,
    },
    "mn": {
        "0": 4.15,
        "1": 4.17,
        "2": 4.26,
        "3": 4.79,
        "4": 5.36,
        "5": 5.94,
        "6": 6.51,
        "-1": 3.61,
    },
    "fe": {
        "0": 4.98,
        "1": 4.51,
        "2": 5.09,
        "3": 5.73,
        "4": 6.33,
        "5": 6.97,
        "6": 7.61,
        "-1": 4.43,
    },
    "co": {
        "0": 5.93,
        "1": 5.40,
        "2": 6.06,
        "3": 6.70,
        "4": 7.42,
        "5": 8.09,
        "6": 8.80,
        "-1": 5.33,
    },
    "ni": {
        "0": 6.98,
        "1": 6.42,
        "2": 7.10,
        "3": 7.82,
        "4": 8.54,
        "5": 9.34,
        "6": 10.09,
        "-1": 6.36,
    },
    "cu": {"0": 7.50, "2": 8.26, "3": 9.02, "4": 9.82, "5": 10.62, "6": 11.51},
}


def calculate_p(metal: str, oxidation_state: int) -> float:
    """
    Multiplies a constant factor for each metal by a value of r^3 (which depends
    on the metal and oxidation state) to calculate the free-ion value of P (MHz).

    Values are adapted from: "Hyperfine coupling constants and atomic parameters
    for electron paramagnetic resonance data" - A. K. Koh and D.J. Miller

    Args:
        metal (str): 3d transition metal
        oxidation_state (int): metal oxidation state

    Returns:
        float: The value of P (in MHz) for the desired metal and oxidation state
    """
    ox_state = str(oxidation_state)
    element = extract_element(metal)
    if ox_state not in R3_VALUES.get(element.lower(), ox_state):
        raise ValueError(
            f"An r^-3 value for the {oxidation_state:+} oxidation state is unavailable..."
        )

    p_value = float(P_PRIME[metal.lower()]) * float(
        R3_VALUES[element.lower()][ox_state]
    )
    return float(np.round(p_value, decimals=0))


def calculate_kappa_free_ion(metal: str, oxidation_state: int) -> float:
    """Calculate the free-ion kappa value for a given 3d metal and oxidation state

        Kappa (free-ion) is calculated using the relation::

            kappa = (r^3 * [1.44 - (0.08 * ox_state)]) + 0.086

        Where r^3 is the mean of the 3d radial distribution function.
        This relation is obtained from correlating free-ion kappa values (from DFT)
        for each metal/oxidation state against r^3

    Args:
        metal (str): 3d transition metal
        oxidation_state (int): metal oxidation state

    Returns:
        float: Free-ion kappa value for a given metal and oxidation state
    """
    r_3_inverse = 1 / R3_VALUES[metal.lower()][str(oxidation_state)]
    kappa = r_3_inverse * (1.44 - (0.08 * oxidation_state)) + 0.086
    kappa = float(np.round(kappa, decimals=4))
    _LOGGER.info(f"kappa (free-ion) = {kappa}")
    return kappa


def estimate_kappa_complex(
    metal: str,
    oxidation_state: int,
    multiplicity: int,
    start: int,
    final: int,
    calculation: Calculation,
    system: ElectronicStructureCalculation,
) -> tuple[float, np.ndarray]:
    """Calculate the kappa value for a 3d transition metal complex

        kappa is defined as the s-electron density at the nucleus and is calculated
        as the sum of three separate contributions:

            1) Core-shell contributions from occupied 1s, 2s and 3s electrons
            2) Valence shell polarisation of the 4s electrons by the ligands.
                This is dependent on the strength of the ligand field
            3) Direct 4s admixture to singly occupied d-orbitals due to 3d-4s mixing.
                This occurs when the sum of e sigma is weak relative to e pi. Hence, the
                4s orbital is sufficiently close to the d-orbital manifold to permit mixing.

    The Fermi contact can then be calculated using the relation: AFC = P * kappa

    Args:
        metal (str): 3d transition metal
        oxidation_state (int): metal oxidation state
        start (int): Start index of the electronic level kappa is calculated for
        final(int): Final index of the electronic level kappa is calculated for.
        multiplicity (int): spin multiplicity (2S+1)
        calculation (Calculation): contains electronic structure calculation results
        system: ElectronicStructureCalculation, parameters and details of
            the molecule system if Molecule is not None

    Returns:
        kappa value for a 3d transition metal complex (sum of all 3 contributions)
        Percentage 4s admixture

    """
    element = extract_element(metal)
    kappa_free_ion = calculate_kappa_free_ion(
        metal=element, oxidation_state=oxidation_state
    )
    kappa_polarisation = estimate_kappa_polarisation(
        ligand_field_trace=calculation.ligand_field.ligand_field_matrix.trace()
    )

    if system.molecule is not None:
        p_value = calculate_p(metal=metal, oxidation_state=oxidation_state)
        e_sigma_minus_pi = _get_e_sigma_minus_pi(
            molecule=system.molecule, parameters=system.ligand_parameters
        )
        config_projection = get_level_configuration_projection(
            start=start,
            final=final,
            configuration_projection=calculation.eigenfunctions_no_soc.configuration_projection,
        )
        kappa_admixture, s_admixture = estimate_kappa_admixture(
            e_sigma_minus_e_pi=e_sigma_minus_pi,
            level_configurations=config_projection,
            ligand_field=calculation.ligand_field,
            unpaired_electrons=(multiplicity - 1),
            p=p_value,
            s_fermi=S_FERMI[element.lower()],
        )
    else:
        _LOGGER.info(
            "e sigma and e pi could not be extracted - skipping calculation of 4s admixture..."
        )
        kappa_admixture = 0.0
        s_admixture = np.zeros(5, dtype=np.float64)

    kappa_complex = kappa_admixture + kappa_free_ion + kappa_polarisation
    return kappa_complex, s_admixture


def estimate_kappa_polarisation(ligand_field_trace: float) -> float:
    """Estimate the valence shell (VS) contribution to kappa

        Kappa (polarisation) is calculated using the relation::

            kappa = Δkappa + kappa (free-ion)
                where: Δkappa = (LF_trace - 29900) / 211000

        This relationship is taken from unpublished work by B. Grace that correlates
        the deviation from free-ion kappa values observed in 3d transition metal
        complexes, with the trace of the 5×5 Ligand Field Matrix.

    Args:
        ligand_field_trace (float): Trace of LF matrix / sum of e parameter values

    Returns:
        float: Kappa value for a 3d transition metal complex
    """
    kappa = (ligand_field_trace - 29900) / 211000
    kappa = float(np.round(kappa, decimals=4))
    _LOGGER.info(f"kappa (VS polarisation) = {kappa}")
    return kappa


def estimate_kappa_admixture(
    e_sigma_minus_e_pi: float,
    level_configurations: list[Configuration],
    ligand_field: LigandField,
    unpaired_electrons: int,
    p: float,
    s_fermi: float,
) -> tuple[float, np.ndarray]:
    """
    Calculate the 4s admixture contribution to kappa.

    The degree of 4s admixture into each d-orbital can be calculated by taking
    the extent of mixing of each 3d orbital with the 3d_z2 orbital and multiplying
    by angular proportionality factors (ki) derived from Wigner 3-j symbols::

        <4s|H|3d_i> = ki * <3d_z2|H|3d_i>

    The result is then divided by the energy gap between the 3d and 4s orbital.
    This is treated according to the empirical correlation below::

        %4s admixture = ([∑e_sigma - ∑e_pi] * -0.00198 + 45.2)

    The result is then weighted by the composition of each singly occupied orbital
    and averaged over the configuration interaction to give the %4s admixture.
    The kappa value for the 4s admixture is then determined according to the
    relationship below::

            kappa(4s_admixture) = (%admixture * 4s(AFC)) / (unpaired_electrons * p)

    The above relationships are taken from unpublished work by B.Grace where the %4s
    admixture is related to the relative strengths of e sigma and e pi. When
    e sigma is relatively weak, the 4s orbital is kept closer in energy to the
    d orbital manifold, permitting 4s admixture.

    Args:
        e_sigma_minus_e_pi (float): sum of e sigma minus sum of e pi
        level_configurations (list[Configuration]): Configuration interaction for the
            electronic levels the 4s admixture is to be calculated for.
        ligand_field: LigandField object containing the 5x5 ligand-field matrix, its
            eigenvalues and eigenvectors.
        unpaired_electrons (int): number of unpaired d-electrons
        p (float): P value (hyperfine scaling parameter)
        s_fermi (float): 4s electron fermi contact value

    Returns:
        kappa_admixture: Isotropic hyperfine coupling constant contribution.
        orbital_4s_admixtures: Array showing 4s spin population for each d orbital

    """
    numerators, mixing_weights = calculate_4s_mixing_strengths(
        lf_matrix=ligand_field.ligand_field_matrix,
        lf_vectors=ligand_field.eigenvectors,
    )
    max_admixtures = calculate_max_4s_orbital_admixtures(
        e_sigma_minus_e_pi=e_sigma_minus_e_pi,
    )
    orbital_admixtures = max_admixtures * mixing_weights

    d_4s_admixture, _ = configuration_weighted_sum(
        level_configurations=level_configurations, values=(orbital_admixtures / 100)
    )
    orbital_4s_admixtures = ligand_field.eigenvectors**2 @ d_4s_admixture

    kappa_admixture = calculate_4s_kappa_value(
        admixture=np.sum(d_4s_admixture),
        s_fermi=s_fermi,
        unpaired_electrons=unpaired_electrons,
        p=p,
    )

    return kappa_admixture, orbital_4s_admixtures


def calculate_4s_kappa_value(
    admixture: float,
    s_fermi: float,
    unpaired_electrons: int,
    p: float,
) -> float:
    """
    Calculate the kappa value for direct 4s admixture.

    The Fermi contact for a pure 4s electron is weighted by the %4s admixture
    into the d orbitals and normalised by the number of unpaired electrons
    according to the relation::

        kappa(4s_admixture) = -[(%admixture * 4s(AFC)) / (unpaired_electrons * p)]

    Args:
        admixture: %4s admixture into the d orbitals
        s_fermi: Fermi contact value for a pure 4s electron
        unpaired_electrons: Number of unpaired electrons
        p: Hyperfine scaling parameter, P.

    Returns:
        kappa value describing the 4s admixture contribution to the Fermi contact
        rounded to 4 d.p.
    """
    if unpaired_electrons > 0 and p != 0:
        kappa = (admixture * s_fermi) / (unpaired_electrons * p)
        kappa = -abs(float(np.round(kappa, decimals=4)))
    else:
        kappa = 0.0
    _LOGGER.info(f"kappa (4s admixture) = {kappa}")
    return kappa


def calculate_max_4s_orbital_admixtures(
    e_sigma_minus_e_pi: float,
) -> np.ndarray:
    """
    Calculate the maximum theoretical 4s admixture into d orbitals
    according to the 4s-3d_i energetic separation. The experimental
    4s admixture will necessarily be lower if mixing is not symmetry allowed.

    This uses a semi-empirical approach based off unpublished work by B. Grace
    where the difference between the sum of e sigma and e pi parameters is used
    as a proxy of the energetic separation of the 3d and 4s orbitals.

    The %admixture is given by the following relation::

        %4s admixture = ([∑e_sigma - ∑e_pi] * -0.00198 + 45.2)

        Args:
            e_sigma_minus_e_pi: Sum of e sigma parameters minus the sum of e pi parameters
                for a given molecule. This acts as a proxy for the 4s-3d energy gap.

        Returns:
            max_percent_admixture: Array of length 5 containing final %4s admixture per orbital.
    """
    max_percent_admixture = np.zeros(5)

    for orbital_idx in range(5):
        # Calculate maximum possible admixture for the given orbital
        percent_admixture = (e_sigma_minus_e_pi * -0.00198) + 45.2
        max_percent_admixture[orbital_idx] = min(max(percent_admixture, 0.0), 100.0)

    return max_percent_admixture


def estimate_ligand_field_trace(metal: str, oxidation_state: int, kappa: float) -> int:
    """Estimate the trace of the 5×5 Ligand Field Matrix from a kappa value

    tr(LF) is calculated using the relation::

        tr(LF) = (211000 * Δkappa) + 29900
            where Δkappa = kappa (complex) - kappa (free-ion)

    This relation is the inverse of that used to calculate the kappa value for a
    3d transition metal complex. Use alongside calculate_kappa_from_fermi
    allows tr(LF) / the sum of e parameter values to be determined from
    known Fermi Contact values

    Args:
        metal (str): 3d transition metal
        oxidation_state (int): metal oxidation state
        kappa (float): kappa parameter value

    Returns:
        int: The ligand field trace (sum of e values)
    """
    kappa_free_ion = calculate_kappa_free_ion(metal, oxidation_state)
    delta_kappa = kappa - kappa_free_ion
    ligand_field_trace = (211000 * delta_kappa) + 29900
    return int(np.round(ligand_field_trace, decimals=0))


def calculate_kappa_from_fermi(metal: str, oxidation_state: int, fermi: float) -> float:
    """Utility function to extract kappa parameter values from the Fermi Contact

    Kappa is calculated using the relation::

        kappa = |AFC / P|
    where P is the hyperfine scaling parameter

    Args:
        metal (str): 3d transition metal
        oxidation_state (int): metal oxidation state
        fermi (float): Fermi contact value

    Returns:
        float: Kappa value
    """
    p = calculate_p(metal, oxidation_state)
    kappa = abs(fermi / p)
    return float(np.round(kappa, decimals=4))


def _get_e_sigma_minus_pi(
    molecule: Molecule, parameters: list[LigandParameters]
) -> float:
    """
    Calculate the sum of e sigma minus the sum of e pi for a given set of ligands

    Args:
        molecule (Molecule): Molecule and the coordinating ligands
        parameters (LigandParameters): Ligand parameter values

    Returns:
        ∑(e_sigma) - ∑(e_pi)
    """
    e_sigma = 0.0
    e_pi = 0.0
    i = 0
    for ligand in molecule.ligands:
        number_ligands = len(ligand.atoms)
        e_sigma += parameters[i].esigma * number_ligands
        e_pi += (parameters[i].epix + parameters[i].epiy) * number_ligands
        i += 1
    return e_sigma - e_pi


def extract_element(metal: str | None) -> str:
    """Converts metal isotope to metal: i.e. '63Cu' is converted to 'Cu'"""
    if metal:
        return "".join(char for char in metal if char.isalpha())
    else:
        return ""
