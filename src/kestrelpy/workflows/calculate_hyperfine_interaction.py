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

from kestrelpy.calculations.eigenfunction_indices import EPRLevelNoSOC
from kestrelpy.calculations.magnetism.effective_g_values import get_xyz_index
from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_calculations import (
    ASquaredTensor,
    calculate_a2tensor,
    calculate_a_values,
    calculate_fermi_contact,
    calculate_orbital_values,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_operators import (
    calculate_dipolar,
    construct_operator_matrices,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_parameters import (
    estimate_kappa_complex,
    extract_element,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.hyperfine_ligand_field import (
    calculate_dipolar_z_sign,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.hyperfine_sign import (
    ASignResults,
    ATensorSign,
    calculate_dipolar_sign,
    calculate_fermi_sign,
)
from kestrelpy.calculations.rotation_matrix import ensure_right_handed_axis
from kestrelpy.calculations.solve_matrix import solve_matrix
from kestrelpy.calculations.utils import (
    FirstRowTransitionMetals,
    calculate_oxidation_state,
)
from kestrelpy.configuration import get_logger
from kestrelpy.parameters import HyperfineParameters
from kestrelpy.workflows.calculate_electronic_structure import (
    Calculation,
    ElectronicStructureCalculation,
)

_LOGGER = get_logger(__name__)


@dataclass
class HyperfineOptions:
    """Class for determining which hyperfine calculations are enabled

    Attributes:
        start: First eigenfunction index
        final: Final eigenfunction index
    """

    start: int = 0
    final: int = 1
    fermi: bool = False
    anisotropic: bool = False
    sign: bool = False


@dataclass
class HyperfineResults:
    """Stores results from hyperfine calculations"""

    fermi_contact: np.ndarray | None = None
    dipolar: ASquaredTensor | None = None
    orbital: ASquaredTensor | None = None
    sign: ASignResults | None = None
    s_admixture: float = 0.0
    p: float = 0.0
    kappa: float = 0.0
    units: str = ""


@dataclass
class ThermalHyperfine:
    dipolar_tensor: np.ndarray
    average_fermi: np.ndarray
    partition_weight: float = 0.0
    paramagnetic_weight: float = 0.0
    total_kappa: float = 0.0
    total_s_admixture: float = 0.0


def _calculate_hyperfine_values(
    ops: dict[str, np.ndarray],
    multiplicity: int,
    start: int,
    final: int,
    p: float,
    description: str,
    units: str | None = "",
) -> ASquaredTensor:
    """Calculates A values for a given contribution (i.e. total, dipolar etc.)

    Args:
        ops (dict): operator matrices for the selected contribution
        multiplicity: Spin multiplicity of ground state
        start: first eigenfunction index,
        final: second eigenfunction index
        p: Hyperfine scaling parameter, P
        units: Units of P parameter (and of A values)

    Returns:
        list[ASquaredTensor]
    """
    _LOGGER.info(
        f"Calculating {description} A values for Eigenfunctions " f"{start}->{final}"
    )

    a2_tensor = calculate_a2tensor(
        ops=ops, start=start, final=final, effective_spin=multiplicity
    )
    hyperfine_results = calculate_a_values(a2_tensor=a2_tensor, p=p, units=units)

    return hyperfine_results


def calculate_hyperfine(
    calculation: Calculation,
    g_factors: GSquaredTensor,
    options: HyperfineOptions,
) -> HyperfineResults:
    """
    Wrapper function to calculate the hyperfine coupling. The function is called
    by calculate_epr_properties and calculates the A values, individual
    contribution values and their signs depending on the selected options.

    Args:
        calculation: Result of electronic structure calculation
        g_factors: object containing EPR g factors
        options: object that determines which hyperfine calculations are undertaken

    Returns:
        Results - object containing the results of hyperfine calculations
    """
    assert calculation.parameters.hyperfine is not None
    assert calculation.eigenfunctions_no_soc is not None

    results = HyperfineResults()
    results.p = calculation.parameters.hyperfine.p
    results.kappa = calculation.parameters.hyperfine.kappa
    element = extract_element(metal=calculation.parameters.hyperfine.metal)

    results.units = str(calculation.parameters.hyperfine.units)
    if options.fermi:
        _LOGGER.info("Calculating the Fermi contact contribution")
        fermi_contact = calculate_fermi_contact(
            p=calculation.parameters.hyperfine.p,
            kappa=calculation.parameters.hyperfine.kappa,
        )
        results.fermi_contact = calculate_fermi_sign(fermi=fermi_contact, metal=element)
    if options.anisotropic:
        _LOGGER.info(
            f"Calculating orbital and dipolar A values for Eigenfunctions "
            f"{options.start}->{options.final}"
        )
        s_mat, y_mat = construct_operator_matrices(
            matrices=calculation.basis_matrices,
            eigenvectors_no_soc=calculation.eigenfunctions_no_soc.eigenvectors,
            metal=element,
        )
        dipolar = calculate_dipolar(s_mat=s_mat, y_mat=y_mat)
        ground_mult = len(
            calculation.eigenfunctions_no_soc.levels[options.start].eigenfunctions
        )
        results.dipolar = _calculate_hyperfine_values(
            ops=dipolar,
            multiplicity=ground_mult,
            start=options.start,
            final=options.final,
            p=calculation.parameters.hyperfine.p,
            units=results.units,
            description="dipolar",
        )

        assert results.dipolar is not None
        results.orbital = calculate_orbital_values(
            g_factors=g_factors,
            a_vectors=results.dipolar.vectors,
            units=results.units,
            p=calculation.parameters.hyperfine.p,
        )

    if options.sign:
        assert results.dipolar is not None
        assert results.orbital is not None
        _LOGGER.info("Calculating signs of each contribution and the total A values")

        dipolar_z_sign = calculate_dipolar_z_sign(
            eigenfunctions=calculation.eigenfunctions_no_soc,
            ligand_field_vectors=calculation.ligand_field.eigenvectors,
            start=options.start,
            final=options.final,
        )
        results.sign = ATensorSign(
            orbital=results.orbital,
            dipolar=results.dipolar,
            fermi_contact=results.fermi_contact,
            g_factors=g_factors,
            dipolar_z_sign=dipolar_z_sign,
            metal=element,
        ).get_sign()

    return results


def calculate_thermal_hyperfine(
    calculation: Calculation,
    multiplicity: int,
    options: HyperfineOptions,
    dipolar_operators: dict[str, np.ndarray] | None = None,
) -> HyperfineResults:
    """
    Wrapper function to calculate EPR properties.

    This is called by calculate_thermal_epr_properties per populated level.
    and calculates signed fermi-contact and spin-dipolar contributions
    depending on the provided options for a populated electronic level.

    Args:
        calculation: Result of electronic structure calculation
        dipolar_operators: Dipolar operator matrices for sx, sy and sz.
        multiplicity: multiplicity of the electronic level
        options: HyperfineOptions specifying the indices of the states
            summed over and options for which contributions to calculate.

    Returns:
        HyperfineResults for the specified electronic level.
    """
    results = HyperfineResults()

    assert calculation.parameters.hyperfine is not None
    results.p = calculation.parameters.hyperfine.p
    results.kappa = calculation.parameters.hyperfine.kappa

    assert calculation.eigenfunctions_no_soc is not None
    element = extract_element(metal=calculation.parameters.hyperfine.metal)
    results.units = str(calculation.parameters.hyperfine.units)

    if options.fermi:
        _LOGGER.info("Calculating the Fermi contact contribution")
        fermi_contact = calculate_fermi_contact(
            p=calculation.parameters.hyperfine.p,
            kappa=calculation.parameters.hyperfine.kappa,
        )
        results.fermi_contact = calculate_fermi_sign(fermi=fermi_contact, metal=element)

    if options.anisotropic:
        assert dipolar_operators is not None
        results.dipolar = _calculate_hyperfine_values(
            ops=dipolar_operators,
            multiplicity=multiplicity,
            start=options.start,
            final=options.final,
            p=calculation.parameters.hyperfine.p,
            units=results.units,
            description="dipolar",
        )
        dipolar_z_sign = calculate_dipolar_z_sign(
            eigenfunctions=calculation.eigenfunctions_no_soc,
            ligand_field_vectors=calculation.ligand_field.eigenvectors,
            start=options.start,
            final=options.final,
        )
        dipolar_sign = calculate_dipolar_sign(
            values=results.dipolar.values,
            vectors=results.dipolar.vectors,
            dipolar_z_sign=dipolar_z_sign,
            metal=element,
        )
        results.dipolar.tensor = (
            results.dipolar.vectors
            @ np.diag(dipolar_sign)
            @ results.dipolar.vectors.conj().T
        )
    return results


def thermally_average_hyperfine(
    epr_levels: list[EPRLevelNoSOC],
    calculation: Calculation,
    options: HyperfineOptions,
    predict_kappa: bool = False,
    dipolar_operators: dict[str, np.ndarray] | None = None,
    system: ElectronicStructureCalculation | None = None,
) -> ThermalHyperfine:
    """Calculate the Boltzmann-weighted thermal average of hyperfine parameters.

    Iterates through a set of provided electronic levels (in the absence of SOC)
    and computes the hyperfine contributions (fermi, orbital and dipolar) for each
    state. The results for each level are weighted by its Boltzmann population
    to yield the thermally averaged hyperfine interaction.

    Args:
        epr_levels: List of electronic levels (EPRLevelNoSOC) containing its
            multiplicity, population and index.
        calculation: Result of electronic structure calculation
        options: HyperfineOptions flag specifying which contributions are calculated
            (i.e. fermi or anisotropic contributions).
        predict_kappa: Flag, if True kappa is estimated otherwise it is set to user-input.
        dipolar_operators: Dictionary containing dipolar spin operators for sx, sy and sz.
        system: Electronic structure calculation parameters, required if kappa is predicted.

    Returns:
    ThermalHyperfine: dataclass containing the averaged dipolar tensor, fermi contact,
        partition weights and, if predict_kappa is True, kappa and s-admixture values.
    """
    total_partition_weight = 0.0
    paramagnetic_partition_weight = 0.0
    total_kappa = 0.0
    total_s_admixture = 0.0
    thermal_dipolar = np.zeros((3, 3), dtype=np.float64)
    average_fermi = np.zeros(3, dtype=np.float64)

    assert calculation.parameters.hyperfine is not None
    isotope = calculation.parameters.hyperfine.metal.title()
    element = FirstRowTransitionMetals(extract_element(isotope).title())
    oxidation_state = calculate_oxidation_state(
        d_electrons=calculation.parameters.d_electrons, element=element
    )

    for lvl_nosoc in epr_levels:
        p_level = lvl_nosoc["level_boltzmann"]
        if p_level <= 1e-6:
            continue

        total_partition_weight += p_level
        if lvl_nosoc["multiplicity"] <= 1:  # Skip diamagnetic S=0 levels
            continue

        paramagnetic_partition_weight += p_level

        if predict_kappa:
            assert system is not None
            level_kappa, level_s_admixture = estimate_kappa_complex(
                metal=isotope,
                oxidation_state=oxidation_state,
                start=lvl_nosoc["start"],
                final=lvl_nosoc["final"],
                multiplicity=lvl_nosoc["multiplicity"],
                calculation=calculation,
                system=system,
            )
            calculation.parameters.hyperfine.kappa = level_kappa
            total_kappa += level_kappa * p_level
            total_s_admixture += np.sum(level_s_admixture) * p_level

        metal_hyperfine = calculate_thermal_hyperfine(
            calculation=calculation,
            dipolar_operators=dipolar_operators,
            multiplicity=lvl_nosoc["multiplicity"],
            options=HyperfineOptions(
                start=lvl_nosoc["start"],
                final=lvl_nosoc["final"],
                anisotropic=options.anisotropic,
                fermi=options.fermi,
            ),
        )
        if options.anisotropic and metal_hyperfine.dipolar is not None:
            thermal_dipolar += np.real(metal_hyperfine.dipolar.tensor) * p_level
        if options.fermi and metal_hyperfine.fermi_contact is not None:
            average_fermi += metal_hyperfine.fermi_contact * p_level

    if predict_kappa:
        _LOGGER.info(f"kappa value set to: {total_kappa}")

    return ThermalHyperfine(
        partition_weight=total_partition_weight,
        paramagnetic_weight=paramagnetic_partition_weight,
        dipolar_tensor=thermal_dipolar,
        average_fermi=average_fermi,
        total_kappa=total_kappa,
        total_s_admixture=total_s_admixture,
    )


def normalise_thermal_hyperfine(
    thermal_hyperfine: ThermalHyperfine,
    predict_kappa: bool = False,
) -> tuple[ThermalHyperfine, bool]:
    """Normalised thermally-averaged hyperfine parameters according to the
    weight of each paramagnetic level.

    Scales the summed dipolar tensor and fermi contact contribution
    (and kappa/s-admixture) by the total paramagnetic weight. If this is
    below 5%, the system is treated as EPR silent.

    Args:
        thermal_hyperfine: Thermal hyperfine object containing the summed
            dipolar tensor, fermi contact and weights.
        predict_kappa: Flag, if true, kappa and s-admixture are normalised.

    Returns:
        ThermalHyperfine: New instance with normalised quantities
        bool: Flag, True if the system is EPR silent (<5% paramagnetic weight)
    """
    thermal_dipolar = thermal_hyperfine.dipolar_tensor
    average_fermi = thermal_hyperfine.average_fermi
    total_kappa = thermal_hyperfine.total_kappa
    total_s_admixture = thermal_hyperfine.total_s_admixture
    epr_silent = False

    para_fraction = (
        thermal_hyperfine.paramagnetic_weight / thermal_hyperfine.partition_weight
        if thermal_hyperfine.partition_weight > 0
        else 0.0
    )
    if para_fraction > 0.05:
        thermal_dipolar /= thermal_hyperfine.paramagnetic_weight
        average_fermi /= thermal_hyperfine.paramagnetic_weight
        if predict_kappa:
            total_kappa /= thermal_hyperfine.paramagnetic_weight
            total_s_admixture /= thermal_hyperfine.paramagnetic_weight
    else:
        _LOGGER.warning(
            f"Paramagnetic population is very low ({para_fraction:.1%})."
            "EPR properties are negligible and system is treated as EPR silent"
        )
        thermal_dipolar = np.zeros((3, 3))
        average_fermi = np.zeros(3)
        epr_silent = True

    normalised_hyperfine = ThermalHyperfine(
        total_kappa=total_kappa,
        total_s_admixture=total_s_admixture,
        dipolar_tensor=thermal_dipolar,
        average_fermi=average_fermi,
    )
    return normalised_hyperfine, epr_silent


def solve_thermal_hyperfine(
    parameters: HyperfineParameters,
    dipolar: np.ndarray,
    fermi: np.ndarray,
    real_g_factors: GSquaredTensor,
    hfc_aniso: bool = False,
    hfc_sign: bool = False,
    hfc_fermi: bool = False,
    epr_silent: bool = False,
) -> HyperfineResults:
    """
    Wrapper function to solve thermal hyperfine properties and return
    thermally-weighted values for each contribution and the overall
    interaction depending on options specified by the user.

    This function is called after the results of calculate_thermal_hyperfine
    have been summed and weighted according to their Boltzmann population.

    Args:
        parameters: HyperfineParameters object containing p, units and metal.
        dipolar: Signed thermally-averaged dipolar tensor (3x3 array)
        fermi: Fermi contact value (1x3 array)
        real_g_factors: GSquaredTensor containing real g values.
        hfc_aniso: Flag, if True dipolar and orbital values are calculated.
        hfc_sign: Flag, if True the sign is calculated and contributions summed
            to give the total hyperfine interaction.
        hfc_fermi: Flag, if True the Fermi contact is calculated.
        epr_silent: Flag, if True the complex is expected to be EPR silent.

    Returns:
        HyperfineResults
    """
    results = HyperfineResults()
    results.p = parameters.p
    units = str(parameters.units)

    if hfc_fermi:
        results.fermi_contact = fermi

    if hfc_aniso:
        _LOGGER.info("Calculating dipolar and orbital A values")
        dipolar_result = solve_matrix(dipolar)
        order = get_xyz_index(dipolar_result.vectors)
        dipolar_vectors = ensure_right_handed_axis(
            vectors=dipolar_result.vectors[:, order]
        )
        results.dipolar = ASquaredTensor(
            tensor=dipolar,
            values=dipolar_result.values[order],
            vectors=dipolar_vectors,
            units=units,
        )

        if epr_silent:
            results.orbital = ASquaredTensor.zero_a_tensor()
        else:
            results.orbital = calculate_orbital_values(
                p=parameters.p,
                g_factors=real_g_factors,
                a_vectors=dipolar_vectors,
                units=units,
            )

    if hfc_sign:
        assert results.dipolar is not None
        assert results.orbital is not None
        _LOGGER.info("Calculating signs of each contribution and the total A values")
        results.sign = ATensorSign(
            orbital=results.orbital,
            dipolar=results.dipolar,
            fermi_contact=fermi,
            g_factors=real_g_factors,
            metal=extract_element(metal=parameters.metal),
        ).get_sign()

    return results


def rotate_hyperfine_to_frame(results: HyperfineResults, rotation: np.ndarray) -> None:
    """Rotate all frame-dependent hyperfine quantities to a specified frame

    Args:
        results: HyperfineResults
        rotation: Right-handed matrix expressing hyperfine frame in the original frame
    """
    for contribution in (results.dipolar, results.orbital):
        if contribution is not None:
            contribution.tensor = rotation @ contribution.tensor @ rotation.conj().T
            contribution.vectors = ensure_right_handed_axis(
                vectors=(rotation @ contribution.vectors)
            )
    if results.sign is not None:
        results.sign.total_vectors = ensure_right_handed_axis(
            vectors=(rotation @ results.sign.total_vectors)
        )
