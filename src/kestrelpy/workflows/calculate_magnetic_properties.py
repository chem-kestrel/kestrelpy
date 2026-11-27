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

import copy
from typing import Iterable

import numpy as np

from kestrelpy.calculations.eigenfunction_indices import EPRLevelNoSOC, EPRLevelSOC
from kestrelpy.calculations.intensities.spectroscopy import (
    max_thermally_occupied_energy,
)
from kestrelpy.calculations.ligand_field_matrix import (
    determine_diagonal_frame,
    rotate_lf_frame,
)
from kestrelpy.calculations.magnetism.effective_g_values import get_xyz_index
from kestrelpy.calculations.magnetism.g_factors import calculate_g_factors
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_calculations import (
    ASquaredTensor,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_operators import (
    calculate_dipolar,
    construct_operator_matrices,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_parameters import (
    calculate_p,
    estimate_kappa_complex,
    extract_element,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.hyperfine_sign import (
    ASignResults,
)
from kestrelpy.calculations.magnetism.susceptibilities import (
    ParamagneticSusceptibilityTensors,
    calculate_paramagnetic_susceptibilities,
)
from kestrelpy.calculations.magnetism.thermal_g_factors import (
    calculate_g_effective,
    calculate_real_g_factors,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    get_configuration_projection,
)
from kestrelpy.calculations.utils import (
    FirstRowTransitionMetals,
    calculate_oxidation_state,
)
from kestrelpy.configuration import get_logger
from kestrelpy.models.input_models.input_models import GFactorType, HyperfineMode
from kestrelpy.models.results.calculations import EPRProperties
from kestrelpy.parameters import ParameterValueError
from kestrelpy.workflows.calculate_electronic_structure import (
    Calculation,
    ElectronicStructureCalculation,
    run_parameters,
)
from kestrelpy.workflows.calculate_hyperfine_interaction import (
    HyperfineOptions,
    HyperfineResults,
    calculate_hyperfine,
    normalise_thermal_hyperfine,
    rotate_hyperfine_to_frame,
    solve_thermal_hyperfine,
    thermally_average_hyperfine,
)

_LOGGER = get_logger(__name__)


def calculate_epr_properties(
    calculation: Calculation,
    start: int,
    final: int,
    system: ElectronicStructureCalculation | None = None,
    calculate_hfc: HyperfineMode = HyperfineMode.NONE,
    predict_p: bool = False,
    predict_kappa: bool = False,
) -> EPRProperties:
    """Calculate EPR properties for the eigenfunctions `start` -> `final`. EPR
        parameters calculated with this function are not thermally averaged.

    This function automatically computes EPR g-factors.

    Args:
        calculation: Results of Ligand Field calculation
        system: ElectronicStructureCalculation, parameters and details of
            the molecule system if Molecule is not None
        start: The first eigenfunction index (inclusive).
        final: The final eigenfunction index (inclusive).
        calculate_hfc: HyperfineMode which configures what hyperfine information
            is computed.
            - "none": metal-hyperfine is not calculated.
            - "values": only metal-hyperfine 'A' (anisotropic) values are computed.
            - "values+contributions+signs": metal-hyperfine 'A' values are calculated
                    along with their fermi-contact, dipolar, and orbital contributions.
                    The signs of each contribution and the 'A' values are computed.
        predict_p: Calculates a P value for the given metal and oxidation state
        predict_kappa: Estimates a value of the kappa parameter for a given complex

    Examples:
        ```
        >>> calculate_epr_properties(start=0, final=1)
        ```
        ... will return the EPR properties for eigenfunctions 0 and 1.

        ```
        >>> calculate_epr_properties(start=0, final=3)
        ```
        ... will return the EPR properties for eigenfunctions 0, 1, 2, and 3.
        ```
         >>> calculate_epr_properties(start=0, final=1, calculate_hfc=HyperfineMode.ALL)
        ```
        ... will return the EPR properties for eigenfunctions 0 and 1, along with the metal
            hyperfine coupling constants, their signs and individual contributions.

        ```
        >>> calculate_epr_properties(start=0, final=1, calculate_hfc=HyperfineMode.ANISO, predict_p=True)
        ```
        ...will return the EPR properties for eigenfunctions 0 and 1, along with the
        anisotropic dipolar and orbital hyperfine (A) coupling. The hyperfine parameter,
        P will also be predicted.
    """
    if max(start, final) > len(calculation.eigenfunctions.eigenfunctions):
        raise ValueError(
            "Eigenfunction indices cannot be larger than the number of eigenfunctions: "
            f"{start=}, {final=} "
            f"and Max Eigenfunctions index = {len(calculation.eigenfunctions.eigenfunctions)}"
        )

    if start > final:
        raise ValueError(
            "Starting Eigenfunction index must be < final Eigenfunction index. "
            f"{start=}, {final=}"
        )

    _LOGGER.info(f"Calculating g-factors for Eigenfunctions {start} -> {final}")
    g_factors = calculate_g_factors(
        muxyz=calculation.get_magnetic_dipole_moments(),
        start=start,
        final=final,
    )

    # Determine which hyperfine contributions to calculate
    metal_hyperfine_coupling = None
    if calculate_hfc != HyperfineMode.NONE:
        hfc_fermi = calculate_hfc in [HyperfineMode.FERMI, HyperfineMode.ALL]
        hfc_aniso = calculate_hfc in [
            HyperfineMode.ANISO,
            HyperfineMode.ALL,
        ]
        hfc_sign = calculate_hfc in [HyperfineMode.ALL]
        _LOGGER.info(
            f"'{calculate_hfc}' chosen: calculating metal-hyperfine values for Eigenfunctions "
            f"{start} -> {final}"
        )
        assert calculation.parameters.hyperfine is not None
        if calculation.parameters.hyperfine.metal is None:
            raise ParameterValueError(
                "fNo isotope or metal assigned for metal-hyperfine calculation. Please assign."
            )

        g_frame, lf_frame, xyz_order = determine_diagonal_frame(
            ligand_field_matrix=calculation.ligand_field.ligand_field_matrix,
            g_vectors=g_factors.vectors,
        )
        lf_in_diagonal_frame = rotate_lf_frame(
            ref_vectors=lf_frame,
            lf_matrix=calculation.ligand_field.ligand_field_matrix,
        )
        g_frame = copy.deepcopy(g_factors)
        g_frame.vectors = g_factors.vectors.conj().T @ g_frame.vectors
        g_frame.apply_g_order(order=get_xyz_index(vectors=g_frame.vectors))

        # Hyperfine is calculated in the g tensor frame
        calculation_parameters = copy.deepcopy(calculation.parameters)
        calculation_parameters.ligand_field = lf_in_diagonal_frame
        calculation_hyperfine = run_parameters(
            parameters=calculation_parameters, calculate_configuration_projections=False
        )
        assert calculation_hyperfine.eigenfunctions_lf is not None
        calculation_hyperfine.eigenfunctions_no_soc = get_configuration_projection(
            many_electron_eigenfunctions=calculation_hyperfine.eigenfunctions_no_soc,
            ligand_field_eigenfunctions=calculation_hyperfine.eigenfunctions_lf,
            ligand_field_orbital_energies=calculation_hyperfine.ligand_field.eigenvalues,
            orbital_occupations=calculation_hyperfine.orbital_occupations,
        )

        # Run hyperfine calculation
        s_admixture = 0.0
        isotope = calculation.parameters.hyperfine.metal.title()
        element = FirstRowTransitionMetals(extract_element(isotope).title())
        assert calculation_hyperfine.parameters.hyperfine is not None

        if predict_p:
            calculation_hyperfine.parameters.hyperfine.p = calculate_p(
                metal=isotope,
                oxidation_state=calculate_oxidation_state(
                    d_electrons=calculation_hyperfine.parameters.d_electrons,
                    element=element,
                ),
            )
            _LOGGER.info(
                f"P value set to: {calculation_hyperfine.parameters.hyperfine.p} MHz"
            )
            calculation_hyperfine.parameters.hyperfine.units = "(MHz)"

        if predict_kappa:
            assert system is not None
            oxidation_state = calculate_oxidation_state(
                d_electrons=calculation_hyperfine.parameters.d_electrons,
                element=element,
            )
            multiplicities = calculation.eigenfunctions.levels[
                start
            ].spin_multiplicity_projection
            multiplicity = max(multiplicities, key=lambda k: multiplicities[k])
            kappa, s_admixture = estimate_kappa_complex(
                metal=isotope,
                oxidation_state=oxidation_state,
                multiplicity=multiplicity,
                start=start,
                final=final,
                calculation=calculation_hyperfine,
                system=system,
            )
            s_admixture = np.sum(s_admixture)
            calculation_hyperfine.parameters.hyperfine.kappa = kappa
            _LOGGER.info(
                f"Kappa value set to: {calculation_hyperfine.parameters.hyperfine.kappa}"
            )

        metal_hyperfine_coupling = calculate_hyperfine(
            calculation=calculation_hyperfine,
            g_factors=g_frame,
            options=HyperfineOptions(
                start=start,
                final=final,
                fermi=hfc_fermi,
                anisotropic=hfc_aniso,
                sign=hfc_sign,
            ),
        )
        rotate_hyperfine_to_frame(results=metal_hyperfine_coupling, rotation=lf_frame)
        metal_hyperfine_coupling.s_admixture = s_admixture

    else:  # Hyperfine not calculated
        _, _, xyz_order = determine_diagonal_frame(
            ligand_field_matrix=calculation.ligand_field.ligand_field_matrix,
            g_vectors=g_factors.vectors,
        )
        g_factors.apply_g_order(order=xyz_order)

    return EPRProperties(
        start=start,
        final=final,
        g_factors=g_factors,
        metal_hyperfine_coupling=metal_hyperfine_coupling,
    )


def calculate_thermal_epr_properties(
    calculation: Calculation,
    epr_levels_soc: list[EPRLevelSOC],
    epr_levels_nosoc: list[EPRLevelNoSOC],
    g_factor_type: GFactorType = GFactorType.EFFECTIVE,
    system: ElectronicStructureCalculation | None = None,
    calculate_hfc: HyperfineMode = HyperfineMode.NONE,
    predict_p: bool = False,
    predict_kappa: bool = False,
) -> EPRProperties:
    """Calculate thermally-weighted EPR properties. The eigenfunctions EPR
    parameters are calculated for require pre-calculation and describe each
    electronic level (in the presence or absence of spin-orbit coupling (SOC))
    weighted by its spin projection and Boltzmann population.

    This function automatically computes EPR g-factors, either real g factors
    or effective g factors according to the g_factor_type.

    Args:
        calculation: Results of Ligand Field calculation
        epr_levels_soc: list of EPRLevelSOC, specifying each SOC-split level and
            its Boltzmann and spin projection weighting.
        epr_levels_nosoc: list of EPRLevelNoSOC, specifying each electronic level
            in the absence of SOC weighted by its Boltzmann population.
        g_factor_type: Specifies whether real or effective g values are calculated.
            - "real": real g values (no zero-field splitting) are calculated.
            - "effective": effective g values (zero-field split) are calculated.
        system: ElectronicStructureCalculation, parameters and details of
            the molecule system if Molecule is not None
        calculate_hfc: HyperfineMode which configures what hyperfine information
            is computed.
            - "none": metal-hyperfine is not calculated.
            - "fermi": only the isotropic fermi contact values are computed.
            - "dipolar+orbital": anisotropic metal-hyperfine 'A' values are calculated
                    to give the dipolar and orbital contributions.
            - "dipolar+orbital+fermi+signs": The signs of each contribution
                    and the total 'A' values are computed.
        predict_p: Calculates a P value for the given metal and oxidation state
        predict_kappa: Estimates a value of the kappa parameter for a given complex

    Returns:
        EPRProperties
    """
    number_levels = (
        len(epr_levels_soc)
        if g_factor_type == GFactorType.EFFECTIVE
        else len(epr_levels_nosoc)
    )
    _LOGGER.info(f"Calculating thermal EPR properties for {number_levels} levels")

    g_factors_real = calculate_real_g_factors(
        eigenfunctions=calculation.eigenfunctions,
        muxyz=calculation.get_magnetic_dipole_moments(),
    )
    if g_factor_type == GFactorType.EFFECTIVE:
        g_factors_output = calculate_g_effective(
            epr_levels_soc=epr_levels_soc,
            muxyz=calculation.get_magnetic_dipole_moments(),
        )
    else:
        g_factors_output = copy.deepcopy(g_factors_real)

    g_output_finite = bool(np.all(np.isfinite(g_factors_output.values)))
    g_real_finite = bool(np.all(np.isfinite(g_factors_output.values)))

    if g_output_finite:
        _, _, xyz_order = determine_diagonal_frame(
            ligand_field_matrix=calculation.ligand_field.ligand_field_matrix,
            g_vectors=g_factors_output.vectors,
        )
        g_factors_output.apply_g_order(order=xyz_order)

    # Return if hyperfine calculations are disabled
    if calculate_hfc == HyperfineMode.NONE:
        return EPRProperties(
            start=epr_levels_nosoc[0]["start"],
            final=epr_levels_nosoc[-1]["final"],
            g_factors=g_factors_output,
        )

    # Return if EPR silent
    if not g_real_finite:
        _LOGGER.info(
            "EPR properties could not be calculated as the system is EPR silent"
        )
        hyperfine_results = HyperfineResults(
            fermi_contact=np.full(3, np.nan),
            dipolar=ASquaredTensor.zero_a_tensor(),
            orbital=ASquaredTensor.zero_a_tensor(),
            sign=ASignResults.zero_a_signs(),
        )
        return EPRProperties(
            start=epr_levels_nosoc[0]["start"],
            final=epr_levels_nosoc[-1]["final"],
            g_factors=g_factors_output,
            metal_hyperfine_coupling=hyperfine_results,
        )

    # Calculate hyperfine interaction
    assert calculation.parameters.hyperfine is not None
    hfc_fermi = calculate_hfc in [HyperfineMode.FERMI, HyperfineMode.ALL]
    hfc_aniso = calculate_hfc in [HyperfineMode.ANISO, HyperfineMode.ALL]
    hfc_sign = calculate_hfc == HyperfineMode.ALL

    isotope = calculation.parameters.hyperfine.metal.title()
    element = FirstRowTransitionMetals(extract_element(isotope).title())
    oxidation_state = calculate_oxidation_state(
        d_electrons=calculation.parameters.d_electrons,
        element=element,
    )

    # Rotate calculation to the diagonal ligand-field frame
    g, lf_frame, real_xyz_order = determine_diagonal_frame(
        ligand_field_matrix=calculation.ligand_field.ligand_field_matrix,
        g_vectors=g_factors_real.vectors,
    )
    g_factors_real.apply_g_order(order=real_xyz_order)

    g_real_frame = copy.deepcopy(g_factors_real)
    g_real_frame.vectors = np.eye(3)

    lf_in_diagonal_frame = rotate_lf_frame(
        ref_vectors=lf_frame,
        lf_matrix=calculation.ligand_field.ligand_field_matrix,
    )

    calculation_parameters = copy.deepcopy(calculation.parameters)
    calculation_parameters.ligand_field = lf_in_diagonal_frame
    calculation_hyperfine = run_parameters(
        parameters=calculation_parameters, calculate_configuration_projections=False
    )
    assert calculation_hyperfine.eigenfunctions_lf is not None
    calculation_hyperfine.eigenfunctions_no_soc = get_configuration_projection(
        many_electron_eigenfunctions=calculation_hyperfine.eigenfunctions_no_soc,
        ligand_field_eigenfunctions=calculation_hyperfine.eigenfunctions_lf,
        ligand_field_orbital_energies=calculation_hyperfine.ligand_field.eigenvalues,
        orbital_occupations=calculation_hyperfine.orbital_occupations,
    )

    # Calculate hyperfine per level in the diagonal lf frame
    dipolar = None
    assert calculation_hyperfine.parameters.hyperfine is not None
    if predict_p:
        calculation_hyperfine.parameters.hyperfine.p = calculate_p(
            metal=isotope,
            oxidation_state=oxidation_state,
        )
        calculation_hyperfine.parameters.hyperfine.units = "(MHz)"
        _LOGGER.info(
            f"P value set to: {calculation_hyperfine.parameters.hyperfine.p} MHz"
        )

    if hfc_aniso:
        s_mat, y_mat = construct_operator_matrices(
            matrices=calculation_hyperfine.basis_matrices,
            eigenvectors_no_soc=calculation_hyperfine.eigenfunctions_no_soc.eigenvectors,
            metal=element,
        )
        dipolar = calculate_dipolar(s_mat=s_mat, y_mat=y_mat)

    thermal_hyperfine = thermally_average_hyperfine(
        epr_levels=epr_levels_nosoc,
        calculation=calculation_hyperfine,
        options=HyperfineOptions(
            fermi=hfc_fermi,
            anisotropic=hfc_aniso,
        ),
        predict_kappa=predict_kappa,
        dipolar_operators=dipolar,
        system=system,
    )
    normalised_hyperfine, epr_silent = normalise_thermal_hyperfine(
        thermal_hyperfine=thermal_hyperfine, predict_kappa=predict_kappa
    )

    dipolar_cartesian = (
        lf_frame @ normalised_hyperfine.dipolar_tensor @ lf_frame.conj().T
    )
    dipolar_g_frame = (
        g_factors_real.vectors.conj().T @ dipolar_cartesian @ g_factors_real.vectors
    )
    hyperfine_results = solve_thermal_hyperfine(
        parameters=calculation_hyperfine.parameters.hyperfine,
        dipolar=dipolar_g_frame,
        real_g_factors=g_real_frame,
        fermi=normalised_hyperfine.average_fermi,
        hfc_sign=hfc_sign,
        hfc_aniso=hfc_aniso,
        hfc_fermi=hfc_fermi,
        epr_silent=epr_silent,
    )
    rotate_hyperfine_to_frame(
        results=hyperfine_results, rotation=g_factors_real.vectors
    )

    if predict_kappa:
        hyperfine_results.kappa = normalised_hyperfine.total_kappa
    else:
        hyperfine_results.kappa = calculation_hyperfine.parameters.hyperfine.kappa
    hyperfine_results.s_admixture = normalised_hyperfine.total_s_admixture

    return EPRProperties(
        start=0,
        final=number_levels,
        g_factors=g_factors_output,
        metal_hyperfine_coupling=hyperfine_results,
    )


def calculate_susceptibilities(
    calculation: Calculation,
    temperatures: Iterable[float],
    second_order_zeeman_energy_cutoff: float = 30000.0,
) -> ParamagneticSusceptibilityTensors:
    """Calculate paramagnetic susceptibilities.

    Args:
        temperatures: Iterable of temperatures (Kelvin).
        second_order_zeeman_energy_cutoff: The energy (cm-1) where contributions
            from higher-lying excited states are not included for the Second-order
            Zeeman contribution.

    Returns:
        The paramagnetic susceptibilities, ordered by the input temperatures.
    """
    _temperatures = np.ascontiguousarray(temperatures)
    energies = np.array(
        [
            f.energy - calculation.eigenfunctions.eigenfunctions[0].energy
            for f in calculation.eigenfunctions.eigenfunctions
        ]
    )
    foz_limit = (
        max_thermally_occupied_energy(
            energies=energies,
            temperature=float(max(_temperatures)),
            limit=0.999,
        )
        + 1
    )
    soz_limit = np.argmax(np.where(energies < second_order_zeeman_energy_cutoff)[0]) + 1
    _LOGGER.info(
        f"Calculate paramagnetic susceptibilities for {len(_temperatures)} temperatures, "
        f"first-order zeeman limit: 0->{foz_limit}, "
        f"second-order zeeman limit: 0->{soz_limit}."
    )
    return calculate_paramagnetic_susceptibilities(
        muxyz=calculation.get_magnetic_dipole_moments(),
        energies=energies,
        temperatures=_temperatures,
        first_order_zeeman_max_index=foz_limit,
        second_order_zeeman_max_index=soz_limit,
    )
