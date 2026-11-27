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

from kestrelpy.calculations.effective_hamiltonian import ConstructEffectiveHamiltonian
from kestrelpy.calculations.intensities.absorption import (
    Absorption,
    calculate_absorption,
)
from kestrelpy.calculations.intensities.circular_dichroism import (
    CircularDichroism,
    calculate_cd,
)
from kestrelpy.calculations.intensities.electronic_transition_dipole_moment import (
    GlobalTransitionDipoleMultipoleExpansion,
    IntensityParameters,
    get_electronic_transition_dipole_moment_matrix_elements,
    get_electronic_transition_dipole_moment_multipole_expansion,
)
from kestrelpy.calculations.intensities.magnetic_circular_dichroism import (
    MagneticCircularDichroism,
    calculate_mcd,
)
from kestrelpy.calculations.intensities.spectroscopy import (
    Spectrum,
    max_thermally_occupied_energy,
)
from kestrelpy.calculations.parameters import MagneticField
from kestrelpy.configuration import get_logger
from kestrelpy.models.input_models.input_models import IntensityBlock
from kestrelpy.models.results.calculations import (
    AbsorptionResults,
    CircularDichroismResults,
    IntensityResults,
    MagneticCircularDichroismResults,
)
from kestrelpy.workflows.calculate_electronic_structure import (
    Calculation,
    ElectronicStructureCalculation,
)
from kestrelpy.workflows.resolve_parameters import get_dipole_parameters

_LOGGER = get_logger(__name__)


@dataclass
class IntensityCalculations:
    electronic_structure: Calculation
    dipole_expansion: GlobalTransitionDipoleMultipoleExpansion
    electronic_transition_dipole_moment_matrix_elements: np.ndarray

    def calculate_absorption_transitions(self) -> Absorption:
        """Calculate the d-d absorption intensities."""
        n_ground_levels = max_thermally_occupied_energy(
            energies=np.array(
                [
                    level.energy
                    for level in self.electronic_structure.eigenfunctions.levels
                ]
            ),  # relative energies
            temperature=300.0,
            limit=0.99,  # 99%
        )
        _LOGGER.info(
            f"Calculating d-d absorption intensities from ground levels 0->{n_ground_levels}"
            if n_ground_levels > 0
            else "Calculating d-d absorption intensities from ground level 0"
        )

        self.absorption = calculate_absorption(
            levels=self.electronic_structure.eigenfunctions.levels,
            electronic_transition_dipole_moments=self.electronic_transition_dipole_moment_matrix_elements,
            n_ground_levels=n_ground_levels + 1,
        )
        return self.absorption

    def calculate_circular_dichroism_transitions(self) -> CircularDichroism:
        """Calculate the d-d circular dichroism intensities."""
        n_ground_levels = max_thermally_occupied_energy(
            energies=np.array(
                [
                    level.energy
                    for level in self.electronic_structure.eigenfunctions.levels
                ]
            ),
            temperature=300.0,
            limit=0.99,  # 99%
        )
        _LOGGER.info(
            f"Calculating circular-dichroism intensities from ground levels 0->{n_ground_levels}"
            if n_ground_levels > 0
            else "Calculating circular-dichroism intensities from ground level 0"
        )
        self.circular_dichroism = calculate_cd(
            levels=self.electronic_structure.eigenfunctions.levels,
            electronic_transition_dipole_moment=self.electronic_transition_dipole_moment_matrix_elements,
            muxyz=self.electronic_structure.get_magnetic_dipole_moments(),
            n_ground_levels=n_ground_levels + 1,
        )
        return self.circular_dichroism

    def calculate_magnetic_circular_dichroism_transitions(
        self,
        magnetic_field: float,
        n_angular_steps: int = 3,
        parallel_processes: int = 0,
    ) -> MagneticCircularDichroism:
        """Calculate the magnetic circular dichroism transition intensities.

        Args:
            magnetic_field: The strength of the magnetic field, in Tesla.
            n_angular_steps: The number of grid steps. At present only a Lebedev
                grid is available so n_angular_steps must be an odd number >= 3.
            parallel_processes: The number of parallel processes. If 0 then
                the calculation of MCD is carried out synchronously.
                Defaults to 0.

        Returns:
            Magnetic Circular Dichroism object.
        """
        if n_angular_steps % 2 == 0:
            raise ValueError(
                f"Number of angular steps must be an odd value >=3, instead got {n_angular_steps=}"
            )

        n_ground_states = max_thermally_occupied_energy(
            energies=np.array(
                [
                    f.energy
                    for f in self.electronic_structure.eigenfunctions.eigenfunctions
                ]
            ),
            temperature=300.0,
            limit=0.99,
        )
        _LOGGER.info(
            f"Calculating MCD intensities from ground states 0->{n_ground_states}"
            if n_ground_states > 0
            else "Calculating MCD intensities from ground state 0"
        )
        return calculate_mcd(
            magnetic_field=magnetic_field,
            angular_grid_steps=n_angular_steps,
            n_ground_levels=n_ground_states + 1,
            global_transition_dipole_multipole=self.dipole_expansion,
            full_hamiltonian=ConstructEffectiveHamiltonian(
                d_electrons=self.electronic_structure.parameters.d_electrons,
                matrices=self.electronic_structure.basis_matrices,
            ).construct(
                multipole_expansion=self.electronic_structure.parameters.ligand_field.multipole_expansion,
                interelectronic_repulsion=self.electronic_structure.parameters.interelectronic_repulsion,
                spin_orbit_coupling=self.electronic_structure.parameters.spin_orbit_coupling,
                magnetic_field=MagneticField(
                    0.0, 0.0, 0.0
                ),  # Need magnetic field to be 0.0.
                orbital_reduction_factors=self.electronic_structure.parameters.orbital_reduction_factors,
            )[0],  # do not need lf_hamiltonian
            effective_hamiltonian=ConstructEffectiveHamiltonian(
                d_electrons=self.electronic_structure.parameters.d_electrons,
                matrices=self.electronic_structure.basis_matrices,
            ),
            orbital_reduction_factors=self.electronic_structure.parameters.orbital_reduction_factors,
            many_electron_basis=self.electronic_structure.many_electron_basis,
            basis_matrices=self.electronic_structure.basis_matrices,
            parallel_processes=parallel_processes,
        )


def calculate_intensities(
    calculation: ElectronicStructureCalculation,
    intensities: IntensityBlock,
) -> list[IntensityResults]:
    if not intensities.has_options():
        raise ValueError("No intensity method was chosen, aborting run!")
    intensity_results: list[IntensityResults] = []
    for resolved_parameters in get_dipole_parameters(assignments=intensities.dipoles):
        intensity_parameters: list[IntensityParameters] = []
        for dipole_parameters in resolved_parameters:
            ligand = calculation.molecule.get_ligand(
                dipole_parameters.name
            )  # TODO: cache this
            for atom in ligand.atoms:
                intensity_parameters.append(
                    IntensityParameters(
                        psigma=dipole_parameters.psigma,
                        ppix=dipole_parameters.ppix,
                        ppiy=dipole_parameters.ppiy,
                        fsigma=dipole_parameters.fsigma,
                        fpix=dipole_parameters.fpix,
                        fpiy=dipole_parameters.fpiy,
                        theta=np.radians(atom.theta),
                        phi=np.radians(atom.phi),
                        psi=np.radians(atom.psi),
                    )
                )
        dipole_expansion = get_electronic_transition_dipole_moment_multipole_expansion(
            *intensity_parameters
        )

        calculations = IntensityCalculations(
            electronic_structure=calculation.calculation,
            dipole_expansion=dipole_expansion,
            electronic_transition_dipole_moment_matrix_elements=(
                get_electronic_transition_dipole_moment_matrix_elements(
                    eigenvectors=calculation.calculation.eigenfunctions.eigenvectors,
                    matrices=calculation.calculation.basis_matrices,
                    global_transition_dipole_multipole_expansion=dipole_expansion,
                )
            ),
        )

        absorption: AbsorptionResults | None = None
        cd: CircularDichroismResults | None = None
        mcd: list[MagneticCircularDichroismResults] = []

        if intensities.absorption is not None:
            absorption_results = calculations.calculate_absorption_transitions()
            abso_spectra: list[Spectrum] = []
            for temperature in intensities.absorption.temperatures:
                abso_spectra.append(absorption_results.get_spectrum(temperature))
            absorption = AbsorptionResults(
                results=absorption_results, spectra=abso_spectra
            )

        if intensities.cd is not None:
            cd_results = calculations.calculate_circular_dichroism_transitions()
            cd_spectra = []
            for temperature in intensities.cd.temperatures:
                cd_spectra.append(cd_results.get_spectrum(temperature))
            cd = CircularDichroismResults(results=cd_results, spectra=cd_spectra)

        if intensities.mcd is not None:
            for magfield in intensities.mcd.magnetic_fields:
                mcd_results = (
                    calculations.calculate_magnetic_circular_dichroism_transitions(
                        magnetic_field=magfield,
                        n_angular_steps=intensities.mcd.n_angular_steps,
                        parallel_processes=intensities.mcd.num_processes,
                    )
                )
                _mcd_spectra: list[Spectrum] = []
                for temperature in intensities.mcd.temperatures:
                    _mcd_spectra.append(mcd_results.get_spectrum(temperature))
                mcd.append(
                    MagneticCircularDichroismResults(
                        results=mcd_results, spectra=_mcd_spectra
                    )
                )
        intensity_results.append(
            IntensityResults(
                dipole_parameters=resolved_parameters,
                global_transition_dipole_moment_expansion=dipole_expansion,
                electronic_transition_dipole_moment_expansion=calculations.electronic_transition_dipole_moment_matrix_elements,
                absorption=absorption,
                cd=cd,
                mcd=mcd if mcd else [],
            )
        )
    return intensity_results
