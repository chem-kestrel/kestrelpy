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
from functools import lru_cache
from typing import Iterator

import numpy as np

from kestrelpy.calculations.effective_hamiltonian import ConstructEffectiveHamiltonian
from kestrelpy.calculations.interelectronic_repulsion import (
    get_interelectronic_repulsion_parameters,
)
from kestrelpy.calculations.ligand_field_matrix import LigandField
from kestrelpy.calculations.ligand_field_matrix import (
    get_ligand_field_parameters as _get_ligand_field_matrix_parameters,
)
from kestrelpy.calculations.magnetism.get_muxyz_matrices import get_muxyz_matrices
from kestrelpy.calculations.magnetism.susceptibilities import (
    ParamagneticSusceptibilityTensors,
)
from kestrelpy.calculations.many_electron_eigenfunctions.basis_set import (
    ManyElectronBasis,
    get_many_electron_basis,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ManyElectronEigenfunctions,
    get_configuration_projection,
    get_many_electron_eigenfunctions,
)
from kestrelpy.calculations.many_electron_eigenfunctions.orbital_occupations import (
    construct_orbital_occupations,
)
from kestrelpy.calculations.many_electron_matrices import (
    NonZeroMatrixElements,
    get_non_zero_matrix_elements,
)
from kestrelpy.calculations.parameters import (
    MagneticField,
    OrbitalReductionFactors,
    SpinOrbitCoupling,
)
from kestrelpy.configuration import get_logger
from kestrelpy.models.input_models.input_models import CalculationBlock, Parameters
from kestrelpy.molecule import Molecule, change_molecule_geometry
from kestrelpy.parameters import (
    GeometryAssignment,
    HyperfineAssignment,
    HyperfineParameters,
    LigandAssignment,
    LigandParameters,
    MetalAssignment,
    MetalParameters,
)
from kestrelpy.workflows.ligand_field_matrices import get_ligand_field_matrices
from kestrelpy.workflows.resolve_parameters import get_ligand_field_parameters

_LOGGER = get_logger(__name__)


def get_calculation_parameters(
    d_electrons: int,
    ligand_field_matrix: np.ndarray,
    ds_stabilisation_matrix: np.ndarray,
    racah_b: float,
    racah_c: float,
    spin_orbit_coupling: float,
    kx: float,
    ky: float,
    kz: float,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.0,
    p: float = 0.0,
    kappa: float = 0.0,
    metal: str = "M",
) -> Parameters:
    return Parameters(
        d_electrons=int(d_electrons),
        ligand_field=_get_ligand_field_matrix_parameters(
            ligand_field_matrix=ligand_field_matrix + ds_stabilisation_matrix,
        ),
        interelectronic_repulsion=get_interelectronic_repulsion_parameters(
            racah_b=racah_b, racah_c=racah_c
        ),
        spin_orbit_coupling=SpinOrbitCoupling(SOC=spin_orbit_coupling),
        orbital_reduction_factors=OrbitalReductionFactors(KX=kx, KY=ky, KZ=kz),
        magnetic_field=MagneticField(BX=bx, BY=by, BZ=bz),
        hyperfine=HyperfineParameters(metal=metal, p=p, kappa=kappa),
    )


@dataclass
class Calculation:
    """Stores the state of a ligand-field calculation."""

    # System
    parameters: Parameters

    # Calculation prerequisites
    basis_matrices: NonZeroMatrixElements
    many_electron_basis: ManyElectronBasis
    orbital_occupations: dict[int, np.ndarray]

    # Results
    ligand_field: LigandField
    eigenfunctions: ManyElectronEigenfunctions
    eigenfunctions_no_soc: ManyElectronEigenfunctions
    eigenfunctions_lf: ManyElectronEigenfunctions | None = None

    # Post-diagonalisation analysis
    susceptibilities: ParamagneticSusceptibilityTensors | None = None

    # Matrices
    magnetic_dipole_moments: np.ndarray | None = None
    electronic_transition_dipole_moments: np.ndarray | None = None

    def get_magnetic_dipole_moments(self) -> np.ndarray:
        if self.magnetic_dipole_moments is None:
            self.magnetic_dipole_moments = get_muxyz_matrices(
                eigenvectors=self.eigenfunctions.eigenvectors,
                matrices=self.basis_matrices,
                reduction_factors=self.parameters.orbital_reduction_factors,
            )
        return self.magnetic_dipole_moments


@lru_cache(maxsize=2)
def _get_calculations_prerequisites(
    d_electrons: int,
) -> tuple[NonZeroMatrixElements, ManyElectronBasis, dict[int, np.ndarray]]:
    return (
        get_non_zero_matrix_elements(d_electrons=d_electrons),
        get_many_electron_basis(d_electrons=d_electrons),
        construct_orbital_occupations(d_electrons=d_electrons),
    )


def run_parameters(
    parameters: Parameters,
    calculate_configuration_projections: bool = False,
) -> Calculation:
    """
    Solves the ligand-field Hamiltonian to generate the ligand-field eigenfunctions.

    Args:
        parameters: The ligand-field parameters which define the system.
        calculate_configuration_projections: Flag to determine whether the configuration
            projection is calculated for each eigenfunction (default=False).

    Returns:
        The Calculation object which can be used to perform post-diagonalisation
        analysis (e.g. calculate g-factors)
    """
    basis_matrices, basis_set, orbital_occupations = _get_calculations_prerequisites(
        d_electrons=parameters.d_electrons
    )

    # Get the full Hamiltonian and ligand-field only Hamiltonian.
    full_hamiltonian, lf_only_hamiltonian, no_soc_hamiltonian = (
        ConstructEffectiveHamiltonian(
            d_electrons=parameters.d_electrons,
            matrices=basis_matrices,
        ).construct(
            multipole_expansion=parameters.ligand_field.multipole_expansion,
            interelectronic_repulsion=parameters.interelectronic_repulsion,
            spin_orbit_coupling=parameters.spin_orbit_coupling,
            magnetic_field=parameters.magnetic_field,
            orbital_reduction_factors=parameters.orbital_reduction_factors,
        )
    )

    # Calculate Eigenfunctions
    lf_eigenfunctions = None
    eigenfunctions = get_many_electron_eigenfunctions(
        many_electron_basis=basis_set,
        effective_hamiltonian=full_hamiltonian,
    )
    eigenfunctions_no_soc = get_many_electron_eigenfunctions(
        many_electron_basis=basis_set, effective_hamiltonian=no_soc_hamiltonian
    )
    if parameters.hyperfine or calculate_configuration_projections:
        lf_eigenfunctions = get_many_electron_eigenfunctions(
            many_electron_basis=basis_set,
            effective_hamiltonian=lf_only_hamiltonian,
        )

    if calculate_configuration_projections:
        assert lf_eigenfunctions is not None
        _LOGGER.info("Calculating the configuration projection...")
        eigenfunctions = get_configuration_projection(
            many_electron_eigenfunctions=eigenfunctions,
            ligand_field_eigenfunctions=lf_eigenfunctions,
            ligand_field_orbital_energies=parameters.ligand_field.eigenvalues,
            orbital_occupations=orbital_occupations,
        )

    return Calculation(
        parameters=parameters,
        basis_matrices=basis_matrices,
        many_electron_basis=basis_set,
        orbital_occupations=orbital_occupations,
        ligand_field=parameters.ligand_field,
        eigenfunctions=eigenfunctions,
        eigenfunctions_no_soc=eigenfunctions_no_soc,
        eigenfunctions_lf=lf_eigenfunctions,
    )


@dataclass(frozen=True)
class ElectronicStructureCalculation:
    molecule: Molecule
    metal_assignments: MetalAssignment
    hyperfine_parameters: HyperfineParameters
    ligand_assignments: list[LigandAssignment]
    variation: dict[str, float]
    metal_parameters: MetalParameters
    ligand_parameters: list[LigandParameters]
    calculation: Calculation


def calculate_electronic_structure(
    molecule: Molecule,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    hyperfine: HyperfineAssignment | None = None,
    geometry: list[GeometryAssignment] | None = None,
    calculation: CalculationBlock = CalculationBlock(),
) -> Iterator[ElectronicStructureCalculation]:
    """Construct the ligand field matrix from a molecule and its parameters and run
    a calculation to determine the electronic eigenfunctions"""

    assignments: list[
        LigandAssignment | MetalAssignment | HyperfineAssignment | GeometryAssignment
    ] = [metal, *ligands, *(geometry or [])]
    if hyperfine is not None:
        assignments.append(hyperfine)

    for resolved_parameters in get_ligand_field_parameters(assignments):
        variation = resolved_parameters.variation
        metal_parameters = resolved_parameters.metal
        ligand_parameters = resolved_parameters.ligands
        hyperfine_parameters = resolved_parameters.hyperfine
        geometry_parameters = resolved_parameters.geometry

        if any(key.startswith("atom.") for key in variation):
            molecule = change_molecule_geometry(
                molecule=molecule,
                geometry_parameters=geometry_parameters,
            )

        # Get ligand-field matrices.
        ligand_field_matrix, ds_stabilisation_matrix = get_ligand_field_matrices(
            molecule=molecule, ligand_field_parameters=ligand_parameters
        )

        parameters = get_calculation_parameters(
            d_electrons=metal_parameters.d_electrons,
            ligand_field_matrix=ligand_field_matrix,
            ds_stabilisation_matrix=ds_stabilisation_matrix,
            racah_b=metal_parameters.racah_b,
            racah_c=metal_parameters.racah_c,
            spin_orbit_coupling=metal_parameters.spin_orbit_coupling,
            p=hyperfine_parameters.p,
            kappa=hyperfine_parameters.kappa,
            metal=hyperfine_parameters.metal,
            kx=metal_parameters.kiso,
            ky=metal_parameters.kiso,
            kz=metal_parameters.kiso,
            bx=0.0,
            by=0.0,
            bz=0.0,
        )
        electronic_structure = run_parameters(
            parameters=parameters,
            calculate_configuration_projections=calculation.configuration_projection,
        )
        yield ElectronicStructureCalculation(
            molecule=molecule,
            metal_assignments=metal,
            ligand_assignments=ligands,
            variation=variation,
            metal_parameters=metal_parameters,
            ligand_parameters=ligand_parameters,
            calculation=electronic_structure,
            hyperfine_parameters=hyperfine_parameters,
        )
