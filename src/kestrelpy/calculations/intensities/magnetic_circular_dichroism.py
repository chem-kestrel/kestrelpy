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
from multiprocessing import Pool
from typing import Any, Callable, Iterable, NamedTuple, TypeVar

import numpy as np
import scipy.integrate

from kestrelpy.calculations.constants import DEBYE_TO_ESU_CM
from kestrelpy.calculations.effective_hamiltonian import ConstructEffectiveHamiltonian
from kestrelpy.calculations.intensities.electronic_transition_dipole_moment import (
    GlobalTransitionDipoleMultipoleExpansion,
    get_electronic_transition_dipole_moment_matrix_elements,
)
from kestrelpy.calculations.intensities.spectroscopy import (
    IntensityResults,
    SpectroscopicTransition,
    Spectrum,
    calculate_molar_absorption_coefficient,
    calculate_oscillator_strength,
    get_boltzmann_weightings,
)
from kestrelpy.calculations.many_electron_eigenfunctions.basis_set import (
    ManyElectronBasis,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    Eigenfunction,
    ManyElectronEigenfunctions,
    get_many_electron_eigenfunctions,
)
from kestrelpy.calculations.many_electron_matrices import NonZeroMatrixElements
from kestrelpy.calculations.parameters import MagneticField, OrbitalReductionFactors
from kestrelpy.calculations.spherical_polar_coordinates import (
    calculate_direction_cosine_matrix,
)
from kestrelpy.configuration import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class MCDTransition:
    energy: float
    d_fosc: float
    from_eigenfunction: Eigenfunction
    to_eigenfunction: Eigenfunction


@dataclass
class MCDOrientation:
    transitions: list[MCDTransition]
    energies: list[float]
    theta: float
    phi: float
    psi: float
    weight: float


@dataclass
class MagneticCircularDichroism(IntensityResults):
    orientations: list[MCDOrientation]
    magnetic_field: float

    def get_spectrum(self, temperature: float) -> Spectrum:
        spectroscopic_transitions: list[SpectroscopicTransition] = []
        for orientation in self.orientations:
            boltzmann_weightings = get_boltzmann_weightings(
                energies=np.array(orientation.energies),
                temperatures=np.ascontiguousarray(temperature),
            )

            for transition in orientation.transitions:
                level_index = transition.from_eigenfunction.index
                spectroscopic_transitions.append(
                    SpectroscopicTransition(
                        energy=transition.energy,
                        intensity=calculate_molar_absorption_coefficient(
                            fosc=transition.d_fosc
                        )
                        * boltzmann_weightings[level_index, 0]
                        * orientation.weight,
                    )
                )

        return Spectrum(
            temperature=temperature,
            magnetic_field=self.magnetic_field,
            transitions=spectroscopic_transitions,
        )


def transform_magnetic_field_from_lab_to_molecular_frame(
    magnetic_field: np.ndarray, euler_rotation_matrix: np.ndarray
) -> np.ndarray:
    """Transform the external magnetic field to the molecule-fixed frame.

    Args:
        magnetic_field: The X, Y, and Z components of the magnetic field.
        euler_rotation_matrix: The euler rotation matrix relating the molecular
            and lab frames in the cartesian coordinate basis.

    Returns:
        Vector with shape(3,).

    References:
        1.  Equation 31 from:
            Dmitry Ganyushin, Frank Neese; First-principles calculations of magnetic circular
            dichroism spectra. J. Chem. Phys. 21 March 2008; 128 (11): 114117.
            https://doi.org/10.1063/1.2894297
    """
    return np.matmul(euler_rotation_matrix.T, magnetic_field)


def transform_etdm_from_molecular_to_lab_frame(
    electronic_transtion_dipole_moment: np.ndarray, euler_rotation_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform the electronic transition-dipole moment vector from the molecular frame
    to the lab frame.

    Args:
        electronic_transition_dipole_moment: X, Y, and Z components of the electronic
            transition-dipole moment vector in the molecular frame.
        euler_rotation_matrix: The euler rotation matrix relating the molecular
            and lab frames in the cartesian coordinate basis.

    Returns:
        Vector with shape(3,).

    References:
        1.  Equation 30 from:
            Dmitry Ganyushin, Frank Neese; First-principles calculations of magnetic circular
            dichroism spectra. J. Chem. Phys. 21 March 2008; 128 (11): 114117.
            https://doi.org/10.1063/1.2894297
    """
    return np.einsum(
        "ab,bij->aij", euler_rotation_matrix, electronic_transtion_dipole_moment
    )


JOULES_TO_WAVENUMBERS = 5.034116e22


class GridPoints(NamedTuple):
    theta: float
    phi: float
    psi: float
    weight: float


def get_lebedev_grid(quadrature_order: int) -> Iterable[GridPoints]:
    """Get the lebedev grid points in spherical polar coordinates."""
    angles, weights = scipy.integrate.lebedev_rule(quadrature_order)
    for (x, y, z), weight in zip(angles.T, weights, strict=True):
        theta = np.arccos(z)
        phi = np.arctan2(y, x) % (2 * np.pi)
        # Psi is set to 0.0 radians and this is because the MCD in the context
        # of our ligand-field model is not dependent on the rotation around the
        # local z-axis of the magnetic field.
        psi = 0.0
        yield GridPoints(
            theta=theta,
            phi=phi,
            psi=psi,
            weight=float(weight) / (4.0 * np.pi),
        )


T = TypeVar("T")
U = TypeVar("U")


def _perform_parallel_integration_over_sphere(
    func: Callable[..., U],
    inputs: Iterable[Iterable[T]],
    number_of_processes: int,
) -> Iterable[U]:
    with Pool(processes=number_of_processes) as p:
        results = p.starmap(func, inputs)
    return results


def get_zeeman_eigenfunctions_and_rotated_etdm(
    theta: float,
    phi: float,
    psi: float,
    magnetic_field: float,
    global_transition_dipole_multipole: GlobalTransitionDipoleMultipoleExpansion,
    full_hamiltonian: np.ndarray,
    effective_hamiltonian: ConstructEffectiveHamiltonian,
    orbital_reduction_factors: OrbitalReductionFactors,
    many_electron_basis: ManyElectronBasis,
    basis_matrices: NonZeroMatrixElements,
) -> tuple[ManyElectronEigenfunctions, np.ndarray]:
    """Get the zeeman eigenfunctions and electronic transition dipole moments in the lab frame.

    Args:
        theta: Theta angle of magnetic field with respect to the molecular frame.
        phi: Phi angle of magnetic field with respect to the molecular frame.
        psi: Psi angle of magnetic field with respect to the molecular frame.
        magnetic_field: The strength of the magnetic field in Tesla.
        global_transition_dipole_multipole: The electronic transition dipole moment of the molecule
            as a multipole expansion.
        full_hamiltonian: Square Hermitian matrix of the molecular ligand-field Hamiltonian.
        effective_hamiltonian: The object to apply the magnetic field perturbation to the full_hamiltonian.
        orbital_reduction_factors: The orbital reduction factors.
        many_electron_basis: The JMLS basis of the molecule.
        basis_matrices: THe basis matrices of the dn system.

    Returns:
        ManyElectronEigenfunctions: The many-electron eigenfunctions perturbed by the magnetic
            field orientated at the zyz angles theta, phi, and psi.
        np.ndarray: The electronic transition-dipole moment matrix elements in the *lab frame*.
    """
    euler_rotation_matrix = calculate_direction_cosine_matrix(
        theta=theta, phi=phi, psi=psi
    )

    # Get magnetic field in the molecular frame.
    magnetic_field_lab_frame = np.array([0.0, 0.0, magnetic_field])
    magnetic_field_molecular_frame = (
        transform_magnetic_field_from_lab_to_molecular_frame(
            magnetic_field=magnetic_field_lab_frame,
            euler_rotation_matrix=euler_rotation_matrix,
        )
        * JOULES_TO_WAVENUMBERS
        * 9.274e-24  # Bohr Magneton
    )
    basis_size = full_hamiltonian.shape[0]
    magnetic_field_perturbation = np.zeros((basis_size, basis_size))

    # Apply the magnetic-field contribution to the Hamiltonian
    # and get the Eigenfunctions.
    hamiltonian = np.copy(
        full_hamiltonian
    ) + effective_hamiltonian.add_magnetic_field_contribution(
        matrix=magnetic_field_perturbation + 0j,
        magnetic_field=MagneticField(*magnetic_field_molecular_frame),
        orbital_reduction_factors=orbital_reduction_factors,
    )
    eigenfunctions = get_many_electron_eigenfunctions(
        many_electron_basis=many_electron_basis, effective_hamiltonian=hamiltonian
    )

    # Get electronic transition dipole moments.
    etdm_lab_frame = transform_etdm_from_molecular_to_lab_frame(
        electronic_transtion_dipole_moment=get_electronic_transition_dipole_moment_matrix_elements(
            eigenvectors=eigenfunctions.eigenvectors,
            matrices=basis_matrices,
            global_transition_dipole_multipole_expansion=global_transition_dipole_multipole,
        ),
        euler_rotation_matrix=euler_rotation_matrix,
    )

    return eigenfunctions, etdm_lab_frame


def _calculate_mcd(
    eigenfunctions: list[Eigenfunction],
    electronic_transtion_dipole_moment: np.ndarray,
    n_ground_levels: int,
) -> list[MCDTransition]:
    transitions: list[MCDTransition] = []
    for ground_state in eigenfunctions[:n_ground_levels]:
        for excited_state in eigenfunctions[n_ground_levels:]:
            i, j = ground_state.index, excited_state.index

            tx1, ty1, _ = electronic_transtion_dipole_moment[:, i, j] * DEBYE_TO_ESU_CM
            tx2, ty2, _ = electronic_transtion_dipole_moment[:, j, i] * DEBYE_TO_ESU_CM
            intensity = np.imag(tx1 * ty2 - ty1 * tx2)

            energy = excited_state.energy - ground_state.energy
            d_fosc = calculate_oscillator_strength(intensity, energy)

            transitions.append(
                MCDTransition(
                    energy=energy,
                    d_fosc=d_fosc,
                    from_eigenfunction=ground_state,
                    to_eigenfunction=excited_state,
                )
            )
    return transitions


def _process_mcd(
    theta: float,
    phi: float,
    psi: float,
    weight: float,
    magnetic_field: float,
    n_ground_levels: int,
    global_transition_dipole_multipole: GlobalTransitionDipoleMultipoleExpansion,
    full_hamiltonian: np.ndarray,
    effective_hamiltonian: ConstructEffectiveHamiltonian,
    orbital_reduction_factors: OrbitalReductionFactors,
    many_electron_basis: ManyElectronBasis,
    basis_matrices: NonZeroMatrixElements,
) -> MCDOrientation:
    """Calculate the MCD transitions and their intensities for the molecular-frame magnetic field.

    Returns:
        The list of MCD transitions for the Magnetic field applied at the angles theta, phi, and psi.
    """
    eigenfunctions, etdm_lab_frame = get_zeeman_eigenfunctions_and_rotated_etdm(
        theta=theta,
        phi=phi,
        psi=psi,
        magnetic_field=magnetic_field,
        global_transition_dipole_multipole=global_transition_dipole_multipole,
        full_hamiltonian=full_hamiltonian,
        effective_hamiltonian=effective_hamiltonian,
        orbital_reduction_factors=orbital_reduction_factors,
        many_electron_basis=many_electron_basis,
        basis_matrices=basis_matrices,
    )
    return MCDOrientation(
        transitions=_calculate_mcd(
            eigenfunctions=eigenfunctions.eigenfunctions,
            electronic_transtion_dipole_moment=etdm_lab_frame,
            n_ground_levels=n_ground_levels,
        ),
        energies=[f.energy for f in eigenfunctions.eigenfunctions],
        theta=theta,
        phi=phi,
        psi=psi,
        weight=weight,
    )


def calculate_mcd(
    magnetic_field: float,
    angular_grid_steps: int,
    n_ground_levels: int,
    global_transition_dipole_multipole: GlobalTransitionDipoleMultipoleExpansion,
    full_hamiltonian: np.ndarray,
    effective_hamiltonian: ConstructEffectiveHamiltonian,
    orbital_reduction_factors: OrbitalReductionFactors,
    many_electron_basis: ManyElectronBasis,
    basis_matrices: NonZeroMatrixElements,
    parallel_processes: int = 0,
) -> MagneticCircularDichroism:
    """Calculate the MCD transitions over the points of a 3d grid."""
    args = (
        magnetic_field,
        n_ground_levels,
        global_transition_dipole_multipole,
        full_hamiltonian,
        effective_hamiltonian,
        orbital_reduction_factors,
        many_electron_basis,
        basis_matrices,
    )

    tasks: list[tuple[Any, ...]] = []
    for theta, phi, psi, weight in get_lebedev_grid(
        quadrature_order=angular_grid_steps
    ):
        _LOGGER.debug(
            f"Queueing task with {theta=:.3f}, {phi=:.3f}, {psi=:.3f}, {weight=:.3f}"
        )
        tasks.append((theta, phi, psi, weight, *args))
    _LOGGER.info(f"Integrating over {len(tasks)} points on the sphere")

    if parallel_processes:
        mcd_orientations = _perform_parallel_integration_over_sphere(
            func=_process_mcd, inputs=tasks, number_of_processes=parallel_processes
        )
    else:
        mcd_orientations = [_process_mcd(*task) for task in tasks]
    return MagneticCircularDichroism(
        orientations=list(mcd_orientations),
        magnetic_field=magnetic_field,
    )
