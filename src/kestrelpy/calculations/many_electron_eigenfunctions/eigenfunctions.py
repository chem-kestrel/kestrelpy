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

This Eigenfunctions module contains methods and classes for representing the Eigenfunctions
of the full many-electron Hamiltonian in Kestrel.

Terminology:
    Eigenfunction - The eigenvalue:eigenvector pair that is a solution of the function
        H_eff x eigenfunction = E x eigenfunction
    Level - A group of degenerate (same energy) eigenfunctions.
    Many-Electron Eigenfunctions - The eigenfunctions for the full Hamiltonian including
        the effects of the ligand field, interelectronic repulsion, and spin-orbit coupling;
        NOTE: can refer to eigenfunctions of the d1 case.
"""

from dataclasses import dataclass, field

import numpy as np
from tabulate import tabulate

from kestrelpy.calculations.many_electron_eigenfunctions.basis_set import (
    ManyElectronBasis,
)
from kestrelpy.calculations.many_electron_eigenfunctions.configuration_projection import (
    Configuration,
    ConfigurationProjection,
    get_eigenfunction_configuration_projection,
    get_level_configuration_projection,
    map_configuration_projection,
    setup_configuration_projection,
)
from kestrelpy.calculations.many_electron_eigenfunctions.spin_projection import (
    SpinProjections,
    get_eigenfunction_spin_multiplicity_projection,
    get_level_spin_multiplicity_projection,
    get_spin_projections,
)
from kestrelpy.calculations.solve_matrix import solve_matrix


@dataclass
class Eigenfunction:
    index: int
    energy: float
    zero_point_energy: float
    spin_multiplicity_projection: dict[int, float]
    configurations: list[Configuration]

    def dominant_multiplicity(self) -> int:
        """Return the dominant spin multiplicity"""
        return max(
            self.spin_multiplicity_projection,
            key=self.spin_multiplicity_projection.__getitem__,
        )


@dataclass
class ElectronicLevel:
    index: int
    eigenfunctions: list[Eigenfunction]
    energy: float
    zero_point_energy: float
    degeneracy: int
    spin_multiplicity_projection: dict[int, float]
    configurations: list[Configuration]

    def dominant_multiplicity(self) -> int:
        """Return the dominant spin multiplicity"""
        return max(
            self.spin_multiplicity_projection,
            key=self.spin_multiplicity_projection.__getitem__,
        )


@dataclass
class ManyElectronEigenfunctions:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    zero_point_energy: float
    spin_projections: SpinProjections
    configuration_projection: ConfigurationProjection
    eigenfunctions: list[Eigenfunction] = field(init=False)
    levels: list[ElectronicLevel] = field(init=False)

    def __post_init__(self) -> None:
        self.eigenfunctions = self._get_eigenfunctions()
        self.levels = self.get_levels()

    def __str__(self) -> str:
        spin_multiplicities = self.spin_projections.spin_multiplicities
        data = []
        for index, level in enumerate(self.levels):
            configurations = [
                config.get_string_representation() for config in level.configurations
            ]
            data.append(
                [
                    index,
                    int(level.degeneracy),
                    round(level.energy, 1),
                    *[
                        float(projection) / level.degeneracy
                        for projection in level.spin_multiplicity_projection.values()
                    ],
                    "\n".join(configurations),
                ]
            )
        return tabulate(
            data,
            headers=[
                "Level",
                "Degeneracy",
                "Energy (cm-1)",
                *[f"2S+1={spin_mult}" for spin_mult in spin_multiplicities],
                # *list(map(str, spin_multiplicities)),
                "Configuration\n <2S+1>: <occupation> (<contribution>)",
            ],
            floatfmt=("", "", ".1f", ".3f", ".3f"),
            tablefmt="grid",
        )

    def _get_eigenfunctions(self) -> list[Eigenfunction]:
        eigenfunctions: list[Eigenfunction] = []
        for index, energy in enumerate(self.eigenvalues):
            eigenfunctions.append(
                Eigenfunction(
                    index=index,
                    energy=energy - self.zero_point_energy,
                    zero_point_energy=self.zero_point_energy,
                    spin_multiplicity_projection=get_eigenfunction_spin_multiplicity_projection(
                        index=index, spin_projections=self.spin_projections
                    ),
                    configurations=get_eigenfunction_configuration_projection(
                        index=index,
                        configuration_projection=self.configuration_projection,
                    ),
                )
            )
        return eigenfunctions

    def get_levels(
        self, energetic_difference: float = 1.0e-01
    ) -> list[ElectronicLevel]:
        """Return the electronic energy levels.

        Args:
            energetic_difference: The difference in eigenvalues required for two eigenfunctions
                to be considered degenerate, or not.

        Returns:
            List of electronic levels.
        """
        decimals = abs(int(np.log10(energetic_difference)))
        unique_levels = np.unique(
            np.round(self.eigenvalues, decimals=decimals), return_counts=True
        )

        electronic_levels: list[ElectronicLevel] = []
        start_eigenfunction_index = 0
        level_index = 0
        for energy, degeneracy in zip(*unique_levels):
            final_eigenfunction_index = start_eigenfunction_index + degeneracy
            electronic_levels.append(
                ElectronicLevel(
                    index=level_index,
                    eigenfunctions=self.eigenfunctions[
                        start_eigenfunction_index:final_eigenfunction_index
                    ],
                    energy=energy - self.zero_point_energy,
                    zero_point_energy=self.zero_point_energy,
                    degeneracy=degeneracy,
                    spin_multiplicity_projection=get_level_spin_multiplicity_projection(
                        start=start_eigenfunction_index,
                        final=final_eigenfunction_index,
                        spin_projections=self.spin_projections,
                    ),
                    configurations=get_level_configuration_projection(
                        start=start_eigenfunction_index,
                        final=final_eigenfunction_index,
                        configuration_projection=self.configuration_projection,
                    ),
                )
            )

            level_index += 1
            start_eigenfunction_index = final_eigenfunction_index

        return electronic_levels


def get_configuration_projection(
    many_electron_eigenfunctions: ManyElectronEigenfunctions,
    ligand_field_eigenfunctions: ManyElectronEigenfunctions,
    ligand_field_orbital_energies: np.ndarray,
    orbital_occupations: dict[int, np.ndarray],
) -> ManyElectronEigenfunctions:
    """Calculate the configuration projection of the eigenfunctions."""
    configuration_mapping = setup_configuration_projection(
        ligand_field_eigenvalues=ligand_field_eigenfunctions.eigenvalues,
        ligand_field_eigenvectors=ligand_field_eigenfunctions.eigenvectors,
        ligand_field_spin_multiplicities=ligand_field_eigenfunctions.spin_projections.spin_multiplicities,
        ligand_field_spin_mapping=ligand_field_eigenfunctions.spin_projections.projection,
        ligand_field_orbital_energies=ligand_field_orbital_energies,
        orbital_occupations=orbital_occupations,
    )
    configuration_projection = map_configuration_projection(
        many_electron_eigenvectors=many_electron_eigenfunctions.eigenvectors,
        configuration_mapping=configuration_mapping,
    )
    return ManyElectronEigenfunctions(
        eigenvalues=many_electron_eigenfunctions.eigenvalues,
        eigenvectors=many_electron_eigenfunctions.eigenvectors,
        zero_point_energy=many_electron_eigenfunctions.zero_point_energy,
        spin_projections=many_electron_eigenfunctions.spin_projections,
        configuration_projection=configuration_projection,
    )


def get_many_electron_eigenfunctions(
    many_electron_basis: ManyElectronBasis,
    effective_hamiltonian: np.ndarray,
) -> ManyElectronEigenfunctions:
    """Return the many electron eigenfunctions for a full Hamiltonian.

    Args:
        many_electron_basis: The basis set of the Effective Hamiltonian.
        effective_hamiltonian: The N x N hermitian Effective Hamiltonian.

    Returns:
        ManyElectronEigenfunctions
    """
    solution = solve_matrix(effective_hamiltonian)
    eigenfunctions = ManyElectronEigenfunctions(
        eigenvalues=solution.values,
        eigenvectors=solution.vectors,
        zero_point_energy=solution.reference_value,
        spin_projections=get_spin_projections(
            spin_multiplicities=many_electron_basis.spin_multiplicities,
            spin_vectors=many_electron_basis.spin_vectors,
            eigenvectors=solution.vectors,
        ),
        # Initialise an empty Configuration Projection
        configuration_projection=ConfigurationProjection(
            configurations=[], mapping=np.zeros(solution.vectors.shape)
        ),
    )
    return eigenfunctions
