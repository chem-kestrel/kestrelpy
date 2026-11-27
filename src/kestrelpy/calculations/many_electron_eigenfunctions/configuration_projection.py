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

Module for calculating the electronic configuration projection.

A 'configuration' in this module refers to the electronic occupation of the five one-electron
ligand-field orbitals by the n d-electrons.

The methodology for this was developed as part of a PhD Thesis on an early version of this
software:
    >>> Nunn, George (2023) Development of a contemporary ligand-field program
        for 3d transition-metal complexes. PhD thesis, University of York.

The methodology is summarised as follows:
1.  Generate all the unique combinations that n-electrons can occupy the five
    d orbitals for each valid spin multiplicity.
2.  Diagonalise the 5x5 one-electron ligand field matrix to get the energies
    of the five ligand-field orbitals.
3.  Calculate the *relative* energy of each unique orbital occupation as a function
    of the energy of the five ligand-field orbitals.

    For example, if the five ligand-field orbitals had relative energies of 0, 100,
    2000, 3000, and 5000 cm-1, then an orbital occupation of 21000 for a d3 system
    would give an energy of (0 cm-1 * 2) + (100 cm-1 * 1) = 100 cm-1.

4.  Diagonalise the full many-electron Hamiltonian under the effects of the ligand
    field *ONLY*. The eigenfunctions will be known as ligand-field eigenfunctions.
5.  For each ligand-field eigenfunction, work out its energy relative to the
    lowest energy ligand-field eigenfunction of the same spin multiplicity and
    find the orbital occupations which have the same energy (there can be more
    than 1 configuration as ligand field orbitals can be degenerate, or as a
    coincidence of arithmetic).

    NOTE: there is no way using this methodology to distinguish between degenerate
    orbital occupations of the same spin multiplicity.
6.  Create a mapping matrix, whose rows are the orbital occupations (configurations)
    and whose columns are the ligand-field eigenfunctions (columns and rows should
    add up to 1.0)
7.  Work out how the ligand-field eigenfunctions map to the full Hamiltonian
    eigenfunctions (under the  effects of interelectronic repulsion and spin-orbit coupling).

    This is done using the inverse of the ligand-field eigenvectors (V_LF):
        (V_LF**-1) * V_full

8.  Apply the configuration mapping matrix to the ligand-field-to-full eigenvector mapping.
"""

from dataclasses import dataclass

import numpy as np


class ConfigurationProjectionError(Exception):
    """Raised if an issue occurred when calculating the configuration projection."""


@dataclass(frozen=True)
class Configuration:
    occupation: tuple[int, ...]
    contribution: float  # 0 to 1.
    spin_multiplicity: int

    def __repr__(self) -> str:
        return f"Configuration({self.get_string_representation()})"

    def get_string_representation(self) -> str:
        orbital_occupation: list[int] = [0, 0, 0, 0, 0]
        for occupation in self.occupation:
            orbital_occupation[occupation] += 1
        occupation_str: list[str] = [
            str(occupation) for occupation in orbital_occupation
        ]
        return f"{self.spin_multiplicity}: {''.join(occupation_str)} ({self.contribution:.3f})"


@dataclass
class ConfigurationMapping:
    inverse_vectors: np.ndarray
    configurations: list[Configuration]
    # Configuration mapping is a mapping matrix
    # whose rows represent configurations and whose
    # columns represent the ligand-field only eigenfunctions.
    configuration_mapping: np.ndarray


@dataclass
class ConfigurationProjection:
    configurations: list[Configuration]
    mapping: np.ndarray


def _get_spin_eigenfunction_indices(
    spin_multiplicities: list[int], spin_projections: np.ndarray, spin_multiplicity: int
) -> np.ndarray:
    """Return the eigenfunction indices which have the spin multiplicity projection given.

    This method just checks if any of the spin multiplicity projections are > 0 as this function
    assumes it is being called with eigenfunctions in the absence of spin-orbit-coupling. Hence,
    eigenfunctions should be pure spin multiplicities.
    """
    index = spin_multiplicities.index(spin_multiplicity)
    return np.argwhere(spin_projections[index, :] > 0.0).T[0]


def setup_configuration_projection(
    ligand_field_eigenvalues: np.ndarray,
    ligand_field_eigenvectors: np.ndarray,
    ligand_field_spin_multiplicities: list[int],
    ligand_field_spin_mapping: np.ndarray,
    ligand_field_orbital_energies: np.ndarray,
    orbital_occupations: dict[int, np.ndarray],
) -> ConfigurationMapping:
    """Generate the mapping between the ligand-field only eigenfunctions and the orbital configurations."""
    # Initialise the empty mapping matrix array
    number_of_configurations = sum(
        [len(occupations) for occupations in orbital_occupations.values()]
    )
    number_of_eigenfunctions = ligand_field_eigenvalues.size
    mapping_matrix: np.ndarray = np.zeros(
        (number_of_configurations, number_of_eigenfunctions), dtype=float
    )

    n0 = 0  # Crude index for creating a stack...
    for spin_mult, occupations in orbital_occupations.items():
        # Get the energies of the ligand-field only eigenfunctions which have the same
        # spin multiplicity.
        spin_indices = _get_spin_eigenfunction_indices(
            spin_multiplicities=ligand_field_spin_multiplicities,
            spin_projections=ligand_field_spin_mapping,
            spin_multiplicity=spin_mult,
        )
        energies = ligand_field_eigenvalues[spin_indices]
        energies -= min(energies)

        # Get the energies of the orbital occupations and normalise.
        occupational_energies = np.sum(
            ligand_field_orbital_energies[occupations], axis=1
        )
        occupational_energies -= min(occupational_energies)

        for energy, index in zip(energies, spin_indices):
            matching_occupations: np.ndarray = np.where(
                np.isclose(occupational_energies, energy)
            )[0]
            if matching_occupations.size == 0:
                raise ValueError(
                    "Unable to match any occupational configuration energies "
                    f"to the ligand-field eigenvalue ({energy=})"
                )
            matching_occupations += n0
            mapping_matrix[matching_occupations, index] = (
                1.0 / matching_occupations.size
            )

        n0 += len(occupations)

    inverse_vectors = np.linalg.inv(ligand_field_eigenvectors)
    return ConfigurationMapping(
        inverse_vectors=inverse_vectors,
        configurations=[
            Configuration(
                occupation=occupation,
                contribution=0.0,
                spin_multiplicity=spin_multiplicity,
            )
            for spin_multiplicity, occupations in orbital_occupations.items()
            for occupation in occupations
        ],
        configuration_mapping=mapping_matrix,
    )


def map_configuration_projection(
    many_electron_eigenvectors: np.ndarray,
    configuration_mapping: ConfigurationMapping,
) -> ConfigurationProjection:
    """Constructs the orbital occupation mapping to the full Hamiltonian eigenfunctions.

    This functions performs two steps:
        1.  Creates a contribution mapping of the ligand-field only eigenfunctions
            and the full Hamiltonian eigenfunctions.
        2.  Creates a contribution mapping of the orbital occupations (configurations)
            and the full Hamiltonian eigenfunctions.
    """
    # This gives the normalised contribution of the ligand-field only eigenfunctions
    # to the final full many-electron eigenfunctions.
    ligand_field_to_full_function_mapping = np.matmul(
        configuration_mapping.inverse_vectors, many_electron_eigenvectors
    )
    ligand_field_to_full_function_mapping *= (
        ligand_field_to_full_function_mapping.conjugate()
    )
    _validate_configuration_mapping(mapping=ligand_field_to_full_function_mapping)

    # Get the configuration --> full eigenvector mapping.
    mapping = np.matmul(
        configuration_mapping.configuration_mapping,
        ligand_field_to_full_function_mapping,
    )
    _validate_configuration_mapping(mapping=mapping)

    return ConfigurationProjection(
        configurations=configuration_mapping.configurations, mapping=mapping
    )


def _validate_configuration_mapping(mapping: np.ndarray) -> None:
    """Check that each column sums to 1, if it does not then something has gone wrong.

    Args:
        mapping: The mapping matrix.

    Raises:
        ConfigurationProjectionError if the rows and columns do not sum to 1.
    """
    columns_sum_to_one = np.all(np.isclose(np.sum(mapping, axis=0), 1))
    if not columns_sum_to_one:
        raise ConfigurationProjectionError(
            "Not every column sums to 1. Something has gone wrong."
        )


def get_eigenfunction_configuration_projection(
    index: int, configuration_projection: ConfigurationProjection
) -> list[Configuration]:
    """Return the contributing configurations for a single eigenfunction.

    Args:
        index: Eigenfunction index.
        configuration_projection: Class representing the configuration projection
            of the eigenfunctions.

    Returns:
        list of configurations whose contributions are >= 0.0005.
    """
    return sorted(
        [
            Configuration(
                occupation=config.occupation,
                contribution=float(projection),
                spin_multiplicity=config.spin_multiplicity,
            )
            for config, projection in zip(
                configuration_projection.configurations,
                configuration_projection.mapping[:, index],
            )
            if float(projection) >= 0.0005
        ],
        key=lambda x: x.contribution,
        reverse=True,
    )


def get_level_configuration_projection(
    start: int, final: int, configuration_projection: ConfigurationProjection
) -> list[Configuration]:
    """Return the contributing configurations for the degenerate eigenfunctions.

    Note: this function does not check that the eigenfunctions are degenerate etc.

    Args:
        start: The starting index of the eigenfunctions.
        final: The final index, which is equal to start + degeneracy of the level.
        configuration_projection: Class representing the configuration projection
            of the eigenfunctions.

    Returns:
        list of configurations whose contributions are >= 0.0005.
    """
    total_projections = np.sum(configuration_projection.mapping[:, start:final], axis=1)
    total_projections /= np.sum(total_projections)

    return sorted(
        [
            Configuration(
                occupation=config.occupation,
                contribution=float(projection),
                spin_multiplicity=config.spin_multiplicity,
            )
            for config, projection in zip(
                configuration_projection.configurations,
                total_projections,
            )
            if float(projection) >= 0.0005
        ],
        key=lambda x: x.contribution,
        reverse=True,
    )


def configuration_weighted_sum(
    level_configurations: list[Configuration], values: np.ndarray, tol: float = 1e-4
) -> tuple[np.ndarray, float]:
    """
    Helper function to sum values for each singly-occupied ligand-field orbital
    for a given level configuration. Values are then weighted by the
    contribution of the configuration.

    Args:
        level_configurations: List of electronic configurations
        values: Array (1x5) consisting of values per ligand-field orbital.
        tol: Cut-off below which configurations are excluded.

    Returns:
        tuple: configuration weighted sum (array) and maximum contribution (float)
    """
    max_contribution = 0.0
    orbital_totals = np.zeros(5, dtype=np.float64)

    for config in level_configurations:
        if config.contribution < tol:
            continue
        singly_occupied = np.bincount(config.occupation, minlength=5) == 1
        n = np.sum(singly_occupied)
        if n == 0:
            continue
        scale = (config.spin_multiplicity - 1) / n
        weight = config.contribution * scale * np.sum(values[singly_occupied])
        orbital_totals[singly_occupied] += weight / n

        if abs(weight) > abs(max_contribution):
            max_contribution = weight

    return orbital_totals, max_contribution
