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

from collections import defaultdict
from dataclasses import fields
from typing import Any

import numpy as np

from kestrelpy.comparison.ligand_field_matrix import (
    compare_ligand_field_matrices,
    get_centered_matrix,
)
from kestrelpy.models.input_models.input_models import PrintingBlock
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.models.input_models.optimisation import OptimisationBlock
from kestrelpy.models.results.optimisation import (
    LigandFieldOptimisationInput,
    LigandFieldOptimisationResult,
    OptimisationResult,
)
from kestrelpy.molecule import Molecule
from kestrelpy.output import (
    write_fit_ligand_field_matrix_results,
    write_molecule_to_file,
)
from kestrelpy.parameters import (
    Bound,
    Fixed,
    LigandOptimisation,
    LigandParameters,
    Link,
)
from kestrelpy.workflows.get_molecule import get_molecule
from kestrelpy.workflows.ligand_field_matrices import get_ligand_field_matrices
from kestrelpy.workflows.optimisation.perform_optimisation import optimise_system
from kestrelpy.workflows.resolve_parameters import key_parser, resolve_linked_parameter


def _fit_ligand_field_matrices(
    fit_values: list[float],
    bounds: list[str],
    fixed: dict[str, float],
    linked: dict[str, Link],
    reserved: dict[str, Any],
    molecule: Molecule,
    reference_ligand_field_matrix: np.ndarray,
) -> float:
    ligand_field_parameters = _get_ligand_field_parameters(
        fit_values, bounds, fixed, linked, reserved
    )

    vlft, ds_vlft = get_ligand_field_matrices(
        molecule=molecule, ligand_field_parameters=ligand_field_parameters
    )
    return compare_ligand_field_matrices(
        calculated_matrix=get_centered_matrix(vlft - ds_vlft),
        reference_matrix=get_centered_matrix(reference_ligand_field_matrix),
    )


def _get_ligand_field_parameters(
    fit_values: list[float],
    bounds: list[str],
    fixed: dict[str, float],
    linked: dict[str, Link],
    reserved: dict[str, Any],
) -> list[LigandParameters]:
    parameter_mapping = {**fixed, **reserved}
    for key, value in zip(bounds, fit_values, strict=True):
        parameter_mapping[key] = value

    for key, link in linked.items():
        parameter_mapping[key] = link.apply_link(
            resolve_linked_parameter(
                key=link.object_reference,
                fixed_values=parameter_mapping,
                linked_values=linked,
            )
        )

    ligand_parameters: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, value in parameter_mapping.items():
        parsed_key = key_parser(key=key)
        if parsed_key.obj == "ligand":
            if not parsed_key.child:
                raise ValueError(
                    f"ligand name was not found in key: {key=}. "
                    "This should not happen!"
                )
            ligand_parameters[parsed_key.child][parsed_key.parameter] = value
        else:
            raise ValueError(f"Unrecognised object key: {key=}")
    ligand_field_parameters = [
        LigandParameters(**values) for values in ligand_parameters.values()
    ]

    return ligand_field_parameters


def _get_bounds(optimisations: list[LigandOptimisation]) -> list[tuple[float, float]]:
    bounds = []
    for optimisation in optimisations:
        for field in fields(optimisation):
            value = getattr(optimisation, field.name)
            if isinstance(value, Bound):
                bounds.append((value.lower_bound, value.upper_bound))
    return bounds


def _get_starting_values(optimisations: list[LigandOptimisation]) -> list[float]:
    starting_values = []
    for optimisation in optimisations:
        for field in fields(optimisation):
            value = getattr(optimisation, field.name)
            if isinstance(value, Bound):
                starting_values.append(value.starting_value)
    return starting_values


def _get_fitting_arguments(
    optimisations: list[LigandOptimisation],
) -> tuple[list[str], dict[str, float], dict[str, Link], dict[str, Any]]:
    bounds: list[str] = []
    fixed: dict[str, float] = {}
    linked: dict[str, Link] = {}
    reserved: dict[str, Any] = {}
    for optimisation in optimisations:
        for field in fields(optimisation):
            value = getattr(optimisation, field.name)
            key = f"ligand.{optimisation.name}.{field.name}"
            if isinstance(value, Fixed):
                fixed[key] = value.value
            elif isinstance(value, Link):
                linked[key] = value
            elif isinstance(value, Bound):
                bounds.append(key)
            else:
                reserved[key] = value
    return bounds, fixed, linked, reserved


def fit_ligand_field_matrix(
    ligand_field_matrix: np.ndarray,
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    ligand_assignments: list[LigandOptimisation],
    optimisation_instructions: OptimisationBlock,
    printing: PrintingBlock | None = None,
) -> LigandFieldOptimisationResult:
    _molecule = get_molecule(molecule)
    result = optimise_system(
        func=_fit_ligand_field_matrices,
        args=(
            *_get_fitting_arguments(optimisations=ligand_assignments),
            _molecule,
            ligand_field_matrix,
        ),
        instructions=optimisation_instructions,
        starting_values=_get_starting_values(optimisations=ligand_assignments),
        bounds=_get_bounds(optimisations=ligand_assignments),
    )

    # Get the final best-fit ligand-field matrix.
    ligand_field_parameters = _get_ligand_field_parameters(
        result.x, *_get_fitting_arguments(optimisations=ligand_assignments)
    )
    vlft, ds_vlft = get_ligand_field_matrices(
        molecule=_molecule, ligand_field_parameters=ligand_field_parameters
    )
    best_fit_ligand_field_matrix = vlft - ds_vlft

    results = LigandFieldOptimisationResult(
        input=LigandFieldOptimisationInput(
            ligand_field_matrix=ligand_field_matrix,
            molecule=molecule,
            ligand_optimisations=ligand_assignments,
            optimisation_instructions=optimisation_instructions,
        ),
        molecule=_molecule,
        best_fit_parameters=_get_ligand_field_parameters(
            result.x,
            *_get_fitting_arguments(optimisations=ligand_assignments),
        ),
        best_fit_ligand_field_matrix=best_fit_ligand_field_matrix,
        result=OptimisationResult.from_scipy(result=result),
    )

    if printing:
        write_fit_ligand_field_matrix_results(
            file_path=printing.output_file, results=results
        )
    if printing and printing.molecule_as_json:
        write_molecule_to_file(file_path=printing.output_file, molecule=_molecule)

    return results
