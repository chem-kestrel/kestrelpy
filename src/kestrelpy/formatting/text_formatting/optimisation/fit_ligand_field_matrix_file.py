import numpy as np

from kestrelpy.comparison.ligand_field_matrix import get_centered_matrix
from kestrelpy.formatting.text_formatting.assignments import (
    format_assignment,
    format_assignments,
)
from kestrelpy.formatting.text_formatting.file_header import get_file_banner
from kestrelpy.formatting.text_formatting.ligand_field import format_ligand_field_matrix
from kestrelpy.formatting.text_formatting.molecule import format_molecule
from kestrelpy.formatting.text_formatting.optimisation.ligand_field_matrix_comparison import (
    format_ligand_field_matrix_comparison,
)
from kestrelpy.models.results.optimisation import LigandFieldOptimisationResult
from kestrelpy.parameters import LigandOptimisation

# =======================================================================================
# Text Templates
# =======================================================================================

CALCULATION_INPUT_TEMPLATE = """
=========================================================================================
                                     INPUT
=========================================================================================
{input_section}
"""

MOLECULE_SECTION_TEMPLATE = """
=========================================================================================
                                    MOLECULE
=========================================================================================
{molecule_section}
"""

BEST_FIT_PARAMETER_TEMPLATE = """
=========================================================================================
                        BEST-FIT PARAMETERS & LIGAND-FIELD
=========================================================================================

... Optimisation success?           {success:10}
... Best fit root-mean-square:      {rms:10.3f} cm-1

*******************
 Ligand Parameters
*******************

{ligand_parameters}

********************************
 Ligand-field matrix comparison
********************************

{ligand_field_matrix_comparison}

Calculated 5x5 one-electron ligand-field matrix:
{calculated_ligand_field_matrix}

Reference 5x5 one-electron ligand-field matrix:
{reference_ligand_field_matrix}
"""


# =======================================================================================
# Generating Output
# =======================================================================================


def format_input_section(
    ligand_field_matrix: np.ndarray, ligand_optimisations: list[LigandOptimisation]
) -> str:
    return "\n".join(
        [
            "Input ligand-field matrix to fit:",
            "---------------------------------",
            format_ligand_field_matrix(
                ligand_field_matrix=np.array(ligand_field_matrix)
            ),
            *[f"\n{format_assignment(lig)}" for lig in ligand_optimisations],
        ]
    )


def format_fit_ligand_field_matrix_file_contents(
    results: LigandFieldOptimisationResult,
) -> str:
    inp = results.input
    input_section = CALCULATION_INPUT_TEMPLATE.format(
        input_section=format_input_section(
            ligand_field_matrix=inp.ligand_field_matrix,
            ligand_optimisations=inp.ligand_optimisations,
        )
    )
    molecule_section = MOLECULE_SECTION_TEMPLATE.format(
        molecule_section=format_molecule(molecule=results.molecule),
    )
    best_fit_section = BEST_FIT_PARAMETER_TEMPLATE.format(
        success=str(results.result.success),
        rms=results.result.fun,
        ligand_parameters=format_assignments(results.best_fit_parameters),
        ligand_field_matrix_comparison=format_ligand_field_matrix_comparison(
            calculated_matrix=get_centered_matrix(results.best_fit_ligand_field_matrix),
            reference_matrix=get_centered_matrix(results.input.ligand_field_matrix),
        ),
        calculated_ligand_field_matrix=format_ligand_field_matrix(
            results.best_fit_ligand_field_matrix
        ),
        reference_ligand_field_matrix=format_ligand_field_matrix(
            results.input.ligand_field_matrix
        ),
    )

    return str(
        get_file_banner("OPTIMISATION")
        + input_section
        + molecule_section
        + best_fit_section
    )
