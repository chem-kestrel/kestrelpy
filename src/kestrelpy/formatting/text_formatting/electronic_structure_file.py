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

from kestrelpy.calculations.intensities.magnetic_circular_dichroism import (
    MagneticCircularDichroism,
)
from kestrelpy.formatting.text_formatting.assignments import (
    format_assignment,
    format_assignments,
)
from kestrelpy.formatting.text_formatting.epr_properties import format_epr_properties
from kestrelpy.formatting.text_formatting.file_header import get_file_banner
from kestrelpy.formatting.text_formatting.hyperfine_properties import (
    format_hyperfine_properties,
)
from kestrelpy.formatting.text_formatting.intensities import (
    format_magnetic_circular_dichroism_table,
    format_rotary_strength_table,
    format_transition_section,
    get_average_mcd_intensities,
)
from kestrelpy.formatting.text_formatting.ligand_field import format_ligand_field
from kestrelpy.formatting.text_formatting.many_electronic_eigenfunctions import (
    format_many_electron_eigenfunctions,
)
from kestrelpy.formatting.text_formatting.molecule import format_molecule
from kestrelpy.formatting.text_formatting.susceptibilities import (
    format_paramagnetic_susceptibilities,
)
from kestrelpy.models.results.calculations import (
    CalculationResults,
    IntensityResults,
    Output,
)
from kestrelpy.parameters import DipoleAssignment, LigandAssignment, MetalAssignment

# =======================================================================================
# Text Templates
# =======================================================================================

INPUT_FILE_TEMPLATE = """
=========================================================================================
INPUT FILE: {input_file_path}
=========================================================================================
{input_file}
"""


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


PARAMETER_SECTION_TEMPLATE = """
=========================================================================================
                            *** PARAMETER RUN: {parameter_run} ***
=========================================================================================

******************
 Metal Parameters
******************

{metal_parameters}

*******************
 Ligand Parameters
*******************

{ligand_parameters}

***************************
 One-Electron Ligand Field
***************************

{ligand_field}

******************************
 Many-Electron Eigenfunctions
******************************

Many-Electron Eigenfunctions (Ligand Field + Interelectronic Repulsion + Spin-Orbit Coupling):
{electronic_structure}
"""

EPR_SECTION_TEMPLATE = """
***************************
 EPR Properties: G-Factors
***************************

{epr_properties}
"""

HYPERFINE_SECTION_TEMPLATE = """
*********************************
 EPR Properties: Metal-hyperfine
*********************************

{hyperfine_properties}
"""

MAGNETISM_SECTION_TEMPLATE = """
*******************************
 Paramagnetic Susceptibilities
*******************************

{paramagnetic_susceptibilities}
"""

INTENSITY_SECTION_TEMPLATE = """
-----------------------------------------------------------------------------------------
                        *** INTENSITY RUN: {intensity_run} ***
-----------------------------------------------------------------------------------------

*******************
 Dipole Parameters
*******************
{parameter_section}
"""

ABSORPTION_SECTION_TEMPLATE = """
************
 Absorption
************

{table}
"""
CIRCULAR_DICHROISM_SECTION_TEMPLATE = """
********************
 Circular Dichroism
********************

{table}
"""

MAGNETIC_CIRCULAR_DICHROISM_SECTION_TEMPLATE = """
*****************************
 Magnetic-Circular Dichroism
*****************************

NOTE: the d-d transition energies and intensities are averaged.
{tables}
"""

MAGNETIC_CIRCULAR_DICHROISM_Table_TEMPLATE = """
Magnetic circular dichroism transitions at {mag_field} T:
{table}
"""


# =======================================================================================
# Generating Output
# =======================================================================================


def format_input_section(
    input_metal: MetalAssignment,
    input_ligands: list[LigandAssignment],
    input_dipole: list[DipoleAssignment],
) -> str:
    return "\n".join(
        [
            format_assignment(input_metal),
            *[f"\n{format_assignment(lig)}" for lig in input_ligands],
            *[f"\n{format_assignment(lig)}" for lig in input_dipole],
        ]
    )


def format_mcd_table(mcd: MagneticCircularDichroism) -> str:
    return MAGNETIC_CIRCULAR_DICHROISM_Table_TEMPLATE.format(
        mag_field=mcd.magnetic_field,
        table=format_magnetic_circular_dichroism_table(
            get_average_mcd_intensities(magnetic_circular_dichroism=mcd)
        ),
    )


def format_intensity_results(intensity_run: int, intensity: IntensityResults) -> str:
    intensity_section = INTENSITY_SECTION_TEMPLATE.format(
        intensity_run=intensity_run,
        parameter_section="\n".join(
            [f"\n{format_assignment(lig)}" for lig in intensity.dipole_parameters]
        ),
    )

    absorption_section = (
        ABSORPTION_SECTION_TEMPLATE.format(
            table=format_transition_section(
                transitions=intensity.absorption.results.transitions
            )
        )
        if intensity.absorption
        else ""
    )

    circular_dichroism_section = (
        CIRCULAR_DICHROISM_SECTION_TEMPLATE.format(
            table=format_rotary_strength_table(
                transitions=intensity.cd.results.transitions
            )
        )
        if intensity.cd
        else ""
    )

    mcd_tables = (
        "\n".join(format_mcd_table(_mcd.results) for _mcd in intensity.mcd)
        if intensity.mcd
        else None
    )

    magnetic_circular_dichroism_section = (
        MAGNETIC_CIRCULAR_DICHROISM_SECTION_TEMPLATE.format(tables=mcd_tables)
        if mcd_tables
        else ""
    )

    return str(
        intensity_section
        + absorption_section
        + circular_dichroism_section
        + magnetic_circular_dichroism_section
    )


def format_parameter_section(index: int, result: CalculationResults) -> str:
    parameter_section = PARAMETER_SECTION_TEMPLATE.format(
        parameter_run=index + 1,
        metal_parameters=format_assignments([result.metal]),
        ligand_parameters=format_assignments(result.ligands),
        ligand_field=format_ligand_field(result.calculation.parameters.ligand_field),
        electronic_structure=format_many_electron_eigenfunctions(
            result.calculation.eigenfunctions
        ),
    )

    epr_section = (
        (
            EPR_SECTION_TEMPLATE.format(
                epr_properties="\n".join(
                    format_epr_properties(epr) + "\n" for epr in result.epr
                )
            )
        )
        if result.epr
        else ""
    )

    hyperfine_section = (
        (
            HYPERFINE_SECTION_TEMPLATE.format(
                hyperfine_properties="\n".join(
                    format_hyperfine_properties(epr) + "\n" for epr in result.epr
                )
            )
        )
        if (
            result.epr
            and len(result.epr) > index
            and result.epr[index].metal_hyperfine_coupling
        )
        else ""
    )

    magnetism_section = (
        MAGNETISM_SECTION_TEMPLATE.format(
            paramagnetic_susceptibilities=(
                format_paramagnetic_susceptibilities(result.susceptibilities)
            )
        )
        if result.susceptibilities
        else ""
    )

    intensity_sections = (
        "\n".join(
            [
                format_intensity_results(index, intensity)
                for index, intensity in enumerate(result.intensities)
            ]
        )
        if result.intensities
        else ""
    )

    return str(
        parameter_section
        + epr_section
        + hyperfine_section
        + magnetism_section
        + intensity_sections
    )


def format_electronic_structure_file_contents(electronic_structure: Output) -> str:
    inp = electronic_structure.input
    input_section = CALCULATION_INPUT_TEMPLATE.format(
        input_section=format_input_section(
            input_metal=inp.metal,
            input_ligands=inp.ligands,
            input_dipole=(
                electronic_structure.input.intensities.dipoles
                if electronic_structure.input.intensities
                else []
            ),
        )
    )
    molecule_section = MOLECULE_SECTION_TEMPLATE.format(
        molecule_section=format_molecule(molecule=electronic_structure.molecule),
    )

    parameter_sections = "".join(
        format_parameter_section(index, results)
        for index, results in enumerate(electronic_structure.results)
    )

    return str(
        get_file_banner("ELECTRONIC STRUCTURE")
        + input_section
        + molecule_section
        + parameter_sections
    )
