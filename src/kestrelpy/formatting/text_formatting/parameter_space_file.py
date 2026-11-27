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

from kestrelpy.formatting.text_formatting.assignments import format_assignment
from kestrelpy.formatting.text_formatting.comparison.data import (
    format_epr_data,
    format_spin_multiplicty_energies,
)
from kestrelpy.formatting.text_formatting.file_header import get_file_banner
from kestrelpy.formatting.text_formatting.molecule import format_molecule
from kestrelpy.models.results.parameter_space import (
    EprParameterSpaceInput,
    NonRelativisticEnergiesParameterSpaceInput,
    ParameterSpaceInput,
    ParameterSpaceResults,
)

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

DATA_SECTION_TEMPLATE = """
=========================================================================================
                        COMPARISON DATA FOR PARAMETER SPACE
=========================================================================================

{data_section}
"""


# =======================================================================================
# Generating Output
# =======================================================================================


def format_input_section(
    input: ParameterSpaceInput,
) -> str:
    return "\n".join(
        [
            format_assignment(input.metal),
            *[f"\n{format_assignment(lig)}" for lig in input.ligands],
        ],
    )


def format_data_section(input: ParameterSpaceInput) -> str:
    if isinstance(input, NonRelativisticEnergiesParameterSpaceInput):
        return format_spin_multiplicty_energies(
            energies=input.spin_multiplicity_assignments
        )
    elif isinstance(input, EprParameterSpaceInput):
        return format_epr_data(epr_data=input.epr_data)
    raise TypeError(f"Unrecognised input data type: {type(input)}")


def format_parameter_space_results(
    results: ParameterSpaceResults,
) -> str:
    input_section = CALCULATION_INPUT_TEMPLATE.format(
        input_section=format_input_section(input=results.input)
    )

    molecule_section = MOLECULE_SECTION_TEMPLATE.format(
        molecule_section=format_molecule(molecule=results.molecule)
    )

    data_section = DATA_SECTION_TEMPLATE.format(
        data_section=format_data_section(input=results.input)
    )

    return str(
        get_file_banner("PARAMETER SPACE")
        + input_section
        + molecule_section
        + data_section
    )
