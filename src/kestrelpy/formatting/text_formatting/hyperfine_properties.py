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

from textwrap import indent

from tabulate import tabulate

from kestrelpy.calculations.magnetism.g_factors import GSquaredTensor
from kestrelpy.calculations.magnetism.hyperfine_tensor.a_tensor_calculations import (
    ASquaredTensor,
)
from kestrelpy.calculations.magnetism.hyperfine_tensor.hyperfine_sign import (
    ASignResults,
)
from kestrelpy.models.results.calculations import EPRProperties
from kestrelpy.workflows.calculate_hyperfine_interaction import HyperfineResults

HYPERFINE_PROPERTY_FORMAT = """Metal-hyperfine values for Eigenfunctions: {first_index} -> {second_index}

{metal_hyperfine}"""


def format_a_values_table(a_tensor: ASquaredTensor) -> str:
    """Format table for anisotropic A values"""
    return tabulate(
        tabular_data=[[a_tensor.values[i], *a_tensor.vectors[:, i]] for i in range(3)],
        headers=[
            f"A value {a_tensor.units}",
            "X".center(7),
            "Y".center(7),
            "Z".center(7),
        ],
        floatfmt=(" .1f", " 5.4f", " 5.4f", " 5.4f"),
    )


def format_hyperfine_sign(hyperfine_sign: ASignResults) -> str:
    """Format table for each contribution, the A values and their signs"""
    assert hyperfine_sign is not None

    contributions_data = [
        ["Fermi", *hyperfine_sign.fermi.tolist()],
        ["Dipolar", *hyperfine_sign.dipolar.tolist()],
        ["Orbital", *hyperfine_sign.orbital.tolist()],
        ["-------"],
        ["Total", *hyperfine_sign.total.tolist()],
    ]
    contributions_table = tabulate(
        tabular_data=contributions_data,
        headers=[
            f"Contribution {hyperfine_sign.units}",
            "A1".center(7),
            "A2".center(7),
            "A3".center(7),
        ],
        floatfmt=(" .1f", "5.1f", "5.1f", "5.1f"),
    )

    orientation_table = tabulate(
        tabular_data=[
            ["X", *hyperfine_sign.total_vectors[0]],
            ["Y", *hyperfine_sign.total_vectors[1]],
            ["Z", *hyperfine_sign.total_vectors[2]],
        ],
        headers=[
            f"Orientation {hyperfine_sign.units}",
            f"{hyperfine_sign.total[0]:.1f}".center(7),
            f"{hyperfine_sign.total[1]:.1f}".center(7),
            f"{hyperfine_sign.total[2]:.1f}".center(7),
        ],
        floatfmt=(" .0f", " 5.4f", " 5.4f", " 5.4f"),
    )

    return f"""* Contributions to hyperfine coupling:

{contributions_table}

* Orientation of A tensor:

{orientation_table}"""


def format_hyperfine_table(
    metal_hyperfine: HyperfineResults, g_factors: GSquaredTensor
) -> str:
    """Determine which hyperfine output table to print"""
    if metal_hyperfine.sign is not None:
        return format_hyperfine_sign(hyperfine_sign=metal_hyperfine.sign)

    if metal_hyperfine.fermi_contact is not None:
        return f"""*Fermi contact interaction: {metal_hyperfine.fermi_contact[0]} {metal_hyperfine.units}"""

    assert metal_hyperfine.dipolar is not None
    assert metal_hyperfine.orbital is not None
    metal_hyperfine.dipolar.vectors = (
        g_factors.vectors @ metal_hyperfine.dipolar.vectors
    )
    dipolar_table = format_a_values_table(
        a_tensor=metal_hyperfine.dipolar,
    )
    orbital_table = format_a_values_table(a_tensor=metal_hyperfine.orbital)
    return f"""*Dipolar hyperfine interaction:

{dipolar_table}

*Orbital hyperfine interaction:

{orbital_table}"""


def format_hyperfine_properties(epr_properties: EPRProperties) -> str:
    return HYPERFINE_PROPERTY_FORMAT.format(
        first_index=epr_properties.g_factors.metadata.eigenfunction_start_index,
        second_index=epr_properties.g_factors.metadata.eigenfunction_final_index,
        metal_hyperfine=indent(
            (
                "Metal-Hyperfine was not calculated."
                if epr_properties.metal_hyperfine_coupling is None
                else format_hyperfine_table(
                    metal_hyperfine=epr_properties.metal_hyperfine_coupling,
                    g_factors=epr_properties.g_factors,
                )
            ),
            "        ",
        ),
    )
