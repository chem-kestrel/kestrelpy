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
from kestrelpy.models.results.calculations import EPRProperties

EPR_PROPERTY_FORMAT = """G-factors for Eigenfunctions: {first_index} -> {second_index}

{g_factors}"""


def format_g_squared_tensor(g_squared_tensor: GSquaredTensor) -> str:
    """Format table for g factors"""
    return tabulate(
        tabular_data=[
            [g_squared_tensor.values[i], *g_squared_tensor.vectors[:, i]]
            for i in range(3)
        ],
        headers=["g-factors", "x", "y", "z"],
        floatfmt=(".4f", ".3f", ".3f", ".3f"),
        tablefmt="outline",
    )


def format_epr_properties(epr_properties: EPRProperties) -> str:
    return EPR_PROPERTY_FORMAT.format(
        first_index=epr_properties.g_factors.metadata.eigenfunction_start_index,
        second_index=epr_properties.g_factors.metadata.eigenfunction_final_index,
        g_factors=indent(
            format_g_squared_tensor(g_squared_tensor=epr_properties.g_factors),
            "        ",
        ),
    )
