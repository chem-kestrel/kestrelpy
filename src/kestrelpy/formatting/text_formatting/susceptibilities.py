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

import numpy as np
from tabulate import tabulate

from kestrelpy.calculations.magnetism.susceptibilities import (
    ParamagneticSusceptibilityTensors,
)


def format_paramagnetic_susceptibilities(
    tensors: ParamagneticSusceptibilityTensors,
) -> str:
    table_rows = []
    for temperature, tensor in zip(tensors.temperatures, tensors.tensors):
        values_cm3_K_mol_1 = tensor.values * temperature
        values_cgs = tensor.values * 1e6
        tensor_cgs = tensor.tensor[np.triu_indices(3)] * 1e6
        table_rows.append(
            [
                temperature,
                np.average(values_cm3_K_mol_1),
                *values_cm3_K_mol_1,
                *values_cgs,
                *tensor_cgs,
            ]
        )

    foz_start, foz_final = 0, tensors.metadata.foz_index - 1
    soz_start, soz_final = foz_final + 1, tensors.metadata.soz_index

    return "\n".join(
        [
            f"First-order Zeeman eigenfunctions:  {foz_start:3} -> {foz_final:3}",
            f"Second-order Zeeman eigenfunctions: {soz_start:3} -> {soz_final:3}\n",
            tabulate(
                tabular_data=table_rows,
                headers=[
                    "Temperature (K)",
                    "Average Chi*T\n(cm3 K mol-1)",
                    "Chi1*T\n(cm3 K mol-1)",
                    "Chi2*T\n(cm3 K mol-1)",
                    "Chi3*T\n(cm3 K mol-1)",
                    "Chi1\n(cgs 1e-6)",
                    "Chi2\n(cgs 1e-6)",
                    "Chi3\n(cgs 1e-6)",
                    "ChiXX\n(cgs 1e-6)",
                    "ChiXY\n(cgs 1e-6)",
                    "ChiXZ\n(cgs 1e-6)",
                    "ChiYY\n(cgs 1e-6)",
                    "ChiYZ\n(cgs 1e-6)",
                    "ChiZZ\n(cgs 1e-6)",
                ],
                floatfmt=(
                    ".1f",
                    ".3f",
                    ".3f",
                    ".3f",
                    ".3f",
                    ".1f",
                    ".1f",
                    ".1f",
                    ".1f",
                    ".1f",
                    ".1f",
                    ".1f",
                    ".1f",
                    ".1f",
                ),
            ),
        ],
    )
