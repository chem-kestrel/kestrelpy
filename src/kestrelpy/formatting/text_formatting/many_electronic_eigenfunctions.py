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

from tabulate import tabulate

from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    ManyElectronEigenfunctions,
)


def format_many_electron_eigenfunctions(
    eigenfunctions: ManyElectronEigenfunctions,
) -> str:
    spin_multiplicities = eigenfunctions.spin_projections.spin_multiplicities
    data = []
    for index, level in enumerate(eigenfunctions.levels):
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
            "Configuration\n <2S+1>: <occupation> (<contribution>)",
        ],
        floatfmt=("", "", ".1f", ".3f", ".3f", ".3f"),
        tablefmt="grid",
    )
