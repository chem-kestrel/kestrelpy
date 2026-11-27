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

from dataclasses import fields
from typing import Any

from tabulate import tabulate

from kestrelpy.parameters import (
    DipoleAssignment,
    DipoleParameters,
    LigandAssignment,
    LigandOptimisation,
    LigandParameters,
    MetalAssignment,
    MetalParameters,
)


def format_assignments(
    assignments: list[DipoleAssignment]
    | list[DipoleParameters]
    | list[LigandAssignment]
    | list[LigandParameters]
    | list[LigandOptimisation]
    | list[MetalAssignment]
    | list[MetalParameters],
) -> str:
    rows: list[Any] = []
    for f in fields(assignments[0]):
        rows.append([f.name, *[getattr(a, f.name) for a in assignments]])
    return tabulate(
        tabular_data=rows,
        headers=["Parameters", *[str(i) for i in range(len(assignments))]],
    )


def format_assignment(
    assignment: DipoleAssignment
    | DipoleParameters
    | LigandAssignment
    | LigandParameters
    | LigandOptimisation
    | MetalAssignment
    | MetalParameters,
) -> str:
    rows: list[tuple[str, str]] = []
    for f in fields(assignment):
        value = getattr(assignment, f.name)
        rows.append((f.name, value))
    return "\n".join(
        [
            f"{assignment.__class__.__name__}:",
            "-" * (len(assignment.__class__.__name__) + 1),
            tabulate(tabular_data=rows, headers=["Field", "Value"]),
        ]
    )
