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

from dataclasses import dataclass

import numpy as np

from kestrelpy.comparison.data import EprData, SpinMultiplicityEnergies
from kestrelpy.models.input_models.input_models import PrintingBlock
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.molecule import Molecule
from kestrelpy.parameters import LigandAssignment, MetalAssignment


@dataclass(frozen=True)
class ParameterSpaceInput:
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile
    metal: MetalAssignment
    ligands: list[LigandAssignment]
    printing: PrintingBlock | None


@dataclass(frozen=True)
class NonRelativisticEnergiesParameterSpaceInput(ParameterSpaceInput):
    spin_multiplicity_assignments: list[SpinMultiplicityEnergies]


@dataclass(frozen=True)
class EprParameterSpaceInput(ParameterSpaceInput):
    epr_data: EprData


@dataclass(frozen=True)
class ParameterSpaceData:
    varied_values: dict[str, list[float]]
    rows: list[dict[str, float]]

    def filter(self, rms: float) -> list[dict[str, float]]:
        filtered_rows: list[dict[str, float]] = []
        for row in self.rows:
            if row["rms"] < rms:
                filtered_rows.append(row)
        return filtered_rows

    def get_2d_slice(
        self, p1: str, p2: str, fixed_values: dict[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not fixed_values:
            fixed_values = {}

        for p in (p1, p2, *list(fixed_values.keys())):
            if p not in self.varied_values:
                raise ValueError(
                    f"'{p}' not found in available parameter labels. Please "
                    f"choose one of {self.varied_values}"
                )

        for p, v in fixed_values.items():
            if v not in self.varied_values[p]:
                raise ValueError(
                    f"Fixed value '{v}' for parameter '{p}' is not an available "
                    f"fixed value. Please choose one of: {self.varied_values[p]}"
                )

        for p in self.varied_values.keys():
            if p not in (p1, p2, *list(fixed_values.keys())):
                raise ValueError(
                    f"'{p}' not found in either the 'p1', 'p2', or the fixed values. "
                    "Please make sure you account for each parameter: "
                    f"{list(self.varied_values.keys())}"
                )

        x = []
        y = []
        z = []
        for row in self.rows:
            # Make sure the row matches the fixed values
            if any(row.get(k) != v for k, v in fixed_values.items()):
                continue
            x.append(row[p1])
            y.append(row[p2])
            z.append(row["rms"])

        x_unique = np.unique(x)
        y_unique = np.unique(y)

        xg, yg = np.meshgrid(x_unique, y_unique)
        zg = np.array(z).reshape(len(x_unique), len(y_unique))
        return xg, yg, zg.T


@dataclass(frozen=True)
class ParameterSpaceResults:
    input: ParameterSpaceInput
    molecule: Molecule
    data: ParameterSpaceData
