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

from kestrelpy.models.input_models.input_models import (
    CalculationBlock,
    EPRBlock,
    IntensityBlock,
    PrintingBlock,
    SusceptibilityBlock,
)
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.models.results.calculations import Output
from kestrelpy.parameters import LigandAssignment, MetalAssignment
from kestrelpy.workflows.e2e_workflow import run_calculation as _run_calculation


def run_calculation(
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    metal: MetalAssignment,
    ligands: list[LigandAssignment],
    calculation: CalculationBlock = CalculationBlock(),
    intensities: IntensityBlock | None = None,
    epr: EPRBlock | None = None,
    susceptibilities: SusceptibilityBlock | None = None,
    printing: PrintingBlock | None = None,
) -> Output:
    return _run_calculation(
        molecule=molecule,
        metal=metal,
        ligands=ligands,
        calculation=calculation,
        intensities=intensities,
        epr=epr,
        susceptibilities=susceptibilities,
        printing=printing,
    )
