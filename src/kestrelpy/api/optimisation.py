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

from kestrelpy.models.input_models.input_models import PrintingBlock
from kestrelpy.models.input_models.molecule import (
    BuildMoleculeFromContents,
    BuildMoleculeFromFile,
    MoleculeFromFile,
)
from kestrelpy.models.input_models.optimisation import OptimisationBlock
from kestrelpy.models.results.optimisation import LigandFieldOptimisationResult
from kestrelpy.parameters import LigandOptimisation
from kestrelpy.workflows.optimisation.fit_ligand_field_matrix import (
    fit_ligand_field_matrix as _fit_ligand_field_matrix,
)


def fit_ligand_field_matrix(
    ligand_field_matrix: np.ndarray,
    molecule: BuildMoleculeFromContents | BuildMoleculeFromFile | MoleculeFromFile,
    ligand_optimisations: list[LigandOptimisation],
    optimisation_instructions: OptimisationBlock = OptimisationBlock(),
    printing: PrintingBlock | None = None,
) -> LigandFieldOptimisationResult:
    """Fit a 5x5 ligand-field matrix using the ligand geometry and AOM e-values.

    Args:
        ligand_field_matrix: 5x5 ligand-field matrix (float).
        molecule: Instructions to read or build a molecule.
        ligand_optimisations: The ligand parameter values, links, and bounds.
        optimisation_instructions: OptimisationBlock which flags which minimizer
            to use and to pass any additional optimisaiton flags.
        printing: Instructions for writing to file and what to print.

    Returns:
        LigandFieldOptimisationResult object.
    """
    return _fit_ligand_field_matrix(
        ligand_field_matrix=ligand_field_matrix,
        molecule=molecule,
        ligand_assignments=ligand_optimisations,
        optimisation_instructions=optimisation_instructions,
        printing=printing,
    )
