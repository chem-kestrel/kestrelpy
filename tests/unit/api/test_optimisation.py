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

import os.path
from pathlib import Path

import numpy as np

import kestrelpy as kp
from kestrelpy.models.results.optimisation import LigandFieldOptimisationResult


def test_fit_ligand_field_matrix(
    tmp_path: Path, molecule_instructions: kp.BuildMoleculeFromContents
) -> None:
    output_file = tmp_path / "test.out"
    output = kp.fit_ligand_field_matrix(
        ligand_field_matrix=np.diag([0.0, 0.0, 4000.0, 0.0, 0.0]),
        molecule=molecule_instructions,
        ligand_optimisations=[
            kp.LigandOptimisation(
                name="ligand",
                esigma=kp.Bound(
                    lower_bound=2000.0,
                    upper_bound=6000.0,
                    starting_value=2000.0,
                ),
            )
        ],
        optimisation_instructions=kp.OptimisationBlock(
            method=kp.OptimisationMethod.MINIMIZE,
            optimisation_options={},
        ),
        printing=kp.PrintingBlock(
            output_file=str(output_file),
            molecule_as_json=True,
        ),
    )

    assert isinstance(output, LigandFieldOptimisationResult)
    assert np.allclose(
        output.best_fit_ligand_field_matrix,
        np.diag([0.0, 0.0, 4000.0, 0.0, 0.0]),
    )
    assert output.best_fit_parameters[0].name == "ligand"
    assert abs(output.best_fit_parameters[0].esigma - 4000.0) < 1.0e-01

    assert os.path.isfile(output_file)
    assert os.path.isfile(tmp_path / "test_molecule.json")
