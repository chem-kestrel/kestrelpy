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

import kestrelpy as kp
from kestrelpy.models.results.parameter_space import (
    EprParameterSpaceInput,
    NonRelativisticEnergiesParameterSpaceInput,
)


def test_slice_non_relativistic_energies(
    tmp_path: Path, molecule_instructions: kp.BuildMoleculeFromContents
) -> None:
    output_file = tmp_path / "test.out"
    parameter_space = kp.slice_non_relativistic_energies(
        molecule=molecule_instructions,
        metal=kp.MetalAssignment(d_electrons=9, racah_b=kp.Varied([0.0, 1000.0])),
        ligands=[
            kp.LigandAssignment(name="ligand", esigma=kp.Varied([2000.0, 3000.0]))
        ],
        spin_multiplicity_assignments=[
            kp.define_spin_multiplicity_assignments(
                spin_multiplicity=2, energies=[2000.0], is_ordered=False
            )
        ],
        printing=kp.PrintingBlock(
            output_file=str(output_file),
            molecule_as_json=True,
        ),
    )

    assert isinstance(parameter_space.input, NonRelativisticEnergiesParameterSpaceInput)
    assert parameter_space.data.varied_values == {
        "metal.racah_b": [0.0, 1000.0],
        "ligand.ligand.esigma": [2000.0, 3000.0],
    }
    assert len(parameter_space.data.rows) == 4  # 2x2 parameter space

    assert os.path.isfile(output_file)
    assert os.path.isfile(tmp_path / "test_molecule.json")
    assert os.path.isfile(tmp_path / "test.csv")

    assert parameter_space.data == kp.analyse_parameter_space_from_file(
        file_path=tmp_path / "test.csv"
    )


def test_slice_epr_data(
    tmp_path: Path, molecule_instructions: kp.BuildMoleculeFromContents
) -> None:
    output_file = tmp_path / "test.out"
    parameter_space = kp.slice_epr_data(
        molecule=molecule_instructions,
        metal=kp.MetalAssignment(d_electrons=9, racah_b=kp.Varied([0.0, 1000.0])),
        ligands=[
            kp.LigandAssignment(name="ligand", esigma=kp.Varied([2000.0, 3000.0]))
        ],
        epr_data=kp.EprData(g_factors=[2.0, 2.0, 2.0]),
        printing=kp.PrintingBlock(
            output_file=str(output_file),
            molecule_as_json=True,
        ),
    )

    assert isinstance(parameter_space.input, EprParameterSpaceInput)
    assert parameter_space.data.varied_values == {
        "metal.racah_b": [0.0, 1000.0],
        "ligand.ligand.esigma": [2000.0, 3000.0],
    }
    assert len(parameter_space.data.rows) == 4  # 2x2 parameter space

    assert os.path.isfile(output_file)
    assert os.path.isfile(tmp_path / "test_molecule.json")
    assert os.path.isfile(tmp_path / "test.csv")

    assert parameter_space.data == kp.analyse_parameter_space_from_file(
        file_path=tmp_path / "test.csv"
    )
