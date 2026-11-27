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


def test_run_calculation(
    tmp_path: Path, molecule_instructions: kp.BuildMoleculeFromContents
) -> None:
    output_file = tmp_path / "test.out"
    output = kp.run_calculation(
        molecule=molecule_instructions,
        metal=kp.MetalAssignment(d_electrons=9, racah_b=kp.Fixed(1000.0)),
        ligands=[kp.LigandAssignment(name="ligand", esigma=kp.Fixed(2000.0))],
        calculation=kp.CalculationBlock(configuration_projection=True),
        intensities=kp.IntensityBlock(
            dipoles=[
                kp.DipoleAssignment(name="ligand", psigma=kp.Varied([50.0, 100.0]))
            ],
            absorption=kp.AbsorptionOptions(temperatures=[300.0]),
            cd=kp.CDOptions(temperatures=[300.0]),
            mcd=kp.MCDOptions(magnetic_fields=[7.0, 5.0], temperatures=[5.0, 10.0]),
        ),
        epr=kp.EPRBlock(selection=kp.EprSelection(levels=[0], manifolds=[(0, 3)])),
        susceptibilities=kp.SusceptibilityBlock(temperatures=[5.0, 100.0, 300.0]),
        printing=kp.PrintingBlock(
            output_file=str(output_file),
            write_spectra=True,
            molecule_as_json=True,
        ),
    )

    assert len(output.results) == 1

    results = output.results[0]

    assert results.intensities is not None
    assert len(results.intensities) == 2
    for intensities in results.intensities:
        assert intensities.absorption is not None
        assert intensities.cd is not None
        assert len(intensities.mcd) == 2

    assert len(results.epr) == 2

    assert results.susceptibilities is not None
    assert len(results.susceptibilities.tensors) == 3

    assert os.path.isfile(output_file)
    assert os.path.isfile(tmp_path / "test_molecule.json")
    assert os.path.isfile(tmp_path / "test_run001_intensity001_spectra.json")
    assert os.path.isfile(tmp_path / "test_run001_intensity002_spectra.json")
