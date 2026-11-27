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
from pathlib import Path
from textwrap import dedent
from typing import Callable

import kestrelpy as kp
from kestrelpy.models.input_models.input_models import GFactorType, HyperfineMode
from kestrelpy.models.results.calculations import Output

RESOURCES = Path(__file__).parent


@dataclass
class ElectronicStructureTestCase:
    id: str
    calculation: Callable[[], Output]
    expected_results_path: str


CUCL4_D9_SQUARE_PLANER = ElectronicStructureTestCase(
    id="CuCl4, d9, square planar, eds mixing",
    calculation=lambda: kp.run_calculation(
        molecule=kp.BuildMoleculeFromFile(
            metal_index=0,
            file_path=str(RESOURCES / "cucl4_d9_square_planar/cucl4.xyz"),
            ligands=[
                kp.BuildLigand(
                    name="Cl",
                    atoms=[
                        kp.AssignAtom(1),
                        kp.AssignAtom(2),
                        kp.AssignAtom(3),
                        kp.AssignAtom(4),
                    ],
                ),
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=9,
            spin_orbit_coupling=kp.Fixed(830.0 * 0.68),
            kiso=kp.Fixed(0.68),
        ),
        ligands=[
            kp.LigandAssignment(
                name="Cl",
                esigma=kp.Fixed(5533.0),
                epix=kp.Fixed(925.0),
                epiy=kp.Link.linear("ligand.Cl.epix"),
                eds=kp.Fixed(1658.0),
            ),
        ],
        calculation=kp.CalculationBlock(configuration_projection=True),
        epr=kp.EPRBlock(
            selection=kp.EprSelection(levels=[0]),
            degeneracy_tolerance=0.1,
        ),
    ),
    expected_results_path=str(RESOURCES / "cucl4_d9_square_planar/results.json"),
)


LSAA9_LPMO_D9 = ElectronicStructureTestCase(
    id="LsAA9, d9, copper(II) complex.",
    calculation=lambda: kp.run_calculation(
        molecule=kp.BuildMoleculeFromFile(
            metal_index=29,
            file_path=str(RESOURCES / "lsaa9_lpmo_d9/lsaa9.xyz"),
            ligands=[
                kp.BuildLigand(
                    name="His", atoms=[kp.AssignAtom(6, 4), kp.AssignAtom(15, 14)]
                ),
                kp.BuildLigand(name="NH2", atoms=[kp.AssignAtom(0)]),
                kp.BuildLigand(name="H2O", atoms=[kp.AssignAtom(65)]),
                kp.BuildLigand(name="Tyr", atoms=[kp.AssignAtom(28)]),
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=9,
            spin_orbit_coupling=kp.Fixed(830.0),
            kiso=kp.Fixed(0.7),
        ),
        ligands=[
            kp.LigandAssignment(
                name="His", esigma=kp.Fixed(6000.0), epiy=kp.Fixed(1300.0)
            ),
            kp.LigandAssignment(name="NH2", esigma=kp.Fixed(5500.0)),
            kp.LigandAssignment(name="H2O", esigma=kp.Link.linear("ligand.NH2.esigma")),
            kp.LigandAssignment(name="Tyr", esigma=kp.Fixed(-1800.0)),
        ],
        calculation=kp.CalculationBlock(configuration_projection=True),
        epr=kp.EPRBlock(
            selection=kp.EprSelection(levels=[0]),
            degeneracy_tolerance=0.1,
            hyperfine_mode=HyperfineMode.FERMI,
            hyperfine_parameters=kp.HyperfineAssignment(
                metal="63Cu", p=kp.Fixed(1168.0), kappa=kp.Fixed(0.20)
            ),
        ),
    ),
    expected_results_path=str(RESOURCES / "lsaa9_lpmo_d9/results.json"),
)


LINEAR_COBALT_II_D7 = ElectronicStructureTestCase(
    id="Linear d7 Cobalt(II) complex",
    calculation=lambda: kp.run_calculation(
        molecule=kp.BuildMoleculeFromContents(
            contents=dedent(
                """
                Co      0.0     0.0     0.0
                C       0.0     0.0     2.0
                C       0.0     0.0    -2.0
                """
            ),
            metal_index=0,
            ligands=[
                kp.BuildLigand(
                    name="CR3",
                    atoms=[
                        kp.AssignAtom(coordinating_atom=1),
                        kp.AssignAtom(coordinating_atom=2),
                    ],
                ),
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=7,
            racah_b=kp.Fixed(775.0),
            racah_c=kp.Link.linear("metal.racah_b", gradient=4.0, constant=0.0),
            spin_orbit_coupling=kp.Fixed(475.0),
            kiso=kp.Fixed(0.99),
        ),
        ligands=[
            kp.LigandAssignment(
                name="CR3",
                esigma=kp.Fixed(1950.0),
                epix=kp.Fixed(375.0),
                epiy=kp.Link.linear("ligand.CR3.epix", gradient=1.0, constant=0.0),
            ),
        ],
        calculation=kp.CalculationBlock(configuration_projection=True),
        susceptibilities=kp.SusceptibilityBlock(temperatures=[5.0, 100.0, 300.0]),
    ),
    expected_results_path=str(RESOURCES / "linear_cobalt_d7/results.json"),
)

CUPY4_D9 = ElectronicStructureTestCase(
    id="CuPy4, d9, copper(II) complex",
    calculation=lambda: kp.run_calculation(
        molecule=kp.BuildMoleculeFromFile(
            metal_index=0,
            file_path=str(RESOURCES / "cupy4_d9/cu_py4.xyz"),
            ligands=[
                kp.BuildLigand(
                    name="py",
                    atoms=[
                        kp.AssignAtom(coordinating_atom=21, orientation_atom=3),
                        kp.AssignAtom(coordinating_atom=22, orientation_atom=11),
                        kp.AssignAtom(coordinating_atom=43, orientation_atom=23),
                        kp.AssignAtom(coordinating_atom=44, orientation_atom=41),
                    ],
                )
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=9,
            spin_orbit_coupling=kp.Fixed(830.0),
            kiso=kp.Fixed(0.74),
        ),
        ligands=[
            kp.LigandAssignment(
                name="py", esigma=kp.Fixed(7400.0), epiy=kp.Fixed(1600.0)
            )
        ],
        calculation=kp.CalculationBlock(configuration_projection=False),
        epr=kp.EPRBlock(
            selection=kp.EprSelection(levels=[0]),
            hyperfine_mode=HyperfineMode.ALL,
            predict_p=True,
            predict_kappa=True,
            hyperfine_parameters=kp.HyperfineAssignment(metal="63Cu"),
        ),
    ),
    expected_results_path=str(RESOURCES / "cupy4_d9/results.json"),
)

COEN2_D7_SQUARE_PLANAR = ElectronicStructureTestCase(
    id="Co(en)2, d7, cobalt(II) complex",
    calculation=lambda: kp.run_calculation(
        molecule=kp.BuildMoleculeFromFile(
            metal_index=0,
            file_path=str(RESOURCES / "coen2_d7_square_planar/co_en2.xyz"),
            ligands=[
                kp.BuildLigand(
                    name="en",
                    atoms=[
                        kp.AssignAtom(coordinating_atom=1),
                        kp.AssignAtom(coordinating_atom=4),
                        kp.AssignAtom(coordinating_atom=13),
                        kp.AssignAtom(coordinating_atom=20),
                    ],
                )
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=7,
            racah_b=kp.Fixed(750.0),
            racah_c=kp.Fixed(3000.0),
            spin_orbit_coupling=kp.Fixed(500.0),
            kiso=kp.Fixed(0.7),
        ),
        ligands=[kp.LigandAssignment(name="en", esigma=kp.Fixed(5200.0))],
        calculation=kp.CalculationBlock(configuration_projection=False),
        epr=kp.EPRBlock(
            hyperfine_mode=HyperfineMode.ALL,
            predict_p=True,
            predict_kappa=True,
            g_type=GFactorType.EFFECTIVE,
            temperature=150.0,
            hyperfine_parameters=kp.HyperfineAssignment(metal="59Co"),
        ),
    ),
    expected_results_path=str(RESOURCES / "coen2_d7_square_planar/results.json"),
)

# All test cases must be added to this list and will be tested against
ELECTRONIC_STRUCTURE_TEST_CASES = [
    CUCL4_D9_SQUARE_PLANER,
    LSAA9_LPMO_D9,
    LINEAR_COBALT_II_D7,
    CUPY4_D9,
    COEN2_D7_SQUARE_PLANAR,
]
