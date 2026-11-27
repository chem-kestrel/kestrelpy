from pathlib import Path
from textwrap import dedent

import numpy as np

import kestrelpy as kp

WORKING_DIRECTORY = Path(__file__).parent

if __name__ == "__main__":
    results = kp.slice_non_relativistic_energies(
        molecule=kp.BuildMoleculeFromContents(
            contents=dedent(
                """
                Cl      -1.230261000      0.394044000     -1.916095000
                Cl       0.883919000      1.917680000      0.857057000
                Cl       1.589922000     -1.461247000     -0.611403000
                Cl      -1.421183000     -0.824209000      1.623496000
                Co       0.000000000      0.000000000      0.000000000
                """
            ),
            metal_index=4,
            ligands=[
                kp.BuildLigand(
                    name="Cl",
                    atoms=[
                        kp.AssignAtom(coordinating_atom=0),
                        kp.AssignAtom(coordinating_atom=1),
                        kp.AssignAtom(coordinating_atom=2),
                        kp.AssignAtom(coordinating_atom=3),
                    ],
                )
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=7,
            # racah_b=kp.Fixed(980.0),
            racah_b=kp.Varied([900.0, 1000.0, 1100.0]),
            racah_c=kp.Fixed(4223.0),
        ),
        ligands=[
            kp.LigandAssignment(
                name="Cl",
                esigma=kp.Varied(
                    np.linspace(
                        5000.0,
                        7500.0,
                        11,
                        endpoint=True,
                    ).tolist()
                ),
                epix=kp.Varied(
                    np.linspace(
                        1500.0,
                        3000.0,
                        11,
                        endpoint=True,
                    ).tolist()
                ),
                epiy=kp.Link.linear("ligand.Cl.epix"),
            ),
        ],
        spin_multiplicity_assignments=[
            kp.define_spin_multiplicity_assignments(
                spin_multiplicity=4,
                energies=[
                    0.0,
                    2977.4,
                    3460.8,
                    3740.6,
                    5736.2,
                    5942.7,
                    6144.4,
                    18451.6,
                    18944.1,
                    19243.0,
                ],
                expand_by_spin_mult=True,
                is_ordered=True,
            ),
            kp.define_spin_multiplicity_assignments(
                spin_multiplicity=2,
                energies=[
                    18397.1,
                    18536.5,
                    18756.0,
                    18802.8,
                    18855.5,
                ],
                expand_by_spin_mult=True,
                is_ordered=True,
            ),
        ],
        printing=kp.PrintingBlock(
            output_file=str(WORKING_DIRECTORY / "cocl4.out"),
            molecule_as_json=True,
        ),
    )
