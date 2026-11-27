from pathlib import Path

import kestrelpy as kp

WORKING_DIRECTORY = Path(__file__).parent
XYZ_FILE = WORKING_DIRECTORY / "cucl4.xyz"


if __name__ == "__main__":
    from kestrelpy.comparison.data import EprData
    from kestrelpy.workflows.parameter_space import slice_epr_properties

    results = slice_epr_properties(
        molecule=kp.BuildMoleculeFromFile(
            file_path=str(XYZ_FILE),
            metal_index=0,
            ligands=[
                kp.BuildLigand(
                    name="Cl",
                    atoms=[
                        kp.AssignAtom(coordinating_atom=1),
                        kp.AssignAtom(coordinating_atom=2),
                        kp.AssignAtom(coordinating_atom=3),
                        kp.AssignAtom(coordinating_atom=4),
                    ],
                )
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=9,
            spin_orbit_coupling=kp.Fixed(830.0),
            kiso=kp.Varied([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        ),
        ligands=[
            kp.LigandAssignment(
                name="Cl",
                esigma=kp.Varied([6000.0, 6500.0, 7000.0, 7500.0, 8000.0]),
                epix=kp.Varied([900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0]),
                epiy=kp.Link.linear("ligand.Cl.epix", gradient=1.0, constant=0.0),
                eds=kp.Varied([1000.0, 1200.0, 1400.0]),
            ),
        ],
        epr_data=EprData(g_factors=[2.046, 2.046, 2.231], level=0),
        printing=kp.PrintingBlock(
            output_file=str(WORKING_DIRECTORY / "cucl4.out"),
            molecule_as_json=True,
        ),
    )
