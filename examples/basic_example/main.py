from pathlib import Path

import kestrelpy as kp

WORKING_DIRECTORY = Path(__file__).parent
XYZ_FILE = str(WORKING_DIRECTORY / "cocl4.xyz")


calculation = kp.run_calculation(
    # Build the molecule from the xyz file.
    # The cobalt metal ion is the first atom so it has the index of 0.
    # The chloride anions have the indices 1, 2, 3, and 4.
    molecule=kp.BuildMoleculeFromFile(
        file_path=XYZ_FILE,
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
    # Next we want to define the metal.
    # The metal is a Co(II) which is a d7 system. To work out the number
    # of d electrons use `d_electrons = group_number - oxidation_state`:
    #   - group_number = 9
    #   - oxidation_state = +2
    #   - d_electrons = 9 - (+2) = 7
    # Spin-orbit coupling is treated by the authors in this way:
    #   - lambda = chi / 2S
    #   - (S=1.5) * 2 * lambda * kiso
    metal=kp.MetalAssignment(
        d_electrons=7,
        racah_b=kp.Fixed(740.0),
        racah_c=kp.Link.linear("metal.racah_b", gradient=4.0),
        spin_orbit_coupling=kp.Fixed(3.0 * 172.0 * 0.92),
        kiso=kp.Fixed(0.92),
    ),
    # The authors use a minimal parameterisation where they can recreate
    # the results just using an esigma parameter.
    ligands=[kp.LigandAssignment(name="Cl", esigma=kp.Fixed(2300.0))],
    # Turn off configuration projection.
    calculation=kp.CalculationBlock(configuration_projection=False),
    # The authors examine the first and second Kramer's doublet
    # chosen by `levels=[0, 1]` (we could do `manifolds=[(0, 1), (2, 3)]`).
    # Also, get the "actual" g-factors of the quartet ground-state by including
    # the span eigenfunction 0 --> 3 which includes 0, 1, 2, 3.
    epr=kp.EPRBlock(selection=kp.EprSelection(levels=[0, 1], manifolds=[(0, 3)])),
    susceptibilities=kp.SusceptibilityBlock(temperatures=[300.0]),
    # Write the output to `cocl4.out`
    printing=kp.PrintingBlock(
        output_file=str(WORKING_DIRECTORY / "cocl4.out"),
        molecule_as_json=True,
    ),
)
