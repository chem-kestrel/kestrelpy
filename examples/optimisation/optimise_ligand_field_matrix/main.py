from pathlib import Path

import numpy as np

import kestrelpy as kp

WORKING_DIRECTORY = Path(__file__).parent
XYZ_FILE = WORKING_DIRECTORY / "cocl4.xyz"


def get_reference_ligand_field_matrix() -> np.ndarray:
    reference_matrix = (
        """
        -1208224.6	-174	402.1	247.6	-557
        -174	-1209204.5	-938	294.1	-873.5
        402.1	-938	-1209303.1	523.4	521
        247.6	294.1	523.4	-1207918.9	-338.6
        -557	-873.5	521	-338.6	-1209817.3
        """
    ).strip()

    return np.array(
        [line.split() for line in reference_matrix.splitlines()], dtype=float
    )


if __name__ == "__main__":
    # Run the optimisation varying esigma(Cl) and epi(Cl).
    # The final best-fit parameters are:
    #   - esigma(Cl) =~ 3337.6 cm-1
    #   - epix(Cl) =~ 1048.4 cm-1
    # The best-fit gave a root-mean-square of 58 cm-1.
    optimisation = kp.fit_ligand_field_matrix(
        # Get the ligand-field matrix from an ORCA ab-initio ligand-field matrix.
        ligand_field_matrix=get_reference_ligand_field_matrix(),
        # Build the molecule from the XYZ_FILE.
        molecule=kp.BuildMoleculeFromFile(
            file_path=str(XYZ_FILE),
            metal_index=4,
            ligands=[
                kp.BuildLigand(
                    "Cl",
                    atoms=[
                        kp.AssignAtom(coordinating_atom=0),
                        kp.AssignAtom(coordinating_atom=1),
                        kp.AssignAtom(coordinating_atom=2),
                        kp.AssignAtom(coordinating_atom=3),
                    ],
                )
            ],
        ),
        # We wish to optimise both esigma and epi.
        # To do so we bound esigma and epix from 0 cm-1 to 10,000 cm-1, with a
        # starting value of 0 cm-1.
        # NOTE: it is important to link epiy to epix, as we are assuming
        # that chloride is a 'linear ligator'.
        ligand_optimisations=[
            kp.LigandOptimisation(
                name="Cl",
                esigma=kp.Bound(0.0, 10000.0, 0.0),
                epix=kp.Bound(0.0, 10000.0, 0.0),
                epiy=kp.Link.linear("ligand.Cl.epix", gradient=1.0, constant=0.0),
            ),
        ],
        # Here we can specify further instructions for optimising.
        # For now, the default settings will suffice.
        optimisation_instructions=kp.OptimisationBlock(),
        # Write the output to `optimise_ligand_field.out`
        printing=kp.PrintingBlock(
            output_file=str(WORKING_DIRECTORY / "optimise_ligand_field.out"),
            molecule_as_json=True,
        ),
    )
