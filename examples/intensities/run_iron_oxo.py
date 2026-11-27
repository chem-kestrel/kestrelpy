from pathlib import Path

import kestrelpy as kp

WORKING_DIRECTORY = Path(__file__).parent
XYZ_FILE = str(WORKING_DIRECTORY / "wusjoj_bp86_def2tzvp_acetonitrile.xyz")

SCALE_DIPOLE = 1.0e-4


if __name__ == "__main__":
    calculation = kp.run_calculation(
        molecule=kp.BuildMoleculeFromFile(
            file_path=XYZ_FILE,
            metal_index=0,
            ligands=[
                kp.BuildLigand(
                    name="O",
                    atoms=[kp.AssignAtom(coordinating_atom=1)],
                ),
                kp.BuildLigand(
                    name="CN",
                    atoms=[kp.AssignAtom(coordinating_atom=52)],
                ),
                kp.BuildLigand(
                    name="Amine",
                    atoms=[
                        kp.AssignAtom(coordinating_atom=2),
                        kp.AssignAtom(coordinating_atom=5),
                        kp.AssignAtom(coordinating_atom=3),
                        kp.AssignAtom(coordinating_atom=4),
                    ],
                ),
            ],
        ),
        metal=kp.MetalAssignment(
            d_electrons=4,
            racah_b=kp.Varied([500.0, 600.0, 700.0]),
            racah_c=kp.Link.linear("metal.racah_b", gradient=4.0),
            spin_orbit_coupling=kp.Fixed(310.0),
            kiso=kp.Fixed(0.60),
        ),
        ligands=[
            kp.LigandAssignment(
                name="O",
                esigma=kp.Fixed(10600.0),
                epix=kp.Fixed(6300.0),
                epiy=kp.Link.linear("ligand.O.epix", gradient=1.0, constant=0.0),
            ),
            kp.LigandAssignment(
                name="CN",
                esigma=kp.Link.linear("ligand.O.esigma", gradient=1.0, constant=0.0),
                epix=kp.Link.linear("ligand.O.epix", gradient=1.0, constant=0.0),
                epiy=kp.Link.linear("ligand.O.epiy", gradient=1.0, constant=0.0),
            ),
            kp.LigandAssignment(
                name="Amine",
                esigma=kp.Fixed(5900.0),
            ),
        ],
        calculation=kp.CalculationBlock(configuration_projection=True),
        intensities=kp.IntensityBlock(
            dipoles=[
                kp.DipoleAssignment(
                    name="O",
                    psigma=kp.Fixed(-90.0 * SCALE_DIPOLE),
                    ppix=kp.Varied(
                        [80.0 * SCALE_DIPOLE, 90.0 * SCALE_DIPOLE, 100.0 * SCALE_DIPOLE]
                    ),
                    ppiy=kp.Link.linear("ligand.O.ppix", gradient=1.0, constant=0.0),
                    fsigma=kp.Fixed(100.0 * SCALE_DIPOLE),
                    fpix=kp.Fixed(0.0 * SCALE_DIPOLE),
                    fpiy=kp.Link.linear("ligand.O.fpix", gradient=1.0, constant=0.0),
                )
            ],
            absorption=kp.AbsorptionOptions(temperatures=[300.0]),
            mcd=kp.MCDOptions(
                magnetic_fields=[7.0], temperatures=[2.0, 10.0, 20.0, 40.0, 80.0]
            ),
        ),
        printing=kp.PrintingBlock(
            output_file=str(WORKING_DIRECTORY / "iron_IV_oxo.out"),
            write_spectra=True,
        ),
    )

    # The calculation will output a lot of "iron_IV_oxo_run<num>_intensity<num>_spectra.json".
    # We can simulate all of these with a common FWHM using one command.
    kp.convolute_spectra_from_files(
        input=str(WORKING_DIRECTORY / "iron_IV_oxo_run*_intensity*_spectra.json"),
        output_directory=str(WORKING_DIRECTORY / "simulations"),
        fwhm=3000.0,
        start=5000.0,
        final=30000.0,
        steps=250,
    )

    # We can also simulate each spectrum object for a calculation type:
    for label, x, y in kp.convolute_spectrum_from_file(
        file_path=str(
            WORKING_DIRECTORY / "iron_IV_oxo_run001_intensity001_spectra.json"
        ),
        calculation_type="mcd",
    ):
        print(f"{label=}, {x[:10]=}, {y[:10]=}")

    # Alternatively we can even simulate the spectrum from the results object directly.
    x, y = kp.convolute_spectrum(
        spectrum=calculation.results[0].intensities[0].mcd[0].spectra[-1],
    )
    print(f"From Spectrum result directly: {x[:10]=}, {y[:10]=}")
