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

from typing import Literal

import numpy as np

from kestrelpy.molecule import Ligand, Molecule
from kestrelpy.parameters import LigandParameters

IntensityParameter = Literal["epix", "epiy", "esigma"]


def calculate_effective_e_values(
    ligand_coordinates: np.ndarray,
    e_value: float,
    target_coordinates: np.ndarray,
) -> float:
    """
    Calculate an effective e value from a ligand at a specified point. This
    gives a measure of the local bonding environment.

    The cosine between the position of a ligand and a point in space are calculated.
    This angle scales the e value of the ligand and returns e_effective at a
    specified point/target.

    Args:
        ligand_coordinates: xyz coordinates of a ligand
        e_value: esigma, epix or epiy parameter for the ligand
        target_coordinates: xyz coordinates e_effective is calculated at.

    Returns:
        e_effective value
    """
    target_norm = np.linalg.norm(target_coordinates)
    ligand_norm = np.linalg.norm(ligand_coordinates)

    if target_norm == 0.0 or ligand_norm == 0.0:
        return 0.0

    target_axis = target_coordinates / target_norm
    ligand_axis = ligand_coordinates / ligand_norm

    cosine = np.dot(ligand_axis, target_axis)

    if cosine < 0:
        return 0.0

    return float(e_value * (cosine**2))


def estimate_intensity_parameter(
    ligand: Ligand,
    molecule: Molecule,
    ligand_parameters: list[LigandParameters],
    parameter: IntensityParameter = "esigma",
    all_ligands: bool = False,
) -> float:
    """
    Calculate psigma (default), ppix or ppiy intensity parameter for a ligand.

    Calculate the sum of effective e values at the ligand and directly opposite.
    The difference is taken as a measure of how similar the bonding environment
    opposite to the ligand is. The p value is returned as follows::
        p_parameter = Δe_effective / 100
    Where larger p values indicate increasingly asymmetric bonding environments.

    Args:
        ligand: Ligand object describing the ligand calculated
        molecule: Molecule object describing the molecular system
        ligand_parameters: list of LigandParameters giving e values
        parameter: which intensity parameter is calculated. Set to "esigma" by default
        all_ligands: Flag to specify if parameters are estimated for all ligands.

    Returns:
        p_sigma, p_pix or p_piy parameter value
    """
    e_eff_difference = 0.0
    ligand_e_value = 0.0
    max_e_value = 0.0

    # Loop through each atom in the ligand p values are calculated for
    for ligand_atom in ligand.atoms:
        target_coords = np.array(ligand_atom.coordinating_atom.coordinates)

        ligand_contribution = 0.0
        antipode_contribution = 0.0

        # Iterate through each ligand in the molecule and store their e values
        for source_ligand, source_params in zip(molecule.ligands, ligand_parameters):
            e_value = getattr(source_params, parameter).value
            if source_ligand == ligand:
                ligand_e_value = e_value
            if abs(e_value) > max_e_value:
                max_e_value = abs(e_value)

            # Iterate through each coordinating atom and calculate effective e values
            # at and opposite to the ligand intensity parameters are calculated for.
            for source_atom in source_ligand.atoms:
                source_coords = np.array(source_atom.coordinating_atom.coordinates)
                ligand_contribution += calculate_effective_e_values(
                    ligand_coordinates=source_coords,
                    e_value=e_value,
                    target_coordinates=target_coords,
                )

                antipode_contribution += calculate_effective_e_values(
                    ligand_coordinates=source_coords,
                    e_value=e_value,
                    target_coordinates=target_coords * -1,
                )

        e_eff_difference += ligand_contribution - antipode_contribution

    # Average over all atoms in the ligand
    e_eff = e_eff_difference / len(ligand.atoms)
    p_value = e_eff / 100

    if all_ligands:
        p_value = 0 if np.sign(ligand_e_value) != np.sign(p_value) else p_value

    return round(p_value, 0)
