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

from collections import defaultdict

import numpy as np

from kestrelpy.calculations.eigenfunction_indices import EPRLevelSOC
from kestrelpy.calculations.magnetism.g_factors import (
    GSquaredTensor,
    GSquaredTensorMetadata,
    calculate_g_factors,
    calculate_g_factors_for_indices,
)
from kestrelpy.calculations.many_electron_eigenfunctions.eigenfunctions import (
    Eigenfunction,
    ManyElectronEigenfunctions,
)
from kestrelpy.calculations.solve_matrix import solve_matrix


def calculate_real_g_factors(
    muxyz: np.ndarray, eigenfunctions: ManyElectronEigenfunctions, zfs_tol: float = 10.0
) -> GSquaredTensor:
    """
    Calculate real g factors weighted by spin projection. Real g factors represent
    g values in the absence of zero-field splitting and are summed over each
    populated spin manifold and are temperature-independent.

    g tensors are calculated for each multiplicity, over the corresponding
    number of states. They are then summed and weighted by their spin projection.

    Args:
        muxyz: Magnetic moment operator matrices.
        eigenfunctions: ManyElectronEigenfunctions object containing the energy of
            each electronic state and its spin projection.
        zfs_tol: For non-Kramers ions, the strength of ZFS between singly
            degenerate states above which EPR silence is expected.

    Returns:
        GSquaredTensor containing real g values.
    """
    soc_microstates = eigenfunctions.eigenfunctions

    is_epr_silent = determine_epr_silence(states=soc_microstates, zfs_tol=zfs_tol)
    if is_epr_silent:
        g_frame_g_values = GSquaredTensor.zero_g_tensor()
        return g_frame_g_values

    number_manifold_states = int(round(soc_microstates[0].dominant_multiplicity()))
    is_non_kramers = number_manifold_states % 2 != 0
    projection = get_g_spin_projection(
        states=soc_microstates,
        is_non_kramers=is_non_kramers,
    )

    states_available = muxyz.shape[1]
    total_g2, used_weight = calculate_projection_weighted_g(
        muxyz=muxyz, states=soc_microstates, projection=projection
    )
    if used_weight < 0.05:
        return GSquaredTensor.zero_g_tensor()

    total_g2 /= used_weight  # re-normalise if any singlet character was dropped

    result = solve_matrix(total_g2)

    return GSquaredTensor(
        tensor=total_g2,
        values=np.sqrt(np.maximum(result.values, 0.0)),
        vectors=result.vectors,
        metadata=GSquaredTensorMetadata(
            eigenfunction_start_index=0,
            eigenfunction_final_index=states_available - 1,
        ),
    )


def calculate_g_effective(
    epr_levels_soc: list[EPRLevelSOC],
    muxyz: np.ndarray,
) -> GSquaredTensor:
    """
    Calculate effective g factors. Effective g factors represent g values
    where zero-field splitting can result in effective spin-doublets (S=1/2).
    Effective g factors are calculated for each populated electronic state.

    The calculated g tensors are then weighted by their boltzmann population
    to give effective g factors comparable to experiment. The result is
    temperature dependent.

    Args:
        epr_levels_soc: List of EPRLevelSOC, containing the populated electronic
            states and their boltzmann and spin projections used to weight the
            calculated effective g values.
        muxyz: Magnetic moment operator matrices.

    Returns:
        GSquaredTensor containing effective g values.
    """
    total_g2_eff = np.zeros((3, 3), dtype=np.float64)
    total_weight_eff = 0.0

    for lvl_soc in epr_levels_soc:
        w_k = lvl_soc["level_boltzmann"]
        if w_k <= 1e-6 or lvl_soc["final"] == lvl_soc["start"]:
            continue

        g_k_soc = calculate_g_factors(
            muxyz=muxyz,
            start=lvl_soc["start"],
            final=lvl_soc["final"],
        )
        total_g2_eff += w_k * np.real(g_k_soc.tensor)
        total_weight_eff += w_k

    if total_weight_eff < 1e-3:
        return GSquaredTensor.zero_g_tensor()

    total_g2_eff /= total_weight_eff
    result = solve_matrix(total_g2_eff)

    return GSquaredTensor(
        tensor=total_g2_eff,
        values=np.sqrt(np.maximum(result.values, 0.0)),
        vectors=result.vectors,
        metadata=GSquaredTensorMetadata(
            eigenfunction_start_index=epr_levels_soc[0]["start"],
            eigenfunction_final_index=epr_levels_soc[-1]["final"],
        ),
    )


def _is_singlet(state: Eigenfunction, singlet_threshold: float = 0.25) -> bool:
    """Determine whether an electronic state has significant singlet character

    This identifies any singlet states mixing into a spin manifold.
    These indices are skipped when calculating g factors for non-Kramers ions.
    """
    for mult, weight in state.spin_multiplicity_projection.items():
        if int(round(mult)) == 1 and weight > singlet_threshold:
            return True
    return False


def _non_singlet_indices(
    states: list[Eigenfunction], manifold_states: int
) -> list[int]:
    """
    Determine the indices for a spin manifold when a singlet state
    has mixed within the manifold. Any singlet indices are skipped
    due to them being diamagnetic and therefore EPR silent.
    """
    non_singlet_states = []
    for i, state in enumerate(states):
        if not _is_singlet(state=state):
            non_singlet_states.append(i)
        if len(non_singlet_states) == manifold_states:
            break
    return non_singlet_states


def determine_epr_silence(states: list[Eigenfunction], zfs_tol: float = 10.0) -> bool:
    """
    Determines whether a system is expected to be EPR silent. This is expected if:
        1) The ground state is diamagnetic and well separated from any excited
            paramagnetic states,
        2) If zero-field splitting results in effective singlet levels separated
            by more than the microwave quantum. In this instance, no EPR transitions
            are expected.

    Args:
        states: List of eigenfunctions that are assessed for EPR activity.
        zfs_tol: Tolerance above which ZFS renders the system EPR silent in cm-1.

    Returns:
        bool, True is an EPR silent system.
    """
    ground_state = states[0]
    if ground_state.dominant_multiplicity() == 1:
        if len(states) < 2:
            return True  # Inactive as ground state is diamagnetic

        energy_gapa = abs(states[1].energy - ground_state.energy)
        if (energy_gapa >= zfs_tol) or (states[1].dominant_multiplicity() == 1):
            return True  # Inactive as no low energy EPR-active states

    if len(states) >= 3:
        energy_gapa = abs(states[1].energy - ground_state.energy)
        energy_gapb = abs(states[2].energy - states[1].energy)

        if (energy_gapa >= zfs_tol) and (energy_gapb >= zfs_tol):
            return True  # Inactive due to large ZFS

    return False  # System is EPR active


def get_g_spin_projection(
    states: list[Eigenfunction], is_non_kramers: bool, manifold_states: int = 1
) -> dict[int, float]:
    """
    Determine the spin projection by which real g values are weighted.
    For Kramers ions this is taken from the ground state. For non-Kramers
    ions the spin projection can be averaged over a specified number of states.

    Args:
        states: List of eigenfunctions for which the spin projection is calculated.
        manifold_states: Number of electronic states to be averaged over for
            non-Kramers ions. Set to 1 by default to average over the ground state.
        is_non_kramers: Flag, True if the complex is a non-Kramers ion

    Returns:
        dict[int, float] containing each spin multiplicity and its projection.
    """
    if is_non_kramers:
        averaged_projection: defaultdict[int, float] = defaultdict(float)
        valid_states = []

        for state in states:
            if not _is_singlet(state=state):
                valid_states.append(state)
            if len(valid_states) == manifold_states:
                break

        for state in valid_states:
            for mult, weight in state.spin_multiplicity_projection.items():
                averaged_projection[mult] += weight / len(valid_states)
        return dict(averaged_projection)

    else:
        return dict(states[0].spin_multiplicity_projection)


def calculate_projection_weighted_g(
    muxyz: np.ndarray, states: list[Eigenfunction], projection: dict[int, float]
) -> tuple[np.ndarray, float]:
    """
    Determine the g tensor per multiplicity where the number of states
    summed over relates to the multiplicity, i.e. a multiplicity of 4 (quartet
    state) sums over the ground 4 states.

    g tensors for each multiplicity are then weighted by the spin projection
    and the weighted sum is returned.

    Args:
        muxyz: Magnetic moment operator matrices.
        states: List of electronic Eigenfunctions.
        projection: dict containing each multiplicity and its spin projection.

    Returns:
        weighted g^2 tensor and the total weight used.
    """
    total_weight = sum(projection.values())
    total_g2 = np.zeros((3, 3), dtype=np.float64)
    used_weight = 0.0

    for mult, weight in projection.items():
        number_states = int(round(mult))
        if weight <= 1e-6 or number_states <= 1:
            continue
        w = weight / total_weight

        non_singlet_indices = _non_singlet_indices(
            states=states, manifold_states=number_states
        )
        g_factors = calculate_g_factors_for_indices(
            muxyz=muxyz, indices=non_singlet_indices[:number_states]
        )
        total_g2 += w * np.real(g_factors.tensor)
        used_weight += w

    return total_g2, used_weight
