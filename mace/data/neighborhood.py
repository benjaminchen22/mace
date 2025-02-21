###########################################################################################
# Neighborhood construction
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Optional, Tuple
from matscipy.neighbours import neighbour_list
import numpy as np
from pymatgen.core import Structure


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False) -> Tuple[np.ndarray, np.ndarray]:

    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)
    assert all(i == False for i in pbc) or all(i == True for i in pbc)  # matscipy nly works with fully periodic or fully non-periodic for now.
    
    # Extend cell in non-periodic directions
    if not np.all(pbc):
        pbc_x = pbc[0]
        pbc_y = pbc[1]
        pbc_z = pbc[2]
        identity = np.identity(3, dtype=float)
        max_positions = np.max(np.absolute(positions)) + 1

        if not pbc_x:
            cell[:,0] = max_positions * 5 * cutoff * identity[:,0]
        if not pbc_y:
            cell[:,1] = max_positions * 5 * cutoff * identity[:,1]
        if not pbc_z:
            cell[:,2] = max_positions * 5 * cutoff * identity[:,2]

    """
    lattice = cell
    symbols = np.ones(len(positions))  # Dummy symbols to create `Structure`...
    struct = Structure(lattice, symbols, positions, coords_are_cartesian=True)
    sender, receiver, unit_shifts, n_distance = struct.get_neighbor_list(
        r=float(cutoff),
        numerical_tol=1e-10,
        exclude_self=False)

    """
    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=float(cutoff),
#        self_interaction=True,  # we want edges from atom to itself in different periodic images
#        use_scaled_positions=False,  # positions are not scaled positions
        )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts
