"""Fixed-grid causal projection of matrix-valued frequency samples."""

import numpy as np

from ._psd import fit_psd_residues
from .sop import SumOfSimplePoles


def pick_matrix_projection(
    G, z, pole_grid, *, first_moment, second_moment=None, static_part=None, eps=1e-8
):
    """Project samples onto a real-pole expansion with Hermitian PSD residues.

    Minimize the squared Frobenius residual over the frequency samples with
    ``G(z) = static_part + sum_l R_l / (z - pole_grid_l)`` and ``R_l >= 0``.
    When supplied, impose ``sum_l R_l = first_moment`` and
    ``sum_l pole_grid_l * R_l = second_moment``. This fixed-grid causal
    projection yields a positive-semidefinite Pick matrix; it is not a direct
    nearest-matrix projection onto the unrestricted Pick cone.

    ``first_moment`` is required explicitly: use the identity for a normalized
    fermionic Green's function, the known moment for a hybridization, or None
    for no sum rule. For a self-energy, supply its constant ``static_part`` and
    coefficients of 1/z and 1/z**2 as the first and second moments.

    This is opt-in preprocessing, independent of AAA/compression. Subsequent
    compression must use ``psd=True`` to constrain its new residues; the
    projection's moment constraints are not propagated to compression.

    Returns ``(G_projected, residues)`` of shapes ``G.shape`` and
    ``(len(pole_grid), n_orb, n_orb)``. Requires the optional cvxpy dependency.
    """
    G = np.asarray(G)
    z = np.asarray(z)
    pole_grid = np.asarray(pole_grid)
    if z.ndim != 1 or pole_grid.ndim != 1 or not np.isrealobj(pole_grid):
        raise ValueError("z and the real pole_grid must be one-dimensional")
    if G.ndim != 3 or G.shape[0] != len(z) or G.shape[1] != G.shape[2]:
        raise ValueError("G must have shape (len(z), n_orb, n_orb)")
    if first_moment is not None:
        first_moment = np.asarray(first_moment)
        if first_moment.shape != G.shape[1:]:
            raise ValueError("first_moment must have shape (n_orb, n_orb)")
    if second_moment is not None:
        second_moment = np.asarray(second_moment)
        if second_moment.shape != G.shape[1:]:
            raise ValueError("second_moment must have shape (n_orb, n_orb)")
    if static_part is not None:
        static_part = np.asarray(static_part)
        if static_part.shape != G.shape[1:]:
            raise ValueError("static_part must have shape (n_orb, n_orb)")

    M = 1.0 / (z[:, None] - pole_grid[None, :])
    dynamic_G = G if static_part is None else G - static_part
    residues = fit_psd_residues(
        M, dynamic_G, eps=eps, poles=pole_grid,
        first_moment=first_moment, second_moment=second_moment,
    )
    projected = SumOfSimplePoles(pole_grid, residues)(z)
    if static_part is not None:
        projected += static_part
    return projected, residues
