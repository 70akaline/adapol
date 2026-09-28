import numpy as np
import pytest

from adapol import pick_matrix_projection


def test_pick_matrix_projection_preserves_matrix_causality_and_self_energy_moments():
    pytest.importorskip("cvxpy")

    z = 1j * np.arange(1, 24, 2) * np.pi / 10
    pole_grid = np.array([-1.2, 0.4, 1.5])
    vectors = np.array([[1, 0.3j], [0.4, 0.8], [0.2j, 0.5]])
    reference_residues = np.einsum("la,lb->lab", vectors, vectors.conj())
    first_moment = reference_residues.sum(axis=0)
    second_moment = np.einsum("l,lab->ab", pole_grid, reference_residues)
    static_part = np.diag([0.3, -0.2])
    kernel = 1 / (z[:, None] - pole_grid[None, :])
    reference = static_part + np.einsum("nl,lab->nab", kernel, reference_residues)

    rng = np.random.default_rng(7)
    noise = 1e-3 * (rng.standard_normal(reference.shape) + 1j * rng.standard_normal(reference.shape))
    projected, residues = pick_matrix_projection(
        reference + noise, z, pole_grid,
        static_part=static_part,
        first_moment=first_moment,
        second_moment=second_moment,
        eps=1e-6,
    )

    assert np.linalg.norm(projected - reference - noise) <= np.linalg.norm(noise) + 1e-4
    assert np.linalg.norm(residues.sum(axis=0) - first_moment) < 1e-5
    assert np.linalg.norm(np.einsum("l,lab->ab", pole_grid, residues) - second_moment) < 1e-5
    assert min(np.linalg.eigvalsh(residue).min() for residue in residues) > -1e-5

    pick = np.block([
        [-(projected[i] - projected[j].conj().T) / (z[i] - z[j].conj())
         for j in range(len(z))]
        for i in range(len(z))
    ])
    assert np.linalg.eigvalsh(pick).min() > -1e-5
