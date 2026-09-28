import numpy as np
import pytest
import scipy.optimize

from adapol.sop import SumOfSimplePoles


def _pole_data(nw=64, norb=2):
    rng = np.random.default_rng(123)
    poles = np.array([-1.4, -0.35, 0.55])
    z = 1j * np.arange(-2 * nw - 1, 2 * nw + 2, 2) * np.pi / 40.0
    z = z[:nw]
    kernel = (1 / (z - poles.reshape(poles.size, 1))).T

    weights = np.empty((poles.size, norb, norb), dtype=complex)
    for p in range(poles.size):
        v = rng.normal(size=norb) + 1j * rng.normal(size=norb)
        weights[p] = np.outer(v, v.conj()) / poles.size

    g = np.einsum("wp,pab->wab", kernel, weights)
    return poles, z, g


def test_scalar_psd_residue_fit_reduces_to_nnls():
    poles, z, g = _pole_data(nw=128, norb=1)
    sop = SumOfSimplePoles(poles, np.zeros((len(poles), 1, 1), dtype=complex))
    sop.fit_residues_to_freq_samples(z, g, psd=True, psd_eps=1e-8)

    kernel = (1 / (z - poles.reshape(poles.size, 1))).T
    mm = np.concatenate([kernel.real, kernel.imag])
    gg = np.concatenate([g[:, 0, 0].real, g[:, 0, 0].imag])
    expected = scipy.optimize.nnls(mm, gg)[0]

    assert np.allclose(sop.R[:, 0, 0].real, expected)
    assert np.allclose(sop.R[:, 0, 0].imag, 0)
    assert np.linalg.norm(sop(z) - g) < 1e-10
    assert min(np.linalg.eigvalsh(weight).min() for weight in sop.R) > -1e-6


def test_matrix_psd_residue_fit_preserves_hermitian_psd_weights():
    pytest.importorskip("cvxpy")
    poles, z, g = _pole_data(nw=64, norb=2)
    sop = SumOfSimplePoles(poles, np.zeros((len(poles), 2, 2), dtype=complex))
    sop.fit_residues_to_freq_samples(z, g, psd=True, psd_eps=1e-6)

    for weight in sop.R:
        assert np.allclose(weight, weight.T.conj(), atol=1e-8)

    assert np.linalg.norm(sop(z) - g) < 1e-4
    assert min(np.linalg.eigvalsh(weight).min() for weight in sop.R) > -1e-5
