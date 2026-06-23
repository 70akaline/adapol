import numpy as np
import pytest
import scipy.optimize

from adapol.fit_utils import check_psd, get_weight


def _pole_data(nw=64, norb=2):
    rng = np.random.default_rng(123)
    pol = np.array([-1.4, -0.35, 0.55])
    z = 1j * np.arange(-2 * nw - 1, 2 * nw + 2, 2) * np.pi / 40.0
    z = z[:nw]
    m = (1 / (z - pol.reshape(pol.size, 1))).T

    weights = np.empty((pol.size, norb, norb), dtype=complex)
    for p in range(pol.size):
        v = rng.normal(size=norb) + 1j * rng.normal(size=norb)
        weights[p] = np.outer(v, v.conj()) / pol.size

    g = np.einsum("wp,pab->wab", m, weights)
    return pol, z, g


def test_scalar_sdp_weight_fit_reduces_to_nnls():
    pol, z, g = _pole_data(nw=128, norb=1)

    r, m, residue = get_weight(pol, z, g, cleanflag=False, complex=True)

    mm = np.concatenate([m.real, m.imag])
    gg = np.concatenate([g[:, 0, 0].real, g[:, 0, 0].imag])
    expected = scipy.optimize.nnls(mm, gg)[0]

    assert np.allclose(r[:, 0, 0].real, expected)
    assert np.allclose(r[:, 0, 0].imag, 0)
    assert np.linalg.norm(residue) < 1e-10
    assert check_psd(r)


def test_matrix_sdp_weight_fit_preserves_hermitian_psd_weights():
    pytest.importorskip("cvxpy")
    pol, z, g = _pole_data(nw=64, norb=2)

    r, _m, residue = get_weight(pol, z, g, cleanflag=False, complex=True, eps=1e-6)

    for weight in r:
        assert np.allclose(weight, weight.T.conj(), atol=1e-8)
        assert np.linalg.eigvalsh(weight).min() > -1e-5

    assert np.linalg.norm(residue) < 1e-4
    assert check_psd(r, atol=1e-5)
