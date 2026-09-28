"""Positive-semidefinite residue fitting for fixed poles."""

import numpy as np
from scipy.optimize import nnls


def fit_psd_residues(
    M, F, *, eps=1e-8, poles=None, first_moment=None, second_moment=None
):
    """Minimize ``||M @ R - F||_F**2`` over nonnegative or Hermitian PSD residues.

    ``F`` must have shape ``(n_samples,)`` or ``(n_samples, n_orb, n_orb)``.
    Optional moments impose ``sum(R)`` and ``sum(poles * R)``, respectively.

    The matrix solve uses the exact real embedding of a Hermitian residue
    ``R = A + i B``: ``A = A.T``, ``B = -B.T`` and
    ``[[A, -B], [B, A]] >= 0``. The full off-diagonal PSD coupling is retained.
    Scalar and 1x1 fits without moments reduce exactly to NNLS.
    """
    M = np.asarray(M)
    F = np.asarray(F)
    if M.ndim != 2 or F.ndim not in (1, 3) or F.shape[0] != M.shape[0]:
        raise ValueError("M and F must have matching sample dimensions; F must be 1D or 3D")
    if F.ndim == 3 and F.shape[1] != F.shape[2]:
        raise ValueError("matrix-valued F must have square trailing dimensions")

    n_poles = M.shape[1]
    if second_moment is not None:
        if poles is None or len(poles) != n_poles:
            raise ValueError("second_moment requires one pole per residue")
        if np.any(np.imag(poles) != 0):
            raise ValueError("PSD residues require real poles")
    expected_shape = () if F.ndim == 1 else F.shape[1:]
    for name, moment in (("first_moment", first_moment), ("second_moment", second_moment)):
        if moment is not None and np.shape(moment) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    if (F.ndim == 1 or F.shape[1] == 1) and first_moment is None and second_moment is None:
        design = np.concatenate([M.real, M.imag])
        target = F.reshape(-1)
        values = nnls(design, np.concatenate([target.real, target.imag]))[0]
        return values.reshape((n_poles,) + F.shape[1:])

    import cvxpy as cp

    if F.ndim == 1:
        residues = cp.Variable(n_poles, nonneg=True)
        constraints = []
        if first_moment is not None:
            constraints.append(cp.sum(residues) == first_moment)
        if second_moment is not None:
            constraints.append(cp.sum(cp.multiply(np.real(poles), residues)) == second_moment)
        residual_re = M.real @ residues - F.real
        residual_im = M.imag @ residues - F.imag
    else:
        n_orb = F.shape[1]
        A = [cp.Variable((n_orb, n_orb), symmetric=True) for _ in range(n_poles)]
        B = [cp.Variable((n_orb, n_orb)) for _ in range(n_poles)]
        constraints = []
        for a, b in zip(A, B):
            constraints += [b + b.T == 0, cp.bmat([[a, -b], [b, a]]) >> 0]
        if first_moment is not None:
            constraints += [sum(A) == np.real(first_moment), sum(B) == np.imag(first_moment)]
        if second_moment is not None:
            constraints += [
                sum(p * a for p, a in zip(np.real(poles), A)) == np.real(second_moment),
                sum(p * b for p, b in zip(np.real(poles), B)) == np.imag(second_moment),
            ]
        A_flat = cp.vstack(
            [cp.reshape(a, (1, n_orb * n_orb), order="C") for a in A]
        )
        B_flat = cp.vstack(
            [cp.reshape(b, (1, n_orb * n_orb), order="C") for b in B]
        )
        target = F.reshape(F.shape[0], -1)
        residual_re = M.real @ A_flat - M.imag @ B_flat - target.real
        residual_im = M.real @ B_flat + M.imag @ A_flat - target.imag

    objective = cp.Minimize(cp.sum_squares(residual_re) + cp.sum_squares(residual_im))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SCS", eps=eps)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"PSD residue fit failed: {problem.status}")

    if F.ndim == 1:
        return residues.value
    values = np.asarray([a.value + 1j * b.value for a, b in zip(A, B)])
    return (values + values.transpose(0, 2, 1).conj()) / 2
