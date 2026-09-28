"""Positive-semidefinite residue fitting for fixed poles."""

import numpy as np


def fit_psd_residues(
    M, F, *, eps=1e-8, poles=None, first_moment=None, second_moment=None
):
    """Minimize ``||M @ R - F||_F**2`` over nonnegative or Hermitian PSD residues.

    ``F`` must have shape ``(n_samples,)`` or ``(n_samples, n_orb, n_orb)``.
    Optional moments impose ``sum(R)`` and ``sum(poles * R)``, respectively.
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

    import cvxpy as cp

    if F.ndim == 1:
        residues = cp.Variable(n_poles, nonneg=True)
        constraints = []
        if first_moment is not None:
            constraints.append(cp.sum(residues) == first_moment)
        if second_moment is not None:
            constraints.append(cp.sum(cp.multiply(np.real(poles), residues)) == second_moment)
        residual = M @ residues - F
    else:
        n_orb = F.shape[1]
        residues = [cp.Variable((n_orb, n_orb), hermitian=True) for _ in range(n_poles)]
        constraints = [residue >> 0 for residue in residues]
        if first_moment is not None:
            constraints.append(sum(residues) == first_moment)
        if second_moment is not None:
            constraints.append(
                sum(pole * residue for pole, residue in zip(np.real(poles), residues))
                == second_moment
            )
        flattened = cp.vstack(
            [cp.reshape(residue, (1, n_orb * n_orb), order="C") for residue in residues]
        )
        residual = M @ flattened - F.reshape(F.shape[0], -1)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(cp.abs(residual))), constraints)
    problem.solve(solver="SCS", eps=eps)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"PSD residue fit failed: {problem.status}")

    if F.ndim == 1:
        return residues.value
    values = np.asarray([residue.value for residue in residues])
    return (values + values.transpose(0, 2, 1).conj()) / 2
