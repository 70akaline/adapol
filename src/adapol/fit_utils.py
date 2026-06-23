# This is a python implementation for analytic continuation of Fermionic Green's functions/self energy
# using PES (ES) method
# Reference: PhysRevB.107.075151
import numpy as np
import scipy
import scipy.optimize

from .aaa import aaa_matrix_real


def _solve_psd_weight_sdp(M, G, complex=True, eps=1e-8):
    """Solve the PSD weight subproblem without CVXPY complex canonicalization.

    Fits the pole weights X_l in the causal representation of
    PhysRevB.107.075151 (arXiv:2210.04187):

        G(z) = sum_l X_l / (z - lambda_l),

    with real poles lambda_l and Hermitian positive-semidefinite weights
    X_l >= 0. This routine only solves for X_l; the lambda_l are fixed.

    To avoid CVXPY's slow complex-Hermitian canonicalization we use the exact
    real embedding of complex Hermitian matrices. Writing X = A + i B,

        X Hermitian PSD  <=>  A = A^T,  B = -B^T (B + B^T == 0),
                              [[A, -B],
                               [B,  A]] >= 0.

    This is an exact isomorphism, not a relaxation: the real block has the same
    eigenvalues as X (each doubled), so the real block is PSD iff X is. The
    objective fits the real and imaginary parts of G simultaneously, so
    complex-valued G(i*omega) data is handled directly while X_l >= 0 enforces
    the matrix causality condition. Do not split a genuinely coupled matrix
    into elementwise NNLS, which would drop the off-diagonal PSD coupling.

    When the weights are block/orbital-diagonal -- either because G is diagonal
    or because only the diagonal is wanted -- this full matrix SDP is not
    needed: a diagonal X_l is PSD iff each diagonal entry is >= 0, so each
    orbital reduces to an independent 1x1 problem and elementwise NNLS per
    orbital is exact and correct. That is the normal diagonal case, not a
    violation of the causality condition.

    For Norb == 1 the 1x1 Hermitian PSD condition reduces to a nonnegative real
    scalar, handled by NNLS in the caller rather than here.
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "Optional dependency 'cvxpy' is required when use_c=True. "
            "Install it via `pip install cvxpy`."
        ) from e

    Nw, Np = M.shape
    Norb = G.shape[1]
    G2 = G.reshape(Nw, Norb * Norb)

    if complex:
        A = [cp.Variable((Norb, Norb), symmetric=True) for _ in range(Np)]
        B = [cp.Variable((Norb, Norb)) for _ in range(Np)]
        constraints = []
        for a, b in zip(A, B):
            # A+iB is Hermitian PSD iff B is skew-symmetric and this real block is PSD.
            constraints += [b + b.T == 0, cp.bmat([[a, -b], [b, a]]) >> 0]

        A_flat = cp.vstack([cp.reshape(a, (1, Norb * Norb), order="C") for a in A])
        B_flat = cp.vstack([cp.reshape(b, (1, Norb * Norb), order="C") for b in B])
        pred_re = M.real @ A_flat - M.imag @ B_flat
        pred_im = M.real @ B_flat + M.imag @ A_flat
        objective = cp.Minimize(
            cp.sum_squares(pred_re - G2.real) + cp.sum_squares(pred_im - G2.imag)
        )
        prob = cp.Problem(objective, constraints)
    else:
        X = [cp.Variable((Norb, Norb), PSD=True) for _ in range(Np)]
        X_flat = cp.vstack([cp.reshape(x, (1, Norb * Norb), order="C") for x in X])
        pred_re = M.real @ X_flat
        pred_im = M.imag @ X_flat
        objective = cp.Minimize(
            cp.sum_squares(pred_re - G2.real) + cp.sum_squares(pred_im - G2.imag)
        )
        prob = cp.Problem(objective)

    prob.solve(solver="SCS", verbose=False, eps=eps)

    R = np.zeros((Np, Norb, Norb), dtype=np.complex128)
    if complex:
        for i in range(Np):
            r = A[i].value + 1j * B[i].value
            R[i] = (r + r.T.conj()) / 2.0
    else:
        for i in range(Np):
            R[i] = X[i].value
    return R


# import mosek
def eval_with_pole(pol, Z, weight, statistics="Fermion"):
    pol_t = np.reshape(pol, [pol.size, 1])
    if statistics == "Fermion":
        M = 1 / (Z - pol_t)
    else:
        M = pol_t / (Z - pol_t)
        M[:, Z == 0] = -1.0
    M = M.transpose()
    if len(weight.shape) == 1:
        return M @ weight
    else:
        G = M @ np.reshape(weight, (weight.shape[0], weight.shape[1] * weight.shape[2]))
        return np.reshape(G, (G.shape[0], weight.shape[1], weight.shape[2]))


def get_weight(
    pol, Z, G, cleanflag=True, maxiter=1000, complex=True, fast=False, eps=1e-8, statistics="Fermion"
):
    pol_t = np.reshape(pol, [pol.size, 1])
    if statistics == "Fermion":
        M = 1 / (Z - pol_t)
    else:
        M = pol_t / (Z - pol_t)
        M[:, Z == 0] = -1.0
    
    M = M.transpose()
    MM = np.concatenate([M.real, M.imag])
    if len(G.shape) == 1:
        GG = np.concatenate([G.real, G.imag])
        if cleanflag:
            R = np.linalg.lstsq(MM, GG, rcond=0)[0]
        else:
            [R, rnorm] = scipy.optimize.nnls(MM, GG, maxiter=maxiter)
        residue = G - M @ R
    else:
        Np = len(pol)
        Norb = G.shape[1]
        R = np.zeros((Np, Norb, Norb), dtype=np.complex128)
        if cleanflag:
            for i in range(Norb):
                GG = np.concatenate([G[:, i, i].real, G[:, i, i].imag])
                R[:, i, i] = np.linalg.lstsq(MM, GG, rcond=0)[0]
                for j in range(i + 1, Norb):
                    g1 = (G[:, j, i] + G[:, i, j]) / 2.0
                    g2 = (G[:, i, j] - G[:, j, i]) / 2.0
                    GG1 = np.concatenate([g1.real, g1.imag])
                    GG2 = np.concatenate([g2.imag, -g2.real])
                    R1 = np.linalg.lstsq(MM, GG1, rcond=0)[0]
                    R2 = np.linalg.lstsq(MM, GG2, rcond=0)[0]
                    R[:, i, j] = R1 + 1j * R2
                    R[:, j, i] = R1 - 1j * R2
        else:
            if not fast:
                if Norb == 1:
                    GG = np.concatenate([G[:, 0, 0].real, G[:, 0, 0].imag])
                    R[:, 0, 0] = scipy.optimize.nnls(MM, GG, maxiter=maxiter)[0]
                else:
                    R = _solve_psd_weight_sdp(M, G, complex=complex, eps=eps)
            else:
                try:
                    import cvxpy as cp
                except ImportError as e:
                    raise ImportError(
                        "Optional dependency 'cvxpy' is required when use_c=True. "
                        "Install it via `pip install cvxpy`."
                    ) from e
                for i in range(Norb):
                    GG = np.concatenate([G[:, i, i].real, G[:, i, i].imag])
                    Rii = scipy.optimize.nnls(MM, GG, maxiter=maxiter)[0]
                    R[:, i, i] = Rii
                MM2 = np.concatenate([M, np.conj(M)])
                for i in range(Norb):
                    for j in range(i + 1, Norb):
                        GG = np.concatenate([G[:, i, j], np.conj(G[:, j, i])])
                        bound = np.sqrt(np.abs(R[:, i, i] * R[:, j, j]))
                        x = cp.Variable(MM.shape[1], complex=True)
                        constraints = [
                            cp.abs(x[k]) <= bound[k] for k in range(MM.shape[1])
                        ]
                        objective = cp.Minimize(cp.sum_squares(MM2 @ x - GG))
                        prob = cp.Problem(objective, constraints)
                        prob.solve(solver="SCS", verbose=False, eps=eps)
                        # result = prob.solve(solver = cp.MOSEK,verbose = False,mosek_params = mosek_params_dict)
                        R[:, i, j] = x.value
                        R[:, j, i] = np.conj(x.value)

        residue = 1.0 * G
        for i in range(Np):
            residue = residue - M[:, i, None, None] * R[i]
    return R, M, residue


def aaa_reduce(pol, R, eps=1e-6):
    Np = R.shape[0]
    Rnorm = np.zeros(Np)
    for i in range(Np):
        Rnorm[i] = np.linalg.norm(R[i])
    nonz_index = Rnorm > eps
    return pol[nonz_index], R[nonz_index]


def erroreval(pol, Z, G, cleanflag=True, maxiter=1000, fast=False, complex=True, statistics="Fermion"):
    R, M, residue = get_weight(
        pol, Z, G, cleanflag=cleanflag, maxiter=maxiter, complex=complex, fast=fast, statistics=statistics
    )
    if len(G.shape) == 1:
        y = np.linalg.norm(residue)
        grad = np.real(np.dot(np.conj(residue), (R * (M**2))))
    else:
        y = np.linalg.norm(residue.flatten())

        Np = len(pol)
        grad = np.zeros(Np)
        Nw = len(Z)
        for k in range(Np):
            for w in range(Nw):
                grad[k] = grad[k] + np.real(
                    np.sum((M[w, k] ** 2) * (np.conj(residue[w, :, :]) * R[k]))
                )

    grad = -grad / y
    return y, grad


def pole_fitting(
    Delta,
    Z,
    tol=None,
    Ns=None,
    mmin=None,
    mmax=50,
    maxiter=50,
    solver="lstsq",
    fast=False,
    disp=False,
    complex=True,
    statistics="Fermion",
):
    # set cleanflag
    if solver == "lstsq":
        cleanflag = True
    if solver == "sdp":
        cleanflag = False

    # pole estimation
    # tol needs to be fixed
    if Ns is None and tol is None:
        raise Exception(
            "One needs to specify either the number of poles or the fitting error tolerance."
        )
    if Ns is not None and tol is not None:
        raise Exception(
            "One can not specify both the number of poles and the fitting error tolerance. Only specify one of them."
        )
    if Ns is None:
        if mmin is None or mmin < 4:
            mmin = 4
        if mmin % 2 == 1:
            mmin = mmin + 1
        if mmax > 2 * (Z.shape[0] // 2):
            mmax = 2 * (Z.shape[0] // 2)
    else:
        if Ns % 2 == 1:
            Ns = Ns + 1
        mmin, mmax = Ns, Ns
    if len(Delta.shape) == 1:
        Delta = Delta.reshape(Delta.shape[0], 1, 1)

    for m in range(mmin, mmax + 1, 2):
        pol, _, _, _ = aaa_matrix_real(Delta, 1j * Z, mmax=m)
        pol = np.real(pol)
        weight, _, residue = get_weight(
            pol, 1.0j * Z, Delta, cleanflag=cleanflag, complex=complex, fast=fast, statistics=statistics
        )
        # print(np.max(np.abs(residue)))
        if tol is not None:
            if np.max(np.abs(residue)) > tol * 10:
                continue
        if Ns is None:
            pol, weight = aaa_reduce(pol, weight, 1e-5)
        # print("Number of poles is ", len(pol))
        if cleanflag:
            if maxiter > 0:

                def fhere(pole):
                    return erroreval(
                        pole, 1j * Z, Delta, cleanflag=cleanflag, complex=complex, statistics=statistics
                    )

                # fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=cleanflag,complex=complex)
                res = scipy.optimize.minimize(
                    fhere,
                    pol,
                    method="L-BFGS-B",
                    jac=True,
                    options={
                        "disp": disp,
                        "maxiter": maxiter,
                        "gtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )
        else:

            def fhere1(pole):
                return erroreval(pole, 1j * Z, Delta, cleanflag=True, complex=complex, statistics=statistics)

            # fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=True,complex=complex)
            res = scipy.optimize.minimize(
                fhere1,
                pol,
                method="L-BFGS-B",
                jac=True,
                options={"disp": False, "gtol": 1e-10, "ftol": 1e-10},
            )
            if maxiter > 0:

                def fhere2(pole):
                    return erroreval(
                        pole, 1j * Z, Delta, cleanflag=False, complex=complex, fast=fast, statistics=statistics
                    )

                # fhere = lambda pole: erroreval(pole,1j*Z,Delta,cleanflag=False,complex= complex,fast = fast)
                res = scipy.optimize.minimize(
                    fhere2,
                    res.x,
                    method="L-BFGS-B",
                    jac=True,
                    options={
                        "disp": disp,
                        "maxiter": maxiter,
                        "gtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )

        weight, _, residuenew = get_weight(
            res.x, 1j * Z, Delta, cleanflag=cleanflag, fast=fast, complex=complex, statistics=statistics
        )
        if not check_psd(weight):
            weight, _, residuenew = get_weight(
                res.x, 1j * Z, Delta, cleanflag=False, complex=complex, statistics=statistics
            )
        err = np.max(np.abs(residuenew))
        if tol is not None:
            if err < tol:
                return res.x, weight, err
        else:
            return res.x, weight, err

    if tol is not None:
        print("Fail to reach desired fitting error!")
    return res.x, weight, np.max(np.abs(residuenew))


def check_psd(weight, atol=1e-6):
    check_psd = True
    for i in range(weight.shape[0]):
        val, _ = np.linalg.eig(weight[i])
        check_psd = check_psd and np.min(val.real) > -atol
    return check_psd
