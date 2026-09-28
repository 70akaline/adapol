# adapol: Adaptive Pole Fitting for Quantum Many-Body Physics

[![CI](https://github.com/flatironinstitute/adapol/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/flatironinstitute/adapol/actions/workflows/python-package.yml)

[`adapol`](https://github.com/flatironinstitute/adapol) ("add-a-pole") is a Python package for constructing compact pole approximations of Matsubara functions,

$$
G(\mathrm{i}\nu_n) \approx \sum_{k=1}^{M} \frac{R_k}{\mathrm{i}\nu_n - p_k},
$$

with real poles $p_k$ and scalar or matrix-valued residues $R_k$, using the AAA rational approximation algorithm and nonlinear optimization. Given Matsubara data, or an existing pole expansion (for example, a discretized spectral density or a discrete Lehmann representation), `adapol` finds an accurate approximation with a specified maximum number of poles, or as few poles as possible. A typical application is hybridization fitting: constructing a compact bath representation of a given hybridization function.

## Installation

```
pip install adapol
```

The core dependencies are `numpy` and `scipy`; install the optional `cvxpy` extra with `pip install 'adapol[cvxpy]'` for matrix-valued PSD residue fitting and projection.

## Usage

`adapol` provides three main functions:

- **`approx_freq_aaa(F, Z, ...)`** fits frequency data `F`, sampled at (typically Matsubara) points `Z`, with a sum of simple poles, using the AAA algorithm. The number of poles is controlled by a pole budget `max_n_poles` and/or a AAA error tolerance `aaa_tol`.
- **`approx_sop_fast(poles, residues, beta, ...)`** approximates a given sum of poles by a (hopefully) smaller one in a single AAA pass. The number of poles is again controlled by `max_n_poles` and/or `aaa_tol`, and an optional `nonlinear_optimization` step refines the pole locations.
- **`approx_sop_tol(poles, residues, tol, beta, ...)`** finds the smallest sum of poles whose actual error (in $L^2(\tau)$ and $l^2(i \omega_n)$ ) is below the tolerance `tol`.

Pass `psd=True` and optionally `psd_eps` to the `approx_*` fitting and compression functions or `adapol.triqs.approx_gf_*` functions to constrain scalar residues to be nonnegative and matrix residues to be Hermitian positive semidefinite; for example, `poles, residues, error = approx_freq_aaa(F, Z, max_n_poles=20, psd=True)`.

This reuses the 0.2.x AAA pole selection and residue-fitting pipeline: frequency data uses least squares at the sample points, while SOP/DLR compression uses the normalized imaginary-time Frobenius L2 norm and the existing optional nonlinear pole optimization. `psd_eps` controls the SDP solver tolerance, not the final approximation error; `approx_sop_tol` still enforces `tol` and may fail if a causal fit cannot meet it. There are no legacy `hybfit`, `anacont`, or `polefitting_dlr` compatibility entry points.

For matrix-valued data, `pick_matrix_projection` optionally fits positive-semidefinite residues on a fixed real pole grid with known spectral moments; this is not a direct nearest-point projection onto the unrestricted Pick cone.

For a self-energy with high-frequency expansion `Sigma(z) = Sigma_inf + M1/z + M2/z**2 + ...`, pass the static part and both moments explicitly:

```python
from adapol import pick_matrix_projection

Sigma_projected, projection_residues = pick_matrix_projection(
    Sigma, z, pole_grid, static_part=Sigma_inf, first_moment=M1, second_moment=M2
)
```

For a normalized canonical fermionic Green's function, use `first_moment=np.eye(n_orb)`; for a hybridization, supply its known moment rather than assuming the identity. Pass `first_moment=None` for PSD-only projection without that sum rule.

The moment constraints apply only during this explicit preprocessing step. Use `psd=True` in subsequent compression to constrain the new residues, but those residue constraints do not preserve the projection moments. Evaluate the resulting pole expansion at other complex frequencies for analytic continuation.

## Examples

Three example notebooks demonstrate the usage of these functions in detail. We recommend reading them in the following order.

- [`semicircle.ipynb`](https://flatironinstitute.github.io/adapol/latest/examples/semicircle.html) — fitting data with a continuous spectrum (semicircular density): the stopping criteria, the nonlinear optimization option, and the error metric.
- [`discrete.ipynb`](https://flatironinstitute.github.io/adapol/latest/examples/discrete.html) — fitting multi-orbital data with a discrete spectrum, including an experiment on how the required number of poles scales with the number of orbitals.
- [`hubbarddimer.ipynb`](https://flatironinstitute.github.io/adapol/latest/examples/hubbarddimer.html) — analytic continuation benchmark for the Hubbard dimer.

## Documentation

The [reference documentation](https://flatironinstitute.github.io/adapol/latest/api.html) for the three functions also describes in detail how to use them, as well as information on the algorithms they implement. The same information is contained in the docstrings, e.g. `help(adapol.approx_freq_aaa)`.

## Citation

If you use this package in your research, please include a reference to this GitHub repository, and cite the following references:

1. Huang, Zhen, Emanuel Gull, and Lin Lin. "[Robust analytic continuation of Green's functions via projection, pole estimation, and semidefinite relaxation](https://doi.org/10.1103/PhysRevB.107.075151)," Phys. Rev. B 107, 075151 (2023).
2. Huang, Zhen, Denis Golež, Hugo U. R. Strand, and Jason Kaye. "[Automated evaluation of imaginary time strong coupling diagrams by sum-of-exponentials hybridization fitting](https://doi.org/10.21468/SciPostPhys.19.5.121)," SciPost Phys. 19 (5), 121 (2025).

## License

`adapol` is distributed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)).
