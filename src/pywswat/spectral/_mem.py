"""
MEM2 directional spectrum estimator.

Maximum-entropy distribution over wave directions, solved via Newton-Raphson
iteration.

The maximum-entropy distribution subject to matching the four observed Fourier
moments a1, b1, a2, b2 takes the form

    p(theta) = exp(-mu . B(theta)) / Z(mu)

where B = [cos theta, sin theta, cos 2*theta, sin 2*theta] is the Fourier basis,
mu in R^4 are the Lagrange multipliers, and Z normalises p so that the integral
over directions equals 1.  The multipliers are found by solving the four
constraint equations

    integral B_k(theta) p(theta; mu) d_theta = target_k     k = 0..3

where target = [a1, b1, a2, b2].

Derivation of the gradient and Hessian of the log-partition function:

    grad_k  log Z = E_p[B_k]                  (first moments under p)
    hess_kl log Z = Cov_p(B_k, B_l)           (covariance matrix of B under p)

The residual at iterate mu is  r(mu) = target - grad log Z, and the Newton
step solves the positive-semidefinite system

    Cov_p(B, B) * delta = -r(mu)

Initial estimate of mu comes from the Kim et al. (1995) "AP2" linearisation.
For bins where Newton does not converge, the distribution at the AP2 estimate
is returned.  The closed-form Lygre & Krogstad (1986) MEM is available
separately as `mem_lk86`.

References
----------
Kim, T., Lin, L. H., & Wang, H. (1995). Application of maximum entropy method
to the real sea data. Coastal Engineering 1994, pp. 340-355.

Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the
directional distribution in ocean wave spectra. JPO 16(12), 2052-2060.
"""

from __future__ import annotations

import numpy as np

_SOLVER = {
    "convergence_tol": 0.01,
    "max_newton_steps": 100,
    "max_ls_steps": 8,
    "lstsq_cutoff": 1e-6,
}


# ---------------------------------------------------------------------------
# Direction-grid helpers
# ---------------------------------------------------------------------------


def _bin_widths(theta: np.ndarray) -> np.ndarray:
    """Midpoint bin width for numerical integration over a (possibly non-uniform) grid.

    For a uniform grid every element equals d_theta.
    """
    fwd = (np.diff(theta, append=theta[0]) + np.pi) % (2 * np.pi) - np.pi
    bwd = (np.diff(theta, prepend=theta[-1]) + np.pi) % (2 * np.pi) - np.pi
    return (fwd + bwd) / 2.0


def _fourier_basis(theta: np.ndarray) -> np.ndarray:
    """Return the (4, n_dirs) Fourier basis matrix B for the direction grid.

    Rows: cos theta, sin theta, cos 2*theta, sin 2*theta.
    """
    B = np.empty((4, len(theta)))
    B[0] = np.cos(theta)
    B[1] = np.sin(theta)
    B[2] = np.cos(2.0 * theta)
    B[3] = np.sin(2.0 * theta)
    return B


# ---------------------------------------------------------------------------
# Kim et al. (1995) AP2 initial estimate
# ---------------------------------------------------------------------------


def _ap2_estimate(
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """AP2 linearisation of the Lagrange multipliers (Kim et al. 1995, Eq. 12).

    Input shapes: any broadcastable leading dims.
    Output shape: (*a1.shape, 4).
    """
    s = 1.0 + a1**2 + b1**2 + a2**2 + b2**2
    mu0 = np.empty((*a1.shape, 4))
    mu0[..., 0] = 2.0 * a1 * a2 + 2.0 * b1 * b2 - 2.0 * a1 * s
    mu0[..., 1] = 2.0 * a1 * b2 - 2.0 * b1 * a2 - 2.0 * b1 * s
    mu0[..., 2] = a1**2 - b1**2 - 2.0 * a2 * s
    mu0[..., 3] = 2.0 * a1 * b1 - 2.0 * b2 * s
    return mu0


# ---------------------------------------------------------------------------
# Core maximum-entropy distribution
# ---------------------------------------------------------------------------


def _maxent_dist(mu: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Normalised max-entropy distribution p(theta) = exp(-mu.B) / Z.

    Shifts the exponent by its minimum before exponentiation so that the
    largest value of exp(-mu.B) is exactly 1; the constant cancels in Z.

    Returns p of shape (n_dirs,).
    """
    phi = B.T @ mu  # (n_dirs,)
    phi -= phi.min()
    unnorm = np.exp(-phi)
    Z = np.sum(unnorm * w)
    if Z < 1e-300:
        return np.full(len(w), 1.0 / np.sum(w))
    return unnorm / Z


def _moments_and_cov(
    mu: np.ndarray, B: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the first moments E_p[B] and covariance Cov_p(B, B) under p(mu).

    Returns
    -------
    m   : shape (4,)  — E_p[B_k] = integral B_k p d_theta
    cov : shape (4,4) — Cov_p(B_k, B_l) = E_p[B_k B_l] - m_k m_l
    """
    p = _maxent_dist(mu, B, w)
    pw = p * w  # integration weights
    m = B @ pw  # (4,)
    cov = (B * pw) @ B.T - np.outer(m, m)  # (4,4)
    return m, cov


# ---------------------------------------------------------------------------
# Newton solver for one (point, frequency) bin
# ---------------------------------------------------------------------------


def _newton_solve(
    target: np.ndarray,
    mu0: np.ndarray,
    B: np.ndarray,
    w: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Newton-Raphson for a single frequency bin.

    Minimises ||r(mu)||  where  r = target - E_p[B].
    The Newton step solves  Cov_p(B,B) * delta = -r  (positive-semidefinite).

    Uses an adaptive backtracking line search: step size is capped at
    ||mu|| / ||delta|| to prevent large steps relative to the current iterate.

    Always returns the max-entropy distribution p(theta; mu) at the best
    iterate reached — the AP2 distribution when Newton cannot make progress.
    """
    tol: float = float(cfg["convergence_tol"])
    n_newton: int = int(cfg["max_newton_steps"])
    n_ls: int = int(cfg["max_ls_steps"])
    cutoff: float = float(cfg["lstsq_cutoff"])

    mu = mu0.copy()
    m, cov = _moments_and_cov(mu, B, w)
    r = target - m

    for _ in range(n_newton):
        if float(np.linalg.norm(r)) < tol:
            break

        # Newton direction: solve  cov * delta = -r
        try:
            step = np.linalg.solve(cov, -r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(cov, -r, rcond=cutoff)[0]

        mu_norm = float(np.linalg.norm(mu))
        step_norm = float(np.linalg.norm(step))
        r_norm = float(np.linalg.norm(r))

        # Adaptive backtracking: try s=1, then shrink by min(||mu||/||step||, s/2)
        s = 1.0
        for _ in range(n_ls):
            mu_try = mu + s * step
            m_try, cov_try = _moments_and_cov(mu_try, B, w)
            r_try = target - m_try
            if float(np.linalg.norm(r_try)) < r_norm:
                mu, m, cov, r = mu_try, m_try, cov_try, r_try
                break
            s = min(mu_norm / step_norm, s / 2.0) if step_norm > 1e-15 else s / 2.0
        else:
            break  # line search exhausted — keep best iterate so far

    return _maxent_dist(mu, B, w)


# ---------------------------------------------------------------------------
# Lygre & Krogstad (1986) closed-form MEM
# ---------------------------------------------------------------------------


def _lk86_one_bin(
    theta: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    w: np.ndarray,
) -> np.ndarray:
    """Lygre & Krogstad (1986) AR(2) spectral estimator for one frequency bin.

    Constructs AR coefficients phi1, phi2 from the circular Yule-Walker
    equations, then evaluates the spectral density:

        p(theta) ∝ sigma^2 / |1 - phi1 e^{-i*theta} - phi2 e^{-2i*theta}|^2

    where sigma^2 = 1 - phi1*conj(c1) - phi2*conj(c2) is the prediction-error
    variance (always real and >= 0 for feasible moments).

    Returns p normalised so sum(p * w) = 1.
    """
    c1 = a1 + 1j * b1
    c2 = a2 + 1j * b2
    denom = 1.0 - abs(c1) ** 2
    if abs(denom) < 1e-10:
        return np.full(len(theta), 1.0 / np.sum(w))
    phi1 = (c1 - c2 * c1.conjugate()) / denom
    phi2 = c2 - phi1 * c1
    sigma2 = float(np.real(1.0 - phi1 * c1.conjugate() - phi2 * c2.conjugate()))
    ar_resp = np.abs(1.0 - phi1 * np.exp(-1j * theta) - phi2 * np.exp(-2j * theta)) ** 2
    p = np.maximum(sigma2 / (2.0 * np.pi * ar_resp), 0.0)
    total = np.sum(p * w)
    if total > 1e-15:
        return p / total
    return np.full(len(theta), 1.0 / np.sum(w))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mem_lk86(
    theta: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Lygre & Krogstad (1986) closed-form MEM, vectorised over leading dims.

    Parameters
    ----------
    theta : (n_dirs,)
        Wave directions in radians.
    a1, b1, a2, b2 : (...,)
        Normalised directional Fourier moments.

    Returns
    -------
    D : (..., n_dirs)
        Directional distribution; sum(D * d_theta) = 1 at each element.
    """
    w = _bin_widths(theta)
    out = np.empty((*a1.shape, len(theta)))
    for idx in np.ndindex(*a1.shape):
        out[idx] = _lk86_one_bin(
            theta,
            float(a1[idx]),
            float(b1[idx]),
            float(a2[idx]),
            float(b2[idx]),
            w,
        )
    return out


def mem_newton(
    theta: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """MEM2 directional distribution via Newton-Raphson (Kim et al. 1995).

    Finds p(theta; f) maximising Shannon entropy  -integral p log p d_theta
    subject to the Fourier moment constraints a1, b1, a2, b2 at each bin.
    Starts from the AP2 linearisation and returns the max-entropy distribution
    at the best Newton iterate, using the AP2 distribution for bins where the
    solver makes no progress.

    Parameters
    ----------
    theta : (n_dirs,)
        Wave directions in radians.
    a1, b1, a2, b2 : (N0, N1)
        Normalised directional Fourier moments.

    Returns
    -------
    D : (N0, N1, n_dirs)
        Directional distribution; sum(D * d_theta) = 1 at each element.
    """
    w = _bin_widths(theta)
    B = _fourier_basis(theta)
    mu0_all = _ap2_estimate(a1, b1, a2, b2)  # (*a1.shape, 4)
    out = np.empty((*a1.shape, len(theta)))

    for idx in np.ndindex(*a1.shape):
        target = np.array([a1[idx], b1[idx], a2[idx], b2[idx]])
        if np.any(np.isnan(target)):
            out[idx] = 0.0
            continue
        out[idx] = _newton_solve(target, mu0_all[idx], B, w, _SOLVER)

    return out
