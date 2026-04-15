"""
estimator.py — Online parameter estimator (Wu 2026, Eq. 3.2)
"""
import numpy as np
from typing import Optional


def learning_rate(t: float, eta0: float, alpha: float) -> float:
    return eta0 / (1.0 + t) ** alpha


def _norm_weight(Dt: float, sigma: float) -> float:
    """Normalized information weight in [0,1]: saturates when Dt >> sigma."""
    x = (Dt / (sigma + 1e-12)) ** 2
    return x / (1.0 + x)


class RecursiveEstimator:
    """
    Step-by-step online estimator for use inside the OW simulation loop.

    Implements Eq. 3.2 from the thesis with a normalized information weight
    to ensure numerical stability when trades push Dt >> sigma.

    The closed-loop feedback is genuine:
        r_hat_n  →  trade mt  →  Dt_{n+1}  →  r_hat_{n+1}
    """

    def __init__(self, sigma: float, r0: float = 2.0,
                 eta0: float = 0.5, alpha: float = 0.75,
                 r_min: float = 0.001, r_max: float = 50.0):
        self.sigma, self.r_hat = sigma, r0
        self.eta0, self.alpha  = eta0, alpha
        self.r_min, self.r_max = r_min, r_max
        self.t = 0.0
        self.history = [r0]

    def step(self, Dt_n: float, dDt_n: float, dt: float) -> float:
        eta   = learning_rate(self.t, self.eta0, self.alpha)
        innov = dDt_n / dt - (-self.r_hat * Dt_n)   # observed - predicted drift
        w     = _norm_weight(Dt_n, self.sigma)
        if abs(Dt_n) > 1e-10:
            update = eta * np.sign(-Dt_n) * np.sqrt(w) * innov * dt
        else:
            update = 0.0
        self.r_hat = float(np.clip(self.r_hat + update, self.r_min, self.r_max))
        self.t += dt
        self.history.append(self.r_hat)
        return self.r_hat

    def reset(self, r0: Optional[float] = None):
        self.r_hat = r0 if r0 is not None else self.history[0]
        self.t = 0.0
        self.history = [self.r_hat]


def run_online_estimator(Dt_grid: np.ndarray, dt: float, sigma: float,
                         r0: float = 2.0, eta0: float = 0.5,
                         alpha: float = 0.75) -> np.ndarray:
    """Batch version: run estimator on a fixed Dt time series."""
    N = len(Dt_grid)
    r_hat = np.zeros(N)
    r_hat[0] = r0
    est = RecursiveEstimator(sigma=sigma, r0=r0, eta0=eta0, alpha=alpha)
    for n in range(N - 1):
        r_hat[n + 1] = est.step(Dt_grid[n], Dt_grid[n+1] - Dt_grid[n], dt)
    return r_hat


def converged_estimate(r_hat: np.ndarray, burn_in: float = 0.5) -> float:
    start = int(len(r_hat) * burn_in)
    return float(np.median(r_hat[start:]))
