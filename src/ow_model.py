"""
ow_model.py
-----------
Simulation environment for the Obizhaeva-Wang (2013) optimal execution model.

Reference:
  Obizhaeva, A. & Wang, J. (2013). "Optimal trading strategy and
  supply/demand dynamics." Journal of Financial Markets 16, 1-32.

Model
-----
State:   Dt = At - Vt - s/2   (LOB deviation above steady state)
Control: mt dt = continuous trade flow  (shares per second)
         x0    = initial discrete trade
         xT    = terminal discrete trade

Dynamics (Eq. 28 in OW):
    dDt = -r Dt dt - k dXt + sigma dBt
    dVt = lam * mt dt                    (permanent price impact)

Execution cost (Eq. 29 in OW):
    dCt = (Vt + Dt + s/2) mt dt          (continuous trades)
    DCt = (Vt + Dt + s/2 + xt/(2q)) xt  (discrete trades)

Optimal strategy (Proposition 3, OW):
    x0 = xT = X0 / (r*T + 2)
    mt = r * X0 / (r*T + 2)   for t in (0, T)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from estimator import RecursiveEstimator


# ══════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class OWParams:
    """Calibrated OW model parameters."""
    r:     float    # resilience (per second)
    sigma: float    # LOB noise volatility ($/sqrt(s))
    q:     float    # market depth (shares)
    lam:   float    # permanent price impact
    k:     float    # = 1/q - lambda
    s:     float    # bid-ask spread ($)
    F0:    float = 580.0   # initial fundamental value ($)


@dataclass
class SimResult:
    """Output of a single simulation run."""
    t:          np.ndarray   # time grid
    Dt:         np.ndarray   # LOB deviation path
    Vt:         np.ndarray   # mid-quote path
    Xt:         np.ndarray   # remaining order
    mt:         np.ndarray   # trading rate (shares/s)
    r_hat:      np.ndarray   # parameter estimate path (adaptive only)
    cost:       float        # total execution cost ($)
    label:      str
    color:      str = '#4a9eff'


# ══════════════════════════════════════════════════════════════════════════
# OW Environment
# ══════════════════════════════════════════════════════════════════════════

class OWEnvironment:
    """
    Simulates the OW LOB dynamics given a trading strategy.

    The simulation is the ground truth: r_true is fixed, and Dt evolves
    according to the SDE driven by the actual trades and Brownian noise.
    In the adaptive case, the controller uses r_hat_t (the online estimate)
    instead of r_true, creating the closed-loop coupling studied in the thesis.
    """

    def __init__(self, params: OWParams, X0: float, T: float, dt: float,
                 seed: Optional[int] = None):
        """
        Parameters
        ----------
        params : OW model parameters (r_true used for simulation)
        X0     : total order size (shares)
        T      : execution horizon (seconds)
        dt     : simulation time step (seconds)
        seed   : random seed for reproducibility
        """
        self.p    = params
        self.X0   = X0
        self.T    = T
        self.dt   = dt
        self.rng  = np.random.default_rng(seed)

        self.N    = int(T / dt)
        self.t    = np.linspace(0, T, self.N + 1)

        # Pre-generate Brownian increments (shared across strategies
        # when using the same seed — ensures fair comparison)
        self._dB  = self.rng.normal(0, np.sqrt(dt), self.N)

    def _ow_discrete_trade(self, r_use: float) -> float:
        """
        OW Proposition 3: optimal initial/terminal discrete trade size.
            x0 = xT = X0 / (r*T + 2)
        """
        return self.X0 / (r_use * self.T + 2.0)

    def _ow_continuous_rate(self, r_use: float) -> float:
        """
        OW Proposition 3: optimal continuous trading rate (shares/second).
            mt = r * X0 / (r*T + 2)
        """
        return r_use * self.X0 / (r_use * self.T + 2.0)

    def _simulate(self, strategy: str,
                  estimator: Optional[RecursiveEstimator] = None,
                  r_fixed: Optional[float] = None,
                  label: str = '', color: str = '#4a9eff') -> SimResult:
        """
        Core simulation loop.

        strategy : 'twap' | 'static_ow' | 'adaptive_ow'
        estimator: RecursiveEstimator instance (adaptive only)
        r_fixed  : r value to use for static strategies
        """
        p   = self.p
        dt  = self.dt
        N   = self.N
        dB  = self._dB

        # State variables
        Dt  = np.zeros(N + 1)
        Vt  = np.zeros(N + 1)
        Xt  = np.zeros(N + 1)
        mt  = np.zeros(N + 1)
        r_h = np.zeros(N + 1)

        Vt[0]  = p.F0
        Xt[0]  = self.X0
        Dt[0]  = 0.0     # start at steady state
        r_h[0] = estimator.r_hat if estimator else (r_fixed or p.r)

        # ── Initial discrete trade ────────────────────────────────────────
        r_use_0 = r_h[0]
        x0 = self._ow_discrete_trade(r_use_0) if strategy != 'twap' else 0.0

        # Apply initial trade
        cost = 0.0
        if x0 > 0:
            ask_before = Vt[0] + p.s / 2 + Dt[0]
            cost += (ask_before + x0 / (2 * p.q)) * x0
            Dt[0] += p.k * x0
            Xt[0] -= x0

        # ── Main simulation loop ──────────────────────────────────────────
        for n in range(N):
            r_now = r_h[n]

            # ── Determine trading rate mt ─────────────────────────────────
            time_remaining = self.T - n * dt

            if strategy == 'twap':
                # Equal-sized trades over the horizon
                mt[n] = Xt[n] / max(time_remaining, dt)

            elif strategy == 'static_ow':
                # OW Prop 3 with fixed r
                mt[n] = self._ow_continuous_rate(r_fixed or p.r)
                mt[n] = min(mt[n], Xt[n] / dt)   # don't overshoot

            elif strategy == 'adaptive_ow':
                # OW Prop 3 with current estimate r_hat (lagged: use r_h[n])
                mt[n] = self._ow_continuous_rate(r_now)
                mt[n] = min(mt[n], Xt[n] / dt)

            mt[n] = max(mt[n], 0.0)

            # ── Execution cost for continuous trade ───────────────────────
            ask_now = Vt[n] + p.s / 2 + Dt[n]
            cost   += ask_now * mt[n] * dt

            # ── State update (Euler-Maruyama) ─────────────────────────────
            # dDt = -r_true * Dt * dt - k * mt * dt + sigma * dB
            dDt     = -p.r * Dt[n] * dt - p.k * mt[n] * dt + p.sigma * dB[n]
            Dt[n+1] = Dt[n] + dDt

            # dVt = lam * mt * dt  (permanent price impact)
            Vt[n+1] = Vt[n] + p.lam * mt[n] * dt

            # Remaining order
            Xt[n+1] = max(Xt[n] - mt[n] * dt, 0.0)

            # ── Estimator update (adaptive only) ──────────────────────────
            if estimator is not None:
                r_h[n+1] = estimator.step(Dt[n], dDt, dt)
            else:
                r_h[n+1] = r_now

        # ── Terminal discrete trade ───────────────────────────────────────
        xT = Xt[N]   # whatever remains
        if xT > 0:
            ask_T = Vt[N] + p.s / 2 + Dt[N]
            cost += (ask_T + xT / (2 * p.q)) * xT
            Xt[N] = 0.0

        return SimResult(t=self.t, Dt=Dt, Vt=Vt, Xt=Xt, mt=mt,
                         r_hat=r_h, cost=cost, label=label, color=color)

    # ── Public API ────────────────────────────────────────────────────────

    def run_twap(self) -> SimResult:
        """TWAP: split order evenly over the horizon."""
        return self._simulate('twap', label='TWAP', color='#ff5555')

    def run_static_ow(self, r: Optional[float] = None) -> SimResult:
        """
        Static OW: optimal strategy using the true r (oracle benchmark).
        If r is None, uses the calibrated r_true from params.
        """
        r_use = r if r is not None else self.p.r
        return self._simulate('static_ow', r_fixed=r_use,
                              label=f'Static OW (r={r_use:.3f})',
                              color='#f1fa8c')

    def run_adaptive_ow(self, r0: float = 2.0,
                        eta0: float = 0.3,
                        alpha: float = 0.75) -> SimResult:
        """
        Adaptive OW: online estimator + OW strategy, fully coupled.

        The estimator and controller evolve on the same time scale.
        The controller uses the lagged estimate r_hat_{t-} (Eq. 2.2 in thesis),
        while the estimator is updated using the new Dt increment driven
        by the actual trade mt(r_hat_{t-}).

        This is the closed-loop system studied in the thesis:
            dDt = -r* Dt dt - k * pi(Dt, r_hat_{t-}) dt + sigma dBt
            d(r_hat) = eta_t * f_theta^T * Sigma^{-1} * [...] dt + ...
        """
        est = RecursiveEstimator(
            sigma=self.p.sigma, r0=r0, eta0=eta0, alpha=alpha
        )
        return self._simulate('adaptive_ow', estimator=est,
                              label=f'Adaptive OW (r0={r0})',
                              color='#50fa7b')

    def run_all(self, r0: float = 2.0, n_mc: int = 1,
                seed_offset: int = 0) -> dict:
        """
        Run all three strategies on the same Brownian path.
        For Monte Carlo, call with different seeds.
        """
        results = {
            'twap':        self.run_twap(),
            'static_ow':   self.run_static_ow(),
            'adaptive_ow': self.run_adaptive_ow(r0=r0),
        }
        return results


# ══════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ══════════════════════════════════════════════════════════════════════════

def monte_carlo(params: OWParams, X0: float, T: float, dt: float,
                n_paths: int = 200, r0: float = 2.0) -> dict:
    """
    Run Monte Carlo simulation for all three strategies.

    Returns dict with keys 'twap', 'static_ow', 'adaptive_ow',
    each containing arrays of shape (n_paths,) for costs and
    (n_paths, N+1) for state trajectories.
    """
    N = int(T / dt)
    costs   = {k: np.zeros(n_paths) for k in ['twap','static_ow','adaptive_ow']}
    r_paths = np.zeros((n_paths, N + 1))   # adaptive r_hat paths
    Dt_paths = {k: np.zeros((n_paths, N+1)) for k in ['twap','static_ow','adaptive_ow']}

    for i in range(n_paths):
        env = OWEnvironment(params, X0, T, dt, seed=i)
        res = env.run_all(r0=r0)
        for k, v in res.items():
            costs[k][i]      = v.cost
            Dt_paths[k][i]   = v.Dt
        r_paths[i] = res['adaptive_ow'].r_hat

    return dict(costs=costs, r_paths=r_paths, Dt_paths=Dt_paths, N=N)
