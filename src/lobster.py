"""
lobster.py
----------
Utilities for loading and preprocessing LOBSTER data.
Computes LOB state variables needed for the OW model.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LOBState:
    """Processed LOB state variables."""
    time:     np.ndarray   # seconds after midnight
    ask1:     np.ndarray   # best ask price ($)
    bid1:     np.ndarray   # best bid price ($)
    mid:      np.ndarray   # mid-quote Vt ($)
    spread:   np.ndarray   # bid-ask spread ($)
    ask_size: np.ndarray   # level-1 ask depth (shares)
    Ft:       np.ndarray   # fundamental value proxy (EMA of mid)
    Dt:       np.ndarray   # LOB deviation = At - (Ft + s/2)
    # execution events
    exec_time:  np.ndarray
    exec_size:  np.ndarray
    exec_dir:   np.ndarray


def load_lobster(
    msg_file: str,
    obk_file: str,
    n_levels: int = 5,
    ema_tau:  float = 30.0,    # seconds, EMA timescale for Ft proxy
    price_scale: float = 10000.0,
) -> LOBState:
    """
    Load LOBSTER message + orderbook files and return LOBState.

    Parameters
    ----------
    msg_file   : path to LOBSTER message CSV
    obk_file   : path to LOBSTER orderbook CSV
    n_levels   : number of price levels in the orderbook file
    ema_tau    : EMA timescale (seconds) for fundamental value proxy Ft
    price_scale: LOBSTER stores prices as integer * 10000

    Notes on Dt
    -----------
    In the OW model, Dt = At - (Ft + s/2) measures how far the current
    ask is above its steady-state level.  At tick resolution,
    At - mid - s/2 = 0 algebraically, so we proxy the fundamental value
    Ft with an exponential moving average of the mid-quote.  This is
    standard in the empirical microstructure literature.
    """
    # ── column names ──────────────────────────────────────────────────────
    msg_cols = ["time", "event_type", "order_id", "size", "price", "direction"]
    obk_cols = []
    for i in range(1, n_levels + 1):
        obk_cols += [f"ask_p{i}", f"ask_s{i}", f"bid_p{i}", f"bid_s{i}"]

    msg = pd.read_csv(msg_file, header=None, names=msg_cols)
    obk = pd.read_csv(obk_file, header=None, names=obk_cols)

    # ── remove halts and dummy levels ─────────────────────────────────────
    valid = (
        (msg["event_type"] != 7) &
        (obk["ask_p1"].between(1, 9_000_000_000)) &
        (obk["bid_p1"].between(1, 9_000_000_000))
    )
    msg = msg[valid].reset_index(drop=True)
    obk = obk[valid].reset_index(drop=True)

    # ── prices to dollars ─────────────────────────────────────────────────
    for i in range(1, n_levels + 1):
        obk[f"ask_p{i}"] /= price_scale
        obk[f"bid_p{i}"] /= price_scale

    ask1   = obk["ask_p1"].values
    bid1   = obk["bid_p1"].values
    ask_s1 = obk["ask_s1"].values
    mid    = (ask1 + bid1) / 2.0
    spread = ask1 - bid1
    time   = msg["time"].values

    # ── fundamental value proxy: EMA of mid ───────────────────────────────
    dt_ticks = np.empty(len(time))
    dt_ticks[0] = 0.0
    dt_ticks[1:] = np.diff(time)
    alpha = 1.0 - np.exp(-np.clip(dt_ticks, 0.0, 60.0) / ema_tau)

    Ft = np.empty(len(mid))
    Ft[0] = mid[0]
    for i in range(1, len(mid)):
        Ft[i] = (1.0 - alpha[i]) * Ft[i - 1] + alpha[i] * mid[i]

    # ── LOB deviation ─────────────────────────────────────────────────────
    Dt = ask1 - (Ft + spread / 2.0)

    # ── execution events ──────────────────────────────────────────────────
    exec_mask = msg["event_type"].isin([4, 5])
    exec_time = time[exec_mask.values]
    exec_size = msg.loc[exec_mask, "size"].values
    exec_dir  = msg.loc[exec_mask, "direction"].values

    return LOBState(
        time=time, ask1=ask1, bid1=bid1, mid=mid,
        spread=spread, ask_size=ask_s1, Ft=Ft, Dt=Dt,
        exec_time=exec_time, exec_size=exec_size, exec_dir=exec_dir,
    )


def resample(lob: LOBState, dt: float = 1.0) -> Tuple[np.ndarray, ...]:
    """
    Resample LOB state onto a regular time grid with spacing dt (seconds).
    Returns (t_grid, Dt_grid, mid_grid, spread_grid, q_grid).
    """
    t_grid = np.arange(lob.time[0], lob.time[-1], dt)
    Dt_g   = np.interp(t_grid, lob.time, lob.Dt)
    mid_g  = np.interp(t_grid, lob.time, lob.mid)
    spr_g  = np.interp(t_grid, lob.time, lob.spread)
    q_g    = np.interp(t_grid, lob.time, lob.ask_size)
    return t_grid, Dt_g, mid_g, spr_g, q_g


def estimate_params(lob: LOBState, dt: float = 1.0,
                    core_start: float = 36000.0,
                    core_end:   float = 54000.0):
    """
    Quick OLS estimates of OW model parameters from LOB data.

    Returns dict with keys: r, sigma, q, lam, k, spread_mean
    """
    t_grid, Dt_g, _, spr_g, q_g = resample(lob, dt)
    dDt = np.diff(Dt_g)
    Dn  = Dt_g[:-1]

    # MLE / OLS for r: dDt = -r Dt dt + noise
    r_hat = -np.dot(dDt, Dn * dt) / (np.dot(Dn, Dn) * dt**2 + 1e-30)
    r_hat = float(max(0.001, r_hat))

    # sigma from residuals
    resid = dDt - (-r_hat * Dn * dt)
    sigma = float(np.std(resid) / np.sqrt(dt))

    # market depth: median level-1 ask size during core hours
    core = (lob.time >= core_start) & (lob.time <= core_end)
    q = float(np.median(lob.ask_size[core]))

    lam = 1.0 / (2.0 * q)   # OW baseline: permanent impact
    k   = lam                # k = 1/q - lambda = 1/(2q)

    return dict(r=r_hat, sigma=sigma, q=q, lam=lam, k=k,
                spread_mean=float(lob.spread.mean()))
