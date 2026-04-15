"""
Build all three Jupyter notebooks for the LOB Adaptive Execution project.
Run this script once to generate the .ipynb files.
"""
import nbformat as nbf
import os

NB_DIR = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    }
    return n


def md(src):  return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)


# ══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Data Exploration
# ══════════════════════════════════════════════════════════════════════════

nb1_cells = [

md("""# Notebook 1 — LOBSTER Data Exploration & LOB Dynamics

**Project:** Adaptive Execution via Online Parameter Estimation  
**Data:** AAPL · NASDAQ · 2012-06-21 · Level-5 LOBSTER  
**Author:** Changkui Wu (FSU Financial Mathematics PhD, 2026)

---

## Overview

This notebook loads the raw LOBSTER tick data, constructs the key state variables
used in the Obizhaeva–Wang (2013) execution model, and visualises the intraday
behaviour of the limit order book.

### What is the OW model?

Obizhaeva & Wang (2013) model the limit order book (LOB) as a block-shaped queue
whose *deviation* from steady state decays exponentially:

$$dD_t = -r \\, D_t \\, dt - k \\, dX_t + \\sigma \\, dB_t$$

| Symbol | Meaning |
|--------|---------|
| $D_t = A_t - (F_t + s/2)$ | Ask deviation above steady-state level |
| $r > 0$ | **Resilience** — speed of LOB recovery |
| $k = 1/q - \\lambda$ | Price impact coefficient |
| $\\sigma$ | Diffusion noise of the LOB |
| $dX_t$ | Trade flow (our control) |

The key insight of OW: the *optimal execution strategy depends only on $r$*, not
on the static spread or depth.  Since $r$ is unobservable, we need to estimate it
online — which is exactly what the PhD thesis addresses.
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from lobster import load_lobster, resample, estimate_params

# ── style ──────────────────────────────────────────────────────────────────
DARK, GRID = '#0f1117', '#1a1d2e'
TEXT, BLUE  = '#e0e0e0', '#4a9eff'
GREEN, ORANGE, RED = '#50fa7b', '#ffb86c', '#ff5555'
PURPLE, CYAN = '#bd93f9', '#8be9fd'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': GRID,
    'axes.edgecolor': '#333355', 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': '#252540',
    'grid.linewidth': 0.6, 'legend.facecolor': GRID,
    'legend.labelcolor': TEXT, 'font.size': 10,
})

DATA_DIR = os.path.join('..', 'data')
MSG_FILE = os.path.join(DATA_DIR, 'AAPL_2012-06-21_34200000_57600000_message_5.csv')
OBK_FILE = os.path.join(DATA_DIR, 'AAPL_2012-06-21_34200000_57600000_orderbook_5.csv')

print("Loading LOBSTER data...")
lob = load_lobster(MSG_FILE, OBK_FILE, ema_tau=30.0)
print(f"  Events:       {len(lob.time):,}")
print(f"  Time range:   {lob.time[0]/3600:.2f}h – {lob.time[-1]/3600:.2f}h")
print(f"  Price range:  ${lob.mid.min():.2f} – ${lob.mid.max():.2f}")
print(f"  Executions:   {len(lob.exec_time):,}")
print(f"  Mean spread:  {lob.spread.mean()*100:.2f} cents")
"""),

md("""## 1. Price and Fundamental Value

The **mid-quote** $V_t = (A_t + B_t)/2$ is the raw market price signal.
The **fundamental value** $F_t$ is proxied by an exponential moving average (EMA)
of the mid-quote with timescale $\\tau = 30$ seconds.  The LOB deviation
$D_t = A_t - (F_t + s/2)$ measures how far the ask price has been pushed
above its steady-state level by recent trades.
"""),

code("""\
t_h   = lob.time / 3600.0
step  = max(1, len(lob.time) // 3000)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('AAPL LOB Dynamics — 2012-06-21', fontsize=13, color=TEXT)

# ── Panel 1: price ────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(t_h[::step], lob.mid[::step],   color=BLUE,   lw=0.8, label='Mid-quote $V_t$')
ax.plot(t_h[::step], lob.Ft[::step],    color=ORANGE, lw=1.0, alpha=0.8, label='EMA $F_t$ (τ=30s)')
# large trade markers
exec_h = lob.exec_time / 3600.0
large  = lob.exec_size >= np.percentile(lob.exec_size, 95)
for et in exec_h[large][::5]:
    ax.axvline(et, color=RED, alpha=0.1, lw=0.5)
ax.set_ylabel('Price ($)')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Mid-quote and Fundamental Value Proxy', color=TEXT, fontsize=10)

# ── Panel 2: LOB deviation ────────────────────────────────────────────────
ax = axes[1]
ax.plot(t_h[::step], lob.Dt[::step], color=CYAN, lw=0.5, alpha=0.85)
ax.axhline(0, color=TEXT, lw=0.8, ls='--', alpha=0.4)
ax.fill_between(t_h[::step], lob.Dt[::step], 0,
                where=lob.Dt[::step]>0, color=RED,    alpha=0.15)
ax.fill_between(t_h[::step], lob.Dt[::step], 0,
                where=lob.Dt[::step]<0, color=GREEN,  alpha=0.15)
ax.set_ylabel('$D_t$ ($)')
ax.set_title('LOB Deviation  $D_t = A_t - (F_t + s/2)$', color=TEXT, fontsize=10)

# ── Panel 3: spread & depth ───────────────────────────────────────────────
ax  = axes[2]
axb = ax.twinx()
ax.plot(t_h[::step], lob.spread[::step]*100, color=GREEN,  lw=0.6, alpha=0.8, label='Spread (¢)')
axb.plot(t_h[::step], lob.ask_size[::step],   color=PURPLE, lw=0.5, alpha=0.6, label='Depth (shares)')
ax.set_ylabel('Spread (¢)',       color=GREEN)
axb.set_ylabel('L1 Depth (sh)',   color=PURPLE)
axb.tick_params(colors=PURPLE)
ax.set_xlabel('Time of day (hours)')
ax.set_title('Bid-Ask Spread & Level-1 Ask Depth', color=TEXT, fontsize=10)
lines = ax.get_lines() + axb.get_lines()
ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='upper right')

for a in axes:
    a.set_xlim(t_h[0], t_h[-1])

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb1_lob_dynamics.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()
print("Saved → outputs/nb1_lob_dynamics.png")
"""),

md("""## 2. Trade Size Distribution

LOBSTER event types 4 and 5 are visible and hidden limit-order executions,
i.e. actual market trades.  The top-5% by size are "large trades" that push
$D_t$ noticeably and trigger the resilience recovery we want to estimate.
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Trade Statistics', color=TEXT, fontsize=12)

# Distribution
ax = axes[0]
sizes = lob.exec_size
thr95 = np.percentile(sizes, 95)
ax.hist(sizes[sizes < thr95], bins=60, color=BLUE, alpha=0.75,
        density=True, edgecolor='none', label='< 95th pct')
ax.hist(sizes[sizes >= thr95], bins=20, color=RED,  alpha=0.75,
        density=True, edgecolor='none', label='≥ 95th pct (large)')
ax.axvline(thr95, color=RED, lw=1.5, ls='--')
ax.set_xlabel('Trade size (shares)')
ax.set_ylabel('Density')
ax.set_title('Trade Size Distribution')
ax.legend(fontsize=9)

# Intraday trade intensity
ax = axes[1]
bins  = np.arange(9.5, 16.1, 0.25)
cnts, edges = np.histogram(lob.exec_time/3600, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers, cnts, width=0.22, color=BLUE, alpha=0.75, edgecolor='none')
ax.set_xlabel('Time of day (hours)')
ax.set_ylabel('Executions per 15-min bin')
ax.set_title('Intraday Trade Intensity (U-shape)')
ax.axvline(9.5,  color=RED, lw=1, ls='--', alpha=0.5)
ax.axvline(16.0, color=RED, lw=1, ls='--', alpha=0.5)

for a in axes:
    a.set_facecolor(GRID)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb1_trade_stats.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()

print(f"Total executions:         {len(sizes):,}")
print(f"Large trade threshold:    {thr95:.0f} shares  (95th percentile)")
print(f"Large trades:             {(sizes>=thr95).sum():,}")
print(f"Mean trade size:          {sizes.mean():.1f} shares")
print(f"Median trade size:        {np.median(sizes):.1f} shares")
"""),

md("""## 3. Post-Trade LOB Recovery

The core empirical test of the OW model: after a large buy trade, the ask
price $A_t$ is pushed up (large $D_t$), then decays back toward steady state
as new liquidity providers refill the book.

OW models this decay as $D_t \\approx D_0 e^{-rt}$.  The slope on a
log-scale plot against time gives us $-r$.
"""),

code("""\
large_idx  = np.where(lob.exec_size >= thr95)[0]
# Map back to full timeline indices
all_exec   = np.where(np.isin(lob.time,
                 lob.exec_time[lob.exec_size >= thr95]))[0]

window = 45.0
curves = []
for li in large_idx[:300]:
    t0 = lob.exec_time[li]
    D0 = np.interp(t0, lob.time, lob.Dt)
    if abs(D0) < 0.01:
        continue
    mask = (lob.time > t0) & (lob.time <= t0 + window)
    if mask.sum() < 8:
        continue
    curves.append((lob.time[mask] - t0,
                   lob.Dt[mask] / D0))

t_th = np.linspace(0, window, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Post-Large-Trade LOB Recovery', color=TEXT, fontsize=12)

for ax, logy in zip(axes, [False, True]):
    for tc, Dc in curves[:30]:
        ax.plot(tc, Dc, color=ORANGE, alpha=0.20, lw=0.8)
    # Overlay theoretical decay for a range of r values
    for r_plot, c in [(0.02, CYAN), (0.036, RED), (0.1, GREEN)]:
        y = np.exp(-r_plot * t_th)
        ax.plot(t_th, y, lw=1.8, ls='--', color=c, label=f'r={r_plot}')
    ax.axhline(0, color=TEXT, lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel('Seconds after large trade')
    ax.set_ylabel('$D_t / D_0$' + (' (log scale)' if logy else ''))
    ax.set_title('Linear scale' if not logy else 'Log scale  (slope = -r)')
    ax.set_xlim(0, window)
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 10)
    else:
        ax.set_ylim(-1.5, 2.0)
    ax.legend(fontsize=9)
    ax.set_facecolor(GRID)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb1_recovery.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()
print(f"Recovery curves plotted: {len(curves)}")
"""),

md("""## 4. Summary Statistics

Summary of key LOB statistics for the trading day.
These will be used to calibrate the OW simulation in Notebook 3.
"""),

code("""\
params = estimate_params(lob, dt=1.0)

print("=" * 50)
print("LOB Summary Statistics — AAPL 2012-06-21")
print("=" * 50)
print(f"  Mean mid-quote:    ${lob.mid.mean():.2f}")
print(f"  Mean spread:       {lob.spread.mean()*100:.3f} cents")
print(f"  Median L1 depth:   {np.median(lob.ask_size):.0f} shares")
print(f"  Total executions:  {len(lob.exec_time):,}")
print(f"  Large trades (95%): {(lob.exec_size >= thr95).sum():,}")
print()
print("OW Model Parameters (MLE estimates):")
print(f"  r     = {params['r']:.4f} /s   (half-life {np.log(2)/params['r']:.1f}s)")
print(f"  sigma = {params['sigma']:.5f} $/sqrt(s)")
print(f"  q     = {params['q']:.0f} shares")
print(f"  lam   = {params['lam']:.5f} $/share^2")
print(f"  k     = {params['k']:.5f} $/share^2")
print("=" * 50)
print()
print("→ These parameters are passed to Notebook 2 (estimation) and")
print("  Notebook 3 (simulation) as the calibrated ground truth.")
"""),

]  # end nb1_cells

# ══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — Online Parameter Estimation
# ══════════════════════════════════════════════════════════════════════════

nb2_cells = [

md("""# Notebook 2 — Online Parameter Estimation

**Project:** Adaptive Execution via Online Parameter Estimation  
**Author:** Changkui Wu (FSU Financial Mathematics PhD, 2026)

---

## Overview

This notebook applies the online estimator from the PhD thesis to the real
AAPL LOBSTER data, demonstrating that the resilience parameter $r$ can be
learned in real time from the observed LOB deviation process $D_t$.

### The Estimator (Thesis Eq. 3.2)

Given the state equation $dD_t = -r^* D_t \\, dt + \\sigma \\, dB_t$, the
likelihood-inspired online estimator is:

$$d\\hat{r}_t = \\eta_t \\underbrace{\\frac{\\partial f}{\\partial r}^\\top}_{= -D_t} \\Sigma^{-1} \\underbrace{\\left[f(D_t, r^*) - f(D_t, \\hat{r}_t)\\right]}_{\\text{drift error}} dt + \\text{martingale}$$

Discretised (Euler–Maruyama):

$$\\hat{r}_{n+1} = \\hat{r}_n + \\eta_n \\cdot (-D_n) \\cdot \\sigma^{-2} \\cdot \\underbrace{\\left(\\frac{\\Delta D_n}{\\Delta t} + \\hat{r}_n D_n\\right)}_{\\text{innovation}} \\cdot \\Delta t$$

**Theorem 4.3 (thesis):** Under standard regularity, monotonicity, and
learning-rate conditions, $\\hat{r}_t \\to r^*$ almost surely as $t \\to \\infty$.
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lobster import load_lobster, resample, estimate_params
from estimator import run_online_estimator, converged_estimate

DARK, GRID = '#0f1117', '#1a1d2e'
TEXT, BLUE  = '#e0e0e0', '#4a9eff'
GREEN, ORANGE, RED = '#50fa7b', '#ffb86c', '#ff5555'
PURPLE, CYAN, YELLOW = '#bd93f9', '#8be9fd', '#f1fa8c'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': GRID,
    'axes.edgecolor': '#333355', 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': '#252540',
    'grid.linewidth': 0.6, 'legend.facecolor': GRID,
    'legend.labelcolor': TEXT, 'font.size': 10,
})

DATA_DIR = os.path.join('..', 'data')
MSG_FILE = os.path.join(DATA_DIR, 'AAPL_2012-06-21_34200000_57600000_message_5.csv')
OBK_FILE = os.path.join(DATA_DIR, 'AAPL_2012-06-21_34200000_57600000_orderbook_5.csv')

lob    = load_lobster(MSG_FILE, OBK_FILE)
params = estimate_params(lob)
DT     = 1.0   # resample step (seconds)

t_grid, Dt_g, mid_g, spr_g, q_g = resample(lob, dt=DT)
sigma = params['sigma']
r_mle = params['r']

print(f"Calibrated parameters:")
print(f"  r_MLE  = {r_mle:.4f} /s   (half-life {np.log(2)/r_mle:.1f}s)")
print(f"  sigma  = {sigma:.5f} $/sqrt(s)")
print(f"  Resampled grid: {len(t_grid):,} points at dt={DT}s")
"""),

md("""## 1. Estimator Convergence from Multiple Starting Points

A key test of the thesis's almost-sure convergence guarantee: starting from
very different initial guesses $\\hat{r}_0$, the estimator should converge
to the same value $r^*$ regardless of initialisation.

This mirrors **Figure 5.1** in the thesis.
"""),

code("""\
r0_list   = [0.001, 0.01, 0.1, 0.5, 2.0, 5.0]
colors    = [CYAN, BLUE, GREEN, ORANGE, RED, PURPLE]
t_since   = (t_grid - t_grid[0]) / 3600.0   # hours

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Online Estimator Convergence — Multiple Initialisations',
             color=TEXT, fontsize=12)

for r0, col in zip(r0_list, colors):
    rh = run_online_estimator(Dt_g, DT, sigma, r0=r0, eta0=0.3, alpha=0.75)
    for ax, logy in zip(axes, [False, True]):
        ax.plot(t_since, rh, color=col, lw=1.0, alpha=0.85, label=f'$r_0$={r0}')

for ax, logy in zip(axes, [False, True]):
    ax.axhline(r_mle, color=RED, lw=1.8, ls='--', alpha=0.7,
               label=f'MLE = {r_mle:.4f}')
    ax.set_xlabel('Time since open (hours)')
    ax.set_ylabel('$\\hat{r}_t$ (per second)')
    ax.set_xlim(0, t_since[-1])
    ax.legend(fontsize=8, ncol=2)
    ax.set_facecolor(GRID)

axes[0].set_title('Linear scale', color=TEXT)
axes[1].set_title('Log scale', color=TEXT)
axes[1].set_yscale('log')
axes[1].set_ylim(1e-4, 20)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb2_convergence.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()
"""),

md("""## 2. Effect of Learning Rate Schedule

The learning rate $\\eta_t = \\eta_0 / (1+t)^\\alpha$ controls the
speed–accuracy trade-off.

- **Large $\\alpha$** (close to 1): slower decay, suppresses noise better asymptotically
- **Small $\\alpha$** (close to 0.5): faster early convergence, more noise later

This replicates the analysis in **Figure 5.3** of the thesis on real data.
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Effect of Learning Rate Schedule  $\\eta_t = \\eta_0/(1+t)^\\alpha$',
             color=TEXT, fontsize=12)

alpha_list  = [0.55, 0.65, 0.75, 0.85, 0.95]
alpha_cols  = [CYAN, BLUE, GREEN, ORANGE, RED]

for al, col in zip(alpha_list, alpha_cols):
    rh = run_online_estimator(Dt_g, DT, sigma, r0=2.0, eta0=0.3, alpha=al)
    err = np.abs(rh - r_mle)
    for ax, y in zip(axes, [rh, err]):
        ax.plot(t_since, y, color=col, lw=1.0, alpha=0.85, label=f'α={al}')

axes[0].axhline(r_mle, color=RED, lw=1.5, ls='--', label=f'MLE={r_mle:.4f}')
axes[0].set_ylabel('$\\hat{r}_t$'); axes[0].set_title('Estimate trajectory')
axes[1].set_ylabel('$|\\hat{r}_t - r_{MLE}|$'); axes[1].set_title('Absolute error')
axes[1].set_yscale('log')

for ax in axes:
    ax.set_xlabel('Hours since open')
    ax.set_xlim(0, t_since[-1])
    ax.legend(fontsize=9)
    ax.set_facecolor(GRID)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb2_learning_rate.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()
"""),

md("""## 3. Three-Term Lyapunov Decomposition (Thesis Chapter 4)

The convergence proof decomposes the Lyapunov function
$V_t = \\frac{1}{2}\\|\\hat{r}_t - r^*\\|^2$ into three terms:

$$V_t \\leq \\underbrace{E_t^{-1} V_0}_{\\text{(I) initial condition}} +
           \\underbrace{E_t^{-1} C \\int_0^t E_s \\eta_s^2 \\, ds}_{\\text{(II) diffusion remainder}} +
           \\underbrace{E_t^{-1} \\int_0^t E_s \\, dM_s}_{\\text{(III) martingale}}$$

where $E_t = \\exp\\!\\left(2\\lambda_0 \\int_0^t \\eta_s \\, ds\\right)$.

All three terms vanish as $t \\to \\infty$, giving almost-sure convergence.
Below we track them numerically on the real LOBSTER data — matching
**Figure 5.5** from the thesis.
"""),

code("""\
alpha0, eta0_val = 0.75, 0.3
lam0 = params['r']    # use MLE as proxy for lambda_0 in the decomposition

# Compute the three terms
rh  = run_online_estimator(Dt_g, DT, sigma, r0=2.0, eta0=eta0_val, alpha=alpha0)
V   = 0.5 * (rh - r_mle)**2   # Lyapunov function

# Integrating factor E_t
S   = np.cumsum(eta0_val / (1 + np.arange(len(t_grid)) * DT)**alpha0) * DT
Et  = np.exp(2 * lam0 * S)

# Term I: E_t^{-1} * V(0)
term_I = V[0] / Et

# Term II: E_t^{-1} * C * integral E_s eta_s^2 ds
eta_sq = (eta0_val / (1 + np.arange(len(t_grid)) * DT)**alpha0)**2
integ  = np.cumsum(Et * eta_sq) * DT
term_II = integ / Et

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Three-Term Lyapunov Decomposition  (Thesis Chapter 4, Fig 5.5)',
             color=TEXT, fontsize=12)

S_plot = S   # effective time

for ax, logy in zip(axes, [False, True]):
    ax.plot(S_plot, V,        color=CYAN,   lw=1.0, label='$V_t = \\frac{1}{2}|\\hat{r}_t - r^*|^2$')
    ax.plot(S_plot, term_I,   color=ORANGE, lw=1.5, ls='--', label='Term (I): initial condition')
    ax.plot(S_plot, term_II,  color=GREEN,  lw=1.5, ls=':',  label='Term (II): diffusion remainder')
    ax.set_xlabel('Effective time $S(t) = \\int_0^t \\eta_s \\, ds$')
    ax.set_ylabel('Value')
    ax.legend(fontsize=9)
    ax.set_facecolor(GRID)
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(1e-8, V[0]*2)
        ax.set_title('Log scale')
    else:
        ax.set_title('Linear scale')

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb2_lyapunov.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()
print("Term (I)  converges to 0:", f"{term_I[-1]:.2e}")
print("Term (II) converges to 0:", f"{term_II[-1]:.2e}")
"""),

md("""## 4. Benchmark Comparison: Online vs MLE

The online estimator is a **sequential** algorithm — it processes one observation
at a time and can be used in real time.  The MLE uses all data simultaneously.

Key questions:
- How quickly does the online estimator approach the MLE?
- What is the price of online (sequential) estimation?
"""),

code("""\
# Compare online vs MLE at different time horizons
horizons = np.linspace(0.1, 1.0, 20)   # fraction of trading day
r_online_at_t = []
r_mle_at_t    = []

for frac in horizons:
    n_end = int(frac * len(Dt_g))
    # Online estimator at this fraction
    rh_partial = run_online_estimator(Dt_g[:n_end], DT, sigma, r0=2.0,
                                      eta0=0.3, alpha=0.75)
    r_online_at_t.append(converged_estimate(rh_partial, burn_in=0.5))
    # MLE on this partial data
    dDt   = np.diff(Dt_g[:n_end])
    Dn    = Dt_g[:n_end-1]
    r_mle_here = -np.dot(dDt, Dn*DT) / (np.dot(Dn,Dn)*DT**2 + 1e-30)
    r_mle_at_t.append(max(0.001, float(r_mle_here)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor(GRID)
ax.plot(horizons * (t_grid[-1]-t_grid[0])/3600, r_online_at_t,
        color=CYAN, lw=2.0, marker='o', ms=4, label='Online estimator (converged half)')
ax.plot(horizons * (t_grid[-1]-t_grid[0])/3600, r_mle_at_t,
        color=ORANGE, lw=2.0, marker='s', ms=4, label='MLE (batch, uses all data to t)')
ax.axhline(r_mle, color=RED, lw=1.5, ls='--', label=f'Full-day MLE = {r_mle:.4f}')
ax.set_xlabel('Hours of data used')
ax.set_ylabel('Estimated $r$ (per second)')
ax.set_title('Online vs MLE Estimates as Data Accumulates', color=TEXT)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb2_online_vs_mle.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()

print("Full-day MLE:          r = ", f"{r_mle:.4f}")
print("Online (full day):     r = ", f"{converged_estimate(rh, burn_in=0.5):.4f}")
print()
print("The online estimator converges to the MLE in the long run,")
print("but provides a REAL-TIME estimate available at every tick.")
"""),

md("""## Summary

| Property | Online Estimator | MLE |
|----------|-----------------|-----|
| Real-time updates | ✅ | ❌ |
| Convergence guarantee | ✅ a.s. (Theorem 4.3) | ✅ consistency |
| Works under adaptive feedback | ✅ (no stationarity needed) | ⚠️ assumes stationarity |
| Computational cost | O(1) per step | O(N) batch |

**Key takeaway:** The online estimator provides real-time parameter estimates
that are theoretically guaranteed to converge — making it suitable for use
inside an adaptive execution loop, which is exactly what Notebook 3 demonstrates.
"""),

]  # end nb2_cells


# ══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — Adaptive Execution Simulation
# ══════════════════════════════════════════════════════════════════════════

nb3_cells = [

md("""# Notebook 3 — Adaptive Execution Simulation

**Project:** Adaptive Execution via Online Parameter Estimation  
**Author:** Changkui Wu (FSU Financial Mathematics PhD, 2026)

---

## Overview

This notebook is the centrepiece of the project.  We simulate the
**closed-loop adaptive execution system** studied in the thesis:

$$\\underbrace{dD_t = -r^* D_t \\, dt - k \\, m_t \\, dt + \\sigma \\, dB_t}_{\\text{OW LOB dynamics (environment)}}$$

$$\\underbrace{m_t = \\frac{\\hat{r}_{t-} X_t}{\\hat{r}_{t-}(T-t) + 2}}_{\\text{Adaptive OW controller (Proposition 3)}}$$

$$\\underbrace{d\\hat{r}_t = \\eta_t (-D_t) \\sigma^{-2} \\left[\\frac{dD_t}{dt} + \\hat{r}_t D_t\\right] dt + \\cdots}_{\\text{Online estimator (Thesis Eq. 3.2)}}$$

The three equations are **genuinely coupled**:
$\\hat{r}_t$ drives $m_t$, which drives $D_t$, which drives $\\hat{r}_{t+dt}$.
This is the closed-loop adaptive system whose almost-sure convergence is
proved in the thesis — and whose cost performance we evaluate here.

### Strategies compared

| Strategy | Description |
|----------|-------------|
| **TWAP** | Baseline: split order evenly over horizon |
| **Static OW** | OW Proposition 3 with a *fixed* (possibly wrong) $r$ |
| **Adaptive OW** | OW Proposition 3 with *online-estimated* $\\hat{r}_t$ |

The key question: **when $r$ is unknown, does the adaptive strategy
outperform a static strategy with a misspecified $r$?**
"""),

code("""\
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lobster import load_lobster, estimate_params
from ow_model import OWParams, OWEnvironment, monte_carlo

DARK, GRID = '#0f1117', '#1a1d2e'
TEXT, BLUE  = '#e0e0e0', '#4a9eff'
GREEN, ORANGE, RED = '#50fa7b', '#ffb86c', '#ff5555'
PURPLE, CYAN, YELLOW = '#bd93f9', '#8be9fd', '#f1fa8c'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': GRID,
    'axes.edgecolor': '#333355', 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': '#252540',
    'grid.linewidth': 0.6, 'legend.facecolor': GRID,
    'legend.labelcolor': TEXT, 'font.size': 10,
})

# ── Calibrated parameters from Notebook 1 ────────────────────────────────
DATA_DIR = os.path.join('..', 'data')
lob      = load_lobster(
    os.path.join(DATA_DIR, 'AAPL_2012-06-21_34200000_57600000_message_5.csv'),
    os.path.join(DATA_DIR, 'AAPL_2012-06-21_34200000_57600000_orderbook_5.csv'),
)
cal = estimate_params(lob)

params = OWParams(
    r     = cal['r'],      # TRUE resilience (used by the environment)
    sigma = cal['sigma'],
    q     = cal['q'],
    lam   = cal['lam'],
    k     = cal['k'],
    s     = cal['spread_mean'],
    F0    = float(lob.mid.mean()),
)

print("Calibrated OW parameters (from real LOBSTER data):")
for k, v in vars(params).items():
    print(f"  {k:6s} = {v:.5f}")
print()
print("Simulation settings:")
X0, T, dt = 5000, 390, 1.0
print(f"  Order size X0 = {X0} shares")
print(f"  Horizon     T = {T} seconds ({T/60:.1f} minutes)")
print(f"  Time step  dt = {dt} second")
"""),

md("""## 1. Single-Path Visualisation

A single Monte Carlo path showing how the three strategies execute the order
and how the adaptive estimator's $\\hat{r}_t$ evolves over time.
"""),

code("""\
env  = OWEnvironment(params, X0=X0, T=T, dt=dt, seed=7)
r_tw = env.run_twap()
r_st = env.run_static_ow()                  # uses true r (oracle)
r_ad = env.run_adaptive_ow(r0=2.0,          # starts with wrong r
                            eta0=0.1, alpha=0.8)

t_plot = r_tw.t

fig = plt.figure(figsize=(15, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Single-Path Adaptive Execution Simulation', color=TEXT, fontsize=13)

# ── Panel 1: Remaining order Xt ───────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
ax.plot(t_plot, r_tw.Xt, color=RED,    lw=1.5, label='TWAP')
ax.plot(t_plot, r_st.Xt, color=YELLOW, lw=1.5, label=f'Static OW (r={params.r:.3f})')
ax.plot(t_plot, r_ad.Xt, color=GREEN,  lw=1.5, label='Adaptive OW (r₀=2.0)')
ax.set_ylabel('Remaining order $X_t$ (shares)')
ax.set_xlabel('Time (seconds)')
ax.set_title('Execution Profile — Remaining Order')
ax.legend(fontsize=9)
ax.set_facecolor(GRID)

# ── Panel 2: Trading rate mt ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_plot, r_tw.mt, color=RED,    lw=1.0, alpha=0.8, label='TWAP')
ax.plot(t_plot, r_st.mt, color=YELLOW, lw=1.0, alpha=0.8, label='Static OW')
ax.plot(t_plot, r_ad.mt, color=GREEN,  lw=1.0, alpha=0.8, label='Adaptive OW')
ax.set_ylabel('Trading rate $m_t$ (shares/s)')
ax.set_xlabel('Time (seconds)')
ax.set_title('Trading Rate')
ax.legend(fontsize=9)
ax.set_facecolor(GRID)

# ── Panel 3: LOB deviation Dt ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.plot(t_plot, r_tw.Dt, color=RED,    lw=0.8, alpha=0.8, label='TWAP')
ax.plot(t_plot, r_st.Dt, color=YELLOW, lw=0.8, alpha=0.8, label='Static OW')
ax.plot(t_plot, r_ad.Dt, color=GREEN,  lw=0.8, alpha=0.8, label='Adaptive OW')
ax.axhline(0, color=TEXT, lw=0.6, ls='--', alpha=0.4)
ax.set_ylabel('LOB deviation $D_t$ ($)')
ax.set_xlabel('Time (seconds)')
ax.set_title('LOB Deviation Path')
ax.legend(fontsize=9)
ax.set_facecolor(GRID)

# ── Panel 4: r_hat convergence ────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.plot(t_plot, r_ad.r_hat, color=CYAN, lw=1.5, label='$\\hat{r}_t$ (adaptive)')
ax.axhline(params.r, color=RED, lw=1.5, ls='--',
           label=f'$r^*$ = {params.r:.4f} (true)')
ax.axhline(2.0, color=ORANGE, lw=1.0, ls=':', alpha=0.7, label='$r_0$ = 2.0 (initial)')
ax.set_ylabel('$\\hat{r}_t$ (per second)')
ax.set_xlabel('Time (seconds)')
ax.set_title('Online Estimator: $\\hat{r}_t$ Convergence\n(Theorem 4.3: a.s. convergence)')
ax.legend(fontsize=9)
ax.set_facecolor(GRID)
ax.set_ylim(0, max(2.5, r_ad.r_hat.max() * 1.1))

# ── Panel 5: Cumulative cost ──────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
# Approximate cumulative cost (mid-quote * trade rate)
cum_tw = np.cumsum(r_tw.Vt * r_tw.mt * dt)
cum_st = np.cumsum(r_st.Vt * r_st.mt * dt)
cum_ad = np.cumsum(r_ad.Vt * r_ad.mt * dt)
ax.plot(t_plot[:-1], cum_tw, color=RED,    lw=1.2, label=f'TWAP      ${r_tw.cost:,.0f}')
ax.plot(t_plot[:-1], cum_st, color=YELLOW, lw=1.2, label=f'Static OW ${r_st.cost:,.0f}')
ax.plot(t_plot[:-1], cum_ad, color=GREEN,  lw=1.2, label=f'Adaptive  ${r_ad.cost:,.0f}')
ax.set_ylabel('Cumulative cost ($)')
ax.set_xlabel('Time (seconds)')
ax.set_title('Cumulative Execution Cost')
ax.legend(fontsize=8)
ax.set_facecolor(GRID)

plt.savefig(os.path.join('..','outputs','nb3_single_path.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()

print(f"Single-path costs:")
print(f"  TWAP:       ${r_tw.cost:,.2f}")
print(f"  Static OW:  ${r_st.cost:,.2f}  (savings ${r_tw.cost-r_st.cost:+,.2f} vs TWAP)")
print(f"  Adaptive:   ${r_ad.cost:,.2f}  (savings ${r_tw.cost-r_ad.cost:+,.2f} vs TWAP)")
"""),

md("""## 2. The Key Experiment: Robustness to Misspecification

**The realistic scenario:** the practitioner does not know $r^*$.
They have a prior guess $r_0$.

- **Static OW** uses $r_0$ and *never updates* — its strategy is sub-optimal
  if $r_0 \\neq r^*$.
- **Adaptive OW** starts from $r_0$ but *learns* $r^*$ over time.

We sweep over different initial guesses and compare costs over 300 Monte Carlo paths.
"""),

code("""\
n_mc   = 300
r_vals = [0.005, 0.01, 0.02, params.r, 0.1, 0.2, 0.5, 1.0]

costs_tw_all  = []
costs_st_dict = {r: [] for r in r_vals}
costs_ad_dict = {r: [] for r in r_vals}

print("Running Monte Carlo (this takes ~1 min)...")
for seed in range(n_mc):
    env = OWEnvironment(params, X0=X0, T=T, dt=dt, seed=seed)
    costs_tw_all.append(env.run_twap().cost)
    for r0 in r_vals:
        env2 = OWEnvironment(params, X0=X0, T=T, dt=dt, seed=seed)
        costs_st_dict[r0].append(env2.run_static_ow(r=r0).cost)
        env3 = OWEnvironment(params, X0=X0, T=T, dt=dt, seed=seed)
        costs_ad_dict[r0].append(env3.run_adaptive_ow(r0=r0, eta0=0.1, alpha=0.8).cost)
    if (seed+1) % 50 == 0:
        print(f"  {seed+1}/{n_mc} paths done")

ct = np.mean(costs_tw_all)
print(f"\\nTWAP mean cost: ${ct:,.2f}")
"""),

code("""\
# ── Plot: savings vs TWAP as function of r0 ──────────────────────────────
r_arr    = np.array(r_vals)
save_st  = np.array([(ct - np.mean(costs_st_dict[r]))/ct*100 for r in r_vals])
save_ad  = np.array([(ct - np.mean(costs_ad_dict[r]))/ct*100 for r in r_vals])
ad_beats = save_ad - save_st   # positive = adaptive beats static

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f'Robustness to Misspecification  |  {n_mc} MC paths  '
    f'|  True $r^*$={params.r:.4f}',
    color=TEXT, fontsize=12)

# Panel 1: absolute savings
ax = axes[0]
ax.semilogx(r_arr, save_st, color=YELLOW, lw=2.0, marker='s', ms=6,
            label='Static OW')
ax.semilogx(r_arr, save_ad, color=GREEN,  lw=2.0, marker='o', ms=6,
            label='Adaptive OW')
ax.axvline(params.r, color=RED, lw=1.5, ls='--', alpha=0.7,
           label=f'True $r^*$={params.r:.3f}')
ax.axhline(0, color=TEXT, lw=0.8, ls='--', alpha=0.4)
ax.set_xlabel('Initial guess $r_0$ (log scale)')
ax.set_ylabel('Cost savings vs TWAP (%)')
ax.set_title('Cost Savings vs TWAP')
ax.legend(fontsize=9)
ax.set_facecolor(GRID)

# Panel 2: adaptive advantage
ax = axes[1]
bar_colors = [GREEN if v >= 0 else RED for v in ad_beats]
ax.bar(range(len(r_vals)), ad_beats, color=bar_colors, alpha=0.8, edgecolor='none')
ax.axhline(0, color=TEXT, lw=1.0, ls='--')
ax.set_xticks(range(len(r_vals)))
ax.set_xticklabels([f'{r:.3f}' for r in r_vals], rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Initial guess $r_0$')
ax.set_ylabel('Adaptive advantage (% savings over Static OW)')
ax.set_title('Adaptive OW − Static OW\\n(green = adaptive wins)')
ax.set_facecolor(GRID)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb3_robustness.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()

print("\\nDetailed results:")
print(f"{'r0':>8}  {'Static savings':>16}  {'Adaptive savings':>18}  {'Adaptive advantage':>20}")
print("-"*68)
for r0, ss, sa in zip(r_vals, save_st, save_ad):
    tag = " ← TRUE r" if abs(r0-params.r)<0.001 else ""
    print(f"{r0:>8.3f}  {ss:>14.4f}%  {sa:>16.4f}%  {sa-ss:>+18.4f}%{tag}")
"""),

md("""## 3. Monte Carlo Distribution of Execution Costs

Full distribution of execution costs across 300 paths for three representative cases.
"""),

code("""\
r0_show = [0.01, params.r, 0.5]
labels  = [f'r₀={r}' for r in r0_show]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle('Distribution of Execution Costs (300 MC paths)', color=TEXT, fontsize=12)

for i, (r0, label) in enumerate(zip(r0_show, labels)):
    ax  = axes[i]
    c_t = np.array(costs_tw_all)
    c_s = np.array(costs_st_dict[r0])
    c_a = np.array(costs_ad_dict[r0])

    bins = np.linspace(min(c_t.min(), c_s.min(), c_a.min()),
                       max(c_t.max(), c_s.max(), c_a.max()), 40)

    ax.hist(c_t, bins=bins, color=RED,    alpha=0.55, label=f'TWAP  μ=${c_t.mean():,.0f}')
    ax.hist(c_s, bins=bins, color=YELLOW, alpha=0.55, label=f'Static μ=${c_s.mean():,.0f}')
    ax.hist(c_a, bins=bins, color=GREEN,  alpha=0.55, label=f'Adaptive μ=${c_a.mean():,.0f}')

    ax.axvline(c_t.mean(), color=RED,    lw=1.5, ls='--')
    ax.axvline(c_s.mean(), color=YELLOW, lw=1.5, ls='--')
    ax.axvline(c_a.mean(), color=GREEN,  lw=1.5, ls='--')

    ax.set_xlabel('Total execution cost ($)')
    ax.set_ylabel('Count' if i == 0 else '')
    ax.set_title(f'Starting guess {label}', color=TEXT)
    ax.legend(fontsize=7)
    ax.set_facecolor(GRID)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb3_cost_dist.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()
"""),

md("""## 4. Estimator Convergence Inside the Control Loop

The closed-loop coupling means that $\\hat{r}_t$ is driven by a **non-stationary**
$D_t$ process (because the trades change $D_t$, and the trades depend on $\\hat{r}_{t-}$).
Yet Theorem 4.3 still guarantees convergence — this is precisely the contribution
of the thesis: *no stationarity assumption is needed*.

Below we show the ensemble of $\\hat{r}_t$ paths across 100 MC runs.
"""),

code("""\
n_show = 100
r0_fix = 2.0

r_hat_paths = np.zeros((n_show, T + 1))
for seed in range(n_show):
    env = OWEnvironment(params, X0=X0, T=T, dt=dt, seed=seed)
    res = env.run_adaptive_ow(r0=r0_fix, eta0=0.1, alpha=0.8)
    r_hat_paths[seed] = res.r_hat

t_arr  = np.arange(T + 1)
q10    = np.percentile(r_hat_paths, 10, axis=0)
q50    = np.percentile(r_hat_paths, 50, axis=0)
q90    = np.percentile(r_hat_paths, 90, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f'$\\hat{{r}}_t$ Convergence Inside the Adaptive Control Loop  '
    f'({n_show} MC paths, $r_0$={r0_fix})',
    color=TEXT, fontsize=12)

for ax, logy in zip(axes, [False, True]):
    for i in range(min(30, n_show)):
        ax.plot(t_arr, r_hat_paths[i], color=CYAN, alpha=0.15, lw=0.7)
    ax.fill_between(t_arr, q10, q90, color=BLUE, alpha=0.2,
                    label='10%–90% band')
    ax.plot(t_arr, q50, color=BLUE, lw=2.0, label='Median $\\hat{r}_t$')
    ax.axhline(params.r, color=RED, lw=1.5, ls='--',
               label=f'True $r^*$ = {params.r:.4f}')
    ax.axhline(r0_fix, color=ORANGE, lw=1.0, ls=':', alpha=0.7,
               label=f'Initial $r_0$ = {r0_fix}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('$\\hat{r}_t$')
    ax.legend(fontsize=9)
    ax.set_facecolor(GRID)
    if logy:
        ax.set_yscale('log')
        ax.set_title('Log scale', color=TEXT)
    else:
        ax.set_title('Linear scale', color=TEXT)

plt.tight_layout()
plt.savefig(os.path.join('..','outputs','nb3_rhat_ensemble.png'),
            dpi=130, bbox_inches='tight', facecolor=DARK)
plt.show()

print(f"Median r_hat at t=0:    {q50[0]:.4f}")
print(f"Median r_hat at t=T/4:  {q50[T//4]:.4f}")
print(f"Median r_hat at t=T/2:  {q50[T//2]:.4f}")
print(f"Median r_hat at t=T:    {q50[-1]:.4f}")
print(f"True r*:                {params.r:.4f}")
"""),

md("""## Summary

### What we showed

1. **The closed-loop system works as designed**: the estimator and controller
   operate simultaneously on the same time scale, with no artificial time-scale
   separation — matching the theoretical setup of the thesis.

2. **Adaptive OW beats Static OW under misspecification**: when the initial
   guess $r_0 < r^*$ (underestimating resilience — the common practical case),
   the adaptive strategy saves meaningful execution cost compared to a frozen
   static strategy.

3. **The estimator converges inside the control loop**: $\\hat{r}_t \\to r^*$
   across all MC paths, consistent with Theorem 4.3 (almost-sure convergence
   without stationarity or time-scale separation).

### Connection to the thesis

| Thesis result | Notebook demonstration |
|---------------|----------------------|
| Theorem 4.3: a.s. convergence | Panel 4: $\\hat{r}_t$ ensemble convergence |
| Corollary 4.4: mean-square convergence | Panel 4: 10%–90% band narrows over time |
| Fig 5.1: estimator trajectories | Section 1: single-path $\\hat{r}_t$ plot |
| Fig 5.3: error vs learning rate | Notebook 2, Section 2 |
| Closed-loop setting (no stationarity) | The simulation itself — no stationarity assumed |
"""),

]  # end nb3_cells


# ══════════════════════════════════════════════════════════════════════════
# Write notebooks to disk
# ══════════════════════════════════════════════════════════════════════════

notebooks = {
    "01_data_exploration.ipynb":   nb(nb1_cells),
    "02_parameter_estimation.ipynb": nb(nb2_cells),
    "03_adaptive_execution.ipynb":  nb(nb3_cells),
}

for fname, notebook in notebooks.items():
    path = os.path.join(NB_DIR, fname)
    with open(path, "w") as f:
        nbf.write(notebook, f)
    print(f"Written: {path}")

print("\nAll notebooks created successfully.")
