# -*- coding: utf-8 -*-
# Closed-form hierarchical Normal bandit demo (modular + dataclass priors)
# Rows = tiers; Cols = groups. Prior dashed; No-pool dotted; Hier solid.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"


# ---------------------------------------------------------------------
# Distribution container (extensible)
# ---------------------------------------------------------------------
@dataclass
class Dist:
    kind: str
    mu: Optional[float] = None
    sigma: Optional[float] = None
    var: Optional[float] = None
    # Future-ready fields for other families
    alpha: Optional[float] = None
    beta: Optional[float] = None
    nu: Optional[float] = None
    tau: Optional[float] = None

    @staticmethod
    def normal(mu: float, var: float) -> "Dist":
        return Dist(kind="normal", mu=mu, var=var, sigma=np.sqrt(var))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.kind != "normal":
            raise NotImplementedError(f"pdf not implemented for kind={self.kind}")
        v = self.var if self.var is not None else (self.sigma ** 2)
        return (1.0 / np.sqrt(2 * np.pi * v)) * np.exp(-(x - self.mu) ** 2 / (2 * v))


# ---------------------------------------------------------------------
# Hierarchical Prior for group–tier mean μ_{g,t}
# ---------------------------------------------------------------------
@dataclass
class Prior:
    """
    Posterior for μ_{g,t} with ESS capping that never discards observations.
    We accumulate RAW precision/sums, then compute EFFECTIVE values via a cap mode.
    """
    a: float                 # discount factor on hierarchical precision
    tau2: float              # between-client variance
    s0_2: float              # prior variance for μ_{g,t}
    m0: float = 0.0          # prior mean for μ_{g,t}
    M_cap: Optional[float] = None
    use_sem: bool = True     # weight ybars by 1/(tau2 + sigma2/n) if True
    cap_mode: str = "soft_saturate"  # "hard_scale", "soft_saturate", or None

    # internal RAW accumulators (sum of raw weights and raw weighted response)
    raw_sum_w: float = 0.0
    raw_sum_wy: float = 0.0

    def add_member_observation(self, ybar: Optional[float], n: int, sigma2: float) -> None:
        """Add one client's summary (ybar, n). n can be 0; ybar may be None when n=0."""
        if n is None or n <= 0 or ybar is None:
            return
        if self.use_sem:
            w_i = self.a / (self.tau2 + sigma2 / n)   # SEM-weighted precision
        else:
            w_i = self.a / self.tau2                  # equal per-mean precision
        self.raw_sum_w  += w_i
        self.raw_sum_wy += w_i * ybar

    # ----- effective (capped) aggregates -----
    def _effective_scale(self) -> float:
        """
        Return the factor k in (0,1] such that:
          eff_sum_w  = k * raw_sum_w
          eff_sum_wy = k * raw_sum_wy
        according to the selected cap mode.
        """
        if (self.M_cap is None) or (self.raw_sum_w <= 0):
            return 1.0

        if self.cap_mode == "hard_scale":
            # Uniformly rescale to fit exactly within M_cap when exceeded (order-invariant).
            return min(1.0, self.M_cap / self.raw_sum_w)

        if self.cap_mode == "soft_saturate":
            # Smooth saturation: eff = M_cap * (1 - exp(-raw/M_cap))
            # Equivalent uniform factor k applied to both sum_w and sum_wy.
            eff = self.M_cap * (1.0 - np.exp(- self.raw_sum_w / self.M_cap))
            return float(eff / self.raw_sum_w) if self.raw_sum_w > 0 else 1.0

        # No cap
        return 1.0

    @property
    def _eff_sum_w(self) -> float:
        return self._effective_scale() * self.raw_sum_w

    @property
    def _eff_sum_wy(self) -> float:
        return self._effective_scale() * self.raw_sum_wy

    # ----- posterior moments for μ_{g,t} -----
    @property
    def mu(self) -> float:
        prec = self._eff_sum_w + 1.0 / self.s0_2
        if prec <= 0:
            return self.m0
        return (self._eff_sum_wy + (1.0 / self.s0_2) * self.m0) / prec

    @property
    def var(self) -> float:
        prec = self._eff_sum_w + 1.0 / self.s0_2
        return 1.0 / prec if prec > 0 else self.s0_2

    def dist(self) -> "Dist":
        return Dist.normal(self.mu, self.var)


# ---------------------------------------------------------------------
# Configs (choose one via CONFIG_INDEX)
# ---------------------------------------------------------------------
CONFIGS = [
    {"name": "Weak",   "a": 0.40, "tau2": 6.0, "s0_2": 25.0, "M_cap": None},
    {"name": "Medium", "a": 0.70, "tau2": 4.0, "s0_2": 10.0, "M_cap": None},
    {"name": "Strong", "a": 1.00, "tau2": 2.0, "s0_2":  4.0, "M_cap": None},
]
CONFIG_INDEX = 2  # pick 0/1/2

# Likelihood variance (known)
SIGMA2 = 1.0
M0 = 0.0

# SEM toggles
USE_SEM_IN_GROUP = True          # recommended: use 1/(tau2 + sigma2/n) to weight ybars in group pooling
USE_SEM_IN_CLIENT_PRIOR = False  # experimental: use 1/(tau2 + sigma2/n) in client prior precision


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
def simulate_data(
    rng: np.random.Generator,
    n_clients: int,
    n_tiers: int,
    groups_map: Dict[int, str],
    client_to_group: Dict[int, int],
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Tuple[Optional[float], int]], Dict[Tuple[str, int], float]]:
    """
    Returns:
      theta_true[(i,t)]
      obs[(i,t)] -> (ybar, n) with possible n=0 => (None, 0)
      mu_true[(gname, t)]
    """
    # True group-tier means (for simulation only)
    mu_true = {
        ("A", 0): 0.0, ("A", 1): 0.5, ("A", 2): -0.5, ("A", 3): 0.3,
        ("B", 0): 0.4, ("B", 1): 1.0, ("B", 2): -1.0, ("B", 3): -0.1,
    }

    theta_true: Dict[Tuple[int, int], float] = {}
    obs: Dict[Tuple[int, int], Tuple[Optional[float], int]] = {}

    for i in range(n_clients):
        gname = groups_map[client_to_group[i]]
        for t in range(n_tiers):
            theta_true[(i, t)] = rng.normal(mu_true[(gname, t)], 4.0)  # SD=1 for variety
            n = int(rng.choice([0,1,2,3,5,10,20,40], p=[0.1,0.05, 0.05, 0.15,0.2,0.2,0.15,0.1]))
            if n > 0:
                y = rng.normal(theta_true[(i, t)], np.sqrt(SIGMA2), size=n)
                obs[(i, t)] = (float(np.mean(y)), n)
            else:
                obs[(i, t)] = (None, 0)
    return theta_true, obs, mu_true


# ---------------------------------------------------------------------
# Group posteriors μ_{g,t} using Prior dataclass
# ---------------------------------------------------------------------
def compute_group_posteriors(
    obs: Dict[Tuple[int, int], Tuple[Optional[float], int]],
    groups_map: Dict[int, str],
    client_to_group: Dict[int, int],
    n_clients: int,
    n_tiers: int,
    *,
    a: float, tau2: float, s0_2: float, m0: float,
    M_cap: Optional[float],
    use_sem: bool
) -> Dict[Tuple[str, int], Prior]:
    # Instantiate a Prior per (group, tier)
    priors: Dict[Tuple[str, int], Prior] = {}
    group_names = sorted(set(groups_map.values()))
    for gname in group_names:
        for t in range(n_tiers):
            priors[(gname, t)] = Prior(
                a=a, tau2=tau2, s0_2=s0_2, m0=m0, M_cap=M_cap, use_sem=use_sem
            )

    # Add each client's (ybar, n) to the corresponding group–tier prior
    for i in range(n_clients):
        gname = groups_map[client_to_group[i]]
        for t in range(n_tiers):
            ybar, n = obs[(i, t)]
            priors[(gname, t)].add_member_observation(ybar, n, SIGMA2)

    return priors


# ---------------------------------------------------------------------
# Client posteriors θ_{i,t}
# ---------------------------------------------------------------------
def compute_client_posteriors(
    obs: Dict[Tuple[int, int], Tuple[Optional[float], int]],
    group_priors: Dict[Tuple[str, int], Prior],
    groups_map: Dict[int, str],
    client_to_group: Dict[int, int],
    n_clients: int,
    n_tiers: int,
    *,
    a: float, tau2: float,
    use_sem_in_prior: bool
) -> pd.DataFrame:
    rows = []
    for i in range(n_clients):
        gname = groups_map[client_to_group[i]]
        for t in range(n_tiers):
            ybar, n = obs[(i, t)]
            g_prior: Prior = group_priors[(gname, t)]
            g_dist = g_prior.dist()  # μ_{g,t} | data

            # Prior precision for client (discounted)
            if (n is not None and n > 0 and use_sem_in_prior):
                prior_prec = a / (tau2 + SIGMA2 / n)          # experimental variant
            else:
                prior_prec = a / (tau2 + g_dist.var)          # recommended: includes μ uncertainty

            if n is not None and n > 0 and ybar is not None:
                like_prec = n / SIGMA2
                post_prec = like_prec + prior_prec
                v_it = 1.0 / post_prec
                m_it = v_it * (like_prec * ybar + prior_prec * g_dist.mu)
                # no-pool for plotting
                m_np, v_np = ybar, SIGMA2 / n
                prior_var_for_plot = 1.0 / prior_prec if prior_prec > 0 else 1e6
            else:
                # n=0: posterior equals the (discounted) prior centered at μ_{g,t}
                v_it = 1.0 / prior_prec if prior_prec > 0 else 1e6
                m_it = g_dist.mu
                m_np, v_np = None, None
                prior_var_for_plot = v_it

            rows.append({
                "client": i,
                "group": gname,
                "tier": t,
                "n": int(n) if n else 0,
                "prior": Dist.normal(g_dist.mu, prior_var_for_plot),
                "post_hier": Dist.normal(m_it, v_it),
                "no_pool": Dist.normal(m_np, v_np) if (m_np is not None) else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting: rows = tiers, cols = groups
# ---------------------------------------------------------------------
def plot_tiers_groups(
    df: pd.DataFrame,
    title_suffix: str,
    client_colors: Optional[List[str]] = None
) -> None:
    tiers = sorted(df["tier"].unique())
    groups = sorted(df["group"].unique())
    n_tiers, n_groups = len(tiers), len(groups)

    # Global x-range across all curves
    means, sds = [], []
    for _, r in df.iterrows():
        means.append(r["post_hier"].mu); sds.append(np.sqrt(r["post_hier"].var))
        if r["no_pool"] is not None:
            means.append(r["no_pool"].mu); sds.append(np.sqrt(r["no_pool"].var))
        means.append(r["prior"].mu); sds.append(np.sqrt(r["prior"].var))
    x_min = float(min(means) - 4 * max(sds))
    x_max = float(max(means) + 4 * max(sds))
    xs = np.linspace(x_min, x_max, 900)

    fig = make_subplots(
        rows=n_tiers, cols=n_groups, shared_xaxes=True, shared_yaxes=False,
        subplot_titles=[f"Tier {t} • Group {g}" for t in tiers for g in groups]
    )

    for r_idx, t in enumerate(tiers, start=1):
        for c_idx, gname in enumerate(groups, start=1):
            df_gt = df[(df["tier"] == t) & (df["group"] == gname)]
            if df_gt.empty:
                continue

            # Representative group prior (median over clients in this group–tier)
            mu_med = float(np.median([d.mu for _, d in df_gt[["prior"]].itertuples()]))
            var_med = float(np.median([d.var for _, d in df_gt[["prior"]].itertuples()]))
            prior_rep = Dist.normal(mu_med, var_med)

            # Prior dashed (grey)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=prior_rep.pdf(xs),
                    mode="lines",
                    line=dict(dash="dash", width=3, color="#7f7f7f"),
                    name=f"Prior μ[{gname},{t}]",
                    legendgroup=f"PRIOR-{gname}",
                    hovertemplate=f"PRIOR [{gname}, tier {t}]<br>mean={mu_med:.3f}, var={var_med:.3f}<extra></extra>",
                    showlegend=(r_idx == 1 and c_idx == 1)
                ),
                row=r_idx, col=c_idx
            )

            # Clients: hier solid, no-pool dotted
            for i in sorted(df_gt["client"].unique()):
                row = df_gt[df_gt["client"] == i].iloc[0]
                color = client_colors[i % len(client_colors)]

                # Hier
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=row["post_hier"].pdf(xs),
                        mode="lines",
                        line=dict(dash="solid", width=2, color=color),
                        name=f"Client {i} hier",
                        legendgroup=f"C{i}",
                        hovertemplate=(
                            f"HIER • C{i} • G{row['group']} • tier {t} • n={row['n']}<br>"
                            f"mean={row['post_hier'].mu:.3f}, var={row['post_hier'].var:.3f}<extra></extra>"
                        ),
                        showlegend=(r_idx == 1 and c_idx == 1)
                    ),
                    row=r_idx, col=c_idx
                )

                # No-pool
                if row["no_pool"] is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=xs, y=row["no_pool"].pdf(xs),
                            mode="lines",
                            line=dict(dash="dot", width=2, color=color),
                            name=f"Client {i} no-pool",
                            legendgroup=f"C{i}",
                            hovertemplate=(
                                f"NO-POOL • C{i} • G{row['group']} • tier {t} • n={row['n']}<br>"
                                f"mean={row['no_pool'].mu:.3f}, var={row['no_pool'].var:.3f}<extra></extra>"
                            ),
                            showlegend=False
                        ),
                        row=r_idx, col=c_idx
                    )

    fig.update_layout(
        title=f"Closed-form Hierarchical Normal — {title_suffix}<br>"
              f"Prior dashed, No-pool dotted, Hier solid (rows=tiers, cols=groups)",
        height=260 * n_tiers,
        legend=dict(orientation="h", y=-0.15)
    )
    fig.update_xaxes(title_text="θ", row=n_tiers, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(13)

    # Layout
    N_CLIENTS = 7
    N_TIERS = 4
    GROUPS_MAP = {0: "A", 1: "B"}                 # group index -> label
    CLIENT_TO_GROUP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5:0, 6:1}  # client -> group index

    client_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Config select
    cfg = CONFIGS[CONFIG_INDEX]
    a, tau2, s0_2, M_cap = cfg["a"], cfg["tau2"], cfg["s0_2"], cfg["M_cap"]

    # Simulate
    theta_true, obs, mu_true = simulate_data(
        rng, N_CLIENTS, N_TIERS, GROUPS_MAP, CLIENT_TO_GROUP
    )

    # Group posteriors
    group_priors = compute_group_posteriors(
        obs, GROUPS_MAP, CLIENT_TO_GROUP, N_CLIENTS, N_TIERS,
        a=a, tau2=tau2, s0_2=s0_2, m0=M0, M_cap=M_cap, use_sem=USE_SEM_IN_GROUP
    )

    # Client posteriors
    df = compute_client_posteriors(
        obs, group_priors, GROUPS_MAP, CLIENT_TO_GROUP, N_CLIENTS, N_TIERS,
        a=a, tau2=tau2, use_sem_in_prior=USE_SEM_IN_CLIENT_PRIOR
    )

    # Plot
    sem_grp = "SEM-group" if USE_SEM_IN_GROUP else "equal-wt"
    sem_cli = "SEM-clientPrior" if USE_SEM_IN_CLIENT_PRIOR else "std-clientPrior"
    title_suffix = f"Config={cfg['name']} (a={a}, τ²={tau2}, s0²={s0_2}) • {sem_grp}, {sem_cli}"
    plot_tiers_groups(df, title_suffix, client_colors)