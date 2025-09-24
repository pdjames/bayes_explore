# -*- coding: utf-8 -*-
# Hierarchical Normal with unknown σ² per client–tier (NIG) + Multi-layer priors
# Rows = tiers; Cols = groups. Prior dashed; No-pool dotted; Hier solid.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"


# ---------------------------------------------------------------------
# Distribution container (normal used for plotting)
# ---------------------------------------------------------------------
@dataclass
class Dist:
    kind: str
    mu: Optional[float] = None
    sigma: Optional[float] = None
    var: Optional[float] = None
    # Future-ready fields
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
# Group-tier pooling container with:
#  - client expected-precision pooling (with order-invariant ESS cap)
#  - multiple prior layers (precision-summed with weights λ_k)
# ---------------------------------------------------------------------
@dataclass
class PriorNIGPool:
    """
    Posterior for μ_{g,t} with:
      - pooled client weights (expected precisions) + order-invariant capping
      - multiple Normal prior layers combined by precision addition
    """
    s0_2: float               # base prior variance for μ_{g,t}
    m0: float = 0.0           # base prior mean
    M_cap: Optional[float] = None
    cap_mode: str = "hard_scale"  # "hard_scale", "soft_saturate", or None
    extra_layers: Optional[List[Dict[str, float]]] = None
    # Each layer: {"mu": float, "var": float, "lambda": float}

    # RAW accumulators from clients (before capping)
    raw_sum_w: float = 0.0
    raw_sum_wy: float = 0.0

    # ----- add client contribution (already a weighted mean with its own weight) -----
    def add_weighted_member(self, mu_i: float, weight_i: float) -> None:
        if weight_i <= 0:
            return
        self.raw_sum_w += weight_i
        self.raw_sum_wy += weight_i * mu_i

    # ----- internal helpers for capping and layered priors -----
    def _effective_scale(self) -> float:
        if (self.M_cap is None) or (self.raw_sum_w <= 0):
            return 1.0
        if self.cap_mode == "hard_scale":
            return min(1.0, self.M_cap / self.raw_sum_w)
        if self.cap_mode == "soft_saturate":
            eff = self.M_cap * (1.0 - np.exp(- self.raw_sum_w / self.M_cap))
            return float(eff / self.raw_sum_w) if self.raw_sum_w > 0 else 1.0
        return 1.0

    def _prior_prec_and_num(self) -> Tuple[float, float]:
        # base layer
        prior_prec = 1.0 / self.s0_2
        prior_num  = (1.0 / self.s0_2) * self.m0
        # extra layers: add λ_k / v_k to precision, and λ_k m_k / v_k to numerator
        if self.extra_layers:
            for L in self.extra_layers:
                lam = float(L.get("lambda", 1.0))
                v   = float(L["var"])
                m   = float(L["mu"])
                prior_prec += lam * (1.0 / v)
                prior_num  += lam * (m / v)
        return prior_prec, prior_num

    # ----- posterior moments for μ_{g,t} -----
    @property
    def mu(self) -> float:
        k = self._effective_scale()
        eff_w  = k * self.raw_sum_w
        eff_wy = k * self.raw_sum_wy
        prior_prec, prior_num = self._prior_prec_and_num()
        tot_prec = eff_w + prior_prec
        if tot_prec <= 0:
            return self.m0
        return (eff_wy + prior_num) / tot_prec

    @property
    def var(self) -> float:
        k = self._effective_scale()
        eff_w = k * self.raw_sum_w
        prior_prec, _ = self._prior_prec_and_num()
        tot_prec = eff_w + prior_prec
        return 1.0 / tot_prec if tot_prec > 0 else self.s0_2

    def dist(self) -> Dist:
        return Dist.normal(self.mu, self.var)


# ---------------------------------------------------------------------
# Configs — choose one via CONFIG_INDEX
# ---------------------------------------------------------------------
CONFIGS = [
    {
        "name": "Weak",
        "a": 0.5,          # discount applied in client->group weights
        "kappa0": 1.0,     # client mean prior ESS
        "alpha0": 2.0,     # IG prior shape for σ²
        "beta0": 1.0,      # IG prior scale for σ²
        "s0_2": 25.0,      # base group prior variance
        "M_cap": 5.0,      # cap on pooled expected precision from clients
        "cap_mode": "hard_scale",
        "group_passes": 2
    },
    {
        "name": "Medium",
        "a": 0.7, "kappa0": 2.0, "alpha0": 2.0, "beta0": 1.0,
        "s0_2": 10.0, "M_cap": 8.0, "cap_mode": "hard_scale", "group_passes": 2
    },
    {
        "name": "Strong",
        "a": 1.0, "kappa0": 4.0, "alpha0": 2.0, "beta0": 1.0,
        "s0_2": 4.0, "M_cap": 10.0, "cap_mode": "soft_saturate", "group_passes": 2
    },
]
CONFIG_INDEX = 1  # pick 0/1/2
M0 = 0.0          # base/global prior mean for group-tier means


# ---------------------------------------------------------------------
# Simulation (returns n, ybar, sse per client–tier)
# ---------------------------------------------------------------------
def simulate_data(
    rng: np.random.Generator,
    n_clients: int,
    n_tiers: int,
    groups_map: Dict[int, str],
    client_to_group: Dict[int, int],
    sigma_true: float = 1.0
) -> Tuple[
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], Dict[str, float]],
    Dict[Tuple[str, int], float]
]:
    """
    Returns:
      theta_true[(i,t)]
      obs_stats[(i,t)] -> dict(n, ybar, sse)
      mu_true[(gname, t)] (for reference)
    """
    mu_true = {
        ("A", 0): 0.0, ("A", 1): 0.5, ("A", 2): -0.5, ("A", 3): 0.3,
        ("B", 0): 0.4, ("B", 1): 1.0, ("B", 2): -1.0, ("B", 3): -0.1,
    }

    theta_true: Dict[Tuple[int, int], float] = {}
    obs_stats: Dict[Tuple[int, int], Dict[str, float]] = {}

    for i in range(n_clients):
        gname = groups_map[client_to_group[i]]
        for t in range(n_tiers):
            theta_true[(i, t)] = rng.normal(mu_true[(gname, t)], 4.0)  # between-client variety
            n = int(rng.choice([0,1,2,3,5,10,20,40], p=[0.1,0.05, 0.05, 0.15,0.2,0.2,0.15,0.1]))
            if n > 0:
                y = rng.normal(theta_true[(i, t)], sigma_true, size=n)
                ybar = float(np.mean(y))
                sse = float(np.sum((y - ybar) ** 2))  # sum of squared deviations from sample mean
                obs_stats[(i, t)] = dict(n=n, ybar=ybar, sse=sse)
            else:
                obs_stats[(i, t)] = dict(n=0, ybar=None, sse=0.0)

    return theta_true, obs_stats, mu_true


# ---------------------------------------------------------------------
# NIG helper (client-tier update from sufficient stats)
# ---------------------------------------------------------------------
def nig_posterior_from_stats(
    ybar: float, n: int, sse: float,
    mu0: float, kappa0: float, alpha0: float, beta0: float
) -> Tuple[float, float, float, float]:
    if n <= 0:
        return mu0, kappa0, alpha0, beta0
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * ybar) / kappa_n
    alpha_n = alpha0 + 0.5 * n
    beta_n = beta0 + 0.5 * (sse + (kappa0 * n / (kappa0 + n)) * (ybar - mu0) ** 2)
    return mu_n, kappa_n, alpha_n, beta_n


# ---------------------------------------------------------------------
# Build extra layers for priors (global/segment/kernel examples)
# ---------------------------------------------------------------------
def build_extra_layers(
    tiers: List[int],
    groups: List[str],
    *,
    # Example layer 1: GLOBAL — same across groups, varies by tier
    m_global: Dict[int, float],
    v_global: Dict[int, float],
    lam_global: float = 0.5,
    # Example layer 2: SEGMENT — same across tiers, varies by group
    m_segment: Dict[str, float] = None,
    v_segment: Dict[str, float] = None,
    lam_segment: float = 0.7,
    # Example layer 3: KERNEL — optional per (g,t)
    m_kernel: Dict[Tuple[str, int], float] = None,
    v_kernel: Dict[Tuple[str, int], float] = None,
    lam_kernel: float = 0.3
) -> Dict[Tuple[str, int], List[Dict[str, float]]]:
    """
    Returns a dict mapping (gname,t) -> list of layer dicts {mu,var,lambda}.
    You can pass any subset; missing ones are ignored.
    """
    out: Dict[Tuple[str, int], List[Dict[str, float]]] = {}
    for g in groups:
        for t in tiers:
            layers: List[Dict[str, float]] = []

            # Global layer (by tier)
            if (m_global is not None) and (v_global is not None) and (t in m_global):
                layers.append({"mu": float(m_global[t]), "var": float(v_global[t]), "lambda": float(lam_global)})

            # Segment layer (by group)
            if (m_segment is not None) and (v_segment is not None) and (g in m_segment):
                layers.append({"mu": float(m_segment[g]), "var": float(v_segment[g]), "lambda": float(lam_segment)})

            # Kernel/neighborhood layer (by group × tier)
            if (m_kernel is not None) and (v_kernel is not None) and ((g, t) in m_kernel):
                layers.append({"mu": float(m_kernel[(g, t)]), "var": float(v_kernel[(g, t)]), "lambda": float(lam_kernel)})

            out[(g, t)] = layers
    return out


# ---------------------------------------------------------------------
# Group posteriors μ_{g,t} via expected precision pooling + layers
# ---------------------------------------------------------------------
def compute_group_posteriors_nig(
    obs_stats: Dict[Tuple[int, int], Dict[str, float]],
    groups_map: Dict[int, str],
    client_to_group: Dict[int, int],
    n_clients: int,
    n_tiers: int,
    *,
    a: float, kappa0: float, alpha0: float, beta0: float,
    s0_2: float, m0: float,
    M_cap: Optional[float], cap_mode: str,
    passes: int,
    extra_layers_by_gt: Dict[Tuple[str, int], List[Dict[str, float]]] = None
) -> Dict[Tuple[str, int], PriorNIGPool]:
    """
    Builds one PriorNIGPool per (group,tier). Does 1–2 EM-style passes:
    pass 0: start μ_{g,t}=layered-prior mean; compute client NIG; pool weights w_i = a * E[prec θ_i] = a*kappa_n*alpha_n/beta_n.
    pass 1: recompute using μ_{g,t} from pass 0 for a small refinement.
    """
    group_names = sorted(set(groups_map.values()))
    priors: Dict[Tuple[str, int], PriorNIGPool] = {}

    # instantiate priors with layers
    for gname in group_names:
        for t in range(n_tiers):
            layers = extra_layers_by_gt.get((gname, t), []) if extra_layers_by_gt else None
            priors[(gname, t)] = PriorNIGPool(
                s0_2=s0_2, m0=m0, M_cap=M_cap, cap_mode=cap_mode, extra_layers=layers
            )

    # EM-style refinement
    for _ in range(max(1, passes)):
        # reset raw client sums each pass
        for key in priors:
            priors[key].raw_sum_w = 0.0
            priors[key].raw_sum_wy = 0.0

        for gname in group_names:
            clients_g = [i for i, gidx in client_to_group.items() if groups_map[gidx] == gname]
            for t in range(n_tiers):
                mu_gt = priors[(gname, t)].mu  # current group mean (includes layers)
                for i in clients_g:
                    stats = obs_stats[(i, t)]
                    n, ybar, sse = stats["n"], stats["ybar"], stats["sse"]
                    if n <= 0 or ybar is None:
                        continue
                    mu_n, kappa_n, alpha_n, beta_n = nig_posterior_from_stats(
                        ybar, n, sse, mu0=mu_gt, kappa0=kappa0, alpha0=alpha0, beta0=beta0
                    )
                    weight_i = a * (kappa_n * alpha_n / beta_n)  # E[precision of θ_i]
                    priors[(gname, t)].add_weighted_member(mu_n, weight_i)

    return priors


# ---------------------------------------------------------------------
# Client posteriors θ_{i,t} + no-pool & prior curves (Normal approximations)
# ---------------------------------------------------------------------
def compute_client_posteriors_nig(
    obs_stats: Dict[Tuple[int, int], Dict[str, float]],
    group_priors: Dict[Tuple[str, int], PriorNIGPool],
    groups_map: Dict[int, str],
    client_to_group: Dict[int, int],
    n_clients: int,
    n_tiers: int,
    *,
    kappa0: float, alpha0: float, beta0: float
) -> pd.DataFrame:
    rows = []
    for i in range(n_clients):
        gname = groups_map[client_to_group[i]]
        for t in range(n_tiers):
            stats = obs_stats[(i, t)]
            n, ybar, sse = stats["n"], stats["ybar"], stats["sse"]
            g_mu = group_priors[(gname, t)].mu

            # Client NIG posterior given current μ_{g,t}
            mu_n, kappa_n, alpha_n, beta_n = nig_posterior_from_stats(
                ybar if ybar is not None else 0.0, n, sse if sse is not None else 0.0,
                mu0=g_mu, kappa0=kappa0, alpha0=alpha0, beta0=beta0
            )

            # Normal approximations for plotting
            if n > 0:
                sigma2_hat = beta_n / (alpha_n - 1) if alpha_n > 1 else beta_n / max(alpha_n, 1e-9)
                var_theta = sigma2_hat / max(kappa_n, 1e-9)
                post_hier = Dist.normal(mu_n, var_theta)

                if n > 1:
                    s2_sample = sse / (n - 1)
                    var_np = s2_sample / n
                else:
                    alpha_np = alpha0 + 0.5 * n
                    beta_np = beta0 + 0.5 * sse
                    sigma2_np = beta_np / (alpha_np - 1) if alpha_np > 1 else beta_np / max(alpha_np, 1e-9)
                    var_np = sigma2_np / max(n, 1)
                no_pool = Dist.normal(ybar, var_np)
            else:
                sigma2_prior = beta0 / (alpha0 - 1) if alpha0 > 1 else beta0 / max(alpha0, 1e-9)
                var_theta = sigma2_prior / max(kappa0, 1e-9) if kappa0 > 0 else 1e6
                post_hier = Dist.normal(g_mu, var_theta)
                no_pool = None

            # Prior curve shown (Normal approx for θ prior under NIG, centered at μ_{g,t})
            sigma2_prior = beta0 / (alpha0 - 1) if alpha0 > 1 else beta0 / max(alpha0, 1e-9)
            prior_var = sigma2_prior / max(kappa0, 1e-9) if kappa0 > 0 else 1e6
            prior = Dist.normal(g_mu, prior_var)

            rows.append({
                "client": i, "group": gname, "tier": t, "n": int(n),
                "prior": prior, "post_hier": post_hier, "no_pool": no_pool
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting: rows = tiers, cols = groups (prior dashed; no-pool dotted; hier solid)
# ---------------------------------------------------------------------
def plot_tiers_groups(
    df: pd.DataFrame,
    title_suffix: str,
    client_colors: Optional[List[str]] = None
) -> None:
    tiers = sorted(df["tier"].unique())
    groups = sorted(df["group"].unique())
    n_tiers, n_groups = len(tiers), len(groups)

    if client_colors is None:
        client_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                         "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

    # Global x-range
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

            # Representative prior (median over clients in group–tier)
            mu_med = float(np.median([d.mu for _, d in df_gt[["prior"]].itertuples()]))
            var_med = float(np.median([d.var for _, d in df_gt[["prior"]].itertuples()]))
            prior_rep = Dist.normal(mu_med, var_med)

            # Prior dashed
            fig.add_trace(
                go.Scatter(
                    x=xs, y=prior_rep.pdf(xs), mode="lines",
                    line=dict(dash="dash", width=2, color="#7f7f7f"),
                    name=f"Prior μ[{gname},{t}]",
                    legendgroup=f"PRIOR-{gname}",
                    hovertemplate=f"PRIOR [{gname}, tier {t}]<br>mean={mu_med:.3f}, var={var_med:.3f}<extra></extra>",
                    showlegend=(r_idx == 1 and c_idx == 1)
                ),
                row=r_idx, col=c_idx
            )

            # Clients
            for i in sorted(df_gt["client"].unique()):
                row = df_gt[df_gt["client"] == i].iloc[0]
                color = client_colors[i % len(client_colors)]

                # Hier (solid)
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=row["post_hier"].pdf(xs), mode="lines",
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

                # No-pool (dotted)
                if row["no_pool"] is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=xs, y=row["no_pool"].pdf(xs), mode="lines",
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
        title=f"NIG Hierarchy + Multi-layer Priors — {title_suffix}<br>"
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
    rng = np.random.default_rng(23)

    # Layout
    N_CLIENTS = 5
    N_TIERS = 4
    GROUPS_MAP = {0: "A", 1: "B"}
    CLIENT_TO_GROUP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}

    tiers = list(range(N_TIERS))
    groups = sorted(set(GROUPS_MAP.values()))

    # Config select
    cfg = CONFIGS[CONFIG_INDEX]
    a        = cfg["a"]
    kappa0   = cfg["kappa0"]
    alpha0   = cfg["alpha0"]
    beta0    = cfg["beta0"]
    s0_2     = cfg["s0_2"]
    M_cap    = cfg["M_cap"]
    cap_mode = cfg["cap_mode"]
    passes   = cfg["group_passes"]

    # Simulate
    theta_true, obs_stats, mu_true = simulate_data(
        rng, N_CLIENTS, N_TIERS, GROUPS_MAP, CLIENT_TO_GROUP, sigma_true=1.0
    )

    # ----- Build multi-layer priors (EXAMPLE LAYERS) -----
    # Global layer (by tier)
    m_global = {t: 0.0 for t in tiers}           # global baseline means per tier
    v_global = {t: 25.0 for t in tiers}          # diffuse
    lam_global = 0.5

    # Segment layer (by group)
    m_segment = {"A": 0.1, "B": -0.1}            # mild offsets per group
    v_segment = {"A": 9.0,  "B": 9.0}
    lam_segment = 0.7

    # (Optional) Kernel layer (by group×tier); here we omit for brevity
    m_kernel = None
    v_kernel = None
    lam_kernel = 0.3

    extra_layers_by_gt = build_extra_layers(
        tiers, groups,
        m_global=m_global, v_global=v_global, lam_global=lam_global,
        m_segment=m_segment, v_segment=v_segment, lam_segment=lam_segment,
        m_kernel=m_kernel, v_kernel=v_kernel, lam_kernel=lam_kernel
    )

    # Group posteriors (μ_{g,t}) with layers + capping
    group_priors = compute_group_posteriors_nig(
        obs_stats, GROUPS_MAP, CLIENT_TO_GROUP, N_CLIENTS, N_TIERS,
        a=a, kappa0=kappa0, alpha0=alpha0, beta0=beta0,
        s0_2=s0_2, m0=M0, M_cap=M_cap, cap_mode=cap_mode, passes=passes,
        extra_layers_by_gt=extra_layers_by_gt
    )

    # Client posteriors (θ_{i,t})
    df = compute_client_posteriors_nig(
        obs_stats, group_priors, GROUPS_MAP, CLIENT_TO_GROUP, N_CLIENTS, N_TIERS,
        kappa0=kappa0, alpha0=alpha0, beta0=beta0
    )

    # Plot
    title_suffix = (f"Config={cfg['name']} (a={a}, κ₀={kappa0}, α₀={alpha0}, β₀={beta0}, "
                    f"s0²={s0_2}, M_cap={M_cap}, cap={cap_mode}, passes={passes}) "
                    f"+ Layers: global(λ={lam_global}), segment(λ={lam_segment})")
    plot_tiers_groups(df, title_suffix)