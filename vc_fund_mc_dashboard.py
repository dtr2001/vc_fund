"""
VC Fund Economics — Monte Carlo Dashboard

Each simulation independently draws company outcomes from the 5-bucket
probability distribution for every tranche.  Running many simulations
reveals the full distribution of fund-level LP returns, not just the
expected value.

Companion to vc_fund_blended_dashboard.py (expected-value benchmark).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="VC Fund — Monte Carlo",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .metric-card {
      background:#f7f8fa; border:1px solid #e0e3ea; border-radius:10px;
      padding:12px 18px; text-align:center;
      height:115px; display:flex; flex-direction:column;
      justify-content:center; overflow:hidden;
  }
  .metric-label { font-size:12px; color:#666; font-weight:600;
                  letter-spacing:0.05em; text-transform:uppercase; }
  .metric-value { font-size:26px; font-weight:700; color:#1a1a2e; margin:4px 0 2px; }
  .metric-sub   { font-size:11px; color:#999;
                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
</style>
""", unsafe_allow_html=True)

BUCKET_LABELS = ["Total loss (0×)", "Return capital (1×)", "Moderate (3×)",
                 "Strong (10×)", "Grand slam (30×)"]
BUCKET_MOICS  = np.array([0.0, 1.0, 3.0, 10.0, 30.0])
BUCKET_COLORS = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#2980b9"]

# ── IRR (Newton-Raphson) ──────────────────────────────────────────────────────
def _irr(cf: np.ndarray) -> float:
    if np.all(cf <= 0) or np.all(cf >= 0):
        return np.nan
    r = 0.15 if np.sum(cf) > 0 else -0.3
    t = np.arange(len(cf), dtype=float)
    for _ in range(500):
        if r <= -1.0:
            return np.nan
        pw   = (1 + r) ** t
        npv  = np.sum(cf / pw)
        dnpv = np.sum(-t * cf / ((1 + r) * pw))
        if abs(dnpv) < 1e-14:
            break
        r_new = r - npv / dnpv
        if not np.isfinite(r_new):
            break
        if abs(r_new - r) < 1e-7:
            r = r_new
            break
        r = r_new
    if not (np.isfinite(r) and r > -1.0):
        return np.nan
    # Verify r is actually a root — NR can converge to a stationary point of
    # NPV (where NPV' ≈ 0) rather than a zero, especially when fees extend past
    # the last exit and no real IRR exists.
    pw  = (1 + r) ** t
    npv = np.sum(cf / pw)
    return float(r) if abs(npv) <= 1e-4 * np.sum(np.abs(cf)) else np.nan

# ── Monte Carlo core ──────────────────────────────────────────────────────────
@st.cache_data
def run_mc(N, H, n_sims,
           p_loss_raw, p_zombie_raw, p_mod_raw, p_win_raw, p_grand_raw,
           carry, hurdle, catchup, waterfall,
           mf1, mf2, gov_share, call_seq, wfall_seq, seed):

    rng = np.random.default_rng(seed)
    C   = 1.0

    invest_yrs    = N + 1
    fee_fund_life = invest_yrs * 2
    fund_life     = max(N + H + 1, fee_fund_life)

    # management fees (1-D, length fund_life)
    mgmt_fees = np.array([
        C * mf1 if (t + 1) <= invest_yrs
        else C * mf2 if (t + 1) <= fee_fund_life
        else 0.0
        for t in range(fund_life)
    ])
    total_fees      = mgmt_fees.sum()
    net_per_tranche = (C - total_fees) / N

    # outcome distribution
    raw   = [p_loss_raw, p_zombie_raw, p_mod_raw, p_win_raw, p_grand_raw]
    tot   = max(sum(raw), 1e-9)
    probs = np.array([r / tot for r in raw])
    exp_moic = float(probs @ BUCKET_MOICS)

    # draw one outcome per tranche per sim: (n_sims, N)
    outcome_idx = rng.choice(5, size=(n_sims, N), p=probs)
    moics_drawn = BUCKET_MOICS[outcome_idx]              # (n_sims, N)
    exit_per_tr = moics_drawn * net_per_tranche          # (n_sims, N)

    # investments (1-D)
    investments = np.zeros(fund_life)
    investments[:N] = net_per_tranche

    # exits (n_sims, fund_life)
    exits_sim = np.zeros((n_sims, fund_life))
    for i in range(N):
        et = i + H
        if et < fund_life:
            exits_sim[:, et] += exit_per_tr[:, i]

    # LP capital call split (deterministic 1-D)
    gov_inv_pp  = gov_share * investments
    gov_inv_seq = np.zeros(fund_life)
    gov_rem = gov_share * N * net_per_tranche
    for t in range(fund_life):
        g = min(gov_rem, max(0.0, investments[t]))
        gov_inv_seq[t] = g
        gov_rem = max(0.0, gov_rem - g)
    gov_inv  = (1.0 - call_seq) * gov_inv_pp + call_seq * gov_inv_seq
    priv_inv = investments - gov_inv
    gov_fees  = gov_share * mgmt_fees
    priv_fees = (1.0 - gov_share) * mgmt_fees

    # waterfall — vectorised over n_sims ──────────────────────────────────────
    lp_sim = np.zeros((n_sims, fund_life))
    gp_sim = np.zeros((n_sims, fund_life))

    if waterfall == "European (Whole-Fund)":
        lp_thresh   = np.full(n_sims, C)
        total_p1    = np.zeros(n_sims)
        ctup_target = np.zeros(n_sims)
        ctup_earned = np.zeros(n_sims)
        hurdle_cl   = np.zeros(n_sims, dtype=bool)

        for t in range(fund_life):
            lp_thresh *= (1.0 + hurdle)
            rem = exits_sim[:, t].copy()

            fill = np.minimum(rem, np.maximum(0.0, lp_thresh))
            lp_sim[:, t] += fill
            total_p1  += fill
            lp_thresh -= fill
            rem       -= fill

            newly = (~hurdle_cl) & (lp_thresh <= 1e-12)
            if newly.any():
                h_pref = np.where(newly, np.maximum(0.0, total_p1 - C), 0.0)
                rate   = carry / (1.0 - carry) if carry < 1.0 else 1.0
                ctup_target += h_pref * rate
                hurdle_cl   |= newly

            if catchup > 0:
                need = np.maximum(0.0, ctup_target - ctup_earned)
                pool = np.where(hurdle_cl & (need > 0),
                                np.minimum(rem, need / catchup), 0.0)
                gp_sim[:, t]    += catchup * pool
                lp_sim[:, t]    += (1.0 - catchup) * pool
                ctup_earned     += catchup * pool
                rem             -= pool

            rem = np.maximum(0.0, rem)
            lp_sim[:, t] += (1.0 - carry) * rem
            gp_sim[:, t] += carry * rem

    else:  # American deal-by-deal
        for i in range(N):
            et = i + H
            if et >= fund_life:
                continue
            dist      = exit_per_tr[:, i].copy()
            preferred = net_per_tranche * ((1.0 + hurdle) ** H - 1.0)
            rem       = dist
            phase1    = np.minimum(rem, net_per_tranche + preferred)
            lp_sim[:, et] += phase1
            rem -= phase1
            if catchup > 0 and carry < 1.0:
                gp_tgt = carry / (1.0 - carry) * preferred
                pool   = np.maximum(0.0, np.minimum(rem, gp_tgt / catchup))
                gp_sim[:, et]   += catchup * pool
                lp_sim[:, et]   += (1.0 - catchup) * pool
                rem -= pool
            rem = np.maximum(0.0, rem)
            lp_sim[:, et] += (1.0 - carry) * rem
            gp_sim[:, et] += carry * rem

    # LP distribution split — vectorised ─────────────────────────────────────
    # Preferred balances start at zero and accrue hurdle only on deployed capital
    # (deterministic call schedule, same for every sim).  Gov called earlier via
    # call_seq accumulates a larger balance, correctly interacting with wfall_seq.
    gov_pp  = gov_share * lp_sim
    gov_seq = np.zeros((n_sims, fund_life))
    g_pref  = np.zeros(n_sims)   # gov LP outstanding balance (scalar per sim)
    p_pref  = np.zeros(n_sims)   # private LP outstanding balance

    for t in range(fund_life):
        g_pref = g_pref * (1.0 + hurdle) + gov_inv[t]  + gov_fees[t]
        p_pref = p_pref * (1.0 + hurdle) + priv_inv[t] + priv_fees[t]
        dist = lp_sim[:, t]
        pf   = np.minimum(dist, np.maximum(0.0, p_pref))
        rem2 = dist - pf
        p_pref = np.maximum(0.0, p_pref - pf)
        gf   = np.minimum(rem2, np.maximum(0.0, g_pref))
        rem3 = rem2 - gf
        g_pref = np.maximum(0.0, g_pref - gf)
        gov_seq[:, t] = gf + gov_share * rem3

    gov_dist  = (1.0 - wfall_seq) * gov_pp  + wfall_seq * gov_seq
    priv_dist = lp_sim - gov_dist

    # cash flows (n_sims, fund_life)
    lp_cf_sim   = lp_sim    - investments[None, :] - mgmt_fees[None, :]
    gov_cf_sim  = gov_dist  - gov_inv[None, :]     - gov_fees[None, :]
    priv_cf_sim = priv_dist - priv_inv[None, :]    - priv_fees[None, :]

    total_out   = float(np.sum(investments + mgmt_fees))
    go_out      = float(np.sum(gov_inv  + gov_fees))
    pr_out      = float(np.sum(priv_inv + priv_fees))

    moics_net   = lp_sim.sum(1)    / total_out if total_out > 0 else np.zeros(n_sims)
    gov_moics   = gov_dist.sum(1)  / go_out    if go_out    > 0 else np.zeros(n_sims)
    priv_moics  = priv_dist.sum(1) / pr_out    if pr_out    > 0 else np.zeros(n_sims)

    # IRR (loop — unavoidable for Newton-Raphson)
    irrs      = np.array([_irr(lp_cf_sim[s])   for s in range(n_sims)])
    gov_irrs  = (np.array([_irr(gov_cf_sim[s])  for s in range(n_sims)])
                 if gov_share > 0 else np.full(n_sims, np.nan))
    priv_irrs = (np.array([_irr(priv_cf_sim[s]) for s in range(n_sims)])
                 if gov_share < 1 else np.full(n_sims, np.nan))

    cumcf_sim      = np.cumsum(lp_cf_sim,   axis=1)  # (n_sims, fund_life)
    gov_cumcf_sim  = np.cumsum(gov_cf_sim,  axis=1)
    priv_cumcf_sim = np.cumsum(priv_cf_sim, axis=1)

    # expected-value benchmark J-curve (deterministic, for comparison) ─────────
    exits_ev     = np.zeros(fund_life)
    lp_ev        = np.zeros(fund_life)
    gp_ev        = np.zeros(fund_life)
    for i in range(N):
        et = i + H
        if et < fund_life:
            exits_ev[et] += net_per_tranche * exp_moic

    if waterfall == "European (Whole-Fund)":
        lth = C; tp1 = 0.0; ct = 0.0; ce = 0.0; hc = False
        for t in range(fund_life):
            lth *= (1.0 + hurdle)
            d = exits_ev[t]
            if d <= 0:
                continue
            r = d
            f = min(r, max(0.0, lth))
            lp_ev[t] += f; tp1 += f; lth -= f; r -= f
            if lth <= 0 and not hc:
                hc = True
                ct = (carry / (1.0 - carry) if carry < 1 else 1.0) * max(0.0, tp1 - C)
            if r > 0 and hc and catchup > 0:
                need = ct - ce
                if need > 0:
                    pool = min(r, need / catchup)
                    gp_ev[t] += catchup * pool; lp_ev[t] += (1.0 - catchup) * pool
                    ce += catchup * pool; r -= pool
            r = max(0.0, r)
            lp_ev[t] += (1.0 - carry) * r; gp_ev[t] += carry * r
    else:
        for i in range(N):
            et = i + H
            if et >= fund_life:
                continue
            d = net_per_tranche * exp_moic
            pref = net_per_tranche * ((1.0 + hurdle) ** H - 1.0)
            r = d
            p1 = min(r, net_per_tranche + pref); lp_ev[et] += p1; r -= p1
            if r > 0 and catchup > 0 and carry < 1:
                pool = min(r, carry / (1.0 - carry) * pref / catchup)
                gp_ev[et] += catchup * pool; lp_ev[et] += (1.0 - catchup) * pool; r -= pool
            r = max(0.0, r)
            lp_ev[et] += (1.0 - carry) * r; gp_ev[et] += carry * r

    cumcf_ev = np.cumsum(lp_ev - investments - mgmt_fees)

    years = [f"Year {t + 1}" for t in range(fund_life)]

    return dict(
        irrs=irrs, moics_net=moics_net,
        gov_irrs=gov_irrs, priv_irrs=priv_irrs,
        gov_moics=gov_moics, priv_moics=priv_moics,
        cumcf_sim=cumcf_sim, gov_cumcf_sim=gov_cumcf_sim,
        priv_cumcf_sim=priv_cumcf_sim, cumcf_ev=cumcf_ev,
        years=years, fund_life=fund_life,
        net_per_tranche=net_per_tranche, exp_moic=exp_moic, probs=probs,
        n_sims=n_sims,
    )

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Assumptions")
    st.caption("All outputs are per unit of committed capital (C = 1).")

    st.subheader("Fund Structure")
    N = st.slider("Drawdown Tranches (N)", 1, 12, 4)
    H = st.slider("Holding Period per Tranche (years)", 1, 15, 7)

    st.subheader("Portfolio Construction")
    n_sims = st.slider("Simulations", 100, 2000, 500, step=100,
                       help="More simulations = smoother distributions, slower compute.")

    st.subheader("VC Return Distribution")
    st.caption("Weights auto-normalized to 100%.")
    p_loss_raw   = st.slider("Total loss (0×)",        0, 100, 50)
    p_zombie_raw = st.slider("Return of capital (1×)", 0, 100, 25)
    p_mod_raw    = st.slider("Moderate return (3×)",   0, 100, 15)
    p_win_raw    = st.slider("Strong return (10×)",    0, 100,  8)
    p_grand_raw  = st.slider("Grand slam (30×)",       0, 100,  2)

    _raw = [p_loss_raw, p_zombie_raw, p_mod_raw, p_win_raw, p_grand_raw]
    _tot = max(sum(_raw), 1)
    _p   = [r / _tot for r in _raw]
    _em  = float(np.array(_p) @ BUCKET_MOICS)
    st.caption(
        f"**Normalized:** {_p[0]:.0%} · {_p[1]:.0%} · {_p[2]:.0%} · {_p[3]:.0%} · {_p[4]:.0%}  \n"
        f"**Expected gross MOIC: {_em:.2f}×**"
    )

    st.subheader("Carried Interest & Waterfall")
    carry     = st.slider("Carry %",       0, 40,  20, format="%d%%") / 100
    hurdle    = st.slider("Hurdle Rate",   0, 25,   8, format="%d%%") / 100
    catchup   = st.slider("GP Catchup",   0, 100, 100, step=5, format="%d%%") / 100
    waterfall = st.selectbox("Waterfall Type",
                             ["European (Whole-Fund)", "American (Deal-by-Deal)"])

    st.subheader("Management Fees")
    mf1 = st.slider("Fee — Investment Period (%)", 0.0, 5.0, 2.5, step=0.25, format="%.2f%%") / 100
    mf2 = st.slider("Fee — Harvest Period (%)",    0.0, 5.0, 1.5, step=0.25, format="%.2f%%") / 100

    st.subheader("LP Structure")
    gov_share = st.slider("Government LP Share (%)", 0, 100, 50, format="%d%%") / 100
    call_seq  = st.slider("Capital Call Sequencing", 0, 100,  0, format="%d%%",
                          help="0% = pari passu. 100% = government fully drawn first.") / 100
    wfall_seq = st.slider("Waterfall Sequencing",    0, 100,  0, format="%d%%",
                          help="0% = pari passu. 100% = private LP preferred return first.") / 100

    st.subheader("Simulation")
    if "seed" not in st.session_state:
        st.session_state.seed = 42
    if st.button("🎲 New random seed"):
        st.session_state.seed += 1
    st.caption(f"Current seed: {st.session_state.seed}")

# ── run ───────────────────────────────────────────────────────────────────────
with st.spinner(f"Running {n_sims} simulations…"):
    R = run_mc(N, H, n_sims,
               p_loss_raw, p_zombie_raw, p_mod_raw, p_win_raw, p_grand_raw,
               carry, hurdle, catchup, waterfall,
               mf1, mf2, gov_share, call_seq, wfall_seq,
               seed=st.session_state.seed)

irrs      = R["irrs"]
moics     = R["moics_net"]
gov_irrs  = R["gov_irrs"]
priv_irrs = R["priv_irrs"]
gov_moics = R["gov_moics"]
pri_moics = R["priv_moics"]
cumcf     = R["cumcf_sim"]      # (n_sims, fund_life)
cumcf_ev  = R["cumcf_ev"]       # (fund_life,)
years_x   = R["years"]
probs     = R["probs"]

def pct(arr, q):
    return float(np.nanpercentile(arr, q))

irr_p = {q: pct(irrs,  q) for q in [10, 25, 50, 75, 90]}
moi_p = {q: pct(moics, q) for q in [10, 25, 50, 75, 90]}

# ── title ─────────────────────────────────────────────────────────────────────
st.title("VC Fund Economics — Monte Carlo")
st.caption(
    f"**{n_sims} simulations** · 1 company / tranche · "
    f"{N} tranches · {H}-yr hold · "
    f"Expected gross MOIC: **{R['exp_moic']:.2f}×** · "
    f"Equity per tranche: **{R['net_per_tranche']:.4f}**"
)
st.markdown("---")

# ── KPI helpers ───────────────────────────────────────────────────────────────
def kpi(col, label, value, sub=""):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

def irr_s(v): return f"{v:.1%}" if np.isfinite(v) else "—"
def moi_s(v): return f"{v:.2f}×"

# fund-level distribution
prob_return = float(np.mean(moics >= 1.0))
prob_2x     = float(np.mean(moics >= 2.0))
prob_3x     = float(np.mean(moics >= 3.0))

cols1 = st.columns(6)
kpi(cols1[0], "Median LP IRR",  irr_s(irr_p[50]), f"P25: {irr_s(irr_p[25])}  P75: {irr_s(irr_p[75])}")
kpi(cols1[1], "P10 LP IRR",     irr_s(irr_p[10]), "bottom decile")
kpi(cols1[2], "P90 LP IRR",     irr_s(irr_p[90]), "top decile")
kpi(cols1[3], "Prob(MOIC ≥ 1×)",f"{prob_return:.0%}", f"2×: {prob_2x:.0%}  3×: {prob_3x:.0%}")
kpi(cols1[4], "Median LP MOIC", moi_s(moi_p[50]), f"P25: {moi_s(moi_p[25])}  P75: {moi_s(moi_p[75])}")
kpi(cols1[5], "Exp. Gross MOIC",f"{R['exp_moic']:.2f}×", "benchmark (expected value)")
st.markdown("<br>", unsafe_allow_html=True)

# per-LP KPIs
st.caption(f"**LP-level performance** (medians) — Gov: {gov_share:.0%}")
cols2 = st.columns(6)
kpi(cols2[0], "Med Gov LP IRR",  irr_s(pct(gov_irrs,  50)), f"P25–P75: {irr_s(pct(gov_irrs,25))} / {irr_s(pct(gov_irrs,75))}")
kpi(cols2[1], "Med Gov MOIC",   moi_s(pct(gov_moics, 50)), f"P25: {moi_s(pct(gov_moics,25))}  P75: {moi_s(pct(gov_moics,75))}")
kpi(cols2[2], "Prob Gov ≥ 1×",  f"{np.mean(gov_moics>=1):.0%}", "gov LP returns capital")
kpi(cols2[3], "Med Priv LP IRR",irr_s(pct(priv_irrs, 50)), f"P25–P75: {irr_s(pct(priv_irrs,25))} / {irr_s(pct(priv_irrs,75))}")
kpi(cols2[4], "Med Priv MOIC",  moi_s(pct(pri_moics, 50)), f"P25: {moi_s(pct(pri_moics,25))}  P75: {moi_s(pct(pri_moics,75))}")
kpi(cols2[5], "Prob Priv ≥ 1×", f"{np.mean(pri_moics>=1):.0%}", "private LP returns capital")
st.markdown("<br>", unsafe_allow_html=True)

# ── chart helpers ─────────────────────────────────────────────────────────────
PCTILE_STYLES = {
    10: ("#c0392b", "dot",   "P10"),
    25: ("#e67e22", "dash",  "P25"),
    50: ("#2980b9", "solid", "Median"),
    75: ("#27ae60", "dash",  "P75"),
    90: ("#8e44ad", "dot",   "P90"),
}

def pctile_vlines(fig, arr, axis="x"):
    for q, (color, dash, label) in PCTILE_STYLES.items():
        v = pct(arr, q)
        if not np.isfinite(v):
            continue
        if axis == "x":
            fig.add_vline(x=v, line_dash=dash, line_color=color, line_width=1.5,
                          annotation_text=f" {label}: {v:.1%}",
                          annotation_position="top", annotation_font_size=9)
        else:
            fig.add_hline(y=v, line_dash=dash, line_color=color, line_width=1.5)

def fan_traces(fig, cumcf_arr, color_hex, name_prefix, alpha_outer=0.10, alpha_inner=0.22):
    """Add P10-P90 fan and median line for a (n_sims, T) cumcf array."""
    T   = cumcf_arr.shape[1]
    yx  = years_x
    rev = list(range(T - 1, -1, -1))

    p10 = np.nanpercentile(cumcf_arr, 10, axis=0)
    p25 = np.nanpercentile(cumcf_arr, 25, axis=0)
    p50 = np.nanpercentile(cumcf_arr, 50, axis=0)
    p75 = np.nanpercentile(cumcf_arr, 75, axis=0)
    p90 = np.nanpercentile(cumcf_arr, 90, axis=0)

    def rgba(h, a):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return f"rgba({r},{g},{b},{a})"

    x_fill = list(yx) + [yx[i] for i in rev]
    fig.add_trace(go.Scatter(
        x=x_fill, y=list(p90) + [p10[i] for i in rev],
        fill="toself", fillcolor=rgba(color_hex, alpha_outer),
        line=dict(width=0), showlegend=True,
        name=f"{name_prefix} P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x_fill, y=list(p75) + [p25[i] for i in rev],
        fill="toself", fillcolor=rgba(color_hex, alpha_inner),
        line=dict(width=0), showlegend=True,
        name=f"{name_prefix} P25–P75", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=yx, y=p50, mode="lines+markers", name=f"{name_prefix} Median",
        line=dict(color=color_hex, width=2.5),
        marker=dict(size=6, color=color_hex)))

LAYOUT = dict(height=360, plot_bgcolor="white", paper_bgcolor="white",
              legend=dict(orientation="h", y=-0.22), margin=dict(t=20, b=70))

# ── charts ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

# Chart 1 — LP IRR distribution
with c1:
    st.subheader("LP IRR Distribution")
    finite_irrs = irrs[np.isfinite(irrs)]
    f1 = go.Figure()
    f1.add_trace(go.Histogram(x=finite_irrs, nbinsx=50, name="LP IRR",
                              marker_color="#2980b9", opacity=0.75))
    pctile_vlines(f1, finite_irrs, axis="x")
    f1.update_layout(**LAYOUT,
        xaxis=dict(title="LP Net IRR", tickformat=".0%"),
        yaxis=dict(title="Simulations"))
    st.plotly_chart(f1, use_container_width=True)

# Chart 2 — LP MOIC distribution
with c2:
    st.subheader("LP MOIC Distribution")
    f2 = go.Figure()
    f2.add_trace(go.Histogram(x=moics, nbinsx=50, name="LP MOIC",
                              marker_color="#27ae60", opacity=0.75))
    for thresh, color in [(1.0, "#c0392b"), (2.0, "#e67e22"), (3.0, "#8e44ad")]:
        p_above = float(np.mean(moics >= thresh))
        f2.add_vline(x=thresh, line_dash="dot", line_color=color, line_width=1.5,
                     annotation_text=f" ≥{thresh:.0f}×: {p_above:.0%}",
                     annotation_position="top", annotation_font_size=9)
    f2.update_layout(**LAYOUT,
        xaxis=dict(title="LP Net MOIC"),
        yaxis=dict(title="Simulations"))
    st.plotly_chart(f2, use_container_width=True)

# Chart 3 — J-curve fan
c3, c4 = st.columns(2)
with c3:
    st.subheader("J-Curve Fan (Combined LP)")
    f3 = go.Figure()
    fan_traces(f3, cumcf, "#2980b9", "LP")
    # Expected-value benchmark
    f3.add_trace(go.Scatter(x=years_x, y=cumcf_ev, mode="lines",
        name="Expected value (benchmark)", line=dict(color="#1a1a2e", width=2, dash="dash")))
    f3.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1.5,
                 annotation_text="  Breakeven", annotation_position="top left",
                 annotation_font=dict(size=10, color="#888"))
    f3.update_layout(**LAYOUT,
        yaxis=dict(title="Cumulative Net CF (× C)", tickformat=".3f", zeroline=False))
    st.plotly_chart(f3, use_container_width=True)

# Chart 4 — Probability of MOIC thresholds
with c4:
    st.subheader("Probability of Reaching MOIC")
    thresholds  = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    prob_thresh = [float(np.mean(moics >= t)) for t in thresholds]
    bar_colors  = ["#27ae60" if p >= 0.5 else "#e67e22" if p >= 0.25 else "#c0392b"
                   for p in prob_thresh]
    f4 = go.Figure(go.Bar(
        x=[f"≥ {t:.1f}×" for t in thresholds],
        y=prob_thresh,
        marker_color=bar_colors,
        text=[f"{p:.0%}" for p in prob_thresh],
        textposition="outside"))
    f4.update_layout(**LAYOUT,
        yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1.1]),
        xaxis=dict(title="LP MOIC threshold"))
    st.plotly_chart(f4, use_container_width=True)

# Chart 5 — Gov vs Priv LP IRR distributions
c5, c6 = st.columns(2)
with c5:
    st.subheader("Gov vs Private LP IRR")
    f5 = go.Figure()
    fg = np.sort(gov_irrs[np.isfinite(gov_irrs)])
    fp = np.sort(priv_irrs[np.isfinite(priv_irrs)])
    # Plot as empirical CDFs so both distributions are always visible
    if len(fg):
        cdf_g = np.arange(1, len(fg) + 1) / len(fg)
        f5.add_trace(go.Scatter(x=fg, y=cdf_g, mode="lines",
                                name=f"Gov LP ({gov_share:.0%})",
                                line=dict(color="#8e44ad", width=2.5)))
        f5.add_vline(x=float(np.median(fg)), line_color="#8e44ad", line_dash="dash",
                     line_width=1.5,
                     annotation_text=f" Gov med: {float(np.median(fg)):.1%}",
                     annotation_font=dict(size=9, color="#8e44ad"),
                     annotation_position="top")
    if len(fp):
        cdf_p = np.arange(1, len(fp) + 1) / len(fp)
        f5.add_trace(go.Scatter(x=fp, y=cdf_p, mode="lines",
                                name=f"Priv LP ({1-gov_share:.0%})",
                                line=dict(color="#e67e22", width=2.5)))
        f5.add_vline(x=float(np.median(fp)), line_color="#e67e22", line_dash="dash",
                     line_width=1.5,
                     annotation_text=f" Priv med: {float(np.median(fp)):.1%}",
                     annotation_font=dict(size=9, color="#e67e22"),
                     annotation_position="bottom")
    f5.add_hline(y=0.5, line_dash="dot", line_color="#aaa", line_width=1,
                 annotation_text=" P50", annotation_font=dict(size=9))
    f5.update_layout(**LAYOUT,
        xaxis=dict(title="LP IRR", tickformat=".0%"),
        yaxis=dict(title="Cumulative probability", tickformat=".0%", range=[0, 1.05]))
    st.plotly_chart(f5, use_container_width=True)

# Chart 6 — Gov vs Priv LP J-curve fans
with c6:
    st.subheader("J-Curve Fan: Gov vs Private LP")
    gov_cumcf  = R["gov_cumcf_sim"]   # (n_sims, fund_life)
    priv_cumcf = R["priv_cumcf_sim"]
    f6 = go.Figure()
    fan_traces(f6, gov_cumcf,  "#8e44ad", f"Gov ({gov_share:.0%})",
               alpha_outer=0.12, alpha_inner=0.25)
    fan_traces(f6, priv_cumcf, "#e67e22", f"Priv ({1-gov_share:.0%})",
               alpha_outer=0.12, alpha_inner=0.25)
    f6.add_hline(y=0, line_dash="dot", line_color="#888", line_width=1.5,
                 annotation_text="  Breakeven", annotation_position="top left",
                 annotation_font=dict(size=10, color="#888"))
    f6.update_layout(**LAYOUT,
        yaxis=dict(title="Cumulative Net CF (× C)", tickformat=".3f", zeroline=False))
    st.plotly_chart(f6, use_container_width=True)

st.markdown("---")
st.caption(
    f"**Model:** {n_sims} independent simulations. Each tranche makes a single investment "
    f"that independently draws an outcome from the 5-bucket distribution. "
    f"Fund-level IRR and MOIC are computed per simulation, then aggregated. "
    f"The dashed line in the J-curve fan is the deterministic expected-value result "
    f"(identical to the benchmark dashboard). "
    f"The median fund result typically lies *below* the expected value because the "
    f"distribution is right-skewed — grand-slam outcomes pull the mean up while the "
    f"typical fund misses them entirely."
)
