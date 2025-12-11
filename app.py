#!/usr/bin/env python3
"""
Streamlit UI for:
 - Single spread simulation (run_analysis)
 - Grid scanning & EV heatmap

Requires:
    pip install streamlit

Run:
    streamlit run app.py
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from enhanced_spx_sim import run_analysis  # assumes file is in same folder


# ------------------------
# Scanner helper (local)
# ------------------------

def scan_spreads(
    base_params: dict,
    short_min: float,
    short_max: float,
    short_step: float,
    prem_min: float,
    prem_max: float,
    prem_step: float,
    width: float,
    min_win_rate: float,
    min_ev: float,
):
    """
    Scan a grid of short strikes & premiums.
    Returns (df_all, df_candidates).
    """
    def build_grid(min_val, max_val, step):
        n_steps = int(round((max_val - min_val) / step)) + 1
        return [min_val + i * step for i in range(n_steps)]

    short_strikes = build_grid(short_min, short_max, short_step)
    premiums = build_grid(prem_min, prem_max, prem_step)

    all_rows = []
    cand_rows = []
    total = len(short_strikes) * len(premiums)
    count = 0

    for ks in short_strikes:
        for prem in premiums:
            count += 1
            st.write(f"Running {count}/{total}: short={ks}, long={ks - width}, premium={prem}")
            params = base_params.copy()
            params["k_short"] = ks
            params["k_long"] = ks - width
            params["premium"] = prem

            res = run_analysis(params)

            row = {
                "k_short": ks,
                "k_long": ks - width,
                "premium": prem,
                "avg_pnl": res["avg_pnl"],
                "std_pnl": res["std_pnl"],
                "win_rate_pct": res["win_rate_pct"],
                "max_profit": res["max_profit"],
                "max_loss": res["max_loss"],
                "prob_expire_short": res["prob_expire_short"],
                "prob_expire_long": res["prob_expire_long"],
                "prob_touch_short": res["prob_touch_short"],
                "prob_touch_long": res["prob_touch_long"],
                "sigma_use": res["sigma_use"],
            }
            all_rows.append(row)

            if (res["avg_pnl"] >= min_ev) and (res["win_rate_pct"] >= min_win_rate):
                cand_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    df_cand = pd.DataFrame(cand_rows)
    return df_all, df_cand


def plot_ev_heatmap(df_all: pd.DataFrame):
    """
    Plot avg_pnl as heatmap vs (k_short, premium).
    Returns a matplotlib figure for st.pyplot.
    """
    if df_all.empty:
        return None

    pivot = df_all.pivot_table(index="premium", columns="k_short", values="avg_pnl")
    premiums = pivot.index.values
    strikes = pivot.columns.values
    Z = pivot.values

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[strikes.min(), strikes.max(), premiums.min(), premiums.max()],
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg P/L per contract ($)")
    ax.set_xlabel("Short Put Strike")
    ax.set_ylabel("Net Premium (per share)")
    ax.set_title("Expected P/L Heatmap")
    return fig


# ------------------------
# Streamlit app
# ------------------------

st.set_page_config(
    page_title="SPX Put Credit Spread Lab",
    layout="wide",
)

st.title("🧪 SPX Put Credit Spread Lab")

st.markdown(
    """
This app wraps:
- A **single spread simulator** (Monte Carlo + IV surface)
- A **grid scanner** that searches over strikes & premiums and shows an EV heatmap
"""
)

# -------- Sidebar: global settings --------
st.sidebar.header("Global Settings")

ticker = st.sidebar.text_input("Ticker", value="^GSPC")
days = st.sidebar.number_input("Days to expiration", min_value=1, max_value=365, value=7)
sims = st.sidebar.number_input("Simulations", min_value=1000, max_value=200000, value=20000, step=5000)

model = st.sidebar.selectbox("Model", options=["gbm", "merton", "heston"], index=1)
r = st.sidebar.number_input("Risk-free rate (annualized)", min_value=-0.05, max_value=0.2, value=0.02, step=0.005, format="%.3f")
seed = st.sidebar.number_input("Random seed (optional)", min_value=0, max_value=1_000_000, value=42, step=1)
use_iv = st.sidebar.checkbox("Use market IV surface (if available)", value=True)

base_params_common = {
    "ticker": ticker,
    "days": days,
    "sims": int(sims),
    "model": model,
    "r": float(r),
    "seed": int(seed),
    "iv_from_market": bool(use_iv),
}

tab1, tab2 = st.tabs(["Single Spread Simulator", "Grid Scanner"])


# -------- Tab 1: Single spread simulator --------
with tab1:
    st.subheader("Single Spread Simulator")

    col1, col2, col3 = st.columns(3)
    with col1:
        k_short = st.number_input("Short put strike (K_short)", min_value=0.0, value=6650.0, step=25.0)
    with col2:
        k_long = st.number_input("Long put strike (K_long)", min_value=0.0, value=6640.0, step=25.0)
    with col3:
        premium = st.number_input("Net premium (credit per share)", min_value=0.0, value=1.50, step=0.10, format="%.2f")

    st.caption("Note: one contract = 100 shares, so net premium × 100 = max profit (approx).")

    run_button = st.button("Run Simulation", type="primary")

    if run_button:
        params = base_params_common.copy()
        params["k_short"] = float(k_short)
        params["k_long"] = float(k_long)
        params["premium"] = float(premium)

        with st.spinner("Running Monte Carlo simulation..."):
            res = run_analysis(params)

        st.success("Simulation complete.")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### P&L & Probabilities")
            st.write(f"**Spot (S0):** {res['S0']:.2f}")
            st.write(f"**Model:** {res['model']}  —  **σ used:** {res['sigma_use']:.3f}")
            st.write(f"**Avg P/L per contract:** ${res['avg_pnl']:.2f}")
            st.write(f"**Std dev P/L:** ${res['std_pnl']:.2f}")
            st.write(f"**Win rate (P/L > 0):** {res['win_rate_pct']:.2f}%")
            st.write(f"**Max profit:** ${res['max_profit']:.2f}")
            st.write(f"**Max loss:** ${res['max_loss']:.2f}")

            # Break-even required win rate
            max_profit = res["max_profit"]
            max_loss = res["max_loss"]
            loss_abs = -max_loss if max_loss < 0 else 0.0
            if max_profit > 0 and loss_abs > 0:
                breakeven_win_rate = loss_abs / (max_profit + loss_abs)
                st.write(
                    f"**Required win rate to break even:** {breakeven_win_rate * 100:.2f}% "
                    f"(vs simulated {res['win_rate_pct']:.2f}%)"
                )
            else:
                st.write("Unable to compute break-even win rate (degenerate max profit/loss).")

            st.markdown("#### Probabilities")
            st.write(f"Prob(expire below short strike): {res['prob_expire_short']:.4f}")
            st.write(f"Prob(expire below long strike): {res['prob_expire_long']:.4f}")
            st.write(f"Prob(touch short strike): {res['prob_touch_short']:.4f}")
            st.write(f"Prob(touch long strike): {res['prob_touch_long']:.4f}")
            if res.get("expected_move_pct") is not None:
                st.write(f"Approx ATM straddle expected move: {res['expected_move_pct']:.2f}%")

        with colB:
            st.markdown("### Greeks (per share)")
            g_short = res["greeks"]["short"]
            g_long = res["greeks"]["long"]

            st.write("**Short put leg:**")
            st.write(
                f"Δ = {g_short['delta']:.4f}, "
                f"Γ = {g_short['gamma']:.6f}, "
                f"Vega = {g_short['vega']:.4f}, "
                f"Theta/day = {g_short['theta']:.4f}"
            )
            st.write("**Long put leg:**")
            st.write(
                f"Δ = {g_long['delta']:.4f}, "
                f"Γ = {g_long['gamma']:.6f}, "
                f"Vega = {g_long['vega']:.4f}, "
                f"Theta/day = {g_long['theta']:.4f}"
            )

            # Net greeks of the spread
            net_delta = g_short["delta"] + g_long["delta"]
            net_gamma = g_short["gamma"] + g_long["gamma"]
            net_vega = g_short["vega"] + g_long["vega"]
            net_theta = g_short["theta"] + g_long["theta"]
            st.write("**Net spread greeks:**")
            st.write(
                f"Δ = {net_delta:.4f}, "
                f"Γ = {net_gamma:.6f}, "
                f"Vega = {net_vega:.4f}, "
                f"Theta/day = {net_theta:.4f}"
            )

        # Histogram of P/L
        st.markdown("### P/L Distribution")
        pnl = res["pnl_samples"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pnl, bins=60, edgecolor="black")
        ax.axvline(res["avg_pnl"], color="red", linestyle="--", label=f"Avg P/L: ${res['avg_pnl']:.2f}")
        ax.set_xlabel("P/L per contract ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of P/L")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


# -------- Tab 2: Grid scanner --------
with tab2:
    st.subheader("Grid Scanner & EV Heatmap")

    st.markdown("Define a grid of **short strikes** and **premiums**; the app will scan them and show EV & candidates.")

    with st.expander("Grid parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            short_min = st.number_input("Short strike min", min_value=0.0, value=6600.0, step=25.0)
            short_max = st.number_input("Short strike max", min_value=0.0, value=6800.0, step=25.0)
            short_step = st.number_input("Short strike step", min_value=1.0, value=25.0, step=1.0)
        with col2:
            prem_min = st.number_input("Premium min (per share)", min_value=0.0, value=1.0, step=0.10, format="%.2f")
            prem_max = st.number_input("Premium max (per share)", min_value=0.0, value=2.5, step=0.10, format="%.2f")
            prem_step = st.number_input("Premium step", min_value=0.01, value=0.25, step=0.05, format="%.2f")
        with col3:
            width = st.number_input("Spread width (K_short - K_long)", min_value=1.0, value=10.0, step=1.0)
            min_win_rate = st.number_input("Min win rate for candidate (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            min_ev = st.number_input("Min avg P/L per contract for candidate", value=0.0, step=25.0)

    scan_button = st.button("Run Grid Scan", type="primary")

    if scan_button:
        base_params = base_params_common.copy()
        # k_short, k_long, premium will be overridden in scan_spreads

        with st.spinner("Running spread grid scan... this may take a bit."):
            df_all, df_cand = scan_spreads(
                base_params=base_params,
                short_min=float(short_min),
                short_max=float(short_max),
                short_step=float(short_step),
                prem_min=float(prem_min),
                prem_max=float(prem_max),
                prem_step=float(prem_step),
                width=float(width),
                min_win_rate=float(min_win_rate),
                min_ev=float(min_ev),
            )

        st.success("Scan complete.")

        st.markdown("### All scan results")
        st.dataframe(df_all)

        st.download_button(
            "Download all results as CSV",
            data=df_all.to_csv(index=False).encode("utf-8"),
            file_name="scan_all_results.csv",
            mime="text/csv",
        )

        if df_cand.empty:
            st.warning("No spreads met the candidate criteria (EV and win rate thresholds).")
        else:
            st.markdown("### Candidate spreads (meeting criteria)")
            st.dataframe(df_cand)

            st.download_button(
                "Download candidate results as CSV",
                data=df_cand.to_csv(index=False).encode("utf-8"),
                file_name="scan_candidate_results.csv",
                mime="text/csv",
            )

        st.markdown("### EV Heatmap (Avg P/L vs Short Strike & Premium)")
        fig_hm = plot_ev_heatmap(df_all)
        if fig_hm is not None:
            st.pyplot(fig_hm)
        else:
            st.info("Not enough data to plot heatmap.")
