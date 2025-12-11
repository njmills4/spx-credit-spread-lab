#!/usr/bin/env python3
"""
enhanced_spx_sim.py

Enhanced put-credit spread analysis for SPX (or any index-like underlying).
Features:
 - Fetches option chains + implied vols (via yfinance)
 - Builds IV surface (dte, strike) -> iv using Black-76
 - Simulates underlying using:
      * GBM
      * Merton jump-diffusion
      * Heston-like stochastic volatility (Euler)
 - Uses IV surface when available to choose sigma
 - Calculates:
      * P/L distribution
      * Probability of expiring below short/long strikes
      * Approximate probability of touching strikes before expiry
      * Greeks (delta, gamma, vega, theta) for short & long legs
 - Optional PNG histogram output

Dependencies:
    pip install numpy pandas matplotlib yfinance scipy

Example:
    python enhanced_spx_sim.py \
        --k-short 6650 \
        --k-long 6640 \
        --premium 1.5 \
        --days 44 \
        --sims 100000 \
        --model merton \
        --save spread_hist.png
"""

from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict

from scipy.interpolate import griddata
from scipy.stats import norm
from scipy.optimize import brentq


# ============================================================
# Black-76 pricing + Greeks helpers
# ============================================================

def black76_d1d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    """Compute d1, d2 for Black-76 (options on forwards)."""
    if T <= 0 or sigma <= 0:
        return float("-inf"), float("-inf")
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def black76_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 European call option price."""
    if T <= 0:
        return max(F - K, 0.0) * math.exp(-r * T)
    d1, d2 = black76_d1d2(F, K, T, sigma)
    df = math.exp(-r * T)
    return df * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_put(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 European put option price."""
    if T <= 0:
        return max(K - F, 0.0) * math.exp(-r * T)
    d1, d2 = black76_d1d2(F, K, T, sigma)
    df = math.exp(-r * T)
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol_from_price(option_price: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """
    Solve put implied vol under Black-76, treating underlying as forward F = S * exp((r - q) * T).
    Returns vol (float) or NaN on failure.
    """
    if T <= 0:
        intrinsic = max(K - S, 0.0)
        return 0.0 if option_price <= intrinsic else float("nan")

    F = S * math.exp((r - q) * T)

    def objective(sigma: float) -> float:
        sigma = max(sigma, 1e-8)
        return black76_put(F, K, T, r, sigma) - option_price

    try:
        vol = brentq(objective, 1e-6, 5.0, maxiter=200)
        return float(vol)
    except Exception:
        return float("nan")


# We’ll keep spot-based BS Greeks for intuition on delta/gamma/vega/theta.
# These are close enough for risk-sense even if pricing uses Black-76.

def bs_vega(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * norm.pdf(d1) * sqrtT


def bs_delta_put(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return -1.0 if S < K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return math.exp(-q * T) * (norm.cdf(d1) - 1.0)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return (math.exp(-q * T) * norm.pdf(d1)) / (S * sigma * sqrtT)


def bs_theta_put(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    first = - (S * norm.pdf(d1) * sigma * math.exp(-q * T)) / (2 * sqrtT)
    second = q * S * norm.cdf(d1) * math.exp(-q * T)
    third = r * K * math.exp(-r * T) * norm.cdf(-d2)
    theta_ann = first - second - third
    return theta_ann / 365.0  # per day per share


# ============================================================
# IV surface construction from option chains (puts only)
# ============================================================

def fetch_option_chain_ivs(ticker_obj, expiries: list, r: float = 0.01) -> pd.DataFrame:
    """
    For a list of expiry strings (YYYY-MM-DD), fetch option chain,
    compute put mid-prices, and then implied vols (Black-76).
    Returns DataFrame with columns: [expiry, dte, strike, mid, iv].
    """
    rows = []
    today = datetime.utcnow().date()
    # spot for IV inversion
    spot_hist = ticker_obj.history(period='1d')
    if spot_hist is None or spot_hist.empty:
        return pd.DataFrame()
    S = spot_hist['Close'].iloc[-1].item()

    for exp in expiries:
        try:
            chain = ticker_obj.option_chain(exp)
        except Exception:
            continue

        puts = chain.puts
        if puts is None or puts.empty:
            continue

        exp_date_obj = datetime.strptime(exp, "%Y-%m-%d").date()
        dte_years = max((exp_date_obj - today).days / 365.0, 1 / 365.0)

        for _, row in puts.iterrows():
            strike = float(row['strike'])
            bid = row.get('bid', np.nan)
            ask = row.get('ask', np.nan)

            bid = float(bid) if not pd.isna(bid) else np.nan
            ask = float(ask) if not pd.isna(ask) else np.nan

            if not np.isnan(bid) and not np.isnan(ask) and ask >= bid:
                mid = 0.5 * (bid + ask)
            elif not np.isnan(ask):
                mid = ask
            elif not np.isnan(bid):
                mid = bid
            else:
                continue

            iv = implied_vol_from_price(mid, float(S), strike, dte_years, r)
            rows.append({
                'expiry': exp,
                'dte': dte_years,
                'strike': strike,
                'mid': mid,
                'iv': iv
            })

    return pd.DataFrame(rows)


def build_iv_surface(df_iv: pd.DataFrame):
    """
    Build 2D interpolator (dte, strike) -> iv using scipy.griddata.
    Returns function iv_interp(T, K) -> iv.
    """
    if df_iv.empty:
        raise RuntimeError("No IV data to build surface.")

    pts = np.vstack([df_iv['dte'].values, df_iv['strike'].values]).T
    vals = df_iv['iv'].values

    def iv_interp(dte: float, strike: float) -> float:
        xi = np.array([dte, strike])
        iv = griddata(pts, vals, xi, method='linear')
        if np.isnan(iv):
            iv = griddata(pts, vals, xi, method='nearest')
        return float(iv)

    return iv_interp


# ============================================================
# Simulation models
# ============================================================

def simulate_gbm(S0: float, r: float, sigma: float, T: float, n: int,
                 seed: Optional[int] = None) -> np.ndarray:
    """Terminal prices under GBM."""
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.standard_normal(n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    return ST


def simulate_merton_jump(S0: float, r: float, sigma: float, T: float, n: int,
                         lam: float = 0.1, mu_j: float = -0.02, sigma_j: float = 0.1,
                         seed: Optional[int] = None) -> np.ndarray:
    """
    Merton jump-diffusion (lognormal jumps).
    lam: average jumps per year
    mu_j: mean jump size (log-space)
    sigma_j: jump volatility (log-space)
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n)
    Nj = np.random.poisson(lam * T, size=n)

    mean_j = Nj * mu_j
    std_j = np.sqrt(Nj) * sigma_j
    jump_component = mean_j + std_j * np.random.standard_normal(n)

    # Merton drift adjustment
    drift = (r - 0.5 * sigma**2 - lam * (math.exp(mu_j + 0.5 * sigma_j**2) - 1.0)) * T
    ST = S0 * np.exp(drift + sigma * math.sqrt(T) * Z + jump_component)
    return ST


def simulate_heston(S0: float, r: float, v0: float, kappa: float, theta: float,
                    xi: float, rho: float, T: float, n: int, steps: int = 50,
                    seed: Optional[int] = None) -> np.ndarray:
    """
    Euler-Maruyama Heston-style SV simulation (terminal prices only).
    dv = kappa*(theta - v) dt + xi*sqrt(v) dW2
    dS = r*S dt + sqrt(v)*S dW1; corr(dW1,dW2)=rho
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    S = np.full(n, S0, dtype=float)
    v = np.full(n, v0, dtype=float)

    for _ in range(steps):
        z1 = np.random.standard_normal(n)
        z2 = np.random.standard_normal(n)
        w1 = z1
        w2 = rho * z1 + math.sqrt(1 - rho**2) * z2

        v = np.maximum(
            v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0.0)) * math.sqrt(dt) * w2,
            1e-8
        )
        S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v) * math.sqrt(dt) * w1)

    return S


# ============================================================
# Probability helpers
# ============================================================

def prob_expire_below_gbm(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """P(S_T < K) under lognormal GBM."""
    if T <= 0:
        return 1.0 if S0 < K else 0.0
    mu = (r - 0.5 * sigma**2)
    denom = sigma * math.sqrt(T)
    z = (math.log(K / S0) - mu * T) / denom
    return float(norm.cdf(z))


def prob_touch_from_paths(full_paths: np.ndarray, K: float) -> float:
    """P(path ever touches or goes below K). full_paths: shape (n_paths, n_steps+1)."""
    touched = np.any(full_paths <= K, axis=1)
    return float(np.sum(touched) / len(touched))


# ============================================================
# Core analysis function
# ============================================================

def run_analysis(params: Dict) -> Dict:
    """
    params keys:
      ticker, k_short, k_long, premium, days, sims, model, r, seed, iv_from_market
      and optionally lam, mu_j, sigma_j, kappa, theta, xi, rho, steps (for heston).
    """
    ticker = params.get('ticker', '^GSPC')
    k_short = float(params['k_short'])
    k_long = float(params['k_long'])
    premium = float(params['premium'])
    days = int(params.get('days', 7))
    sims = int(params.get('sims', 10000))
    model = params.get('model', 'gbm')
    r = float(params.get('r', 0.02))
    seed = params.get('seed', None)
    iv_from_market = bool(params.get('iv_from_market', True))

    if k_short <= k_long:
        raise ValueError("k_short must be > k_long for a credit put spread.")

    # Spot from yfinance
    tk = yf.Ticker(ticker)
    spot_hist = tk.history(period='1d')
    if spot_hist is None or spot_hist.empty:
        raise RuntimeError("Unable to fetch spot price.")
    S0 = spot_hist['Close'].iloc[-1].item()

    # ================= IV: market surface or historical =================
    iv_surface = None
    sigma_h = None

    if iv_from_market:
        expiries = list(tk.options)[:6]  # first few expiries
        if not expiries:
            iv_from_market = False
        else:
            try:
                df_iv = fetch_option_chain_ivs(tk, expiries, r=r)
                if df_iv.empty:
                    iv_from_market = False
                else:
                    iv_surface = build_iv_surface(df_iv)
                    print("Using market IV surface from option chains.")
            except Exception:
                iv_from_market = False

    if not iv_from_market:
        # Historical vol fallback
        series = yf.download(ticker, period='1y', auto_adjust=True)['Close']
        lr = np.log(series / series.shift(1)).dropna()
        # std() returns a scalar; product may be a 0-d array or scalar – use item()
        sigma_h = (lr.std() * np.sqrt(252)).item()
        print(f"Using historical vol sigma={sigma_h:.3f}")

    # ================= Choose sigma for simulation =================
    T = max(days / 365.0, 1 / 365.0)

    if iv_surface is not None:
        # approximate ATM strike for IV lookup
        atm_strike = round(S0 / 5) * 5
        sigma_use = max(1e-6, iv_surface(T, atm_strike))
        print(f"Market ATM IV at T={T:.4f}y: {sigma_use:.3f}")
    else:
        sigma_use = sigma_h

    # ================= Run the chosen model =================
    full_paths = None

    if model == 'gbm':
        ST = simulate_gbm(S0, r, sigma_use, T, sims, seed=seed)

        # To estimate touch probabilities, create discretized GBM paths
        steps = 100
        dt = T / steps
        full_paths = np.full((sims, steps + 1), S0, dtype=float)
        for i in range(1, steps + 1):
            Z = np.random.standard_normal(sims)
            full_paths[:, i] = full_paths[:, i - 1] * np.exp(
                (r - 0.5 * sigma_use**2) * dt + sigma_use * math.sqrt(dt) * Z
            )

    elif model == 'merton':
        lam = float(params.get('lam', 0.15))
        mu_j = float(params.get('mu_j', -0.02))
        sigma_j = float(params.get('sigma_j', 0.12))
        ST = simulate_merton_jump(S0, r, sigma_use, T, sims,
                                  lam=lam, mu_j=mu_j, sigma_j=sigma_j, seed=seed)
    elif model == 'heston':
        v0 = sigma_use**2
        kappa = float(params.get('kappa', 1.5))
        theta = float(params.get('theta', v0))
        xi = float(params.get('xi', 0.6))
        rho = float(params.get('rho', -0.6))
        steps = int(params.get('steps', 200))
        ST = simulate_heston(S0, r, v0, kappa, theta, xi, rho, T, sims,
                             steps=steps, seed=seed)
    else:
        raise ValueError(f"Unknown model: {model}")

    # ================= P/L computation =================
    payoff_short = np.maximum(k_short - ST, 0.0)
    payoff_long = np.maximum(k_long - ST, 0.0)
    net_per_share = payoff_long - payoff_short + premium
    pnl_per_contract = net_per_share * 100.0  # 1 contract = 100 shares

    avg_pnl = float(np.mean(pnl_per_contract))
    std_pnl = float(np.std(pnl_per_contract, ddof=1))
    win_rate = float(np.sum(pnl_per_contract > 0) / sims * 100.0)
    max_profit = float(np.max(pnl_per_contract))
    max_loss = float(np.min(pnl_per_contract))

    # ================= Probabilities =================
    if model == 'gbm':
        prob_expire_short = prob_expire_below_gbm(S0, k_short, r, sigma_use, T)
        prob_expire_long = prob_expire_below_gbm(S0, k_long, r, sigma_use, T)
    else:
        # approximate from terminal distribution
        prob_expire_short = float(np.mean(ST < k_short))
        prob_expire_long = float(np.mean(ST < k_long))

    # Touch probabilities
    if full_paths is not None:
        prob_touch_short = prob_touch_from_paths(full_paths, k_short)
        prob_touch_long = prob_touch_from_paths(full_paths, k_long)
    else:
        # fallback crude approximation: scale terminal probability
        prob_touch_short = min(1.0, prob_expire_short * 1.3)
        prob_touch_long = min(1.0, prob_expire_long * 1.3)

    # ================= Greeks =================
    if iv_surface is not None:
        iv_short = iv_surface(T, k_short)
        iv_long = iv_surface(T, k_long)
    else:
        iv_short = sigma_use
        iv_long = sigma_use

    greeks = {
        'short': {
            'delta': bs_delta_put(S0, k_short, T, r, iv_short),
            'gamma': bs_gamma(S0, k_short, T, r, iv_short),
            'vega': bs_vega(S0, k_short, T, r, iv_short),
            'theta': bs_theta_put(S0, k_short, T, r, iv_short)
        },
        'long': {
            'delta': bs_delta_put(S0, k_long, T, r, iv_long),
            'gamma': bs_gamma(S0, k_long, T, r, iv_long),
            'vega': bs_vega(S0, k_long, T, r, iv_long),
            'theta': bs_theta_put(S0, k_long, T, r, iv_long)
        }
    }

    # ================= Expected move (straddle) =================
    expected_move_pct = None
    if iv_surface is not None:
        try:
            # reuse df_iv + earliest expiry for approximate straddle
            df_iv = fetch_option_chain_ivs(tk, list(tk.options)[:3], r=r)
            if not df_iv.empty:
                earliest = df_iv['expiry'].iloc[0]
                df_e = df_iv[df_iv['expiry'] == earliest]
                atm_idx = (df_e['strike'] - S0).abs().idxmin()
                atm_row = df_e.loc[atm_idx]
                Katm = float(atm_row['strike'])
                iv_atm = float(atm_row['iv'])
                T_e = float(atm_row['dte'])

                F_e = S0 * math.exp((r - 0.0) * T_e)
                put_mid = float(atm_row['mid'])
                call_price = black76_call(F_e, Katm, T_e, r, iv_atm)
                straddle_price = call_price + put_mid
                expected_move_pct = (straddle_price / S0) * 100.0
        except Exception:
            expected_move_pct = None

    results = {
        'S0': S0,
        'sigma_use': sigma_use,
        'model': model,
        'avg_pnl': avg_pnl,
        'std_pnl': std_pnl,
        'win_rate_pct': win_rate,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'prob_expire_short': prob_expire_short,
        'prob_expire_long': prob_expire_long,
        'prob_touch_short': prob_touch_short,
        'prob_touch_long': prob_touch_long,
        'expected_move_pct': expected_move_pct,
        'greeks': greeks,
        'pnl_samples': pnl_per_contract
    }
    return results

# ============================================================
# Sensitivity analysis helper
# ============================================================

def sensitivity_analysis(base_params: Dict, vary_param: str, values: list) -> pd.DataFrame:
    """
    Run multiple simulations, varying a single parameter while
    keeping all others fixed.

    base_params : dict
        Baseline parameters for run_analysis (ticker, strikes, premium, etc.)
    vary_param : str
        Name of the parameter key in base_params you want to vary
        (e.g. 'premium', 'k_short', 'k_long', 'days', 'sims', 'r', 'model').
    values : list
        List of values to assign to that parameter across runs.

    Returns
    -------
    DataFrame summarizing each run with columns:
      param, avg_pnl, win_rate_pct, max_profit, max_loss,
      prob_expire_short, prob_expire_long, prob_touch_short, prob_touch_long
    """
    rows = []
    for val in values:
        # copy base params and override the one we're varying
        params = base_params.copy()
        params[vary_param] = val

        print(f"Running sensitivity case: {vary_param} = {val}")
        res = run_analysis(params)

        rows.append({
            'param': val,
            'avg_pnl': res['avg_pnl'],
            'std_pnl': res['std_pnl'],
            'win_rate_pct': res['win_rate_pct'],
            'max_profit': res['max_profit'],
            'max_loss': res['max_loss'],
            'prob_expire_short': res['prob_expire_short'],
            'prob_expire_long': res['prob_expire_long'],
            'prob_touch_short': res['prob_touch_short'],
            'prob_touch_long': res['prob_touch_long'],
            'sigma_use': res['sigma_use'],
        })

    return pd.DataFrame(rows)

# ============================================================
# Spread scanner helper (find EV >= 0 & win_rate >= X%)
# ============================================================

def find_viable_spreads(
    base_params: Dict,
    short_strikes: list,
    premiums: list,
    width: float,
    min_win_rate: float = 70.0,
    min_ev: float = 0.0
) -> pd.DataFrame:
    """
    Scan over a grid of (short strike, premium) for a fixed width, and
    return spreads that have:
        - avg_pnl >= min_ev
        - win_rate_pct >= min_win_rate

    base_params : dict
        Baseline params for run_analysis (ticker, days, sims, model, r, iv_from_market, etc.)
        k_short, k_long, premium in base_params will be overridden.
    short_strikes : list[float]
        List of short put strikes to test.
    premiums : list[float]
        List of net credits (per share) to test.
    width : float
        Distance between short and long strikes (e.g. 10.0 for a 10-point spread).
    min_win_rate : float
        Minimum simulated win rate (%) required for a spread to be considered.
    min_ev : float
        Minimum average P/L per contract required (e.g. 0.0 for non-negative EV).

    Returns
    -------
    DataFrame with columns:
        k_short, k_long, premium, avg_pnl, win_rate_pct, max_profit, max_loss,
        prob_expire_short, prob_touch_short, sigma_use
    """
    rows = []

    for ks in short_strikes:
        for prem in premiums:
            params = base_params.copy()
            params['k_short'] = ks
            params['k_long'] = ks - width
            params['premium'] = prem

            print(f"Testing spread: short={ks}, long={ks - width}, premium={prem}")
            res = run_analysis(params)

            if res['avg_pnl'] >= min_ev and res['win_rate_pct'] >= min_win_rate:
                rows.append({
                    'k_short': ks,
                    'k_long': ks - width,
                    'premium': prem,
                    'avg_pnl': res['avg_pnl'],
                    'win_rate_pct': res['win_rate_pct'],
                    'max_profit': res['max_profit'],
                    'max_loss': res['max_loss'],
                    'prob_expire_short': res['prob_expire_short'],
                    'prob_touch_short': res['prob_touch_short'],
                    'sigma_use': res['sigma_use'],
                })

    return pd.DataFrame(rows)

# ============================================================
# CLI wrapper
# ============================================================

def parse_cli(argv=None):
    p = argparse.ArgumentParser(description="Enhanced SPX put credit spread Monte Carlo analyzer.")
    p.add_argument("--ticker", default="^GSPC")
    p.add_argument("--k-short", type=float, required=True, help="Short put strike")
    p.add_argument("--k-long", type=float, required=True, help="Long put strike")
    p.add_argument("--premium", type=float, required=True, help="Net credit per share")
    p.add_argument("--days", type=int, default=7, help="Days to expiration")
    p.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo paths")
    p.add_argument("--model", choices=['gbm', 'merton', 'heston'], default='merton', help="Underlying model")
    p.add_argument("--r", type=float, default=0.02, help="Risk-free rate")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--no-iv", dest="iv_from_market", action="store_false",
                   help="Do NOT use market IV, fall back to historical vol")
    p.add_argument("--save", default=None, help="Optional PNG path to save histogram")
    return p.parse_args(argv)


def main_cli():
    args = parse_cli()
    params = {
        'ticker': args.ticker,
        'k_short': args.k_short,
        'k_long': args.k_long,
        'premium': args.premium,
        'days': args.days,
        'sims': args.sims,
        'model': args.model,
        'r': args.r,
        'seed': args.seed,
        'iv_from_market': args.iv_from_market
    }

    print("Running analysis... this may take a bit depending on sims and model.")
    res = run_analysis(params)

    print("\n--- Summary ---")
    print(f"Spot: {res['S0']:.2f}")
    print(f"Model: {res['model']}, sigma used: {res['sigma_use']:.3f}")
    print(f"Avg P/L per contract: ${res['avg_pnl']:.2f}")
    print(f"Std dev P/L: ${res['std_pnl']:.2f}")
    print(f"Win rate (simulated): {res['win_rate_pct']:.2f}%")
    print(f"Max profit: ${res['max_profit']:.2f}")
    print(f"Max loss: ${res['max_loss']:.2f}")

# ---- Break-even win rate calculation ----
    max_profit = res['max_profit']          # should be positive
    max_loss = res['max_loss']              # should be negative for a credit spread
    loss_abs = -max_loss if max_loss < 0 else 0.0

    breakeven_win_rate = None
    if max_profit > 0 and loss_abs > 0:
        # Solve p*P + (1-p)*(-L) = 0  =>  p = L / (P + L)
        breakeven_win_rate = loss_abs / (max_profit + loss_abs)
        print(f"Required win rate to break even (given max profit/loss): {breakeven_win_rate*100:.2f}%")
    else:
        print("Unable to compute break-even win rate (degenerate max profit/loss).")

    print(f"Prob expire below short strike: {res['prob_expire_short']:.4f}")
    print(f"Prob expire below long strike: {res['prob_expire_long']:.4f}")
    print(f"Prob touch short strike (approx): {res['prob_touch_short']:.4f}")
    print(f"Prob touch long strike (approx): {res['prob_touch_long']:.4f}")
    if res['expected_move_pct'] is not None:
        print(f"Approx ATM straddle expected move: {res['expected_move_pct']:.2f}%")

    print("\nGreeks (per share, BS-style):")
    for leg, g in res['greeks'].items():
        print(f"  {leg:5s}: delta={g['delta']:.4f}, gamma={g['gamma']:.6f}, "
              f"vega={g['vega']:.4f}, theta/day={g['theta']:.4f}")

    # Plot histogram
    pnl = res['pnl_samples']
    plt.figure(figsize=(10, 6))
    plt.hist(pnl, bins=60, color="skyblue", edgecolor="black")
    plt.axvline(res['avg_pnl'], color="red", linestyle="--",
                label=f"Avg P/L: ${res['avg_pnl']:.2f}")
    plt.title(f"Put Credit Spread P&L ({args.model})")
    plt.xlabel("P&L per contract ($)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved histogram to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main_cli()
