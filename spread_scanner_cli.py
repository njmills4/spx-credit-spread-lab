#!/usr/bin/env python3
"""
spread_scanner_cli.py

CLI-powered scanner for put credit spreads using enhanced_spx_sim.py.

Features:
 - Scans a grid of (short strike, premium) for a fixed spread width
 - Uses run_analysis() from enhanced_spx_sim.py for each combination
 - Filters for spreads with:
      avg_pnl >= min_ev
      win_rate_pct >= min_win_rate
 - Saves:
      * full_results CSV
      * candidates CSV (passing filters)
      * PNG heatmap of avg_pnl vs (k_short, premium)

Example usage:

python spread_scanner_cli.py \
  --ticker ^GSPC \
  --days 44 \
  --width 10 \
  --short-min 6600 --short-max 6800 --short-step 10 \
  --prem-min 1.0 --prem-max 5.0 --prem-step 0.25 \
  --min-win-rate 70 \
  --min-ev 0 \
  --model merton \
  --sims 100000 \
  --heatmap-out ev_heatmap.png \
  --csv-prefix ev_scan
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enhanced_spx_sim import run_analysis


def build_grid(min_val: float, max_val: float, step: float) -> list:
    """Build an inclusive grid from min_val to max_val with given step."""
    n_steps = int(round((max_val - min_val) / step)) + 1
    return [min_val + i * step for i in range(n_steps)]


def parse_cli(argv=None):
    p = argparse.ArgumentParser(description="Scan SPX put credit spreads over short strikes & premiums.")

    # Underlying & environment
    p.add_argument("--ticker", default="^GSPC", help="Underlying ticker (default: ^GSPC)")
    p.add_argument("--days", type=int, default=7, help="Days to expiration for all spreads (default: 7)")
    p.add_argument("--sims", type=int, default=20000, help="Number of Monte Carlo sims per spread (default: 20000)")
    p.add_argument("--model", choices=['gbm', 'merton', 'heston'], default='merton', help="Underlying model")
    p.add_argument("--r", type=float, default=0.02, help="Risk-free rate (annualized, default: 0.02)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--no-iv", dest="iv_from_market", action="store_false",
                   help="Disable market IV surface, use historical vol instead")

    # Spread geometry
    p.add_argument("--width", type=float, default=10.0,
                   help="Width of the put spread (short strike - long strike). Default: 10")

    # Short strike grid
    p.add_argument("--short-min", type=float, required=True, help="Minimum short strike to scan")
    p.add_argument("--short-max", type=float, required=True, help="Maximum short strike to scan")
    p.add_argument("--short-step", type=float, default=25.0, help="Step size for short strikes (default: 25)")

    # Premium grid
    p.add_argument("--prem-min", type=float, required=True, help="Minimum net premium (credit per share) to scan")
    p.add_argument("--prem-max", type=float, required=True, help="Maximum net premium (credit per share) to scan")
    p.add_argument("--prem-step", type=float, default=0.25, help="Step size for premium (default: 0.25)")

    # Filters
    p.add_argument("--min-win-rate", type=float, default=70.0, help="Min win rate %% for candidate (default: 70)")
    p.add_argument("--min-ev", type=float, default=0.0,
                   help="Min avg P/L per contract for candidate (default: 0)")

    # Outputs
    p.add_argument("--heatmap-out", default="ev_heatmap.png",
                   help="Path to save EV heatmap PNG (default: ev_heatmap.png)")
    p.add_argument("--csv-prefix", default="scan_results",
                   help="Prefix for CSV outputs (default: scan_results)")

    return p.parse_args(argv)


def generate_heatmap(df: pd.DataFrame, heatmap_path: str):
    """
    Generate a PNG heatmap of avg_pnl as a function of (k_short, premium).

    df must contain columns: k_short, premium, avg_pnl
    """
    if df.empty:
        print("No data for heatmap.")
        return

    # Pivot to matrix: rows = premium, cols = k_short
    pivot = df.pivot_table(index="premium", columns="k_short", values="avg_pnl")
    premiums = pivot.index.values
    strikes = pivot.columns.values
    Z = pivot.values

    plt.figure(figsize=(10, 6))
    # imshow expects [row, col] => [premium_index, strike_index]
    # origin = 'lower' so low premiums appear at bottom
    im = plt.imshow(
        Z,
        origin='lower',
        aspect='auto',
        extent=[strikes.min(), strikes.max(), premiums.min(), premiums.max()]
    )
    plt.colorbar(im, label="Avg P/L per contract ($)")
    plt.xlabel("Short Put Strike")
    plt.ylabel("Net Premium (per share)")
    plt.title("Expected P/L Heatmap for Put Credit Spreads")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved EV heatmap to {heatmap_path}")


def main(argv=None):
    args = parse_cli(argv)

    # Build grids
    short_strikes = build_grid(args.short_min, args.short_max, args.short_step)
    premiums = build_grid(args.prem_min, args.prem_max, args.prem_step)

    print("Short strike grid:", short_strikes)
    print("Premium grid:", premiums)
    print(f"Spread width: {args.width}")
    print(f"Model: {args.model}, Sims per spread: {args.sims}")
    print(f"Min candidate win rate: {args.min_win_rate}%, Min candidate EV: {args.min_ev}")
    print("Starting scan...\n")

    all_rows = []
    candidate_rows = []

    # Baseline parameters for each run; k_short, k_long, premium will be overridden
    base_params = {
        'ticker': args.ticker,
        'k_short': 0.0,        # placeholder
        'k_long': 0.0,         # placeholder
        'premium': 0.0,        # placeholder
        'days': args.days,
        'sims': args.sims,
        'model': args.model,
        'r': args.r,
        'seed': args.seed,
        'iv_from_market': args.iv_from_market
    }

    total = len(short_strikes) * len(premiums)
    count = 0

    for ks in short_strikes:
        for prem in premiums:
            count += 1
            print(f"[{count}/{total}] short={ks}, long={ks - args.width}, premium={prem}")
            params = base_params.copy()
            params['k_short'] = ks
            params['k_long'] = ks - args.width
            params['premium'] = prem

            res = run_analysis(params)

            row = {
                'k_short': ks,
                'k_long': ks - args.width,
                'premium': prem,
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
            }
            all_rows.append(row)

            if res['avg_pnl'] >= args.min_ev and res['win_rate_pct'] >= args.min_win_rate:
                candidate_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    df_candidates = pd.DataFrame(candidate_rows)

    # Save CSVs
    all_path = f"{args.csv_prefix}_all.csv"
    cand_path = f"{args.csv_prefix}_candidates.csv"
    df_all.to_csv(all_path, index=False)
    print(f"\nSaved all scan results to {all_path}")

    if df_candidates.empty:
        print("No spreads met the candidate criteria (EV >= min_ev and win_rate_pct >= min_win_rate).")
    else:
        print("\n=== Candidate Spreads (meeting criteria) ===")
        print(df_candidates.to_string(index=False))
        df_candidates.to_csv(cand_path, index=False)
        print(f"Saved candidate spreads to {cand_path}")

    # Generate EV heatmap from all rows
    generate_heatmap(df_all, args.heatmap_out)


if __name__ == "__main__":
    main()
