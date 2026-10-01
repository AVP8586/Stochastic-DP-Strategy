import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import *


def max_drawdown(equity):
    rolling_max = np.maximum.accumulate(equity)
    return float(np.max((rolling_max - equity) / rolling_max))


def calculate_calmar_ratio(equity):
    total_return = (equity[-1] - equity[0]) / equity[0]
    mdd = max_drawdown(equity)
    return 0.0 if mdd == 0 else total_return / mdd


def engine(df: pd.DataFrame):
    """Continuous mark-to-market backtest over the whole file (capital is NOT reset).
    Position at bar i is entered at Open[i] and earns Open[i+1]-Open[i]. Fee on traded value."""
    price = df["Open"].to_numpy(float)
    delta = df["signals"].to_numpy(float)
    pos = df["position"].to_numpy(float)

    fees = np.abs(delta) * price * COST_PER_SIDE
    step = np.zeros(len(df))
    step[:-1] = pos[:-1] * np.diff(price)
    net = step - fees                                   # per-bar PnL in $

    # Benchmark: buy UNITS at the first bar, hold to the last
    b_step = np.zeros(len(df))
    b_step[:-1] = UNITS * np.diff(price)
    b_net = b_step.copy()
    b_net[0] -= UNITS * price[0] * COST_PER_SIDE
    b_net[-1] -= UNITS * price[-1] * COST_PER_SIDE

    equity = CAPITAL + np.r_[0.0, np.cumsum(net)]
    b_equity = CAPITAL + np.r_[0.0, np.cumsum(b_net)]

    day = df["Time"].dt.date.to_numpy()
    daily = pd.Series(net).groupby(day).sum()
    b_daily = pd.Series(b_net).groupby(day).sum()

    prev = np.r_[0.0, pos[:-1]]
    return dict(
        net=net, equity=equity, b_equity=b_equity, daily=daily, b_daily=b_daily,
        pnl=equity[-1] - CAPITAL, b_pnl=b_equity[-1] - CAPITAL,
        mdd=max_drawdown(equity), b_mdd=max_drawdown(b_equity),
        calmar=calculate_calmar_ratio(equity),
        entries=int(((delta != 0) & (prev == 0)).sum()), fills=int((delta != 0).sum()),
        fees=float(fees.sum()),
    )


def report(name, r):
    d, bd = r["daily"], r["b_daily"]
    print(f"\n============ {name} ============")
    print(f"Net PnL:          ${r['pnl']:.2f}  ({r['pnl'] / CAPITAL * 100:.2f}% on ${CAPITAL:,.0f})")
    print(f"Gross PnL:        ${r['pnl'] + r['fees']:.2f}  (before fees)")
    print(f"Benchmark PnL:    ${r['b_pnl']:.2f}  ({r['b_pnl'] / CAPITAL * 100:.2f}%)")
    print(f"Max drawdown:     {r['mdd'] * 100:.2f}%  (benchmark {r['b_mdd'] * 100:.2f}%)")
    print(f"Calmar:           {r['calmar']:.2f}")
    print(f"Entries / fills:  {r['entries']} / {r['fills']}   fees paid: ${r['fees']:.2f}")
    print(f"Days:             {len(d)}  win {100 * (d > 0).mean():.1f}%  loss {100 * (d < 0).mean():.1f}%")
    print("Daily PnL ($):");  print(d.describe().round(2).to_string())
    print(f"Benchmark daily win%: {100 * (bd > 0).mean():.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["predicted", "ideal", "both"], default=MODE)
    ap.add_argument("--tag", default="", help="matches --tag used in sdp_strat.py")
    a = ap.parse_args()
    modes = ["predicted", "ideal"] if a.mode == "both" else [a.mode]

    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(14, 7))
    for m in modes:
        df = pd.read_csv(os.path.join(LOG_DIR, f"sdp_{m}{a.tag}.csv"), parse_dates=["Time"])
        r = engine(df)
        report(f"SDP [{m}]", r)
        plt.plot(df["Time"].iloc[:1].tolist() + df["Time"].tolist(), r["equity"], label=f"SDP {m}")
    plt.plot(df["Time"].iloc[:1].tolist() + df["Time"].tolist(), r["b_equity"],
             color="gray", ls="--", label=f"Buy & hold {UNITS} BTC")
    plt.axhline(CAPITAL, color="red", lw=1)
    plt.title("SDP strategy vs benchmark"); plt.ylabel("Portfolio value ($)"); plt.legend(); plt.grid(True)
    out = os.path.join(PLOT_DIR, f"equity{a.tag}.png")
    plt.savefig(out); print(f"\nsaved {out}")