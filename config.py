# config.py -- single source of truth (replaces parameters.py)
FLAT, LONG, SHORT = 0, 1, -1          # position += action requires these exact values

# ---- data / paths (relative) ----
DATA_PATH = "data/btc_30m.csv"        # Time, Open, High, Low, Close, Volume (+ indicators, ignored)
LOG_DIR = "logs"
PLOT_DIR = "results"

# ---- strategy ----
MODE = "predicted"                    # "predicted" | "ideal"
DEPTH = 6
LEN_PREDS = 2 ** DEPTH + 1            # 65 bars: bridge length and forecast horizon
SHIFT = 13                            # bridge[SHIFT:] feeds the DP (kept from original)
N_COLS = LEN_PREDS - SHIFT            # 52 planned bars per window
RF_LAGS = 3                           # RandomForest input window
N_TREES = 50
N_ITERS = 100                         # Adam iterations per window (main speed knob)
STD_REL = 0.0005                      # bridge per-bar sigma as a FRACTION of price (was absolute 0.005)
VOL_WINDOW = 50                       # rolling window for adaptive doubt
MIN_DOUBT = 1e-4
DP_UP_PROB = 0.125
DP_DOWN_PROB = 0.125

# ---- sizing / costs (shared by DP and backtest) ----
CAPITAL = 10_000.0
UNITS = 0.1                           # BTC held when in position (~half of capital at ~$55k)
COST_PER_SIDE = 0.0005                # 5 bps on traded value
DP_COST_PER_SIDE = 0.0015
SEED = 0