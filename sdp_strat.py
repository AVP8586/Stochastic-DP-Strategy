import argparse
import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import *

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

COLS = [f"Column_{k + 1}" for k in range(N_COLS)]
ACTION_MAP = {"BUY": 1, "SELL": -1, "SHORT": -1, "COVER": 1, "WAIT/HOLD": 0}


# ------------------------------------------------------------------ data
def load_data(path: str = DATA_PATH, start: int = 0, end: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["Time", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"missing columns: {missing}"
    df = df[need].copy()                      # indicator columns are dropped
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)
    return df.iloc[start:end].reset_index(drop=True)


def window_starts(n: int):
    """Shared by GenWindow and ExpertAction so both walk the same windows."""
    return range(LEN_PREDS, n - LEN_PREDS + SHIFT, LEN_PREDS - SHIFT)


# ------------------------------------------------------------------ endpoint forecast
def predictpoint(query, prices: np.ndarray, bandwidth: int) -> float:
    """RandomForest recursive forecast of the bar at end+bandwidth-1."""
    start, end, _ = query
    X_train, y_train = [], []
    for j in range(start + RF_LAGS, end):
        X_train.append(prices[j - RF_LAGS:j])
        y_train.append(prices[j])
    model = RandomForestRegressor(n_estimators=N_TREES, random_state=42)
    model.fit(np.array(X_train), np.array(y_train))

    current_window = prices[end - RF_LAGS:end].tolist()
    pred = None
    for _ in range(bandwidth):
        pred = model.predict(np.array(current_window[-RF_LAGS:]).reshape(1, -1))[0]
        current_window.append(pred)
    return pred


# ------------------------------------------------------------------ bridge
def dtw(x, y):
    x, y = np.array(x), np.array(y)
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = abs(x[i - 1] - y[j - 1]) + min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1])
    return dp[n][m]


class BrownianBridgeFractalNaive:
    def __init__(self, start: Tuple[float, float], end: Tuple[float, float],
                 standard_deviation: float = 0.005, depth: int = 10,
                 tail: Optional[np.ndarray] = None, head: Optional[np.ndarray] = None,
                 param: float = 0.1) -> None:
        self.start, self.end = start, end
        self.standard_deviation = standard_deviation
        self.depth = depth
        self.tail, self.head, self.param = tail, head, param

    def _build_intermediate(self, times, values):
        new_time = 0.5 * (times[:-1] + times[1:])
        m = 0.5 * (values[:-1] + values[1:])
        # NOTE: drift only applies when head is given; forecastday passes head=None,
        # so `param` is currently inert (see README notes).
        if self.tail is not None and self.head is not None:
            dist = dtw(self.tail, self.head)
            alpha = np.clip(1 / (1 + np.exp(-self.param * (0.2 - dist))), 0, 1)
            mu = m + alpha * (np.mean(self.tail) - np.mean(self.head))
        else:
            mu = m
        delta = times[1:] - times[:-1]
        time_scale = np.sqrt((new_time - times[:-1]) * (times[1:] - new_time) / delta)
        new_values = mu + time_scale * self.standard_deviation * np.random.normal(0, 1, size=len(new_time))

        times_out = np.empty(len(times) + len(new_time))
        values_out = np.empty(len(values) + len(new_values))
        times_out[0::2], times_out[1::2] = times, new_time
        values_out[0::2], values_out[1::2] = values, new_values
        return times_out, values_out

    def sample(self) -> np.ndarray:
        times = np.array([self.start[0], self.end[0]])
        values = np.array([self.start[1], self.end[1]])
        for _ in range(self.depth):
            times, values = self._build_intermediate(times, values)
        return values


def smooth_trend_similarity(tail, head, alpha=6.0):
    st, sh = np.tanh(alpha * np.diff(tail)), np.tanh(alpha * np.diff(head))
    return float(np.mean(st * sh))


def normalized_mse(tail, head, eps=1e-8):
    return float(np.mean((tail - head) ** 2)) / (np.var(tail) + eps)


def fitness_from_head(tail, head, alpha=6.0, w1=1.0, w2=1.0):
    return w1 * smooth_trend_similarity(tail, head, alpha) - w2 * normalized_mse(tail, head)


class SimpleAdam:
    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = self.v = 0
        self.t = 0
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def forecastday(prices: np.ndarray, idx: int, mode: Optional[str] = None,
                param_init=0.1, std_rel=STD_REL, n_iters=N_ITERS, lr=1e-2,
                eps_fd=1e-4, n_eval=2, alpha=6.0, w1=1.0, w2=1.0):
    """Build the bridge for the window starting at bar idx.
    predicted: endpoint = RF forecast.  ideal: endpoint = true future price (oracle)."""
    mode = mode or MODE
    bandwidth = 2 ** DEPTH + 1
    start, end, target = idx - bandwidth, idx, idx + bandwidth

    if mode == "ideal":
        pred = prices[min(target - 1, len(prices) - 1)]      # bar predictpoint would estimate
    else:
        pred = predictpoint([start, end, target], prices, bandwidth)
    end1 = (target, pred)
    start1 = (end, prices[end - 1])
    std = std_rel * start1[1]                                 # relative noise, works at any price level
    lookback = int(0.5 * bandwidth)
    tail = np.array(prices[end - lookback:end])               # FIX: bars before idx, not end of day

    def make_bridge(p):
        return BrownianBridgeFractalNaive(start=start1, end=end1, standard_deviation=std,
                                          depth=DEPTH, param=p, tail=tail, head=None).sample()

    def loss_fn(p):
        return np.mean([-fitness_from_head(tail, make_bridge(p)[:lookback], alpha, w1, w2)
                        for _ in range(n_eval)])

    adam, param = SimpleAdam(lr=lr), float(param_init)
    best_loss, best_param, best_bridge = np.inf, param, None
    for _ in range(n_iters):
        grad = (loss_fn(param + eps_fd) - loss_fn(param - eps_fd)) / (2 * eps_fd)
        param = float(np.clip(param - adam.step(grad), 0.01, 0.15))
        cur_loss = loss_fn(param)
        if cur_loss < best_loss:
            best_loss, best_param, best_bridge = cur_loss, param, make_bridge(param)
    if best_bridge is None:
        best_bridge = make_bridge(param)
    return best_bridge, best_param, end, target


def plot_bridge(prices, bridge, end, target, mode, save_path=None):
    lo, hi = max(0, end - LEN_PREDS), min(len(prices), target + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(range(lo, hi), prices[lo:hi], color="gray", label="actual Open")
    plt.plot(np.linspace(end, target, len(bridge)), bridge, color="blue", label=f"bridge ({mode})")
    plt.scatter([target], [bridge[-1]], color="red", zorder=5, label="endpoint")
    plt.axvline(end, color="k", ls=":")
    plt.title(f"window @ bar {end} [{mode}]")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class GenWindow:
    def __init__(self, data: pd.DataFrame, mode=MODE, plot_dir=None, plot_every=0, n_iters=N_ITERS):
        self.data, self.mode = data, mode
        self.prices = data["Open"].to_numpy(dtype=float)
        self.plot_dir, self.plot_every, self.n_iters = plot_dir, plot_every, n_iters

    def generate(self):
        for w, i in enumerate(window_starts(len(self.prices))):
            bridge, _, end, target = forecastday(self.prices, i, mode=self.mode, n_iters=self.n_iters)
            if self.plot_dir and self.plot_every and w % self.plot_every == 0:
                os.makedirs(self.plot_dir, exist_ok=True)
                plot_bridge(self.prices, bridge, end, target, self.mode,
                            os.path.join(self.plot_dir, f"bridge_{self.mode}_{i}.png"))
            self.data.loc[i, COLS] = bridge[SHIFT:]


# ------------------------------------------------------------------ stochastic DP
class StochasticDPSolver:
    def __init__(self, doubt_pct=0.005, up_prob=0.3, down_prob=0.3, cost_bips=DP_COST_PER_SIDE):
        self.doubt_tick = 1 + doubt_pct
        self.up_prob, self.down_prob = up_prob, down_prob
        self.centre_prob = 1 - up_prob - down_prob
        self.tx_cost_pct = cost_bips
        self.price_tree, self.prices, self.n = {}, [], 0
        self.STATE_TO_INDEX_MAP = {FLAT: 0, LONG: 1, SHORT: 2}

    def _build_stochastic_price_tree(self):
        self.price_tree = {}
        for t, c in enumerate(self.prices):
            self.price_tree[t] = [round(c / self.doubt_tick, 4), c, round(c * self.doubt_tick, 4)]

    def solve(self, prices, start_posn=FLAT, initial_entry_price=0.0):
        self.n, self.prices = len(prices), prices
        if self.n == 0:
            return [], start_posn, initial_entry_price
        self._build_stochastic_price_tree()
        V, policy = self._calculate_policy()
        first = self._decide_first_action(policy, start_posn)
        return self._get_final_state_from_policy(first, policy, start_posn, initial_entry_price)

    def _calculate_policy(self):
        V = np.zeros((self.n + 1, 3, 3))
        policy = np.full((self.n, 3, 3), "WAIT/HOLD", dtype=object)
        for t in range(self.n - 1, -1, -1):
            efv = {s: (self.down_prob * V[t + 1, s, 0] + self.centre_prob * V[t + 1, s, 1]
                       + self.up_prob * V[t + 1, s, 2]) for s in range(3)}
            for j, price_t in enumerate(self.price_tree[t]):
                cost = price_t * self.tx_cost_pct

                q_wait, q_buy, q_short = efv[0], -cost + efv[1], -cost + efv[2]
                V[t, 0, j] = max(q_wait, q_buy, q_short)
                if V[t, 0, j] == q_buy: policy[t, 0, j] = "BUY"
                elif V[t, 0, j] == q_short: policy[t, 0, j] = "SHORT"

                exp_change = 0
                if t < self.n - 1:
                    nxt = self.price_tree[t + 1]
                    exp_change = (self.down_prob * nxt[0] + self.centre_prob * nxt[1]
                                  + self.up_prob * nxt[2]) - price_t

                q_hold_long, q_sell = exp_change + efv[1], -cost + efv[0]
                V[t, 1, j] = max(q_hold_long, q_sell)
                if V[t, 1, j] == q_sell: policy[t, 1, j] = "SELL"

                q_hold_short, q_cover = -exp_change + efv[2], -cost + efv[0]
                V[t, 2, j] = max(q_hold_short, q_cover)
                if V[t, 2, j] == q_cover: policy[t, 2, j] = "COVER"
        return V, policy

    def _decide_first_action(self, policy, start_posn):
        node = np.argmin([abs(p - self.prices[0]) for p in self.price_tree[0]])
        return policy[0, self.STATE_TO_INDEX_MAP[start_posn], node]

    def _get_final_state_from_policy(self, first_action, policy, start_posn, entry_price):
        action_seq, pos = [], start_posn
        a = ACTION_MAP[first_action]
        action_seq.append(a)
        prev, pos = pos, pos + a
        if prev == FLAT and pos != FLAT: entry_price = self.prices[0]
        elif prev != FLAT and pos == FLAT: entry_price = 0.0

        for t in range(1, self.n):
            action_str = policy[t, self.STATE_TO_INDEX_MAP[pos], 1]
            if (pos == LONG and action_str == "BUY") or (pos == SHORT and action_str == "SHORT"):
                action_str = "WAIT/HOLD"
            a = ACTION_MAP[action_str]
            action_seq.append(a)
            prev, pos = pos, pos + a
            if prev == FLAT and pos != FLAT: entry_price = self.prices[t]
            elif prev != FLAT and pos == FLAT: entry_price = 0.0
        return action_seq, pos, entry_price


class ExpertAction:
    def __init__(self, data: pd.DataFrame):
        self.data, self.n = data, len(data)

    def gen_len(self):
        actions = np.zeros(self.n, dtype=int)
        pos, entry = FLAT, 0.0
        for i in window_starts(self.n):
            prices = list(self.data.loc[i, COLS].astype(float))
            doubt = self.data.loc[i, "adaptive_doubt_pct"]
            solver = StochasticDPSolver(doubt_pct=doubt / 3, up_prob=DP_UP_PROB,
                                        down_prob=DP_DOWN_PROB, cost_bips=DP_COST_PER_SIDE)
            seq, pos, entry = solver.solve(prices, start_posn=pos, initial_entry_price=entry)
            for k, a in enumerate(seq):
                if i + k < self.n:
                    actions[i + k] = a
        self.data["expert_action"] = actions


# ------------------------------------------------------------------ pipeline
class TradingEnv:
    def __init__(self, df_path=DATA_PATH, mode=MODE, start=0, end=None, tag="", plot_every=0, n_iters=N_ITERS):
        self.mode, self.tag = mode, tag
        self.df = load_data(df_path, start, end)

        # placeholder forecast columns = current Open (used for rows with no window)
        fill = pd.DataFrame(np.repeat(self.df[["Open"]].values, N_COLS, axis=1),
                            columns=COLS, index=self.df.index)
        self.df = pd.concat([self.df, fill], axis=1)

        plot_dir = os.path.join(PLOT_DIR, f"bridges_{mode}")
        GenWindow(self.df, mode, plot_dir, plot_every, n_iters).generate()

        # FIX: shift(1) so vol at bar i only uses closes known before bar i opens
        rets = self.df["Close"].pct_change().shift(1)
        vol = rets.rolling(window=VOL_WINDOW).std().bfill()
        self.df["adaptive_doubt_pct"] = vol.clip(lower=MIN_DOUBT)

        ExpertAction(self.df).gen_len()

        # backtest.py contract: signals = delta in units, position = net units
        self.df["signals"] = self.df["expert_action"] * UNITS
        self.df["position"] = self.df["expert_action"].cumsum() * UNITS
        assert self.df["position"].abs().max() <= UNITS + 1e-9, "position left {-UNITS, 0, UNITS}"
        print(self.df["expert_action"].value_counts())

    def save_logs(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"sdp_{self.mode}{self.tag}.csv")
        self.df.drop(columns=COLS).to_csv(path, index=False)
        print(f"saved {path}")


def run(mode=MODE, data=DATA_PATH, **kw):
    np.random.seed(SEED)
    env = TradingEnv(df_path=data, mode=mode, **kw)
    env.save_logs()
    return env.df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["predicted", "ideal", "both"], default=MODE)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--slice", default=":", help="bar range a:b, e.g. :40000 (tune) or 40000: (test)")
    ap.add_argument("--tag", default="", help="suffix for log files, e.g. _train / _test")
    ap.add_argument("--plot-every", type=int, default=0, help="save a bridge plot every k windows")
    ap.add_argument("--n-iters", type=int, default=N_ITERS)
    a = ap.parse_args()
    for m in (["predicted", "ideal"] if a.mode == "both" else [a.mode]):
        lo, hi = (int(x) if x else None for x in a.slice.split(":"))
        run(m, data=a.data, start=lo or 0, end=hi, tag=a.tag, plot_every=a.plot_every, n_iters=a.n_iters)