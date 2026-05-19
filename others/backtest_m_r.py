# =========================================================
# MONTHLY REBALANCE - DEBUG VERSION
# GIỮ NGUYÊN RANDOM STRATEGY
# Chỉ nâng cấp MODEL-DRIVEN strategy
# =========================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import defaultdict
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# IMPORT
# ─────────────────────────────────────────────────────────
try:
    import a_ML3
except ImportError:
    print("❌ Không tìm thấy a_ML3")
    sys.exit(1)

# =========================================================
# CONFIG
# =========================================================
INITIAL_CAPITAL = 100_000_000
TRANSACTION_FEE = 0.0015
LOT_SIZE = 100

TICKERS = ["TCB", "VRE", "MSN", "VCB", "GAS"]

BACKTEST_START = "2025-01-01"
BACKTEST_END = "2026-01-15"
MODEL_TRAIN_START = "2019-06-15"

MODEL_HORIZON = 21

# ─────────────────────────────────────────────────────────
# MPT CONFIG
# ─────────────────────────────────────────────────────────
ANNUAL_RF = 0.05
MONTHLY_RF = (1 + ANNUAL_RF) ** (1 / 12) - 1

LAMBDA_REG = 0.001
MAX_TURNOVER = 0.50

MAX_POSITION_SIZE = 0.35
MIN_POSITION_SIZE = 0.05

# ─────────────────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────────────────
MAX_DRAWDOWN_STOP = -0.20
DRAWDOWN_WARNING = -0.12
RECOVERY_THRESHOLD = -0.05

# ─────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────
DEBUG = True

RESULT_DIR = "backtest_results_debug"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# DATA CACHE
# =========================================================
class DataCache:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache = {}
        self.feat_cache = {}

    def load_all_data(self):

        print("\n" + "=" * 70)
        print("📦 PRELOAD DATA")
        print("=" * 70)

        for ticker in self.tickers:

            print(f"⏳ {ticker}...", end=" ")

            try:
                df = a_ML3.load_data(
                    ticker,
                    MODEL_TRAIN_START,
                    self.end_date,
                    "d"
                )

                if df is None or df.empty:
                    raise ValueError("No data")

                df = a_ML3.validate_data(df)

                df["time"] = pd.to_datetime(df["time"])

                df = df.sort_values("time").reset_index(drop=True)

                self.cache[ticker] = df

                print(f"✅ {len(df)} rows")

            except Exception as e:
                print(f"❌ {e}")

    def get_featured_df(self, ticker):

        if ticker not in self.feat_cache:

            raw = self.cache.get(ticker)

            if raw is None:
                return None

            feat = a_ML3.compute_features(
                raw.copy(),
                zlema_period=a_ML3.CONFIG["zlema_period"]
            )

            self.feat_cache[ticker] = feat

        return self.feat_cache[ticker]

    def get_price_on_or_after(self, ticker, target_date):

        df = self.cache.get(ticker)

        if df is None:
            return None

        td = pd.to_datetime(target_date).date()

        sub = df[df["time"].dt.date >= td]

        if sub.empty:
            return None

        row = sub.iloc[0]

        return {
            "date": row["time"].strftime("%Y-%m-%d"),
            "price": float(row["close"])
        }

    def get_prices_on_date(self, tickers, target_date):

        prices = {}

        for t in tickers:

            r = self.get_price_on_or_after(t, target_date)

            if r:
                prices[t] = r["price"]

        return prices

    def build_price_maps(self):

        out = {}

        for ticker, df in self.cache.items():

            tmp = df.copy()

            tmp["ds"] = tmp["time"].dt.strftime("%Y-%m-%d")

            out[ticker] = dict(
                zip(tmp["ds"], tmp["close"].astype(float))
            )

        return out

    def get_history_before(self, ticker, before_date, n_days):

        feat = self.get_featured_df(ticker)

        if feat is None:
            return pd.DataFrame()

        bd = pd.to_datetime(before_date)

        sub = feat[feat["time"] < bd].tail(n_days)

        return sub.copy()

    def get_historical_returns(self, tickers, before_date, n_days=252):

        returns_dict = {}

        for ticker in tickers:

            df = self.cache.get(ticker)

            if df is None:
                continue

            bd = pd.to_datetime(before_date)

            sub = df[df["time"] < bd].tail(n_days + 1).copy()

            if len(sub) < 60:
                continue

            sub["ret"] = sub["close"].pct_change()

            ret = sub["ret"].dropna()

            if len(ret) >= 30:
                returns_dict[ticker] = ret.values

        if not returns_dict:
            return pd.DataFrame()

        min_len = min(len(v) for v in returns_dict.values())

        aligned = {
            k: v[-min_len:]
            for k, v in returns_dict.items()
        }

        return pd.DataFrame(aligned)

    def get_union_trading_dates(self, start_date, end_date):

        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)

        all_dates = set()

        for df in self.cache.values():

            sub = df[(df["time"] >= s) & (df["time"] <= e)]

            all_dates.update(
                sub["time"].dt.strftime("%Y-%m-%d")
            )

        return sorted(all_dates)

    def get_first_trading_day_of_month(self, year, month, ticker_ref="TCB"):

        df = self.cache.get(ticker_ref)

        if df is None:
            return None

        sub = df[
            (df["time"].dt.year == year) &
            (df["time"].dt.month == month)
        ]

        if sub.empty:
            return None

        return sub.iloc[0]["time"].strftime("%Y-%m-%d")

    def get_all_first_trading_days_of_months(
        self,
        start_date,
        end_date,
        ticker_ref="TCB"
    ):

        df = self.cache.get(ticker_ref)

        if df is None:
            return []

        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)

        first_days = []

        current = pd.to_datetime(
            f"{s.year}-{s.month:02d}-01"
        )

        while current <= e:

            y = current.year
            m = current.month

            sub = df[
                (df["time"].dt.year == y) &
                (df["time"].dt.month == m)
            ]

            if not sub.empty:
                first_days.append(
                    sub.iloc[0]["time"].strftime("%Y-%m-%d")
                )

            if m == 12:
                current = pd.to_datetime(f"{y+1}-01-01")
            else:
                current = pd.to_datetime(f"{y}-{m+1:02d}-01")

        return first_days


# =========================================================
# MODEL MANAGER
# =========================================================
class ModelManager:

    def __init__(self):
        self._models = {}

    def get_model(self, ticker):

        t = ticker.upper()

        if t not in self._models:

            print(f"🔧 Loading model {t}...")

            try:

                pkg = a_ML3.ensure_model(t)

                self._models[t] = pkg

                print(f"✅ {t}")

            except Exception as e:

                print(f"❌ {t}: {e}")

                self._models[t] = None

        return self._models[t]

    def predict_price(self, ticker, past_window_df):

        pkg = self.get_model(ticker)

        if pkg is None:
            return None

        model, f_sc, t_sc, meta = pkg

        lookback = a_ML3.CONFIG["lookback"]

        if len(past_window_df) < lookback:
            return None

        try:

            pred = a_ML3.predict_next_price(
                past_window=past_window_df,
                feature_list=a_ML3.CORE_MOMENTUM_FEATURES,
                model=model,
                f_scaler=f_sc,
                t_scaler=t_sc,
                lookback=lookback,
            )

            return float(pred)

        except:
            return None


# =========================================================
# ER CALCULATOR
# =========================================================
class ERCalculator:

    def __init__(
        self,
        data_cache,
        model_manager,
        horizon=MODEL_HORIZON
    ):

        self.dc = data_cache
        self.mm = model_manager
        self.horizon = horizon

    def get_er_monthly(self, ticker, date_str):

        past = self.dc.get_history_before(
            ticker,
            date_str,
            a_ML3.CONFIG["lookback"] + 10
        )

        if past.empty:
            return None

        current_info = self.dc.get_price_on_or_after(
            ticker,
            date_str
        )

        if current_info is None:
            return None

        p0 = current_info["price"]

        pred = self.mm.predict_price(ticker, past)

        if pred is None or p0 <= 0:
            return None

        er1 = (pred / p0) - 1.0

        er1 = np.clip(er1, -0.20, 0.20)

        return (1 + er1) ** self.horizon - 1.0


# =========================================================
# PORTFOLIO OPTIMIZER
# =========================================================
class PortfolioOptimizer:

    def __init__(self):

        self.rf = MONTHLY_RF
        self.lambda_reg = LAMBDA_REG

    def optimize(
        self,
        expected_returns,
        cov_matrix,
        current_weights=None
    ):

        tickers = list(expected_returns.keys())

        n = len(tickers)

        if n == 0:
            return {}

        er_array = np.array([
            expected_returns[t]
            for t in tickers
        ])

        cov_array = cov_matrix.loc[
            tickers,
            tickers
        ].values

        # ─────────────────────────────────────────
        # OBJECTIVE
        # ─────────────────────────────────────────
        def objective(weights):

            port_ret = np.sum(er_array * weights)

            port_std = np.sqrt(
                np.dot(
                    weights.T,
                    np.dot(cov_array, weights)
                )
            )

            if port_std <= 0:
                return 1e10

            sharpe = (port_ret - self.rf) / port_std

            penalty = self.lambda_reg * np.sum(weights ** 2)

            return -sharpe + penalty

        # ─────────────────────────────────────────
        # CONSTRAINTS
        # ─────────────────────────────────────────
        constraints = [
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - 1
            }
        ]

        # ✅ FIX TURNOVER
        if current_weights is not None:

            current_array = np.array([
                current_weights.get(t, 0)
                for t in tickers
            ])

            def turnover_constraint(x):

                turnover = np.sum(
                    np.abs(x - current_array)
                )

                return MAX_TURNOVER - turnover

            constraints.append({
                "type": "ineq",
                "fun": turnover_constraint
            })

        # ─────────────────────────────────────────
        # BOUNDS
        # ─────────────────────────────────────────
        bounds = tuple(
            (MIN_POSITION_SIZE, MAX_POSITION_SIZE)
            for _ in range(n)
        )

        # ─────────────────────────────────────────
        # INITIAL
        # ─────────────────────────────────────────
        if current_weights is not None:

            x0 = np.array([
                current_weights.get(t, 1/n)
                for t in tickers
            ])

        else:

            x0 = np.array([1/n] * n)

        # ─────────────────────────────────────────
        # OPTIMIZE
        # ─────────────────────────────────────────
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 5000,
                "ftol": 1e-12
            }
        )

        # ─────────────────────────────────────────
        # FALLBACK
        # ─────────────────────────────────────────
        if not result.success:

            if DEBUG:
                print(f"⚠️ Optimize fail: {result.message}")

            w = np.array([1/n] * n)

        else:

            w = result.x

        w = w / np.sum(w)

        # ─────────────────────────────────────────
        # DEBUG
        # ─────────────────────────────────────────
        if DEBUG:

            print(f"      ER: {dict(zip(tickers, np.round(er_array,4)))}")

            if current_weights is not None:

                turnover = np.sum(
                    np.abs(w - current_array)
                )

                print(f"      Turnover: {turnover*100:.1f}%")

            print(f"      Weights: {dict(zip(tickers, np.round(w,3)))}")

        return dict(zip(tickers, w))


# =========================================================
# RISK MANAGER
# =========================================================
class RiskManager:

    def __init__(self):

        self.peak_value = INITIAL_CAPITAL

        self.risk_mode = "NORMAL"

    def update(self, current_value):

        self.peak_value = max(
            self.peak_value,
            current_value
        )

        dd = (
            current_value - self.peak_value
        ) / self.peak_value

        if dd <= MAX_DRAWDOWN_STOP:
            self.risk_mode = "STOP"

        elif dd <= DRAWDOWN_WARNING:
            self.risk_mode = "DEFENSIVE"

        elif dd >= RECOVERY_THRESHOLD:
            self.risk_mode = "NORMAL"

        return self.risk_mode, dd

    def get_position_multiplier(self):

        if self.risk_mode == "STOP":
            return 0.0

        elif self.risk_mode == "DEFENSIVE":
            return 0.6

        return 1.0


# =========================================================
# UTILS
# =========================================================
def round_lot_down(qty):

    return int(qty // LOT_SIZE) * LOT_SIZE


def allocate_by_weights(capital, weights, prices):

    alloc = {}

    total_tv = 0
    total_fee = 0

    for ticker, w in weights.items():

        if ticker not in prices:
            alloc[ticker] = 0
            continue

        tv = capital * w / (1 + TRANSACTION_FEE)

        qty = round_lot_down(
            tv / prices[ticker]
        )

        alloc[ticker] = qty

        total_tv += qty * prices[ticker]

        total_fee += (
            qty * prices[ticker] * TRANSACTION_FEE
        )

    needed = total_tv + total_fee

    while needed > capital:

        cands = [
            (
                alloc[t] * prices[t],
                t
            )
            for t in alloc
            if alloc[t] >= LOT_SIZE
        ]

        if not cands:
            break

        _, tk = max(cands)

        alloc[tk] -= LOT_SIZE

        total_tv = sum(
            alloc[t] * prices[t]
            for t in alloc
        )

        total_fee = sum(
            alloc[t] * prices[t] * TRANSACTION_FEE
            for t in alloc
        )

        needed = total_tv + total_fee

    return alloc


def compute_max_drawdown(values):

    arr = np.array(values)

    rm = np.maximum.accumulate(arr)

    return ((arr - rm) / rm).min()


def generate_random_weights(tickers, seed=None):

    # ✅ GIỮ NGUYÊN RANDOM CŨ

    if seed is not None:
        np.random.seed(seed)

    n = len(tickers)

    alpha = np.ones(n) * 2.0

    weights = np.random.dirichlet(alpha)

    weights = np.clip(weights, 0.05, 0.30)

    weights = weights / weights.sum()

    return dict(zip(tickers, weights))


# =========================================================
# PORTFOLIO
# =========================================================
class Portfolio:

    def __init__(self, initial_capital, name="Portfolio"):

        self.name = name

        self.initial_capital = float(initial_capital)

        self.cash = float(initial_capital)

        self.holdings = defaultdict(int)

        self.transactions = []

    def buy(self, date, ticker, qty, price):

        if qty <= 0:
            return False

        tv = qty * price

        cost = tv + tv * TRANSACTION_FEE

        if cost > self.cash:
            return False

        self.cash -= cost

        self.holdings[ticker] += qty

        self.transactions.append({
            "date": date,
            "type": "BUY",
            "ticker": ticker,
            "qty": qty,
            "price": price,
            "trade_value": tv,
            "fee": tv * TRANSACTION_FEE,
            "cash_after": self.cash
        })

        return True

    def sell(self, date, ticker, qty, price):

        if qty <= 0:
            return False

        if self.holdings[ticker] < qty:
            return False

        tv = qty * price

        self.cash += tv - tv * TRANSACTION_FEE

        self.holdings[ticker] -= qty

        self.transactions.append({
            "date": date,
            "type": "SELL",
            "ticker": ticker,
            "qty": qty,
            "price": price,
            "trade_value": tv,
            "fee": tv * TRANSACTION_FEE,
            "cash_after": self.cash
        })

        return True

    def sell_all(self, date, prices):

        for ticker, qty in list(self.holdings.items()):

            if qty > 0 and ticker in prices:

                self.sell(
                    date,
                    ticker,
                    qty,
                    prices[ticker]
                )

    def total_assets(self, prices):

        stock = sum(
            qty * prices[t]
            for t, qty in self.holdings.items()
            if qty > 0 and t in prices
        )

        return self.cash + stock

    def get_current_weights(self, prices):

        total = self.total_assets(prices)

        if total <= 0:
            return {}

        weights = {}

        for t, qty in self.holdings.items():

            if qty > 0 and t in prices:

                weights[t] = (
                    qty * prices[t]
                ) / total

        return weights

    def get_transactions_df(self):

        if not self.transactions:
            return pd.DataFrame()

        return pd.DataFrame(self.transactions)


# =========================================================
# MODEL DRIVEN STRATEGY
# =========================================================
def run_model_driven(data_cache, er_calc):

    optimizer = PortfolioOptimizer()

    port = Portfolio(
        INITIAL_CAPITAL,
        name="Model-Driven MPT"
    )

    risk_mgr = RiskManager()

    price_maps = data_cache.build_price_maps()

    rebalance_dates = data_cache.get_all_first_trading_days_of_months(
        BACKTEST_START,
        BACKTEST_END
    )

    print("\n" + "=" * 80)
    print("🤖 MODEL DRIVEN MPT")
    print("=" * 80)

    event_log = []

    for i, rebalance_date in enumerate(rebalance_dates):

        prices = {
            t: price_maps[t][rebalance_date]
            for t in TICKERS
            if rebalance_date in price_maps[t]
        }

        current_value = port.total_assets(prices)

        risk_mode, dd = risk_mgr.update(current_value)

        multiplier = risk_mgr.get_position_multiplier()

        print(f"\n📅 {rebalance_date}")

        print(f"   Risk Mode: {risk_mode}")

        print(f"   Drawdown: {dd*100:.2f}%")

        # ✅ FIX TURNOVER BUG
        current_weights = port.get_current_weights(prices)

        if DEBUG:
            print(f"   Current Weights: {current_weights}")

        # SELL OLD
        if i > 0:
            port.sell_all(rebalance_date, prices)

        if risk_mode == "STOP":

            print("   STOP → CASH")

            continue

        # ─────────────────────────────────────────
        # ER
        # ─────────────────────────────────────────
        ers = {}

        for t in TICKERS:

            er = er_calc.get_er_monthly(
                t,
                rebalance_date
            )

            if er is not None and er > 0:

                ers[t] = er

        if not ers:

            print("   No positive ER")

            continue

        # ─────────────────────────────────────────
        # COV MATRIX
        # ─────────────────────────────────────────
        returns_df = data_cache.get_historical_returns(
            list(ers.keys()),
            rebalance_date
        )

        print(f"   Covariance Data: {len(returns_df)} rows")

        if returns_df.empty or len(returns_df) < 30:

            print("   Fallback Equal Weight")

            n = len(ers)

            weights = {
                t: 1/n
                for t in ers
            }

        else:

            lw = LedoitWolf()

            lw.fit(returns_df)

            cov = pd.DataFrame(
                lw.covariance_ * 30,
                index=returns_df.columns,
                columns=returns_df.columns
            )

            print(f"   Shrinkage: {lw.shrinkage_:.4f}")

            weights = optimizer.optimize(
                ers,
                cov,
                current_weights=current_weights if i > 0 else None
            )

        # ─────────────────────────────────────────
        # BUY
        # ─────────────────────────────────────────
        capital_to_use = port.cash * multiplier

        alloc = allocate_by_weights(
            capital_to_use,
            weights,
            prices
        )

        for ticker, qty in alloc.items():

            if qty > 0:

                port.buy(
                    rebalance_date,
                    ticker,
                    qty,
                    prices[ticker]
                )

        total_assets = port.total_assets(prices)

        print(f"   Total Assets: {total_assets:,.0f}")

        event_log.append({
            "date": rebalance_date,
            "weights": weights,
            "ers": ers,
            "total_assets": total_assets
        })

    # FINAL SELL
    sell_date = data_cache.get_first_trading_day_of_month(
        2026,
        1
    )

    if sell_date:

        sp = data_cache.get_prices_on_date(
            TICKERS,
            sell_date
        )

        port.sell_all(
            sell_date,
            sp
        )

    return port, sell_date, event_log


# =========================================================
# RANDOM STRATEGY (UNCHANGED)
# =========================================================
def run_random_monthly(data_cache):

    # ✅ GIỮ NGUYÊN HOÀN TOÀN

    port = Portfolio(
        INITIAL_CAPITAL,
        name="Random Monthly"
    )

    price_maps = data_cache.build_price_maps()

    rebalance_dates = data_cache.get_all_first_trading_days_of_months(
        BACKTEST_START,
        BACKTEST_END
    )

    print("\n" + "=" * 80)
    print("🎲 RANDOM MONTHLY")
    print("=" * 80)

    event_log = []

    for i, rebalance_date in enumerate(rebalance_dates):

        prices = {
            t: price_maps[t][rebalance_date]
            for t in TICKERS
            if rebalance_date in price_maps.get(t, {})
        }

        if len(prices) < len(TICKERS):
            continue

        if i > 0:
            port.sell_all(rebalance_date, prices)

        # ✅ RANDOM CŨ
        month_seed = int(
            pd.to_datetime(rebalance_date).strftime("%Y%m")
        )

        weights = generate_random_weights(
            TICKERS,
            seed=month_seed
        )

        alloc = allocate_by_weights(
            port.cash,
            weights,
            prices
        )

        for ticker, qty in alloc.items():

            if qty > 0:

                port.buy(
                    rebalance_date,
                    ticker,
                    qty,
                    prices[ticker]
                )

        total_val = port.total_assets(prices)

        print(f"{rebalance_date}: {weights}")

        event_log.append({
            "date": rebalance_date,
            "weights": weights,
            "total_assets": total_val
        })

    sell_date = data_cache.get_first_trading_day_of_month(
        2026,
        1
    )

    if sell_date:

        sp = data_cache.get_prices_on_date(
            TICKERS,
            sell_date
        )

        port.sell_all(
            sell_date,
            sp
        )

    return port, sell_date, event_log


# =========================================================
# EQUITY CURVE
# =========================================================
def build_total_assets_curve(
    data_cache,
    portfolio,
    start_date,
    end_date
):

    dates = data_cache.get_union_trading_dates(
        start_date,
        end_date
    )

    price_maps = data_cache.build_price_maps()

    tx_df = portfolio.get_transactions_df().copy()

    if tx_df.empty:
        return pd.DataFrame()

    tx_df["date"] = pd.to_datetime(tx_df["date"])

    tx_df = tx_df.sort_values("date")

    rows = []

    holdings = defaultdict(int)

    cash = portfolio.initial_capital

    tx_idx = 0

    for date_str in dates:

        if not (start_date <= date_str <= end_date):
            continue

        date_ts = pd.to_datetime(date_str).date()

        while (
            tx_idx < len(tx_df)
            and tx_df.iloc[tx_idx]["date"].date() <= date_ts
        ):

            tx = tx_df.iloc[tx_idx]

            qty = int(tx["qty"])

            if tx["type"] == "BUY":
                holdings[tx["ticker"]] += qty
            else:
                holdings[tx["ticker"]] -= qty

            cash = float(tx["cash_after"])

            tx_idx += 1

        stock_value = sum(
            holdings[t] * price_maps[t][date_str]
            for t in TICKERS
            if holdings[t] > 0 and date_str in price_maps[t]
        )

        rows.append({
            "date": date_str,
            "total_assets": cash + stock_value
        })

    return pd.DataFrame(rows)


# =========================================================
# SUMMARY
# =========================================================
def summarize(name, portfolio, curve_df):

    fv = float(
        curve_df["total_assets"].iloc[-1]
    )

    ret = fv / portfolio.initial_capital - 1

    max_dd = compute_max_drawdown(
        curve_df["total_assets"].tolist()
    )

    tx_df = portfolio.get_transactions_df()

    fees = tx_df["fee"].sum()

    turnover = tx_df["trade_value"].sum()

    sharpe = 0

    if len(curve_df) > 1:

        dr = curve_df["total_assets"].pct_change().dropna()

        excess = dr - MONTHLY_RF / 21

        if excess.std() > 0:

            sharpe = (
                excess.mean()
                / excess.std()
                * np.sqrt(252)
            )

    return {
        "Strategy": name,
        "Final Value": fv,
        "Return (%)": ret * 100,
        "Max Drawdown (%)": max_dd * 100,
        "Sharpe": sharpe,
        "Total Fees": fees,
        "Total Turnover": turnover
    }


# =========================================================
# MAIN
# =========================================================
def main():

    print("\n" + "=" * 80)
    print("📈 DEBUG BACKTEST")
    print("=" * 80)

    data_cache = DataCache(
        TICKERS,
        BACKTEST_START,
        BACKTEST_END
    )

    data_cache.load_all_data()

    model_mgr = ModelManager()

    er_calc = ERCalculator(
        data_cache,
        model_mgr
    )

    # ─────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────
    md_port, md_sell_date, md_events = run_model_driven(
        data_cache,
        er_calc
    )

    # ─────────────────────────────────────────
    # RANDOM (UNCHANGED)
    # ─────────────────────────────────────────
    rd_port, rd_sell_date, rd_events = run_random_monthly(
        data_cache
    )

    # ─────────────────────────────────────────
    # CURVES
    # ─────────────────────────────────────────
    curve_md = build_total_assets_curve(
        data_cache,
        md_port,
        BACKTEST_START,
        md_sell_date
    )

    curve_rd = build_total_assets_curve(
        data_cache,
        rd_port,
        BACKTEST_START,
        rd_sell_date
    )

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    s1 = summarize(
        "Model-Driven MPT",
        md_port,
        curve_md
    )

    s2 = summarize(
        "Random",
        rd_port,
        curve_rd
    )

    summary_df = pd.DataFrame([s1, s2])

    print("\n" + "=" * 100)
    print(summary_df)
    print("=" * 100)

    summary_df.to_csv(
        os.path.join(RESULT_DIR, "summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n✅ DONE")


if __name__ == "__main__":
    main()