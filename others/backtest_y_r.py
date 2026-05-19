# =========================================================
# BUY AND HOLD BACKTEST
# Mua đầu 2025 → Bán đầu 2026 (không rebalance)
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

BUY_DATE = "2025-01-02"   # Phiên đầu năm 2025
SELL_DATE = "2026-01-02"  # Phiên đầu năm 2026

MODEL_TRAIN_START = "2019-06-15"
MODEL_HORIZON = 21

# ─────────────────────────────────────────────────────────
# MPT CONFIG
# ─────────────────────────────────────────────────────────
ANNUAL_RF = 0.05
MONTHLY_RF = (1 + ANNUAL_RF) ** (1 / 12) - 1

LAMBDA_REG = 0.001

MAX_POSITION_SIZE = 0.35
MIN_POSITION_SIZE = 0.05

# ─────────────────────────────────────────────────────────
# CORE-SATELLITE
# ─────────────────────────────────────────────────────────
MIN_CORE_ER = 0.01
MAX_SATELLITE_LOSS = -0.03

# ─────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────
DEBUG = True

RESULT_DIR = "backtest_results_buy_hold"
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
# CORE-SATELLITE STRATEGY
# =========================================================
def build_portfolio_strategy(all_ers, min_core_er=MIN_CORE_ER, 
                            max_satellite_loss=MAX_SATELLITE_LOSS):
    """
    Core-Satellite cho buy & hold
    """
    
    core = {t: er for t, er in all_ers.items() if er >= min_core_er}
    satellite = {t: er for t, er in all_ers.items() 
                 if max_satellite_loss < er < min_core_er}
    
    print("\n" + "="*70)
    print("📊 CORE-SATELLITE STRATEGY (BUY & HOLD)")
    print("="*70)
    
    if core:
        print(f"\n🎯 CORE ({len(core)} mã - ER >= {min_core_er*100}%):")
        for t, er in sorted(core.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {er*100:+6.2f}%")
    
    if satellite:
        print(f"\n🔍 SATELLITE ({len(satellite)} mã):")
        for t, er in sorted(satellite.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {er*100:+6.2f}%")
    
    # ══════════════════════════════════════════════════════
    # Logic selection (simplified for buy & hold)
    # ══════════════════════════════════════════════════════
    if not core and not satellite:
        print("\n❌ KHÔNG CÓ MÃ NÀO ĐỦ ĐIỀU KIỆN!")
        print(f"   Tất cả mã đều có ER < {max_satellite_loss*100}%")
        
        # Fallback: top 3
        sorted_all = sorted(all_ers.items(), key=lambda x: x[1], reverse=True)
        top_3 = dict(sorted_all[:min(3, len(sorted_all))])
        
        print(f"\n📌 Fallback: Top {len(top_3)} mã ít thua nhất")
        portfolio = top_3
    
    elif not core:
        print("\n⚠️ KHÔNG CÓ CORE - Chỉ Satellite")
        portfolio = satellite
    
    elif len(core) == 1:
        core_ticker = list(core.keys())[0]
        print(f"\n⚠️ CHỈ CÓ 1 CORE: {core_ticker}")
        
        if satellite:
            print(f"   → Thêm {len(satellite)} Satellite")
            portfolio = {**core, **satellite}
        else:
            # Thêm 1-2 mã defensive
            others = {t: er for t, er in all_ers.items() if t not in core}
            sorted_others = sorted(others.items(), key=lambda x: x[1], reverse=True)
            top_2 = dict(sorted_others[:min(2, len(sorted_others))])
            
            print(f"   → Thêm {len(top_2)} defensive")
            portfolio = {**core, **top_2}
    
    else:
        print(f"\n✅ CÓ {len(core)} CORE")
        portfolio = {**core, **satellite} if satellite else core
    
    # Shift ER nếu âm
    min_er = min(portfolio.values())
    if min_er < 0:
        print(f"\n📊 Shift ER: min {min_er*100:.2f}% → 0.1%")
        portfolio = {t: er - min_er + 0.001 for t, er in portfolio.items()}
    
    print(f"\n✅ Danh mục cuối: {len(portfolio)} mã")
    return portfolio


# =========================================================
# PORTFOLIO OPTIMIZER
# =========================================================
class PortfolioOptimizer:
    def __init__(self):
        self.rf = MONTHLY_RF
        self.lambda_reg = LAMBDA_REG

    def optimize(self, expected_returns, cov_matrix):
        tickers = list(expected_returns.keys())
        n = len(tickers)

        if n == 0:
            return {}

        er_array = np.array([expected_returns[t] for t in tickers])
        cov_array = cov_matrix.loc[tickers, tickers].values

        def objective(weights):
            port_ret = np.sum(er_array * weights)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_array, weights)))

            if port_std <= 0:
                return 1e10

            sharpe = (port_ret - self.rf) / port_std
            penalty = self.lambda_reg * np.sum(weights ** 2)

            return -sharpe + penalty

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        bounds = tuple(
            (MIN_POSITION_SIZE, MAX_POSITION_SIZE)
            for _ in range(n)
        )

        x0 = np.array([1/n] * n)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 5000, "ftol": 1e-12}
        )

        if not result.success:
            if DEBUG:
                print(f"   ⚠️ Optimize fail: {result.message}")
            w = np.array([1/n] * n)
        else:
            w = result.x

        w = w / np.sum(w)

        if DEBUG:
            print(f"   ER: {dict(zip(tickers, np.round(er_array,4)))}")
            print(f"   Weights: {dict(zip(tickers, np.round(w,3)))}")

        return dict(zip(tickers, w))


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
        qty = round_lot_down(tv / prices[ticker])

        alloc[ticker] = qty
        total_tv += qty * prices[ticker]
        total_fee += qty * prices[ticker] * TRANSACTION_FEE

    needed = total_tv + total_fee

    while needed > capital:
        cands = [
            (alloc[t] * prices[t], t)
            for t in alloc
            if alloc[t] >= LOT_SIZE
        ]

        if not cands:
            break

        _, tk = max(cands)
        alloc[tk] -= LOT_SIZE

        total_tv = sum(alloc[t] * prices[t] for t in alloc)
        total_fee = sum(alloc[t] * prices[t] * TRANSACTION_FEE for t in alloc)
        needed = total_tv + total_fee

    return alloc


def compute_max_drawdown(values):
    arr = np.array(values)
    rm = np.maximum.accumulate(arr)
    return ((arr - rm) / rm).min()


def generate_random_weights(tickers, seed=None):
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
                self.sell(date, ticker, qty, prices[ticker])

    def total_assets(self, prices):
        stock = sum(
            qty * prices[t]
            for t, qty in self.holdings.items()
            if qty > 0 and t in prices
        )

        return self.cash + stock

    def get_transactions_df(self):
        if not self.transactions:
            return pd.DataFrame()

        return pd.DataFrame(self.transactions)


# =========================================================
# BUY & HOLD STRATEGIES
# =========================================================
def run_buy_hold_mpt(data_cache, er_calc, buy_date, sell_date):
    """
    MPT + Core-Satellite: Mua đầu 2025 → Bán đầu 2026
    """
    
    port = Portfolio(INITIAL_CAPITAL, name="MPT Buy & Hold")
    optimizer = PortfolioOptimizer()

    print("\n" + "=" * 80)
    print("🤖 MPT BUY & HOLD")
    print("=" * 80)

    # ══════════════════════════════════════════════════════
    # BUY DATE
    # ══════════════════════════════════════════════════════
    print(f"\n📅 BUY: {buy_date}")

    buy_prices = data_cache.get_prices_on_date(TICKERS, buy_date)

    if len(buy_prices) < len(TICKERS):
        print(f"⚠️ Missing prices for some tickers")

    # Calculate ERs
    all_ers = {}
    for t in TICKERS:
        er = er_calc.get_er_monthly(t, buy_date)
        if er is not None:
            all_ers[t] = er

    if not all_ers:
        print("❌ No ER available")
        return port, None

    # Core-Satellite
    ers = build_portfolio_strategy(all_ers)

    if not ers:
        print("❌ No portfolio")
        return port, None

    # Covariance
    returns_df = data_cache.get_historical_returns(
        list(ers.keys()),
        buy_date
    )

    if returns_df.empty or len(returns_df) < 30:
        print("   ⚠️ Fallback to equal weight")
        weights = {t: 1/len(ers) for t in ers}
    else:
        lw = LedoitWolf()
        lw.fit(returns_df)

        cov = pd.DataFrame(
            lw.covariance_ * 30,
            index=returns_df.columns,
            columns=returns_df.columns
        )

        print(f"   Shrinkage: {lw.shrinkage_:.4f}")

        weights = optimizer.optimize(ers, cov)

    # Allocate
    alloc = allocate_by_weights(
        port.cash,
        weights,
        {t: buy_prices[t] for t in weights}
    )

    # Buy
    for ticker, qty in alloc.items():
        if qty > 0:
            port.buy(buy_date, ticker, qty, buy_prices[ticker])

    buy_value = port.total_assets(buy_prices)
    print(f"\n   Total Assets: {buy_value:,.0f}")
    print(f"   Weights: {weights}")

    # ══════════════════════════════════════════════════════
    # SELL DATE
    # ══════════════════════════════════════════════════════
    print(f"\n📅 SELL: {sell_date}")

    sell_prices = data_cache.get_prices_on_date(TICKERS, sell_date)
    port.sell_all(sell_date, sell_prices)

    sell_value = port.total_assets(sell_prices)
    print(f"   Total Assets: {sell_value:,.0f}")

    return port, (buy_date, sell_date)


def run_buy_hold_random(data_cache, buy_date, sell_date):
    """
    Random: Mua đầu 2025 → Bán đầu 2026
    """
    
    port = Portfolio(INITIAL_CAPITAL, name="Random Buy & Hold")

    print("\n" + "=" * 80)
    print("🎲 RANDOM BUY & HOLD")
    print("=" * 80)

    # ══════════════════════════════════════════════════════
    # BUY DATE
    # ══════════════════════════════════════════════════════
    print(f"\n📅 BUY: {buy_date}")

    buy_prices = data_cache.get_prices_on_date(TICKERS, buy_date)

    # Random weights (seed by year)
    weights = generate_random_weights(TICKERS, seed=2025)

    print(f"   Weights: {weights}")

    # Allocate
    alloc = allocate_by_weights(port.cash, weights, buy_prices)

    # Buy
    for ticker, qty in alloc.items():
        if qty > 0:
            port.buy(buy_date, ticker, qty, buy_prices[ticker])

    buy_value = port.total_assets(buy_prices)
    print(f"   Total Assets: {buy_value:,.0f}")

    # ══════════════════════════════════════════════════════
    # SELL DATE
    # ══════════════════════════════════════════════════════
    print(f"\n📅 SELL: {sell_date}")

    sell_prices = data_cache.get_prices_on_date(TICKERS, sell_date)
    port.sell_all(sell_date, sell_prices)

    sell_value = port.total_assets(sell_prices)
    print(f"   Total Assets: {sell_value:,.0f}")

    return port, (buy_date, sell_date)


# =========================================================
# EQUITY CURVE
# =========================================================
def build_total_assets_curve(data_cache, portfolio, start_date, end_date):
    dates = data_cache.get_union_trading_dates(start_date, end_date)
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
    fv = float(curve_df["total_assets"].iloc[-1])
    ret = fv / portfolio.initial_capital - 1
    max_dd = compute_max_drawdown(curve_df["total_assets"].tolist())

    tx_df = portfolio.get_transactions_df()
    fees = tx_df["fee"].sum()
    turnover = tx_df["trade_value"].sum()

    sharpe = 0

    if len(curve_df) > 1:
        dr = curve_df["total_assets"].pct_change().dropna()
        excess = dr - MONTHLY_RF / 21

        if excess.std() > 0:
            sharpe = excess.mean() / excess.std() * np.sqrt(252)

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
# PLOT
# =========================================================
def plot_comparison(curve_mpt, curve_rand):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("MPT Buy & Hold vs Random Buy & Hold", 
                 fontsize=14, fontweight="bold")

    # ══════════════════════════════════════════════════════
    # EQUITY
    # ══════════════════════════════════════════════════════
    ax1 = axes[0]
    
    if not curve_mpt.empty:
        ax1.plot(
            pd.to_datetime(curve_mpt["date"]),
            curve_mpt["total_assets"],
            label="MPT Buy & Hold",
            color="#2196F3",
            lw=2.5
        )
    
    if not curve_rand.empty:
        ax1.plot(
            pd.to_datetime(curve_rand["date"]),
            curve_rand["total_assets"],
            label="Random Buy & Hold",
            color="#FF9800",
            lw=2.5,
            linestyle="--"
        )
    
    ax1.axhline(INITIAL_CAPITAL, color="gray", lw=1, ls=":", alpha=0.5)
    ax1.set_ylabel("Total Assets (VND)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    # ══════════════════════════════════════════════════════
    # DRAWDOWN
    # ══════════════════════════════════════════════════════
    ax2 = axes[1]
    
    for curve, label, color in [
        (curve_mpt, "MPT", "#2196F3"),
        (curve_rand, "Random", "#FF9800"),
    ]:
        if curve.empty:
            continue
        
        arr = curve["total_assets"].values.astype(float)
        rm = np.maximum.accumulate(arr)
        dd = (arr - rm) / rm * 100
        
        ax2.plot(
            pd.to_datetime(curve["date"]),
            dd,
            label=label,
            color=color,
            lw=1.5
        )
        
        ax2.fill_between(
            pd.to_datetime(curve["date"]),
            dd,
            0,
            alpha=0.1,
            color=color
        )

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    
    out = os.path.join(RESULT_DIR, "comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    
    print(f"\n✅ Chart: {out}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 80)
    print("📈 BUY & HOLD BACKTEST (2025 → 2026)")
    print("=" * 80)

    data_cache = DataCache(TICKERS, BUY_DATE, SELL_DATE)
    data_cache.load_all_data()

    model_mgr = ModelManager()
    er_calc = ERCalculator(data_cache, model_mgr)

    # Load models
    print("\n🔧 Loading models...")
    for ticker in TICKERS:
        model_mgr.get_model(ticker)

    # ══════════════════════════════════════════════════════
    # RUN STRATEGIES
    # ══════════════════════════════════════════════════════
    mpt_port, mpt_dates = run_buy_hold_mpt(
        data_cache,
        er_calc,
        BUY_DATE,
        SELL_DATE
    )

    rand_port, rand_dates = run_buy_hold_random(
        data_cache,
        BUY_DATE,
        SELL_DATE
    )

    # ══════════════════════════════════════════════════════
    # EQUITY CURVES
    # ══════════════════════════════════════════════════════
    curve_mpt = build_total_assets_curve(
        data_cache,
        mpt_port,
        BUY_DATE,
        SELL_DATE
    )

    curve_rand = build_total_assets_curve(
        data_cache,
        rand_port,
        BUY_DATE,
        SELL_DATE
    )

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    s1 = summarize("MPT Buy & Hold", mpt_port, curve_mpt)
    s2 = summarize("Random Buy & Hold", rand_port, curve_rand)

    summary_df = pd.DataFrame([s1, s2])

    print("\n" + "=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # ══════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════
    summary_df.to_csv(
        os.path.join(RESULT_DIR, "summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    mpt_port.get_transactions_df().to_csv(
        os.path.join(RESULT_DIR, "transactions_mpt.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    rand_port.get_transactions_df().to_csv(
        os.path.join(RESULT_DIR, "transactions_random.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    curve_mpt.to_csv(
        os.path.join(RESULT_DIR, "equity_curve_mpt.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    curve_rand.to_csv(
        os.path.join(RESULT_DIR, "equity_curve_random.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ══════════════════════════════════════════════════════
    # PLOT
    # ══════════════════════════════════════════════════════
    plot_comparison(curve_mpt, curve_rand)

    print("\n✅ DONE - BUY & HOLD BACKTEST")


if __name__ == "__main__":
    main()