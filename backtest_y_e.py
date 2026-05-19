# =========================================================
# BUY AND HOLD BACKTEST v2 (a_ML5_weekly)
# Mua đầu kỳ → Bán cuối kỳ (không rebalance)
# Model-Driven (PURE RF weekly) vs Equal Weight Baseline
# =========================================================

import os
import sys
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
# IMPORT a_ML5_weekly
# ─────────────────────────────────────────────────────────
try:
    import a_ML4_weekly as ml
    print("✅ Loaded a_ML5_weekly")
    print(f"   interval     : {ml.CONFIG.get('interval','w')}")
    print(f"   start_date   : {ml.CONFIG.get('start_date','N/A')}")
    print(f"   zlema_period : {ml.CONFIG.get('zlema_period','N/A')}")
    print(f"   lookback     : {ml.CONFIG.get('lookback','N/A')}")
    print(f"   forecast_steps: {ml.FORECAST_STEPS}")
except ImportError as e:
    print(f"❌ Không tìm thấy a_ML5_weekly: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# AUTO-MAP API
# ─────────────────────────────────────────────────────────
def _get_attr(module, candidates, what="function"):
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)
    available = [x for x in dir(module) if not x.startswith("_")]
    raise AttributeError(
        f"Không tìm thấy {what}.\n"
        f"Candidates: {candidates}\n"
        f"Available : {available[:60]}"
    )

_compute_features = _get_attr(ml, [
    "compute_features_DELTA",
    "compute_features_delta",
    "compute_features",
], what="compute_features")

_predict_next_price = _get_attr(ml, [
    "predict_next_price",
    "predict_next",
    "forecast_next_price",
], what="predict_next_price")

_ensure_model = _get_attr(ml, [
    "ensure_model",
    "load_model",
    "get_model",
], what="ensure_model")

_load_data = _get_attr(ml, [
    "load_data",
    "load_stock_data",
    "fetch_data",
], what="load_data")

_validate_data = _get_attr(ml, [
    "validate_data",
    "clean_data",
    "preprocess_data",
], what="validate_data")

_FEATURE_LIST   = _get_attr(ml, [
    "CORE_MOMENTUM_FEATURES_DELTA",
    "CORE_MOMENTUM_FEATURES",
    "FEATURE_LIST",
], what="feature_list")

_CONFIG         = ml.CONFIG
_FORECAST_STEPS = getattr(ml, "FORECAST_STEPS", 4)

print(f"✅ API mapped:")
print(f"   compute_features → {_compute_features.__name__}")
print(f"   predict_next     → {_predict_next_price.__name__}")
print(f"   ensure_model     → {_ensure_model.__name__}")
print(f"   feature_list     → {len(_FEATURE_LIST)} features")


# =========================================================
# CONFIG
# =========================================================
INITIAL_CAPITAL = 100_000_000
TRANSACTION_FEE = 0.0015
LOT_SIZE        = 100

TICKERS = ["TCB", "VRE", "VCB", "SSI", "FPT"]

BUY_DATE  = "2025-01-02"   # Phiên đầu kỳ backtest
SELL_DATE = "2026-01-02"   # Phiên cuối kỳ backtest

# Forecast horizon: 4 tuần (dùng để tính ER)
FORECAST_HORIZON = _FORECAST_STEPS   # = 4

# MPT
ANNUAL_RF  = 0.05
WEEKLY_RF  = (1 + ANNUAL_RF) ** (1 / 52) - 1
MONTHLY_RF = (1 + ANNUAL_RF) ** (1 / 12) - 1

LAMBDA_REG   = 0.001
MAX_POSITION = 0.35
MIN_POSITION = 0.05

# Core-Satellite (4-week scale)
MIN_CORE_ER        =  0.01   # ER >= 1% / 4 tuần → Core
MAX_SATELLITE_LOSS = -0.03   # ER >= -3% / 4 tuần → Satellite

DEBUG      = True
RESULT_DIR = "backtest_results_buy_hold_v2"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# DATA CACHE (weekly)
# =========================================================
class DataCache:

    def __init__(self, tickers, start_date, end_date):
        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date
        self.cache      = {}
        self.feat_cache = {}

    def load_all_data(self):
        print("\n" + "=" * 70)
        print("📦 PRELOAD DATA (weekly)")
        print("=" * 70)
        train_start = _CONFIG.get("start_date", "2010-01-01")

        for ticker in self.tickers:
            print(f"⏳ {ticker}...", end=" ", flush=True)
            try:
                df = _load_data(ticker, train_start,
                                self.end_date, "w")
                df = _validate_data(df)
                df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time").reset_index(drop=True)
                self.cache[ticker] = df
                print(f"✅ {len(df)} weeks "
                      f"({df['time'].iloc[0].strftime('%Y-%m-%d')}"
                      f" → {df['time'].iloc[-1].strftime('%Y-%m-%d')})")
            except Exception as e:
                print(f"❌ {e}")

    def get_featured_df(self, ticker):
        if ticker not in self.feat_cache:
            raw = self.cache.get(ticker)
            if raw is None:
                return None
            feat = _compute_features(
                raw.copy(),
                zlema_period=_CONFIG["zlema_period"],
            )
            self.feat_cache[ticker] = feat
        return self.feat_cache[ticker]

    def get_price_on_or_after(self, ticker, target_date):
        df = self.cache.get(ticker)
        if df is None:
            return None
        td  = pd.to_datetime(target_date).date()
        sub = df[df["time"].dt.date >= td]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "date" : row["time"].strftime("%Y-%m-%d"),
            "price": float(row["close"]),
        }

    def get_prices_on_date(self, tickers, target_date):
        out = {}
        for t in tickers:
            r = self.get_price_on_or_after(t, target_date)
            if r:
                out[t] = r["price"]
        return out

    def build_price_maps(self):
        out = {}
        for ticker, df in self.cache.items():
            tmp       = df.copy()
            tmp["ds"] = tmp["time"].dt.strftime("%Y-%m-%d")
            out[ticker] = dict(
                zip(tmp["ds"], tmp["close"].astype(float))
            )
        return out

    def get_history_before(self, ticker, before_date, n_rows):
        feat = self.get_featured_df(ticker)
        if feat is None:
            return pd.DataFrame()
        bd  = pd.to_datetime(before_date)
        sub = feat[feat["time"] < bd].tail(n_rows)
        return sub.copy()

    def get_historical_returns(self, tickers, before_date,
                               n_weeks=104):
        """
        Weekly returns, align by tail position.
        Không convert RangeIndex → tránh lỗi join shape(1,N).
        """
        raw_series = {}
        for ticker in tickers:
            df = self.cache.get(ticker)
            if df is None:
                continue
            bd  = pd.to_datetime(before_date)
            sub = df[df["time"] < bd].tail(n_weeks + 1).copy()
            if len(sub) < 20:
                continue
            sub["ret"] = sub["close"].pct_change()
            ret        = sub["ret"].dropna()
            if len(ret) >= 20:
                raw_series[ticker] = ret.values

        if not raw_series:
            return pd.DataFrame()

        min_len = min(len(v) for v in raw_series.values())
        aligned = {k: v[-min_len:] for k, v in raw_series.items()}
        return pd.DataFrame(aligned)

    def get_union_trading_dates(self, start_date, end_date):
        s, e   = pd.to_datetime(start_date), pd.to_datetime(end_date)
        all_ds = set()
        for df in self.cache.values():
            sub = df[(df["time"] >= s) & (df["time"] <= e)]
            all_ds.update(sub["time"].dt.strftime("%Y-%m-%d"))
        return sorted(all_ds)


# =========================================================
# MODEL MANAGER
# =========================================================
class ModelManager:

    def __init__(self):
        self._models = {}

    def get_model(self, ticker):
        t = ticker.upper()
        if t not in self._models:
            print(f"🔧 Loading model {t}...", end=" ", flush=True)
            try:
                pkg = _ensure_model(t)
                self._models[t] = pkg
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
                self._models[t] = None
        return self._models[t]

    def predict_price(self, ticker, past_window_df):
        pkg = self.get_model(ticker)
        if pkg is None:
            return None
        model, f_sc, t_sc, meta = pkg
        lookback = _CONFIG["lookback"]
        if len(past_window_df) < lookback:
            return None
        try:
            pred = _predict_next_price(
                past_window  = past_window_df,
                feature_list = _FEATURE_LIST,
                model        = model,
                f_scaler     = f_sc,
                t_scaler     = t_sc,
                lookback     = lookback,
            )
            return float(pred)
        except Exception as e:
            if DEBUG:
                print(f"      ⚠️ predict {ticker}: {e}")
            return None


# =========================================================
# ER CALCULATOR (weekly → compound 4 tuần)
# =========================================================
class ERCalculator:

    def __init__(self, data_cache, model_manager,
                 horizon=FORECAST_HORIZON):
        self.dc      = data_cache
        self.mm      = model_manager
        self.horizon = horizon   # 4 tuần

    def get_er(self, ticker, date_str):
        """
        Dự báo 1 tuần tới → compound horizon lần = ER 4 tuần.
        """
        lookback = _CONFIG["lookback"]
        past     = self.dc.get_history_before(
            ticker, date_str, lookback + 10
        )
        if past.empty:
            return None

        current = self.dc.get_price_on_or_after(ticker, date_str)
        if current is None:
            return None

        p0   = current["price"]
        pred = self.mm.predict_price(ticker, past)

        if pred is None or p0 <= 0:
            return None

        er1 = (pred / p0) - 1.0
        er1 = np.clip(er1, -0.20, 0.20)
        return (1 + er1) ** self.horizon - 1.0


# =========================================================
# CORE-SATELLITE STRATEGY
# =========================================================
def build_portfolio_strategy(all_ers,
                              min_core_er=MIN_CORE_ER,
                              max_satellite_loss=MAX_SATELLITE_LOSS):
    core      = {t: er for t, er in all_ers.items()
                 if er >= min_core_er}
    satellite = {t: er for t, er in all_ers.items()
                 if max_satellite_loss < er < min_core_er}

    print("\n" + "=" * 70)
    print("📊 CORE-SATELLITE STRATEGY (BUY & HOLD)")
    print("=" * 70)

    if core:
        print(f"\n🎯 CORE ({len(core)} mã — ER >= {min_core_er*100}%):")
        for t, er in sorted(core.items(),
                            key=lambda x: x[1], reverse=True):
            print(f"  {t}: {er*100:+6.2f}%")

    if satellite:
        print(f"\n🔍 SATELLITE ({len(satellite)} mã):")
        for t, er in sorted(satellite.items(),
                            key=lambda x: x[1], reverse=True):
            print(f"  {t}: {er*100:+6.2f}%")

    # ── Chọn portfolio ────────────────────────────────────
    if not core and not satellite:
        print(f"\n❌ KHÔNG CÓ MÃ NÀO ĐỦ ĐIỀU KIỆN!")
        print(f"   Fallback: top 3 ít thua nhất")
        sorted_all = sorted(all_ers.items(),
                            key=lambda x: x[1], reverse=True)
        portfolio  = dict(sorted_all[:min(3, len(sorted_all))])

    elif not core:
        print(f"\n⚠️  KHÔNG CÓ CORE — Chỉ Satellite")
        portfolio = satellite

    elif len(core) == 1:
        t_core = list(core.keys())[0]
        print(f"\n⚠️  CHỈ CÓ 1 CORE: {t_core}")
        if satellite:
            print(f"   → Thêm {len(satellite)} Satellite")
            portfolio = {**core, **satellite}
        else:
            others = {t: er for t, er in all_ers.items()
                      if t not in core}
            top2   = dict(
                sorted(others.items(),
                       key=lambda x: x[1], reverse=True)[:2]
            )
            print(f"   → Thêm 2 defensive: {list(top2.keys())}")
            portfolio = {**core, **top2}

    else:
        print(f"\n✅ CÓ {len(core)} CORE")
        portfolio = {**core, **satellite} if satellite else core

    # Shift ER nếu có âm
    min_er = min(portfolio.values())
    if min_er < 0:
        print(f"\n📊 Shift ER: {min_er*100:.2f}% → 0.1%")
        portfolio = {
            t: er - min_er + 0.001
            for t, er in portfolio.items()
        }

    print(f"\n✅ Danh mục cuối: {len(portfolio)} mã — "
          f"{list(portfolio.keys())}")
    return portfolio


# =========================================================
# PORTFOLIO OPTIMIZER (MPT — weekly RF)
# =========================================================
class PortfolioOptimizer:

    def __init__(self):
        self.rf         = WEEKLY_RF
        self.lambda_reg = LAMBDA_REG

    def optimize(self, expected_returns, cov_matrix):
        tickers = list(expected_returns.keys())
        n       = len(tickers)
        if n == 0:
            return {}

        er_arr  = np.array([expected_returns[t] for t in tickers])
        cov_arr = cov_matrix.loc[tickers, tickers].values

        def objective(w):
            ret = np.sum(er_arr * w)
            std = np.sqrt(w @ cov_arr @ w)
            if std <= 0:
                return 1e10
            return (-(ret - self.rf) / std
                    + self.lambda_reg * np.sum(w**2))

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]
        bounds = tuple((MIN_POSITION, MAX_POSITION)
                       for _ in range(n))
        x0     = np.array([1/n] * n)

        res = minimize(objective, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 5000, "ftol": 1e-12})

        w = res.x if res.success else x0
        w = w / w.sum()

        if DEBUG:
            print(f"\n   Optimizer Results:")
            print(f"   ER    : { {t:f'{er_arr[i]*100:+.2f}%' for i,t in enumerate(tickers)} }")
            print(f"   Weights: { {t:f'{w[i]*100:.1f}%' for i,t in enumerate(tickers)} }")
            if not res.success:
                print(f"   ⚠️  {res.message}")

        return dict(zip(tickers, w))


# =========================================================
# UTILS
# =========================================================
def round_lot(qty):
    return int(qty // LOT_SIZE) * LOT_SIZE

def allocate_by_weights(capital, weights, prices):
    alloc     = {}
    total_tv  = 0
    total_fee = 0
    for ticker, w in weights.items():
        if ticker not in prices:
            alloc[ticker] = 0
            continue
        tv  = capital * w / (1 + TRANSACTION_FEE)
        qty = round_lot(tv / prices[ticker])
        alloc[ticker]  = qty
        total_tv      += qty * prices[ticker]
        total_fee     += qty * prices[ticker] * TRANSACTION_FEE
    while total_tv + total_fee > capital:
        cands = [
            (alloc[t] * prices[t], t)
            for t in alloc
            if alloc[t] >= LOT_SIZE and t in prices
        ]
        if not cands:
            break
        _, tk = max(cands)
        alloc[tk] -= LOT_SIZE
        total_tv   = sum(alloc[t] * prices[t]
                         for t in alloc if t in prices)
        total_fee  = total_tv * TRANSACTION_FEE
    return alloc

def compute_max_drawdown(values):
    arr = np.array(values)
    rm  = np.maximum.accumulate(arr)
    dd  = (arr - rm) / np.where(rm > 0, rm, 1)
    return dd.min()

def generate_equal_weights(tickers):
    return {t: 1.0 / len(tickers) for t in tickers}


# =========================================================
# PORTFOLIO
# =========================================================
class Portfolio:

    def __init__(self, initial_capital, name="Portfolio"):
        self.name            = name
        self.initial_capital = float(initial_capital)
        self.cash            = float(initial_capital)
        self.holdings        = defaultdict(int)
        self.transactions    = []

    def buy(self, date, ticker, qty, price):
        if qty <= 0:
            return False
        cost = qty * price * (1 + TRANSACTION_FEE)
        if cost > self.cash:
            return False
        self.cash -= cost
        self.holdings[ticker] += qty
        self.transactions.append({
            "date": date, "type": "BUY",
            "ticker": ticker, "qty": qty, "price": price,
            "trade_value": qty * price,
            "fee"        : qty * price * TRANSACTION_FEE,
            "cash_after" : self.cash,
        })
        return True

    def sell(self, date, ticker, qty, price):
        if qty <= 0 or self.holdings[ticker] < qty:
            return False
        tv = qty * price
        self.cash += tv * (1 - TRANSACTION_FEE)
        self.holdings[ticker] -= qty
        self.transactions.append({
            "date": date, "type": "SELL",
            "ticker": ticker, "qty": qty, "price": price,
            "trade_value": tv,
            "fee"        : tv * TRANSACTION_FEE,
            "cash_after" : self.cash,
        })
        return True

    def sell_all(self, date, prices):
        for t, qty in list(self.holdings.items()):
            if qty > 0 and t in prices:
                self.sell(date, t, qty, prices[t])

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
# MODEL-DRIVEN BUY & HOLD
# =========================================================
def run_buy_hold_model_driven(data_cache, er_calc,
                               buy_date, sell_date):
    port      = Portfolio(INITIAL_CAPITAL,
                          "Model-Driven Buy & Hold")
    optimizer = PortfolioOptimizer()

    print("\n" + "=" * 80)
    print("🤖 MODEL-DRIVEN BUY & HOLD (a_ML5_weekly | PURE RF)")
    print(f"   Buy : {buy_date}")
    print(f"   Sell: {sell_date}")
    print(f"   ER horizon: {FORECAST_HORIZON} tuần")
    print("=" * 80)

    # ── Giá mua ──────────────────────────────────────────
    buy_prices = data_cache.get_prices_on_date(TICKERS, buy_date)
    if len(buy_prices) < len(TICKERS):
        missing = set(TICKERS) - set(buy_prices.keys())
        print(f"⚠️  Thiếu giá: {missing}")

    print(f"\n📅 Giá mua ({buy_date}):")
    for t in sorted(buy_prices):
        print(f"   {t}: {buy_prices[t]:,.0f}")

    # ── Expected Returns ──────────────────────────────────
    print(f"\n🔮 Expected Returns ({FORECAST_HORIZON} tuần):")
    all_ers = {}
    for t in TICKERS:
        er = er_calc.get_er(t, buy_date)
        if er is not None:
            all_ers[t] = er
            print(f"   {t}: {er*100:+6.2f}%")
        else:
            print(f"   {t}: ❌ Failed")

    if not all_ers:
        print("❌ Không có ER → Abort")
        return port, None

    # ── Core-Satellite ────────────────────────────────────
    ers = build_portfolio_strategy(all_ers)
    if not ers:
        print("❌ Không có portfolio → Abort")
        return port, None

    # ── Covariance ────────────────────────────────────────
    print(f"\n📊 Covariance Matrix (Ledoit-Wolf)...")
    ret_df = data_cache.get_historical_returns(
        list(ers.keys()), buy_date, n_weeks=104
    )
    n_obs = len(ret_df)
    p     = len(ers)
    print(f"   Data: {n_obs} tuần / {p} mã "
          f"(n/p = {n_obs/p:.1f})")

    if n_obs < max(20, p + 5):
        print("   ⚠️  Thiếu data → Fallback Equal Weight")
        weights = {t: 1/p for t in ers}
    else:
        lw = LedoitWolf()
        lw.fit(ret_df)
        cov = pd.DataFrame(
            lw.covariance_ * FORECAST_HORIZON,
            index   = ret_df.columns,
            columns = ret_df.columns,
        )
        print(f"   Shrinkage: {lw.shrinkage_:.4f}")
        weights = optimizer.optimize(ers, cov)

    # ── Mua ──────────────────────────────────────────────
    print(f"\n💰 Mua tại {buy_date}:")
    alloc = allocate_by_weights(
        port.cash, weights,
        {t: buy_prices[t] for t in weights if t in buy_prices}
    )
    total_qty = 0
    for t, qty in sorted(alloc.items()):
        if qty > 0:
            port.buy(buy_date, t, qty, buy_prices[t])
            tv = qty * buy_prices[t]
            print(f"   {t}: {qty:,} cp @ {buy_prices[t]:,.0f}"
                  f" = {tv:,.0f}"
                  f"  ({weights.get(t,0)*100:.1f}%)")
            total_qty += qty

    buy_val = port.total_assets(buy_prices)
    print(f"\n   Total mua  : {buy_val:,.0f}")
    print(f"   Cash còn   : {port.cash:,.0f}")
    print(f"   Tổng cổ phần: {total_qty:,}")

    # ── Giữ ──────────────────────────────────────────────
    print(f"\n🔒 HOLD: {buy_date} → {sell_date} (không rebalance)")

    # ── Bán ──────────────────────────────────────────────
    sell_prices = data_cache.get_prices_on_date(TICKERS, sell_date)

    print(f"\n📅 Giá bán ({sell_date}):")
    for t in sorted(TICKERS):
        if t in sell_prices and t in buy_prices:
            pct = (sell_prices[t] / buy_prices[t] - 1) * 100
            arrow = "📈" if pct >= 0 else "📉"
            print(f"   {arrow} {t}: {sell_prices[t]:,.0f}"
                  f"  ({pct:+.2f}%)")

    port.sell_all(sell_date, sell_prices)
    sell_val = port.total_assets(sell_prices)
    final_ret = sell_val / INITIAL_CAPITAL - 1
    print(f"\n   Total bán  : {sell_val:,.0f}")
    print(f"   Return     : {final_ret*100:+.2f}%")

    return port, (buy_date, sell_date)


# =========================================================
# EQUAL WEIGHT BUY & HOLD
# =========================================================
def run_buy_hold_equal_weight(data_cache, buy_date, sell_date):
    port = Portfolio(INITIAL_CAPITAL,
                     "Equal Weight Buy & Hold")

    print("\n" + "=" * 80)
    print("⚖️  EQUAL WEIGHT (1/N) BUY & HOLD")
    print(f"   Buy : {buy_date}")
    print(f"   Sell: {sell_date}")
    print("=" * 80)

    # ── Giá mua ──────────────────────────────────────────
    buy_prices = data_cache.get_prices_on_date(TICKERS, buy_date)
    weights    = generate_equal_weights(
        [t for t in TICKERS if t in buy_prices]
    )

    print(f"\n📅 Giá mua ({buy_date}):")
    for t in sorted(buy_prices):
        print(f"   {t}: {buy_prices[t]:,.0f}"
              f"  ({weights.get(t,0)*100:.1f}%)")

    # ── Mua ──────────────────────────────────────────────
    print(f"\n💰 Mua tại {buy_date}:")
    alloc = allocate_by_weights(port.cash, weights, buy_prices)
    total_qty = 0
    for t, qty in sorted(alloc.items()):
        if qty > 0:
            port.buy(buy_date, t, qty, buy_prices[t])
            tv = qty * buy_prices[t]
            print(f"   {t}: {qty:,} cp @ {buy_prices[t]:,.0f}"
                  f" = {tv:,.0f}"
                  f"  ({weights.get(t,0)*100:.1f}%)")
            total_qty += qty

    buy_val = port.total_assets(buy_prices)
    print(f"\n   Total mua  : {buy_val:,.0f}")
    print(f"   Cash còn   : {port.cash:,.0f}")
    print(f"   Tổng cổ phần: {total_qty:,}")

    # ── Giữ ──────────────────────────────────────────────
    print(f"\n🔒 HOLD: {buy_date} → {sell_date} (không rebalance)")

    # ── Bán ──────────────────────────────────────────────
    sell_prices = data_cache.get_prices_on_date(TICKERS, sell_date)

    print(f"\n📅 Giá bán ({sell_date}):")
    for t in sorted(TICKERS):
        if t in sell_prices and t in buy_prices:
            pct   = (sell_prices[t] / buy_prices[t] - 1) * 100
            arrow = "📈" if pct >= 0 else "📉"
            print(f"   {arrow} {t}: {sell_prices[t]:,.0f}"
                  f"  ({pct:+.2f}%)")

    port.sell_all(sell_date, sell_prices)
    sell_val  = port.total_assets(sell_prices)
    final_ret = sell_val / INITIAL_CAPITAL - 1
    print(f"\n   Total bán  : {sell_val:,.0f}")
    print(f"   Return     : {final_ret*100:+.2f}%")

    return port, (buy_date, sell_date)


# =========================================================
# EQUITY CURVE
# =========================================================
def build_equity_curve(data_cache, portfolio,
                       start_date, end_date):
    dates      = data_cache.get_union_trading_dates(
        start_date, end_date
    )
    price_maps = data_cache.build_price_maps()
    tx_df      = portfolio.get_transactions_df().copy()

    if tx_df.empty:
        return pd.DataFrame()

    tx_df["date"] = pd.to_datetime(tx_df["date"])
    tx_df = tx_df.sort_values("date")

    rows     = []
    holdings = defaultdict(int)
    cash     = portfolio.initial_capital
    tx_idx   = 0

    for ds in dates:
        if not (start_date <= ds <= end_date):
            continue
        dts = pd.to_datetime(ds).date()
        while (tx_idx < len(tx_df) and
               tx_df.iloc[tx_idx]["date"].date() <= dts):
            tx  = tx_df.iloc[tx_idx]
            qty = int(tx["qty"])
            if tx["type"] == "BUY":
                holdings[tx["ticker"]] += qty
            else:
                holdings[tx["ticker"]] -= qty
            cash   = float(tx["cash_after"])
            tx_idx += 1
        stock = sum(
            holdings[t] * price_maps[t][ds]
            for t in TICKERS
            if holdings[t] > 0 and ds in price_maps.get(t, {})
        )
        rows.append({"date": ds, "total_assets": cash + stock})

    return pd.DataFrame(rows)


# =========================================================
# SUMMARY
# =========================================================
def summarize(name, portfolio, curve_df):

    if curve_df.empty:
        return {
            "Strategy": name, "Final Value (VND)": "0",
            "Return (%)": 0, "Max Drawdown (%)": 0,
            "Sharpe (annul)": 0, "Total Fees": "0",
            "Total Turnover": "0",
        }

    fv     = float(curve_df["total_assets"].iloc[-1])
    ret    = fv / portfolio.initial_capital - 1
    max_dd = compute_max_drawdown(
        curve_df["total_assets"].tolist()
    )
    tx_df    = portfolio.get_transactions_df()
    fees     = tx_df["fee"].sum()         if not tx_df.empty else 0
    turnover = tx_df["trade_value"].sum() if not tx_df.empty else 0

    # Weekly bars → annualize ×√52
    sharpe = 0.0
    if len(curve_df) > 2:
        dr     = curve_df["total_assets"].pct_change().dropna()
        excess = dr - WEEKLY_RF
        if excess.std() > 0:
            sharpe = excess.mean() / excess.std() * np.sqrt(52)

    return {
        "Strategy"         : name,
        "Final Value (VND)": f"{fv:,.0f}",
        "Return (%)"       : round(ret * 100, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Sharpe (annul)"   : round(sharpe, 4),
        "Total Fees"       : f"{fees:,.0f}",
        "Total Turnover"   : f"{turnover:,.0f}",
    }


# =========================================================
# PLOT
# =========================================================
def plot_comparison(curve_md, curve_ew, s1, s2):

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Buy & Hold Backtest: {BUY_DATE} → {SELL_DATE}\n"
        f"Model-Driven (PURE RF weekly) vs Equal Weight (1/N)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Equity curve ─────────────────────────────
    ax1 = axes[0]
    if not curve_md.empty:
        ax1.plot(
            pd.to_datetime(curve_md["date"]),
            curve_md["total_assets"],
            label=(f"Model-Driven "
                   f"({s1['Return (%)']:+.1f}% | "
                   f"Sharpe {s1['Sharpe (annul)']:.2f})"),
            color="royalblue", lw=2.5,
        )
    if not curve_ew.empty:
        ax1.plot(
            pd.to_datetime(curve_ew["date"]),
            curve_ew["total_assets"],
            label=(f"Equal Weight "
                   f"({s2['Return (%)']:+.1f}% | "
                   f"Sharpe {s2['Sharpe (annul)']:.2f})"),
            color="tomato", lw=2.5, ls="--",
        )
    ax1.axhline(INITIAL_CAPITAL, color="gray",
                ls=":", alpha=0.5, label="Initial Capital")
    ax1.set_ylabel("Portfolio Value (VND)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    # ── Panel 2: Drawdown ─────────────────────────────────
    ax2 = axes[1]
    for curve, label, color in [
        (curve_md, "Model-Driven", "royalblue"),
        (curve_ew, "Equal Weight", "tomato"),
    ]:
        if curve.empty:
            continue
        arr = curve["total_assets"].values.astype(float)
        rm  = np.maximum.accumulate(arr)
        dd  = (arr - rm) / np.where(rm > 0, rm, 1) * 100
        ax2.plot(pd.to_datetime(curve["date"]), dd,
                 label=label, color=color, lw=1.5)
        ax2.fill_between(pd.to_datetime(curve["date"]),
                         dd, 0, alpha=0.15, color=color)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=1)
        )
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=45, ha="right")

    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Chart: {out}")


# =========================================================
# MAIN
# =========================================================
def main():

    print("\n" + "=" * 80)
    print("📈 BUY & HOLD BACKTEST v2 (a_ML5_weekly)")
    print(f"   Period : {BUY_DATE} → {SELL_DATE}")
    print(f"   Tickers: {TICKERS}")
    print(f"   Capital: {INITIAL_CAPITAL:,.0f} VND")
    print(f"   ER horizon: {FORECAST_HORIZON} tuần")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────
    dc = DataCache(TICKERS, BUY_DATE, SELL_DATE)
    dc.load_all_data()

    # ── Load models ───────────────────────────────────────
    print("\n🔧 Pre-loading models...")
    mm = ModelManager()
    for t in TICKERS:
        mm.get_model(t)

    er_calc = ERCalculator(dc, mm, horizon=FORECAST_HORIZON)

    # ── Run strategies ────────────────────────────────────
    md_port, md_dates = run_buy_hold_model_driven(
        dc, er_calc, BUY_DATE, SELL_DATE
    )
    ew_port, ew_dates = run_buy_hold_equal_weight(
        dc, BUY_DATE, SELL_DATE
    )

    # ── Equity curves ─────────────────────────────────────
    curve_md = build_equity_curve(dc, md_port, BUY_DATE, SELL_DATE)
    curve_ew = build_equity_curve(dc, ew_port, BUY_DATE, SELL_DATE)

    # ── Summary ───────────────────────────────────────────
    s1 = summarize("Model-Driven Buy&Hold", md_port, curve_md)
    s2 = summarize("Equal Weight 1/N",      ew_port, curve_ew)
    summary_df = pd.DataFrame([s1, s2])

    print("\n" + "=" * 100)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # ── Save ──────────────────────────────────────────────
    def _save(df, name):
        df.to_csv(
            os.path.join(RESULT_DIR, name),
            index=False, encoding="utf-8-sig"
        )

    _save(summary_df,                      "summary.csv")
    _save(curve_md,                        "equity_curve_model_driven.csv")
    _save(curve_ew,                        "equity_curve_equal_weight.csv")
    _save(md_port.get_transactions_df(),   "tx_model_driven.csv")
    _save(ew_port.get_transactions_df(),   "tx_equal_weight.csv")

    # ── Plot ──────────────────────────────────────────────
    plot_comparison(curve_md, curve_ew, s1, s2)

    print(f"\n✅ DONE → {RESULT_DIR}/")
    for f in ["summary.csv",
              "equity_curve_model_driven.csv",
              "equity_curve_equal_weight.csv",
              "tx_model_driven.csv",
              "tx_equal_weight.csv",
              "comparison.png"]:
        print(f"   • {f}")


if __name__ == "__main__":
    main()