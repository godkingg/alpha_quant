# =========================================================
# WEEKLY REBALANCE BACKTEST v2 — FIXED for a_ML2_pure.py
# 
# Thay đổi so với bản gốc:
#   ✅ Import đúng: a_ML2_pure (daily interval)
#   ✅ Rebalance: hàng tuần nhưng dùng daily data
#   ✅ Horizon = 5 phiên (1 tuần giao dịch)
#   ✅ Feature list: CORE_MOMENTUM_FEATURES_DELTA
#   ✅ Config sync: zlema_period=5, lookback=22
#   ✅ ER calc: predict 5 bước recursive → compound
#   ✅ Core-Satellite threshold điều chỉnh cho daily scale
#   ✅ Sharpe annualize ×√252 (daily)
#   ✅ Final sell: ngày cuối backtest (không hardcode 2025-01)
# =========================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from collections import defaultdict
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# IMPORT a_ML2_pure  (daily RF model)
# ─────────────────────────────────────────────────────────
try:
    import a_ML4_daily as ml
    print("✅ Loaded a_ML4_daily")
    print(f"   interval     : {ml.CONFIG.get('interval','d')}")
    print(f"   start_date   : {ml.CONFIG.get('start_date','N/A')}")
    print(f"   zlema_period : {ml.CONFIG.get('zlema_period','N/A')}")
    print(f"   lookback     : {ml.CONFIG.get('lookback','N/A')}")
except ImportError as e:
    print(f"❌ Không tìm thấy a_ML2_pure: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────
# AUTO-MAP API  (giữ lại để dễ swap module)
# ─────────────────────────────────────────────────────────
def _get_attr(module, candidates, what="function"):
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)
    available = [x for x in dir(module) if not x.startswith("_")]
    raise AttributeError(
        f"Không tìm thấy {what}.\n"
        f"Candidates : {candidates}\n"
        f"Available  : {available[:60]}"
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

_FEATURE_LIST = _get_attr(ml, [
    "CORE_MOMENTUM_FEATURES_DELTA",
    "CORE_MOMENTUM_FEATURES",
    "FEATURE_LIST",
    "FEATURES",
], what="feature_list")

_CONFIG = ml.CONFIG

print(f"\n✅ API mapped:")
print(f"   compute_features → {_compute_features.__name__}")
print(f"   predict_next     → {_predict_next_price.__name__}")
print(f"   ensure_model     → {_ensure_model.__name__}")
print(f"   feature_list     → {len(_FEATURE_LIST)} features")
print(f"   interval         → {_CONFIG.get('interval','d')} (daily)")


# =========================================================
# CONFIG
# =========================================================
INITIAL_CAPITAL  = 100_000_000
TRANSACTION_FEE  = 0.0015        # 0.15% mỗi chiều
LOT_SIZE         = 100

# ── Danh sách cổ phiếu ────────────────────────────────────
TICKERS = ["TCB", "VRE", "VCB", "SSI", "FPT"]

# ── Khoảng backtest ───────────────────────────────────────
BACKTEST_START = "2025-01-01"
BACKTEST_END   = "2025-12-31"   # ← sửa: tránh hardcode tương lai

# ── Forecast horizon ─────────────────────────────────────
# Daily model → 1 tuần ≈ 5 phiên giao dịch
FORECAST_HORIZON = 5            # ← sửa: 5 phiên thay vì 1 (weekly)

# ── MPT — daily scale ─────────────────────────────────────
ANNUAL_RF = 0.05
DAILY_RF  = (1 + ANNUAL_RF) ** (1 / 252) - 1
WEEKLY_RF = (1 + ANNUAL_RF) ** (5 / 252) - 1   # 5 phiên

LAMBDA_REG    = 0.001
MAX_TURNOVER  = 0.50
MAX_POSITION  = 0.40
MIN_POSITION  = 0.05

# ── Core-Satellite — 5 phiên scale ───────────────────────
# ← sửa: threshold cho 5-phiên ER thay vì weekly %
MIN_CORE_ER        =  0.01    # ER >= +1%/5 phiên → Core
MAX_SATELLITE_LOSS = -0.02    # ER >= -2%/5 phiên → Satellite (cắt lỗ)

# ── Risk management ───────────────────────────────────────
MAX_DRAWDOWN_STOP  = -0.20
DRAWDOWN_WARNING   = -0.12
RECOVERY_THRESHOLD = -0.05

DEBUG      = True
RESULT_DIR = "backtest_results_weekly_v2"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# DATA CACHE  (daily interval)
# =========================================================
class DataCache:

    def __init__(self, tickers, start_date, end_date):
        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date
        self.cache      = {}      # ticker → raw daily df
        self.feat_cache = {}      # ticker → featured df

    # ── Load daily data ───────────────────────────────────
    def load_all_data(self):
        print("\n" + "=" * 70)
        print("📦 PRELOAD DATA (daily interval)")
        print("=" * 70)
        # Lấy thêm lịch sử để train feature
        train_start = _CONFIG.get("start_date", "2019-06-15")

        for ticker in self.tickers:
            print(f"⏳ {ticker}...", end=" ", flush=True)
            try:
                # ← sửa: interval="d" thay vì "w"
                df = _load_data(ticker, train_start,
                                self.end_date, "d")
                df = _validate_data(df)
                df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time").reset_index(drop=True)
                self.cache[ticker] = df
                print(
                    f"✅ {len(df)} ngày "
                    f"({df['time'].iloc[0].strftime('%Y-%m-%d')}"
                    f" → {df['time'].iloc[-1].strftime('%Y-%m-%d')})"
                )
            except Exception as e:
                print(f"❌ {e}")

    # ── Featured df ───────────────────────────────────────
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

    # ── Giá tại / sau ngày ───────────────────────────────
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

    # ── Price map {date_str → price} ─────────────────────
    def build_price_maps(self):
        out = {}
        for ticker, df in self.cache.items():
            tmp       = df.copy()
            tmp["ds"] = tmp["time"].dt.strftime("%Y-%m-%d")
            out[ticker] = dict(
                zip(tmp["ds"], tmp["close"].astype(float))
            )
        return out

    # ── Historical daily returns (cho Cov) ───────────────
    def get_historical_returns(self, tickers, before_date,
                               n_days=252):
        """
        Daily returns, align by tail position.
        ← sửa: n_days thay vì n_weeks
        """
        raw_series = {}
        for ticker in tickers:
            df = self.cache.get(ticker)
            if df is None:
                continue
            bd  = pd.to_datetime(before_date)
            sub = df[df["time"] < bd].tail(n_days + 1).copy()
            if len(sub) < 30:
                continue
            sub = sub.copy()
            sub["ret"] = sub["close"].pct_change()
            ret = sub["ret"].dropna()
            if len(ret) >= 30:
                raw_series[ticker] = ret.values

        if not raw_series:
            return pd.DataFrame()

        min_len = min(len(v) for v in raw_series.values())
        aligned = {k: v[-min_len:] for k, v in raw_series.items()}
        return pd.DataFrame(aligned)

    # ── Past window cho model ────────────────────────────
    def get_history_before(self, ticker, before_date, n_rows):
        feat = self.get_featured_df(ticker)
        if feat is None:
            return pd.DataFrame()
        bd  = pd.to_datetime(before_date)
        sub = feat[feat["time"] < bd].tail(n_rows)
        return sub.copy()

    # ── Union trading dates (daily) ───────────────────────
    def get_union_trading_dates(self, start_date, end_date):
        s, e   = pd.to_datetime(start_date), pd.to_datetime(end_date)
        all_ds = set()
        for df in self.cache.values():
            sub = df[(df["time"] >= s) & (df["time"] <= e)]
            all_ds.update(sub["time"].dt.strftime("%Y-%m-%d"))
        return sorted(all_ds)

    # ── Ngày rebalance: thứ Hai mỗi tuần ─────────────────
    def get_all_monday_dates(self, start_date, end_date,
                             ticker_ref=None):
        """
        ← sửa: thay get_all_weekly_dates bằng hàm lấy ngày
          giao dịch đầu tuần (thứ Hai hoặc ngày GD gần nhất).
        """
        if ticker_ref is None:
            ticker_ref = self.tickers[0]
        df = self.cache.get(ticker_ref)
        if df is None:
            return []

        s, e  = pd.to_datetime(start_date), pd.to_datetime(end_date)
        sub   = df[(df["time"] >= s) & (df["time"] <= e)].copy()
        sub["week"] = sub["time"].dt.isocalendar().week.astype(str)
        sub["year"] = sub["time"].dt.year.astype(str)
        sub["yw"]   = sub["year"] + "_" + sub["week"]

        # Lấy ngày đầu tiên mỗi tuần
        first_of_week = (
            sub.groupby("yw")["time"]
               .min()
               .sort_values()
               .dt.strftime("%Y-%m-%d")
               .tolist()
        )
        return first_of_week

    def get_first_trading_day_of_month(self, year, month,
                                       ticker_ref=None):
        if ticker_ref is None:
            ticker_ref = self.tickers[0]
        df = self.cache.get(ticker_ref)
        if df is None:
            return None
        sub = df[
            (df["time"].dt.year  == year) &
            (df["time"].dt.month == month)
        ]
        if sub.empty:
            return None
        return sub.iloc[0]["time"].strftime("%Y-%m-%d")

    # ← thêm: lấy ngày giao dịch cuối cùng trong backtest
    def get_last_trading_date(self, end_date, ticker_ref=None):
        if ticker_ref is None:
            ticker_ref = self.tickers[0]
        df = self.cache.get(ticker_ref)
        if df is None:
            return None
        e   = pd.to_datetime(end_date)
        sub = df[df["time"] <= e]
        if sub.empty:
            return None
        return sub.iloc[-1]["time"].strftime("%Y-%m-%d")


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
# ER CALCULATOR — RECURSIVE 5 BƯỚC (daily → 1 tuần)
# =========================================================
class ERCalculator:
    """
    ← sửa hoàn toàn: recursive forecast 5 phiên daily
      thay vì dùng predict 1 bước weekly.
    
    Quy trình:
      1. Lấy past window (daily features)
      2. Predict bước 1 → append synthetic row
      3. Recompute features → predict bước 2
      4. Lặp FORECAST_HORIZON lần
      5. ER = (price_step5 / price_now) - 1
    """

    def __init__(self, data_cache, model_manager,
                 horizon=FORECAST_HORIZON):
        self.dc      = data_cache
        self.mm      = model_manager
        self.horizon = horizon

    def get_er_weekly(self, ticker, date_str):
        lookback = _CONFIG["lookback"]
        # Lấy raw data trước ngày này
        raw_df = self.dc.cache.get(ticker)
        if raw_df is None:
            return None

        bd      = pd.to_datetime(date_str)
        raw_sub = raw_df[raw_df["time"] < bd].copy()
        if len(raw_sub) < lookback + 50:
            return None

        current = self.dc.get_price_on_or_after(ticker, date_str)
        if current is None:
            return None
        p0 = current["price"]

        # ── Recursive forecast HORIZON bước ───────────────
        working_raw = raw_sub.copy()

        try:
            for step in range(self.horizon):
                # Recompute features
                working_feat = _compute_features(
                    working_raw.tail(300).copy(),
                    zlema_period=_CONFIG["zlema_period"],
                )
                if len(working_feat) < lookback:
                    return None

                past_window = working_feat.tail(lookback).copy()
                pred_price  = self.mm.predict_price(ticker, past_window)

                if pred_price is None or pred_price <= 0:
                    return None

                # Tạo synthetic row
                prev        = working_raw.iloc[-1].copy()
                prev_close  = float(working_raw["close"].iloc[-1])
                new_row     = prev.copy()
                new_row["time"]   = prev["time"] + timedelta(days=1)
                # Bỏ qua cuối tuần
                while new_row["time"].weekday() >= 5:
                    new_row["time"] += timedelta(days=1)
                new_row["open"]   = prev_close
                new_row["close"]  = pred_price
                new_row["high"]   = max(prev_close, pred_price)
                new_row["low"]    = min(prev_close, pred_price)
                new_row["volume"] = float(
                    working_raw["volume"].tail(20).mean()
                )

                working_raw = pd.concat(
                    [working_raw, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

            # Giá cuối sau HORIZON bước
            p_final = float(working_raw["close"].iloc[-1])
            er      = (p_final / p0) - 1.0
            # Cap ER: ±15% / 5 phiên (hợp lý cho VN)
            return float(np.clip(er, -0.15, 0.15))

        except Exception as e:
            if DEBUG:
                print(f"      ⚠️ ER {ticker} {date_str}: {e}")
            return None


# =========================================================
# CORE-SATELLITE STRATEGY
# =========================================================
def build_portfolio_strategy(all_ers,
                              min_core_er=MIN_CORE_ER,
                              max_satellite_loss=MAX_SATELLITE_LOSS):
    """
    Phân loại theo ER 5-phiên:
      Core      : ER >= +1%
      Satellite : -2% <= ER < +1%
      Loại bỏ  : ER < -2%
    """
    core      = {t: er for t, er in all_ers.items()
                 if er >= min_core_er}
    satellite = {t: er for t, er in all_ers.items()
                 if max_satellite_loss <= er < min_core_er}

    if DEBUG:
        print(f"   Core({len(core)}): "
              f"{[(t, f'{v*100:.2f}%') for t, v in core.items()]}")
        print(f"   Sat ({len(satellite)}): "
              f"{[(t, f'{v*100:.2f}%') for t, v in satellite.items()]}")

    # ── Chọn portfolio ────────────────────────────────────
    if not core and not satellite:
        # Defensive: top 3 ít thua nhất
        sorted_all = sorted(all_ers.items(),
                            key=lambda x: x[1], reverse=True)
        portfolio  = dict(sorted_all[:min(3, len(sorted_all))])
        if DEBUG:
            print(f"   → Fallback defensive: {list(portfolio.keys())}")

    elif not core:
        portfolio = satellite

    elif len(core) == 1:
        # 1 Core: bổ sung satellite hoặc 2 mã tốt nhất
        if satellite:
            portfolio = {**core, **satellite}
        else:
            others = {t: er for t, er in all_ers.items()
                      if t not in core}
            top2   = dict(sorted(others.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:2])
            portfolio = {**core, **top2}
    else:
        portfolio = {**core, **satellite} if satellite else core

    # Shift ER nếu có âm (để optimizer hoạt động)
    min_er = min(portfolio.values())
    if min_er < 0:
        shift     = abs(min_er) + 0.0001
        portfolio = {t: er + shift for t, er in portfolio.items()}

    return portfolio


# =========================================================
# PORTFOLIO OPTIMIZER (MPT — Sharpe maximize)
# =========================================================
class PortfolioOptimizer:

    def __init__(self):
        self.rf         = WEEKLY_RF      # ← 5-phiên RF
        self.lambda_reg = LAMBDA_REG

    def optimize(self, expected_returns, cov_matrix,
                 current_weights=None):
        tickers = list(expected_returns.keys())
        n       = len(tickers)
        if n == 0:
            return {}

        er_arr  = np.array([expected_returns[t] for t in tickers])

        # Scale cov: daily cov × 5 để ra 5-phiên cov
        cov_arr = cov_matrix.loc[tickers, tickers].values * FORECAST_HORIZON

        def objective(w):
            ret = np.sum(er_arr * w)
            std = np.sqrt(w @ cov_arr @ w)
            if std <= 0:
                return 1e10
            return (-(ret - self.rf) / std
                    + self.lambda_reg * np.sum(w ** 2))

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]
        if current_weights is not None:
            cur = np.array([
                current_weights.get(t, 0) for t in tickers
            ])
            constraints.append({
                "type": "ineq",
                "fun" : lambda x, c=cur:
                    MAX_TURNOVER - np.sum(np.abs(x - c))
            })

        bounds = tuple(
            (MIN_POSITION, MAX_POSITION) for _ in range(n)
        )
        x0 = np.array([
            current_weights.get(t, 1 / n)
            if current_weights else 1 / n
            for t in tickers
        ])
        x0 /= x0.sum()

        res = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 5000, "ftol": 1e-12},
        )

        w = res.x if res.success else x0
        w = w / w.sum()

        if DEBUG:
            wdict = {t: round(float(w[i]), 3)
                     for i, t in enumerate(tickers)}
            print(f"      Weights : {wdict}")
            if not res.success:
                print(f"      ⚠️  {res.message}")
            if current_weights is not None:
                cur = np.array([
                    current_weights.get(t, 0) for t in tickers
                ])
                to  = np.sum(np.abs(w - cur)) * 100
                print(f"      Turnover: {to:.1f}%")

        return dict(zip(tickers, w.tolist()))


# =========================================================
# RISK MANAGER
# =========================================================
class RiskManager:

    def __init__(self):
        self.peak_value = INITIAL_CAPITAL
        self.risk_mode  = "NORMAL"

    def update(self, current_value):
        self.peak_value = max(self.peak_value, current_value)
        dd = (current_value - self.peak_value) / self.peak_value
        if dd <= MAX_DRAWDOWN_STOP:
            self.risk_mode = "STOP"
        elif dd <= DRAWDOWN_WARNING:
            self.risk_mode = "DEFENSIVE"
        elif dd >= RECOVERY_THRESHOLD:
            self.risk_mode = "NORMAL"
        return self.risk_mode, dd

    def get_position_multiplier(self):
        return {
            "STOP"     : 0.0,
            "DEFENSIVE": 0.6,
            "NORMAL"   : 1.0,
        }.get(self.risk_mode, 1.0)


# =========================================================
# UTILS
# =========================================================
def round_lot(qty):
    return int(qty // LOT_SIZE) * LOT_SIZE


def allocate_by_weights(capital, weights, prices):
    alloc      = {}
    total_tv   = 0
    total_fee  = 0

    for ticker, w in weights.items():
        if ticker not in prices or prices[ticker] <= 0:
            alloc[ticker] = 0
            continue
        tv  = capital * w / (1 + TRANSACTION_FEE)
        qty = round_lot(tv / prices[ticker])
        alloc[ticker] = qty
        total_tv  += qty * prices[ticker]
        total_fee += qty * prices[ticker] * TRANSACTION_FEE

    # Giảm lot nếu vượt vốn
    while total_tv + total_fee > capital:
        cands = [
            (alloc[t] * prices[t], t)
            for t in alloc
            if alloc.get(t, 0) >= LOT_SIZE and t in prices
        ]
        if not cands:
            break
        _, tk  = max(cands)
        alloc[tk] -= LOT_SIZE
        total_tv   = sum(alloc[t] * prices[t]
                         for t in alloc if t in prices)
        total_fee  = total_tv * TRANSACTION_FEE

    return alloc


def compute_max_drawdown(values):
    arr = np.array(values, dtype=float)
    rm  = np.maximum.accumulate(arr)
    dd  = (arr - rm) / np.where(rm > 0, rm, 1.0)
    return float(dd.min())


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
        if cost > self.cash + 1:       # +1 tránh float error
            return False
        self.cash -= cost
        self.holdings[ticker] += qty
        self.transactions.append({
            "date"       : date,
            "type"       : "BUY",
            "ticker"     : ticker,
            "qty"        : qty,
            "price"      : price,
            "trade_value": qty * price,
            "fee"        : qty * price * TRANSACTION_FEE,
            "cash_after" : self.cash,
        })
        return True

    def sell(self, date, ticker, qty, price):
        if qty <= 0 or self.holdings[ticker] < qty:
            return False
        tv         = qty * price
        self.cash += tv * (1 - TRANSACTION_FEE)
        self.holdings[ticker] -= qty
        self.transactions.append({
            "date"       : date,
            "type"       : "SELL",
            "ticker"     : ticker,
            "qty"        : qty,
            "price"      : price,
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

    def get_current_weights(self, prices):
        total = self.total_assets(prices)
        if total <= 0:
            return {}
        return {
            t: qty * prices[t] / total
            for t, qty in self.holdings.items()
            if qty > 0 and t in prices
        }

    def get_transactions_df(self):
        if not self.transactions:
            return pd.DataFrame()
        return pd.DataFrame(self.transactions)


# =========================================================
# MODEL-DRIVEN WEEKLY STRATEGY
# =========================================================
def run_model_driven_weekly(data_cache, er_calc):

    optimizer = PortfolioOptimizer()
    port      = Portfolio(INITIAL_CAPITAL, "Model-Driven Weekly")
    risk_mgr  = RiskManager()
    price_maps = data_cache.build_price_maps()

    # ← sửa: dùng get_all_monday_dates thay vì get_all_weekly_dates
    rebalance_dates = data_cache.get_all_monday_dates(
        BACKTEST_START, BACKTEST_END
    )

    print("\n" + "=" * 80)
    print("🤖 MODEL-DRIVEN WEEKLY (a_ML2_pure | PURE RF daily)")
    print(f"   Horizon      : {FORECAST_HORIZON} phiên (~1 tuần)")
    print(f"   Weekly RF    : {WEEKLY_RF*100:.4f}%")
    print(f"   Core ER >=   : {MIN_CORE_ER*100:.2f}% / 5 phiên")
    print(f"   Sat  ER >=   : {MAX_SATELLITE_LOSS*100:.2f}% / 5 phiên")
    print(f"   Rebalance    : {len(rebalance_dates)} tuần")
    print("=" * 80)

    event_log = []

    for i, rb in enumerate(rebalance_dates):

        # Giá tại ngày rebalance
        prices = {
            t: price_maps[t][rb]
            for t in TICKERS
            if rb in price_maps.get(t, {})
        }
        if len(prices) < 2:
            if DEBUG:
                print(f"{rb} | ⚠️ Thiếu giá → bỏ qua")
            continue

        current_val   = port.total_assets(prices)
        risk_mode, dd = risk_mgr.update(current_val)
        multiplier    = risk_mgr.get_position_multiplier()
        cw            = port.get_current_weights(prices)

        # Bán hết kỳ trước
        if i > 0:
            port.sell_all(rb, prices)

        if risk_mode == "STOP":
            print(f"{rb} | ⛔ STOP  | DD:{dd*100:5.1f}%"
                  f" | Giữ tiền mặt")
            continue

        # ── Expected Returns (recursive 5 phiên) ──────────
        all_ers = {}
        for t in TICKERS:
            if t not in prices:
                continue
            er = er_calc.get_er_weekly(t, rb)
            if er is not None:
                all_ers[t] = er

        if not all_ers:
            print(f"{rb} | ⚠️ Không có ER → bỏ qua")
            continue

        if DEBUG:
            er_str = "  ".join(
                f"{t}:{v*100:+.2f}%" for t, v in all_ers.items()
            )
            print(f"\n{rb} | {risk_mode:8s} | DD:{dd*100:5.1f}%")
            print(f"   ER: {er_str}")

        # ── Core-Satellite ─────────────────────────────────
        ers = build_portfolio_strategy(all_ers)
        if not ers:
            continue

        # ── Covariance (daily returns, align by position) ──
        ret_df = data_cache.get_historical_returns(
            list(ers.keys()), rb, n_days=252
        )
        n_obs = len(ret_df)
        p     = len(ers)

        if n_obs < max(30, p + 5):
            if DEBUG:
                print(f"   Cov: {n_obs}d/{p} mã → Fallback EW")
            weights = {t: 1 / p for t in ers}
        else:
            lw = LedoitWolf()
            lw.fit(ret_df)
            cov = pd.DataFrame(
                lw.covariance_,          # daily cov
                index  = ret_df.columns,
                columns= ret_df.columns,
            )
            if DEBUG:
                print(f"   Cov: {n_obs}d/{p} mã "
                      f"(n/p={n_obs/p:.1f}) "
                      f"shrink={lw.shrinkage_:.3f}")
            weights = optimizer.optimize(
                ers, cov,
                current_weights=cw if i > 0 else None,
            )

        # ── Buy ────────────────────────────────────────────
        capital_to_use = port.cash * multiplier
        alloc = allocate_by_weights(
            capital_to_use,
            weights,
            {t: prices[t] for t in weights if t in prices},
        )
        bought = 0
        for t, qty in alloc.items():
            if qty > 0 and t in prices:
                if port.buy(rb, t, qty, prices[t]):
                    bought += 1

        total_val = port.total_assets(prices)
        w_str = " ".join(
            f"{t}:{weights.get(t, 0)*100:.0f}%"
            for t in sorted(weights)
        )
        print(f"   → {w_str} | Mua {bought} mã | "
              f"Total {total_val:>12,.0f} | "
              f"Cash {port.cash:>12,.0f}")

        event_log.append({
            "date"        : rb,
            "weights"     : weights,
            "ers"         : all_ers,
            "total_assets": total_val,
            "risk_mode"   : risk_mode,
            "drawdown"    : dd,
        })

    # ── Final sell: ngày GD cuối backtest ─────────────────
    # ← sửa: không hardcode năm/tháng
    last_date = data_cache.get_last_trading_date(BACKTEST_END)
    if last_date:
        sp = data_cache.get_prices_on_date(TICKERS, last_date)
        port.sell_all(last_date, sp)
        print(f"\n🏁 Final sell: {last_date} | "
              f"Cash: {port.cash:,.0f}")

    return port, event_log


# =========================================================
# EQUAL WEIGHT WEEKLY (baseline)
# =========================================================
def run_equal_weight_weekly(data_cache):

    port       = Portfolio(INITIAL_CAPITAL, "Equal Weight 1/N")
    price_maps = data_cache.build_price_maps()

    # ← sửa: dùng get_all_monday_dates
    rebalance_dates = data_cache.get_all_monday_dates(
        BACKTEST_START, BACKTEST_END
    )

    print("\n" + "=" * 80)
    print("⚖️  EQUAL WEIGHT (1/N) WEEKLY REBALANCE")
    print(f"   Rebalance: {len(rebalance_dates)} tuần")
    print("=" * 80)

    event_log = []

    for i, rb in enumerate(rebalance_dates):

        prices = {
            t: price_maps[t][rb]
            for t in TICKERS
            if rb in price_maps.get(t, {})
        }
        if len(prices) < 2:
            print(f"{rb}: ⚠️ Thiếu giá → bỏ qua")
            continue

        if i > 0:
            port.sell_all(rb, prices)

        weights = generate_equal_weights(list(prices.keys()))
        alloc   = allocate_by_weights(port.cash, weights, prices)

        for t, qty in alloc.items():
            if qty > 0 and t in prices:
                port.buy(rb, t, qty, prices[t])

        total_val = port.total_assets(prices)
        w_str     = " ".join(
            f"{t}:{weights[t]*100:.0f}%"
            for t in sorted(prices.keys())
        )
        print(f"{rb} | {w_str} | {total_val:>12,.0f}")

        event_log.append({
            "date"        : rb,
            "weights"     : weights,
            "total_assets": total_val,
        })

    # ← sửa: final sell động
    last_date = data_cache.get_last_trading_date(BACKTEST_END)
    if last_date:
        sp = data_cache.get_prices_on_date(TICKERS, last_date)
        port.sell_all(last_date, sp)

    return port, event_log


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
        return pd.DataFrame(
            columns=["date", "total_assets"]
        )

    tx_df["date"] = pd.to_datetime(tx_df["date"])
    tx_df = tx_df.sort_values("date").reset_index(drop=True)

    rows     = []
    holdings = defaultdict(int)
    cash     = portfolio.initial_capital
    tx_idx   = 0
    n_tx     = len(tx_df)

    for ds in dates:
        if not (start_date <= ds <= end_date):
            continue
        dts = pd.to_datetime(ds).date()

        # Apply tất cả giao dịch đến ngày ds
        while tx_idx < n_tx:
            tx_date = tx_df.iloc[tx_idx]["date"].date()
            if tx_date > dts:
                break
            tx  = tx_df.iloc[tx_idx]
            qty = int(tx["qty"])
            if tx["type"] == "BUY":
                holdings[tx["ticker"]] += qty
            else:
                holdings[tx["ticker"]] = max(
                    0, holdings[tx["ticker"]] - qty
                )
            cash   = float(tx["cash_after"])
            tx_idx += 1

        stock = sum(
            holdings[t] * price_maps[t][ds]
            for t in TICKERS
            if holdings.get(t, 0) > 0
            and ds in price_maps.get(t, {})
        )
        rows.append({
            "date"        : ds,
            "total_assets": cash + stock,
        })

    return pd.DataFrame(rows)


# =========================================================
# SUMMARY
# =========================================================
def summarize(name, portfolio, curve_df):

    tx_df    = portfolio.get_transactions_df()
    fees     = float(tx_df["fee"].sum())         if not tx_df.empty else 0.0
    turnover = float(tx_df["trade_value"].sum()) if not tx_df.empty else 0.0
    n_trades = len(tx_df)

    if curve_df.empty or len(curve_df) < 2:
        return {
            "Strategy"         : name,
            "Final Value (VND)": "N/A",
            "Return (%)"       : 0.0,
            "Max Drawdown (%)" : 0.0,
            "Sharpe (annul)"   : 0.0,
            "Total Fees"       : f"{fees:,.0f}",
            "Total Turnover"   : f"{turnover:,.0f}",
            "Num Trades"       : n_trades,
        }

    fv     = float(curve_df["total_assets"].iloc[-1])
    ret    = fv / portfolio.initial_capital - 1
    max_dd = compute_max_drawdown(
        curve_df["total_assets"].tolist()
    )

    # ← sửa: annualize Sharpe theo daily (×√252)
    sharpe = 0.0
    dr     = curve_df["total_assets"].pct_change().dropna()
    excess = dr - DAILY_RF
    if len(excess) > 2 and excess.std() > 0:
        sharpe = float(
            excess.mean() / excess.std() * np.sqrt(252)
        )

    return {
        "Strategy"         : name,
        "Final Value (VND)": f"{fv:,.0f}",
        "Return (%)"       : round(ret * 100, 2),
        "Max Drawdown (%)" : round(max_dd * 100, 2),
        "Sharpe (annul)"   : round(sharpe, 4),
        "Total Fees"       : f"{fees:,.0f}",
        "Total Turnover"   : f"{turnover:,.0f}",
        "Num Trades"       : n_trades,
    }


# =========================================================
# PLOT
# =========================================================
def plot_comparison(curve_md, curve_ew, s1, s2):

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Weekly Rebalance Backtest  {BACKTEST_START} → {BACKTEST_END}\n"
        f"Model-Driven (PURE RF daily×5) vs Equal Weight (1/N)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Equity curve ─────────────────────────────
    ax1 = axes[0]
    for curve, s, color, ls, lbl in [
        (curve_md, s1, "royalblue", "-",  "Model-Driven"),
        (curve_ew, s2, "tomato",    "--", "Equal Weight"),
    ]:
        if curve.empty:
            continue
        label = (f"{lbl} "
                 f"({s['Return (%)']:+.1f}% | "
                 f"Sharpe {s['Sharpe (annul)']:.2f})")
        ax1.plot(pd.to_datetime(curve["date"]),
                 curve["total_assets"],
                 label=label, color=color, lw=2, ls=ls)

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
        dd  = (arr - rm) / np.where(rm > 0, rm, 1.0) * 100
        ax2.plot(pd.to_datetime(curve["date"]), dd,
                 label=label, color=color, lw=1.5)
        ax2.fill_between(
            pd.to_datetime(curve["date"]),
            dd, 0, alpha=0.15, color=color,
        )

    ax2.axhline(MAX_DRAWDOWN_STOP * 100, color="red",
                ls="--", lw=1,
                label=f"Stop {MAX_DRAWDOWN_STOP*100:.0f}%")
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
    out_path = os.path.join(RESULT_DIR, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Chart saved: {out_path}")


# =========================================================
# MAIN
# =========================================================
def main():

    print("\n" + "=" * 80)
    print("📈 WEEKLY REBALANCE BACKTEST v2  (fixed for a_ML2_pure)")
    print(f"   Period  : {BACKTEST_START} → {BACKTEST_END}")
    print(f"   Tickers : {TICKERS}")
    print(f"   Capital : {INITIAL_CAPITAL:,.0f} VND")
    print(f"   Horizon : {FORECAST_HORIZON} phiên/tuần (daily recursive)")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────
    dc = DataCache(TICKERS, BACKTEST_START, BACKTEST_END)
    dc.load_all_data()

    if not dc.cache:
        print("❌ Không load được dữ liệu nào. Dừng.")
        return

    # ── Pre-load models ───────────────────────────────────
    print("\n🔧 Pre-loading models...")
    mm = ModelManager()
    for t in TICKERS:
        mm.get_model(t)

    er_calc = ERCalculator(dc, mm, horizon=FORECAST_HORIZON)

    # ── Run strategies ────────────────────────────────────
    md_port, md_events = run_model_driven_weekly(dc, er_calc)
    ew_port, ew_events = run_equal_weight_weekly(dc)

    # ── Equity curves ─────────────────────────────────────
    curve_md = build_equity_curve(
        dc, md_port, BACKTEST_START, BACKTEST_END
    )
    curve_ew = build_equity_curve(
        dc, ew_port, BACKTEST_START, BACKTEST_END
    )

    # ── Summary ───────────────────────────────────────────
    s1 = summarize("Model-Driven Weekly", md_port, curve_md)
    s2 = summarize("Equal Weight 1/N",    ew_port, curve_ew)
    summary_df = pd.DataFrame([s1, s2])

    print("\n" + "=" * 100)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # ── Save results ──────────────────────────────────────
    def _save(df, fname):
        if df is None or (hasattr(df, "empty") and df.empty):
            return
        path = os.path.join(RESULT_DIR, fname)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"   💾 {fname}")

    print(f"\n📁 Saving to {RESULT_DIR}/")
    _save(summary_df,                    "summary.csv")
    _save(curve_md,                      "equity_curve_model_driven.csv")
    _save(curve_ew,                      "equity_curve_equal_weight.csv")
    _save(md_port.get_transactions_df(), "tx_model_driven.csv")
    _save(ew_port.get_transactions_df(), "tx_equal_weight.csv")

    # Event log
    if md_events:
        ev_df = pd.DataFrame([{
            "date"        : e["date"],
            "risk_mode"   : e["risk_mode"],
            "drawdown_pct": round(e["drawdown"] * 100, 2),
            "total_assets": e["total_assets"],
        } for e in md_events])
        _save(ev_df, "event_log_model_driven.csv")

    # ── Plot ──────────────────────────────────────────────
    plot_comparison(curve_md, curve_ew, s1, s2)

    print(f"\n✅ DONE → {RESULT_DIR}/")


if __name__ == "__main__":
    main()