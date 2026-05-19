# =========================================================
# MONTHLY REBALANCE - BACKTEST v3  (a_ML5_weekly)
# Model-Driven (PURE RF weekly) vs Equal Weight Baseline
# Backtest: 2024-01-01 → 2025-01-15
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
# IMPORT MODEL WEEKLY (a_ML5_weekly)
# ─────────────────────────────────────────────────────────
try:
    import a_ML4_weekly as ml
    print("✅ Loaded a_ML5_weekly")
    print(f"   interval      : {ml.CONFIG.get('interval', 'w')}")
    print(f"   start_date    : {ml.CONFIG.get('start_date', 'N/A')}")
    print(f"   zlema_period  : {ml.CONFIG.get('zlema_period', 'N/A')}")
    print(f"   lookback      : {ml.CONFIG.get('lookback', 'N/A')}")
    print(f"   forecast_steps: {ml.FORECAST_STEPS}")
except ImportError as e:
    print(f"❌ Không tìm thấy a_ML5_weekly: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────
# AUTO-MAP API (đề phòng tên hàm khác nhau)
# ─────────────────────────────────────────────────────────
def _get_attr(module, candidates, what="function"):
    for name in candidates:
        if hasattr(module, name):
            return getattr(module, name)
    available = [x for x in dir(module) if not x.startswith("_")]
    raise AttributeError(
        f"Không tìm thấy {what} trong module.\n"
        f"Các thuộc tính có sẵn: {available[:60]}"
    )

# Map các hàm/constant chính
_compute_features = _get_attr(ml, [
    "compute_features_DELTA",
    "compute_features_delta",
    "compute_features",
    "build_features",
], what="compute_features")

_predict_next_price = _get_attr(ml, [
    "predict_next_price",
    "predict_next",
    "forecast_next_price",
    "forecast",
], what="predict_next_price")

_ensure_model = _get_attr(ml, [
    "ensure_model",
    "load_model",
    "get_model",
    "train_model",
], what="ensure_model")

_load_data = _get_attr(ml, [
    "load_data",
    "load_stock_data",
    "fetch_data",
    "get_data",
], what="load_data")

_validate_data = _get_attr(ml, [
    "validate_data",
    "clean_data",
    "preprocess_data",
], what="validate_data")

_forecast_future = _get_attr(ml, [
    "forecast_future",
    "forecast",
    "predict_future",
], what="forecast_future")

_get_expected_return_for_portfolio = _get_attr(ml, [
    "get_expected_return_for_portfolio",
    "get_er_for_portfolio",
], what="get_expected_return_for_portfolio")

# Feature list & config
_FEATURE_LIST = _get_attr(ml, [
    "CORE_MOMENTUM_FEATURES_DELTA",
    "CORE_MOMENTUM_FEATURES",
    "FEATURE_LIST",
    "FEATURES",
], what="feature_list")

_CONFIG         = ml.CONFIG
_FORECAST_STEPS = getattr(ml, "FORECAST_STEPS", 4)

print(f"✅ API mapped: compute_features={_compute_features.__name__}, "
      f"predict_next={_predict_next_price.__name__}, "
      f"feature_list={_FEATURE_LIST.__class__.__name__ if hasattr(_FEATURE_LIST,'__class__') else type(_FEATURE_LIST)}")

# =========================================================
# CONFIG BACKTEST
# =========================================================
INITIAL_CAPITAL = 100_000_000
TRANSACTION_FEE = 0.0015
LOT_SIZE        = 100

TICKERS = ["TCB", "VRE", "VCB", "SSI", "FPT"]

BACKTEST_START = "2024-01-01"
BACKTEST_END   = "2025-01-15"

FORECAST_HORIZON = _FORECAST_STEPS   # 4 tuần

# MPT
ANNUAL_RF  = 0.05
WEEKLY_RF  = (1 + ANNUAL_RF) ** (1 / 52) - 1
MONTHLY_RF = (1 + ANNUAL_RF) ** (1 / 12) - 1

LAMBDA_REG   = 0.001
MAX_TURNOVER = 0.50
MAX_POSITION = 0.35
MIN_POSITION = 0.05

# Risk
MAX_DRAWDOWN_STOP  = -0.20
DRAWDOWN_WARNING   = -0.12
RECOVERY_THRESHOLD = -0.05

DEBUG      = True
RESULT_DIR = "backtest_results_v3"
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
                df = _load_data(ticker, train_start, self.end_date, "w")
                df = _validate_data(df)
                df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time").reset_index(drop=True)
                self.cache[ticker] = df
                print(f"✅ {len(df)} weeks")
            except Exception as e:
                print(f"❌ {e}")

    def get_featured_df(self, ticker):
        if ticker not in self.feat_cache:
            raw = self.cache.get(ticker)
            if raw is None:
                return None
            feat = _compute_features(raw.copy(),
                                     zlema_period=_CONFIG["zlema_period"])
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
        return {"date": row["time"].strftime("%Y-%m-%d"),
                "price": float(row["close"])}

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
            tmp = df.copy()
            tmp["ds"] = tmp["time"].dt.strftime("%Y-%m-%d")
            out[ticker] = dict(zip(tmp["ds"], tmp["close"].astype(float)))
        return out

    def get_historical_returns(self, tickers, before_date, n_weeks=104):
        """Weekly returns, align by position (tail)."""
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
            ret = sub["ret"].dropna()
            if len(ret) >= 20:
                raw_series[ticker] = ret.values
        if not raw_series:
            return pd.DataFrame()
        min_len = min(len(v) for v in raw_series.values())
        aligned = {k: v[-min_len:] for k, v in raw_series.items()}
        return pd.DataFrame(aligned)

    def get_history_before(self, ticker, before_date, n_rows):
        feat = self.get_featured_df(ticker)
        if feat is None:
            return pd.DataFrame()
        bd  = pd.to_datetime(before_date)
        sub = feat[feat["time"] < bd].tail(n_rows)
        return sub.copy()

    def get_union_trading_dates(self, start_date, end_date):
        s, e   = pd.to_datetime(start_date), pd.to_datetime(end_date)
        all_ds = set()
        for df in self.cache.values():
            sub = df[(df["time"] >= s) & (df["time"] <= e)]
            all_ds.update(sub["time"].dt.strftime("%Y-%m-%d"))
        return sorted(all_ds)

    def get_all_rebalance_dates(self, start_date, end_date, ticker_ref=None):
        if ticker_ref is None:
            ticker_ref = self.tickers[0]
        df = self.cache.get(ticker_ref)
        if df is None:
            return []
        s, e    = pd.to_datetime(start_date), pd.to_datetime(end_date)
        dates   = []
        current = pd.to_datetime(f"{s.year}-{s.month:02d}-01")
        while current <= e:
            y, m = current.year, current.month
            sub  = df[(df["time"].dt.year==y)&(df["time"].dt.month==m)]
            if not sub.empty:
                dates.append(sub.iloc[0]["time"].strftime("%Y-%m-%d"))
            if m == 12:
                current = pd.to_datetime(f"{y+1}-01-01")
            else:
                current = pd.to_datetime(f"{y}-{m+1:02d}-01")
        return dates

    def get_first_trading_day_of_month(self, year, month, ticker_ref=None):
        if ticker_ref is None:
            ticker_ref = self.tickers[0]
        df = self.cache.get(ticker_ref)
        if df is None:
            return None
        sub = df[(df["time"].dt.year==year)&(df["time"].dt.month==month)]
        if sub.empty:
            return None
        return sub.iloc[0]["time"].strftime("%Y-%m-%d")


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
# ER CALCULATOR
# =========================================================
class ERCalculator:

    def __init__(self, data_cache, model_manager, horizon=FORECAST_HORIZON):
        self.dc = data_cache
        self.mm = model_manager
        self.horizon = horizon

    def get_er(self, ticker, date_str):
        lookback = _CONFIG["lookback"]
        past = self.dc.get_history_before(ticker, date_str, lookback + 10)
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
# PORTFOLIO OPTIMIZER
# =========================================================
class PortfolioOptimizer:

    def __init__(self):
        self.rf = WEEKLY_RF
        self.lambda_reg = LAMBDA_REG

    def optimize(self, expected_returns, cov_matrix, current_weights=None):
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
            return -(ret - self.rf) / std + self.lambda_reg * np.sum(w**2)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x)-1}]
        if current_weights is not None:
            cur = np.array([current_weights.get(t,0) for t in tickers])
            constraints.append({
                "type": "ineq",
                "fun" : lambda x, c=cur: MAX_TURNOVER - np.sum(np.abs(x-c))
            })

        bounds = tuple((MIN_POSITION, MAX_POSITION) for _ in range(n))
        x0 = (np.array([current_weights.get(t,1/n) for t in tickers])
              if current_weights else np.array([1/n]*n))
        x0 /= x0.sum()

        res = minimize(objective, x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter":5000,"ftol":1e-12})

        w = res.x if res.success else x0
        w = w / w.sum()

        if DEBUG:
            print(f"      ER    : {{ {', '.join([f'{t}:{er_arr[i]:+.4f}' for i,t in enumerate(tickers)])} }}")
            if current_weights is not None:
                cur = np.array([current_weights.get(t,0) for t in tickers])
                print(f"      TO    : {np.sum(np.abs(w-cur))*100:.1f}%")
            print(f"      Weights: {{ {', '.join([f'{t}:{w[i]:.3f}' for i,t in enumerate(tickers)])} }}")
            if not res.success:
                print(f"      ⚠️  {res.message}")
        return dict(zip(tickers, w))


# =========================================================
# RISK MANAGER / UTILS / PORTFOLIO
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
    def get_multiplier(self):
        return {"STOP":0.0,"DEFENSIVE":0.6,"NORMAL":1.0}.get(self.risk_mode,1.0)

def round_lot(qty): return int(qty // LOT_SIZE) * LOT_SIZE

def allocate_by_weights(capital, weights, prices):
    alloc= {}; tv=0; fee=0
    for ticker,w in weights.items():
        if ticker not in prices: alloc[ticker]=0; continue
        tv_use = capital * w / (1+TRANSACTION_FEE)
        qty    = round_lot(tv_use / prices[ticker])
        alloc[ticker]=qty; tv+=qty*prices[ticker]; fee+=qty*prices[ticker]*TRANSACTION_FEE
    while tv+fee > capital:
        cands=[(alloc[t]*prices[t],t) for t in alloc if alloc[t]>=LOT_SIZE and t in prices]
        if not cands: break
        _,tk=max(cands); alloc[tk]-=LOT_SIZE
        tv=sum(alloc[t]*prices[t] for t in alloc if t in prices); fee=tv*TRANSACTION_FEE
    return alloc

def compute_max_drawdown(values):
    arr=np.array(values); rm=np.maximum.accumulate(arr)
    dd=(arr-rm)/np.where(rm>0,rm,1); return dd.min()

def generate_equal_weights(tickers):
    return {t:1.0/len(tickers) for t in tickers}

class Portfolio:
    def __init__(self, initial_capital, name="Portfolio"):
        self.name= name; self.initial_capital=float(initial_capital)
        self.cash=float(initial_capital); self.holdings=defaultdict(int); self.transactions=[]
    def buy(self,date,ticker,qty,price):
        if qty<=0: return False
        cost=qty*price*(1+TRANSACTION_FEE)
        if cost>self.cash: return False
        self.cash-=cost; self.holdings[ticker]+=qty
        self.transactions.append({"date":date,"type":"BUY","ticker":ticker,"qty":qty,"price":price,
                                 "trade_value":qty*price,"fee":qty*price*TRANSACTION_FEE,"cash_after":self.cash})
        return True
    def sell(self,date,ticker,qty,price):
        if qty<=0 or self.holdings[ticker]<qty: return False
        tv=qty*price; self.cash+=tv*(1-TRANSACTION_FEE); self.holdings[ticker]-=qty
        self.transactions.append({"date":date,"type":"SELL","ticker":ticker,"qty":qty,"price":price,
                                 "trade_value":tv,"fee":tv*TRANSACTION_FEE,"cash_after":self.cash})
        return True
    def sell_all(self,date,prices):
        for t,qty in list(self.holdings.items()):
            if qty>0 and t in prices: self.sell(date,t,qty,prices[t])
    def total_assets(self,prices):
        stock=sum(qty*prices[t] for t,qty in self.holdings.items() if qty>0 and t in prices)
        return self.cash+stock
    def get_current_weights(self,prices):
        total=self.total_assets(prices)
        if total<=0: return {}
        return {t: qty*prices[t]/total for t,qty in self.holdings.items() if qty>0 and t in prices}
    def get_transactions_df(self):
        if not self.transactions: return pd.DataFrame()
        return pd.DataFrame(self.transactions)


# =========================================================
# STRATEGIES
# =========================================================
def run_model_driven(data_cache, er_calc):
    optimizer=PortfolioOptimizer(); port=Portfolio(INITIAL_CAPITAL,"Model-Driven MPT")
    risk_mgr=RiskManager(); price_maps=data_cache.build_price_maps()
    rebalance_dates=data_cache.get_all_rebalance_dates(BACKTEST_START,BACKTEST_END)
    print("\n"+"="*80)
    print("🤖 MODEL-DRIVEN MPT (a_ML5_weekly | 4-week horizon)")
    print("="*80); print(f"   Rebalance: {len(rebalance_dates)}")
    event_log=[]
    for i,rb in enumerate(rebalance_dates):
        prices={t: price_maps[t][rb] for t in TICKERS if rb in price_maps.get(t,{})}
        if not prices: print(f"\n📅 {rb} ⚠️ thiếu giá"); continue
        current_val=port.total_assets(prices)
        risk_mode,dd=risk_mgr.update(current_val); mult=risk_mgr.get_multiplier()
        cw=port.get_current_weights(prices)
        print(f"\n📅 {rb} | Assets {current_val:>14,.0f} | Risk {risk_mode} | DD {dd*100:.2f}%")
        if i>0: port.sell_all(rb,prices)
        if risk_mode=="STOP": print("   → STOP"); continue
        ers={}
        for t in TICKERS:
            er=er_calc.get_er(t,rb)
            if DEBUG: print(f"   ER {t}: {er*100:+.2f}%" if er is not None else f"   ER {t}: N/A")
            if er is not None and er>0: ers[t]=er
        if not ers: print("   ⚠️ Không ER dương"); continue
        ret_df=data_cache.get_historical_returns(list(ers.keys()), rb, n_weeks=104)
        n_obs,p=len(ret_df),len(ers)
        print(f"   Cov: {n_obs}w/{p} mã (n/p={n_obs/p:.1f})")
        if n_obs < max(20,p+5):
            print("   → Fallback Equal Weight"); weights={t:1/p for t in ers}
        else:
            lw=LedoitWolf(); lw.fit(ret_df)
            cov=pd.DataFrame(lw.covariance_*FORECAST_HORIZON,
                             index=ret_df.columns, columns=ret_df.columns)
            print(f"   Shrinkage: {lw.shrinkage_:.4f}")
            weights=optimizer.optimize(ers,cov, current_weights=cw if i>0 else None)
        alloc=allocate_by_weights(port.cash*mult, weights, prices)
        bought=0
        for tk,qty in alloc.items():
            if qty>0 and port.buy(rb,tk,qty,prices[tk]): bought+=1
        total_val=port.total_assets(prices)
        print(f"   → Mua {bought} mã | Total {total_val:>14,.0f} | Cash {port.cash:>14,.0f}")
        event_log.append({"date":rb,"weights":weights,"ers":ers,"total_assets":total_val,"risk_mode":risk_mode})
    sell_date=data_cache.get_first_trading_day_of_month(2025,1)
    if sell_date:
        sp=data_cache.get_prices_on_date(TICKERS,sell_date); port.sell_all(sell_date,sp)
    return port,sell_date,event_log

def run_equal_weight(data_cache):
    port=Portfolio(INITIAL_CAPITAL,"Equal Weight 1/N"); price_maps=data_cache.build_price_maps()
    rebalance_dates=data_cache.get_all_rebalance_dates(BACKTEST_START,BACKTEST_END)
    print("\n"+"="*80); print("⚖️  EQUAL WEIGHT (1/N)"); print("="*80)
    event_log=[]
    for i,rb in enumerate(rebalance_dates):
        prices={t: price_maps[t][rb] for t in TICKERS if rb in price_maps.get(t,{})}
        if len(prices)<len(TICKERS):
            missing=set(TICKERS)-set(prices.keys()); print(f"{rb}: thiếu {missing}"); continue
        if i>0: port.sell_all(rb,prices)
        weights=generate_equal_weights(list(prices.keys()))
        alloc=allocate_by_weights(port.cash,weights,prices)
        for tk,qty in alloc.items():
            if qty>0: port.buy(rb,tk,qty,prices[tk])
        total_val=port.total_assets(prices)
        print(f"{rb}: {weights} → {total_val:,.0f}")
        event_log.append({"date":rb,"weights":weights,"total_assets":total_val})
    sell_date=data_cache.get_first_trading_day_of_month(2025,1)
    if sell_date:
        sp=data_cache.get_prices_on_date(TICKERS,sell_date); port.sell_all(sell_date,sp)
    return port,sell_date,event_log


# =========================================================
# EQUITY CURVE & SUMMARY
# =========================================================
def build_equity_curve(data_cache, portfolio, start_date, end_date):
    dates=data_cache.get_union_trading_dates(start_date,end_date)
    price_maps=data_cache.build_price_maps()
    tx_df=portfolio.get_transactions_df().copy()
    if tx_df.empty: return pd.DataFrame()
    tx_df["date"]=pd.to_datetime(tx_df["date"]); tx_df=tx_df.sort_values("date")
    rows=[]; holdings=defaultdict(int); cash=portfolio.initial_capital; tx_idx=0
    for ds in dates:
        if not (start_date<=ds<=end_date): continue
        dts=pd.to_datetime(ds).date()
        while tx_idx<len(tx_df) and tx_df.iloc[tx_idx]["date"].date()<=dts:
            tx=tx_df.iloc[tx_idx]; qty=int(tx["qty"])
            if tx["type"]=="BUY": holdings[tx["ticker"]]+=qty
            else: holdings[tx["ticker"]]-=qty
            cash=float(tx["cash_after"]); tx_idx+=1
        stock=sum(holdings[t]*price_maps[t][ds] for t in TICKERS if holdings[t]>0 and ds in price_maps.get(t,{}))
        rows.append({"date":ds,"total_assets":cash+stock})
    return pd.DataFrame(rows)

def summarize(name, portfolio, curve_df):
    if curve_df.empty: return {"Strategy":name,"Error":"No data"}
    fv=float(curve_df["total_assets"].iloc[-1])
    ret=fv/portfolio.initial_capital-1
    max_dd=compute_max_drawdown(curve_df["total_assets"].tolist())
    tx_df=portfolio.get_transactions_df()
    fees=tx_df["fee"].sum() if not tx_df.empty else 0
    turnover=tx_df["trade_value"].sum() if not tx_df.empty else 0
    sharpe=0.0
    if len(curve_df)>2:
        dr=curve_df["total_assets"].pct_change().dropna()
        excess=dr-WEEKLY_RF
        if excess.std()>0: sharpe=excess.mean()/excess.std()*np.sqrt(52)
    return {
        "Strategy":name,
        "Final Value (VND)": f"{fv:,.0f}",
        "Return (%)": round(ret*100,2),
        "Max Drawdown (%)": round(max_dd*100,2),
        "Sharpe (annul)": round(sharpe,4),
        "Total Fees": f"{fees:,.0f}",
        "Total Turnover": f"{turnover:,.0f}",
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n"+"="*80)
    print("📈 BACKTEST v3 — a_ML5_weekly (PURE RF weekly)")
    print(f"   Period: {BACKTEST_START} → {BACKTEST_END}")
    print(f"   Tickers: {TICKERS}")
    print(f"   Capital: {INITIAL_CAPITAL:,.0f} VND")
    print("="*80)

    dc=DataCache(TICKERS,BACKTEST_START,BACKTEST_END)
    dc.load_all_data()

    mm=ModelManager(); er_calc=ERCalculator(dc,mm,horizon=FORECAST_HORIZON)

    md_port,md_sell,md_events=run_model_driven(dc,er_calc)
    ew_port,ew_sell,ew_events=run_equal_weight(dc)

    curve_md=build_equity_curve(dc,md_port,BACKTEST_START,md_sell or BACKTEST_END)
    curve_ew=build_equity_curve(dc,ew_port,BACKTEST_START,ew_sell or BACKTEST_END)

    s1=summarize("Model-Driven MPT",md_port,curve_md)
    s2=summarize("Equal Weight 1/N",ew_port,curve_ew)
    summary_df=pd.DataFrame([s1,s2])

    print("\n"+"="*100); print("📊 PERFORMANCE SUMMARY"); print("="*100)
    print(summary_df.to_string(index=False)); print("="*100)

    # save
    summary_df.to_csv(os.path.join(RESULT_DIR,"summary.csv"),index=False,encoding="utf-8-sig")
    curve_md.to_csv(os.path.join(RESULT_DIR,"curve_model_driven.csv"),index=False,encoding="utf-8-sig")
    curve_ew.to_csv(os.path.join(RESULT_DIR,"curve_equal_weight.csv"),index=False,encoding="utf-8-sig")
    md_port.get_transactions_df().to_csv(os.path.join(RESULT_DIR,"tx_model_driven.csv"),index=False,encoding="utf-8-sig")
    ew_port.get_transactions_df().to_csv(os.path.join(RESULT_DIR,"tx_equal_weight.csv"),index=False,encoding="utf-8-sig")

    # plot
    if not curve_md.empty and not curve_ew.empty:
        fig,axes=plt.subplots(2,1,figsize=(14,10))
        ax1=axes[0]
        curve_md["date"]=pd.to_datetime(curve_md["date"]); curve_ew["date"]=pd.to_datetime(curve_ew["date"])
        ax1.plot(curve_md["date"],curve_md["total_assets"],label=f"Model-Driven ({s1['Return (%)']:.1f}%)",lw=2,color="royalblue")
        ax1.plot(curve_ew["date"],curve_ew["total_assets"],label=f"Equal Weight ({s2['Return (%)']:.1f}%)",lw=2,color="tomato",ls="--")
        ax1.axhline(INITIAL_CAPITAL,color="gray",ls=":",alpha=0.6,label="Initial Capital")
        ax1.set_ylabel("Portfolio Value (VND)"); ax1.set_title("Backtest 2024-01-01 → 2025-01-15\nModel-Driven (ML5 Weekly) vs Equal Weight",fontweight="bold")
        ax1.legend(); ax1.grid(True,alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(),rotation=45)
        ax2=axes[1]
        def dd(s): v=s["total_assets"].values; rm=np.maximum.accumulate(v); return (v-rm)/np.where(rm>0,rm,1)*100
        ax2.fill_between(curve_md["date"],dd(curve_md),0,alpha=0.4,color="royalblue",label="Model DD")
        ax2.fill_between(curve_ew["date"],dd(curve_ew),0,alpha=0.4,color="tomato",label="EW DD")
        ax2.set_ylabel("Drawdown (%)"); ax2.set_xlabel("Date"); ax2.legend(); ax2.grid(True,alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(),rotation=45)
        plt.tight_layout()
        out=os.path.join(RESULT_DIR,"equity_curve_comparison.png"); plt.savefig(out,dpi=150,bbox_inches="tight")
        print(f"\n✅ Chart: {out}"); plt.show()

    print(f"\n✅ Done → {RESULT_DIR}/")

if __name__ == "__main__":
    main()