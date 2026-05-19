# ============================================================
# a_ML5_weekly.py — PURE ML RECURSIVE (RF + ZLEMA5 + Core+Momentum DELTA)
#
# Phiên bản WEEKLY cho portfolio optimization:
#   - interval = "w" (dữ liệu tuần)
#   - start_date = 2010-01-01 (~780 tuần để train đủ, ~15 năm)
#   - Forecast 4 bước = 4 tuần tới
#   - PURE ML: 100% Random Forest, không blend
#   - Dùng thay a_ML3 trong portfolio system
#
# Config có thể điều chỉnh:
#   WEEKLY_START_DATE : mốc lấy dữ liệu (mặc định 2010-01-01)
#   FORECAST_STEPS    : số tuần dự báo (mặc định 4)
# ============================================================

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from typing import List

import talib as ta
from talipp.indicators import ZLEMA as TalippZLEMA
from vnstock import Quote

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

# ============================================================
# GLOBAL CONFIG
# ============================================================

SAVE_DIR = "saved_model_weekly"
os.makedirs(SAVE_DIR, exist_ok=True)

WEEKLY_START_DATE = "2010-01-01"   # ~780 tuần (~15 năm) — đủ data train
FORECAST_STEPS    = 4              # 4 tuần = ~1 tháng

CONFIG = {
    "start_date"         : WEEKLY_START_DATE,
    "end_date"           : "2025-12-31",
    "interval"           : "w",        # ← WEEKLY
    "lookback"           : 16,         # ~4 tháng weekly (daily dùng 22)
    "zlema_period"       : 5,
    "retrain_every"      : 7,          # retrain mỗi 7 ngày (1 tuần)
    # RF hyperparameters
    "rf_n_estimators"    : 200,
    "rf_max_depth"       : 8,
    "rf_min_samples_leaf": 4,
    "rf_min_samples_split": 8,
    "rf_max_features"    : "sqrt",
    "random_state"       : 42,
}

# Core + Momentum features — DELTA format (giống a_ML5)
CORE_MOMENTUM_FEATURES_DELTA = [
    "residual_delta_1",
    "residual_delta_2",
    "residual_delta_3",
    "close_delta",
    "zlema_delta",
    "slope_5_close_delta",
    "rsi_delta",
    "macd_hist_delta",
    "stoch_diff_delta",
    "adx_delta",
    "mfi_delta",
]

TARGET_ADJ = "reg_target_adj"


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_data(symbol: str,
              start_date: str = None,
              end_date: str = None,
              interval: str = None) -> pd.DataFrame:
    """
    Load weekly OHLCV từ VCI.
    start_date / end_date / interval có thể override CONFIG.
    """
    sd  = start_date or CONFIG["start_date"]
    ed  = end_date   or datetime.now().strftime("%Y-%m-%d")
    iv  = interval   or CONFIG["interval"]

    quote = Quote(symbol=symbol.upper(), source="VCI")
    df    = quote.history(start=sd, end=ed, interval=iv)
    if df is None or df.empty:
        raise ValueError(f"❌ Không có dữ liệu cho {symbol} (interval={iv})")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    required = ["time", "open", "high", "low", "close", "volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Thiếu cột: {missing}")

    df["time"] = pd.to_datetime(df["time"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset="time", keep="last").reset_index(drop=True)

    df[["open", "high", "low", "close"]] = (
        df[["open", "high", "low", "close"]].ffill().bfill()
    )
    df["volume"] = df["volume"].fillna(0).clip(lower=0)

    mask = df["high"] < df["low"]
    if mask.any():
        df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)
    df.loc[df["high"] < max_oc, "high"] = max_oc[df["high"] < max_oc]
    df.loc[df["low"]  > min_oc, "low"]  = min_oc[df["low"]  > min_oc]

    return df


# ============================================================
# FEATURE ENGINEERING — DELTA VERSION (weekly-safe)
# ============================================================

def linreg_slope(arr) -> float:
    try:
        y = np.asarray(arr, dtype=np.float64)
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        x = np.arange(len(y), dtype=np.float64)
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan


def compute_features_DELTA(df: pd.DataFrame,
                            zlema_period: int = None) -> pd.DataFrame:
    if zlema_period is None:
        zlema_period = CONFIG["zlema_period"]

    df     = df.copy().sort_values("time").reset_index(drop=True)
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0)

    # ── Base indicators ───────────────────────────────────
    df["log_return"] = np.log(close / close.shift(1))
    # Weekly HV: annualize dùng sqrt(52) thay vì sqrt(252)
    df["hv_14"]      = df["log_return"].rolling(14).std() * np.sqrt(52)
    df["rsi"]        = ta.RSI(close, timeperiod=14)
    _, _, df["macd_hist"] = ta.MACD(close, 12, 26, 9)
    df["stoch_k"], df["stoch_d"] = ta.STOCH(high, low, close, 14, 3, 3)
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]
    df["adx"]        = ta.ADX(high, low, close, timeperiod=14)
    df["mfi"]        = ta.MFI(high, low, close, volume, timeperiod=14)
    df["atr_14"]     = ta.ATR(high, low, close, timeperiod=14)

    # ── ZLEMA ────────────────────────────────────────────
    try:
        z = TalippZLEMA(period=zlema_period)
        z.update(close.tolist())
        z_vals = [np.nan if v is None else float(v) for v in z]
        df["zlema_val"] = np.nan
        df.iloc[-len(z_vals):, df.columns.get_loc("zlema_val")] = z_vals
    except Exception:
        df["zlema_val"] = ta.EMA(close, timeperiod=zlema_period)

    # ── Target ───────────────────────────────────────────
    df["close_next"]  = close.shift(-1)
    df["reg_target"]  = df["close_next"] - df["zlema_val"]
    df["target_expanding_mean"] = df["reg_target"].expanding(min_periods=20).mean()
    df["reg_target_adj"] = df["reg_target"] - df["target_expanding_mean"].shift(1)

    # ── DELTA features ───────────────────────────────────
    residual = close - df["zlema_val"]
    df["residual_delta_1"] = residual.diff(1)
    df["residual_delta_2"] = residual.diff(2)
    df["residual_delta_3"] = residual.diff(3)

    df["close_delta"] = close.diff(1)
    df["zlema_delta"] = df["zlema_val"].diff(1)

    df["rsi_delta"]        = df["rsi"].diff(1)
    df["macd_hist_delta"]  = df["macd_hist"].diff(1)
    df["stoch_diff_delta"] = df["stoch_diff"].diff(1)
    df["adx_delta"]        = df["adx"].diff(1)
    df["mfi_delta"]        = df["mfi"].diff(1)

    df["slope_5_close_delta"] = df["close_delta"].rolling(5).apply(
        linreg_slope, raw=True
    )

    # ── Clean ─────────────────────────────────────────────
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df.dropna().reset_index(drop=True)


# ============================================================
# TICK SIZE UTILITIES
# ============================================================

def detect_tick_size(df: pd.DataFrame, price_col: str = "close",
                     tail_n: int = 200) -> float:
    s     = df[price_col].dropna().tail(tail_n).astype(float)
    diffs = s.diff().abs().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.01

    diffs_r   = diffs.round(4)
    counts    = diffs_r.value_counts().sort_index()
    min_count = max(2, int(len(diffs_r) * 0.01))
    candidates = counts[counts >= min_count]

    tick = float(candidates.index.min()) if len(candidates) > 0 \
           else float(max(diffs_r.quantile(0.05), 0.01))

    common = np.array([0.001, 0.005, 0.01, 0.02, 0.05,
                       0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    return float(common[np.argmin(np.abs(common - tick))])


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    return round(round(price / tick) * tick, 6)


# ============================================================
# SEQUENCE BUILDER (giống a_ML5)
# ============================================================

def create_single_sequence(X_window: np.ndarray) -> np.ndarray:
    n_features = X_window.shape[1]
    row = []
    for f in range(n_features):
        col       = X_window[:, f]
        col_clean = col[np.isfinite(col)]
        if len(col_clean) == 0:
            row.extend([0.0] * 6)
            continue
        last_val = float(col[-1]) if np.isfinite(col[-1]) else 0.0
        mean_val = float(np.nanmean(col_clean))
        std_val  = float(np.nanstd(col_clean))
        if len(col_clean) >= 2 and np.all(np.isfinite(col)):
            slope_val = float(np.polyfit(range(len(col)), col, 1)[0])
        else:
            slope_val = 0.0
        col_range = np.nanmax(col_clean) - np.nanmin(col_clean)
        position  = float(np.clip(
            (last_val - np.nanmin(col_clean)) / col_range, 0.0, 1.0
        )) if col_range > 1e-10 else 0.5
        delta = float(col[-1] - col[-6]) if len(col) >= 6 \
                else (float(col[-1] - col[0]) if len(col) > 0 else 0.0)
        row.extend([last_val, mean_val, std_val, slope_val, position, delta])
    return np.array(row, dtype=np.float32)


def create_sequences_summary_xy(X_data: np.ndarray,
                                 y_data: np.ndarray,
                                 lookback: int):
    X_out, y_out = [], []
    for i in range(lookback, len(X_data)):
        window = X_data[i - lookback:i, :]
        X_out.append(create_single_sequence(window))
        y_out.append(y_data[i])
    if len(X_out) == 0:
        return (np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=np.float32))
    return (np.array(X_out, dtype=np.float32),
            np.array(y_out, dtype=np.float32))


# ============================================================
# FIT / PREDICT
# ============================================================

def fit_final_model(df: pd.DataFrame,
                    feature_list: List[str],
                    target: str,
                    lookback: int = None) -> tuple:
    if lookback is None:
        lookback = CONFIG["lookback"]

    feats_ok = [f for f in feature_list if f in df.columns]
    X_raw    = df[feats_ok].values.astype(np.float64)
    y_raw    = df[target].values.astype(np.float64)
    X_raw    = np.where(np.isfinite(X_raw), X_raw, 0.0)
    y_raw    = np.where(np.isfinite(y_raw), y_raw, 0.0)

    f_sc = StandardScaler()
    t_sc = StandardScaler()
    X_s  = f_sc.fit_transform(X_raw)
    y_s  = t_sc.fit_transform(y_raw.reshape(-1, 1)).ravel()

    X_seq, y_seq = create_sequences_summary_xy(X_s, y_s, lookback)

    model = RandomForestRegressor(
        n_estimators   = CONFIG["rf_n_estimators"],
        max_depth      = CONFIG["rf_max_depth"],
        min_samples_leaf  = CONFIG["rf_min_samples_leaf"],
        min_samples_split = CONFIG["rf_min_samples_split"],
        max_features   = CONFIG["rf_max_features"],
        random_state   = CONFIG["random_state"],
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_seq, y_seq)
    return model, f_sc, t_sc


def predict_next_price(past_window: pd.DataFrame,
                       feature_list: List[str],
                       model,
                       f_scaler,
                       t_scaler,
                       lookback: int = None) -> float:
    """
    PURE ML: close_next = zlema_now + predicted_residual
    """
    if lookback is None:
        lookback = CONFIG["lookback"]

    feats_ok = [f for f in feature_list if f in past_window.columns]
    window   = past_window[feats_ok].values[-lookback:].astype(np.float64)
    window   = np.where(np.isfinite(window), window, 0.0)

    window_s  = f_scaler.transform(window)
    feat_vec  = create_single_sequence(window_s).reshape(1, -1)

    pred_s        = model.predict(feat_vec)
    pred_residual = float(
        t_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()[0]
    )
    zlema_now = float(past_window["zlema_val"].iloc[-1])
    return zlema_now + pred_residual


# ============================================================
# SAVE / LOAD MODEL
# ============================================================

def get_model_paths(symbol: str) -> dict:
    s = symbol.upper()
    return {
        "model"   : os.path.join(SAVE_DIR, f"rf_weekly_{s}.joblib"),
        "f_scaler": os.path.join(SAVE_DIR, f"f_scaler_weekly_{s}.joblib"),
        "t_scaler": os.path.join(SAVE_DIR, f"t_scaler_weekly_{s}.joblib"),
        "meta"    : os.path.join(SAVE_DIR, f"meta_weekly_{s}.json"),
    }


def model_exists(symbol: str) -> bool:
    return all(os.path.exists(p) for p in get_model_paths(symbol).values())


def needs_retrain(symbol: str) -> bool:
    paths = get_model_paths(symbol)
    if not os.path.exists(paths["meta"]):
        return True
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    saved_at = datetime.fromisoformat(meta.get("saved_at", "2000-01-01"))
    return (datetime.now() - saved_at).days >= CONFIG["retrain_every"]


def save_model_package(symbol: str, model, f_scaler, t_scaler,
                       feature_list: List[str]) -> None:
    s     = symbol.upper()
    paths = get_model_paths(s)
    joblib.dump(model,    paths["model"])
    joblib.dump(f_scaler, paths["f_scaler"])
    joblib.dump(t_scaler, paths["t_scaler"])
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump({
            "symbol"       : s,
            "model"        : "RandomForest_PURE_WEEKLY",
            "interval"     : CONFIG["interval"],
            "zlema_period" : CONFIG["zlema_period"],
            "feature_list" : feature_list,
            "target"       : TARGET_ADJ,
            "lookback"     : CONFIG["lookback"],
            "rf_params"    : {
                "n_estimators": CONFIG["rf_n_estimators"],
                "max_depth"   : CONFIG["rf_max_depth"],
                "min_samples_leaf": CONFIG["rf_min_samples_leaf"],
            },
            "saved_at"     : datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved WEEKLY RF model for {s}")


def load_model_package(symbol: str) -> tuple:
    s     = symbol.upper()
    paths = get_model_paths(s)
    model    = joblib.load(paths["model"])
    f_scaler = joblib.load(paths["f_scaler"])
    t_scaler = joblib.load(paths["t_scaler"])
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, f_scaler, t_scaler, meta


# ============================================================
# TRAIN PIPELINE
# ============================================================

def train_model_for_symbol(symbol: str) -> tuple:
    s = symbol.upper()
    print(f"\n{'='*60}")
    print(f"TRAIN WEEKLY (PURE RF)  {s}  |  RF + ZLEMA({CONFIG['zlema_period']})")
    print(f"  interval={CONFIG['interval']}  start={CONFIG['start_date']}")
    print(f"{'='*60}")

    end_fetch = datetime.now().strftime("%Y-%m-%d")
    df = load_data(s, CONFIG["start_date"], end_fetch, CONFIG["interval"])
    df = validate_data(df)

    n_rows = len(df)
    print(f"  Tổng số tuần: {n_rows}")
    if n_rows < 100:
        print(f"  ⚠️ Chỉ có {n_rows} tuần — có thể không đủ để train tốt")

    df = compute_features_DELTA(df, zlema_period=CONFIG["zlema_period"])

    model, f_sc, t_sc = fit_final_model(
        df=df,
        feature_list=CORE_MOMENTUM_FEATURES_DELTA,
        target=TARGET_ADJ,
        lookback=CONFIG["lookback"],
    )
    save_model_package(s, model, f_sc, t_sc, CORE_MOMENTUM_FEATURES_DELTA)
    print(f"✅ Training done — {len(df)} rows (weekly), "
          f"{len(CORE_MOMENTUM_FEATURES_DELTA)} features (DELTA)")
    return model, f_sc, t_sc


def ensure_model(symbol: str) -> tuple:
    s = symbol.upper()
    if not model_exists(s):
        print(f"⚙️  Chưa có model → Train {s} (weekly)...")
        train_model_for_symbol(s)
    return load_model_package(s)


# ============================================================
# FORECAST — PURE ML RECURSIVE (WEEKLY)
# ============================================================

def _next_week_start(date: datetime) -> datetime:
    """Ngày đầu tuần kế tiếp (thứ Hai)."""
    days_ahead = 7 - date.weekday()   # weekday: 0=Mon
    return date + timedelta(days=days_ahead)


def forecast_future(symbol: str,
                    forecast_steps: int = None) -> pd.DataFrame:
    """
    PURE ML RECURSIVE FORECAST — weekly.

    Trả về DataFrame gồm:
        Phiên, Ngày (tuần bắt đầu), Giá dự báo, Thay đổi_%, Bước giá, Loại

    forecast_steps: số tuần cần dự báo (mặc định = FORECAST_STEPS = 4)
    """
    if forecast_steps is None:
        forecast_steps = FORECAST_STEPS

    s = symbol.upper()
    model, f_sc, t_sc, meta = ensure_model(s)

    end_fetch  = datetime.now().strftime("%Y-%m-%d")
    df_raw     = load_data(s, CONFIG["start_date"], end_fetch,
                           CONFIG["interval"])
    df_raw     = validate_data(df_raw)
    df         = compute_features_DELTA(df_raw,
                                        zlema_period=CONFIG["zlema_period"])
    tick       = detect_tick_size(df)

    working_raw = df_raw.copy()
    results     = []

    print(f"\n{'='*60}")
    print(f"PURE RF WEEKLY FORECAST {forecast_steps} TUẦN  |  {s}")
    print(f"RF + ZLEMA({CONFIG['zlema_period']}) | NO HYBRID")
    print(f"{'='*60}")

    last_real_date = pd.to_datetime(df["time"].iloc[-1])
    next_date      = _next_week_start(last_real_date)
    prev_close     = float(df["close"].iloc[-1])

    for step in range(1, forecast_steps + 1):

        # ── Recompute features ──────────────────────────
        working_feat = compute_features_DELTA(
            working_raw.tail(400).copy(),
            zlema_period=CONFIG["zlema_period"],
        )
        past_window = working_feat.tail(CONFIG["lookback"]).copy()

        # ── PURE ML prediction ──────────────────────────
        pred_price_raw = predict_next_price(
            past_window  = past_window,
            feature_list = CORE_MOMENTUM_FEATURES_DELTA,
            model        = model,
            f_scaler     = f_sc,
            t_scaler     = t_sc,
            lookback     = CONFIG["lookback"],
        )

        pred_price = round_to_tick(pred_price_raw, tick)
        pct_change = (pred_price / prev_close - 1) * 100

        results.append({
            "Phiên"      : step,
            "Ngày"       : next_date.strftime("%d/%m/%Y"),
            "Giá dự báo" : pred_price,
            "Thay đổi_%" : pct_change,
            "Bước giá"   : tick,
            "Loại"       : "PURE RF (Weekly)",
        })

        print(f"  Tuần {step} [PURE RF] | "
              f"{next_date.strftime('%d/%m/%Y')} | "
              f"Giá: {pred_price:,.2f} | {pct_change:+.2f}%")

        # ── Append synthetic weekly row ─────────────────
        new_row = working_raw.iloc[-1].copy()
        new_row["time"]   = next_date
        new_row["open"]   = prev_close
        new_row["close"]  = pred_price
        new_row["high"]   = max(prev_close, pred_price)
        new_row["low"]    = min(prev_close, pred_price)
        new_row["volume"] = working_raw["volume"].tail(20).mean()

        working_raw = pd.concat(
            [working_raw, pd.DataFrame([new_row])],
            ignore_index=True
        )
        prev_close = pred_price
        next_date  = _next_week_start(next_date)

    result_df = pd.DataFrame(results)
    print("\n" + result_df.to_string(index=False, formatters={
        "Giá dự báo" : "{:,.2f}".format,
        "Thay đổi_%" : "{:+.2f}".format,
        "Bước giá"   : "{:.4f}".format,
    }))
    return result_df


# ============================================================
# PORTFOLIO HELPERS
# ============================================================

def get_expected_return_for_portfolio(symbol: str,
                                      cost_basis: float = None,
                                      forecast_steps: int = None,
                                      auto_confirm: bool = False
                                      ) -> tuple:
    """
    Hàm tiện ích cho portfolio optimizer — thay thế get_data_for_symbol của a_ML3.

    Returns:
        (expected_return, std_dev, hist_returns_weekly,
         current_price, predicted_price_4w, base_price)

    expected_return = (predicted_4w / base_price) - 1
    std_dev         = weekly std × sqrt(forecast_steps)  ← weekly-consistent
    hist_returns    : pd.Series weekly returns (252 tuần gần nhất)
    """
    import sys, os as _os

    if forecast_steps is None:
        forecast_steps = FORECAST_STEPS

    s = symbol.upper()

    try:
        # 1. Lấy dữ liệu weekly thô
        end_fetch = datetime.now().strftime("%Y-%m-%d")
        df_raw    = load_data(s, CONFIG["start_date"], end_fetch, CONFIG["interval"])
        df_raw    = validate_data(df_raw)
        last_close = float(df_raw["close"].iloc[-1])

        # 2. Ensure + run forecast
        ensure_model(s)

        # Suppress output khi chạy batch
        old_out = sys.stdout
        sys.stdout = open(_os.devnull, "w", encoding="utf-8")
        try:
            fc_df = forecast_future(s, forecast_steps)
        finally:
            sys.stdout.close()
            sys.stdout = old_out

        last_pred = float(fc_df["Giá dự báo"].iloc[-1])

        # 3. Base price
        if cost_basis is not None and cost_basis > 0:
            base_price = float(cost_basis)
            ratio = base_price / last_close
            if ratio > 100 or ratio < 0.01:
                print(f"\n⚠️ {s}: Giá vốn ({base_price:,.2f}) vs giá HT "
                      f"({last_close:,.2f}) chênh lệch lớn!")
                if auto_confirm:
                    base_price = last_close
                    print(f"   → [Auto] Dùng giá hiện tại: {last_close:,.2f}")
                else:
                    while True:
                        ch = input("   [1] Giá HT  [2] Nhập lại  [3] Giữ: ").strip()
                        if ch == "1":
                            base_price = last_close; break
                        elif ch == "2":
                            try:
                                v = float(input(f"   Nhập lại giá vốn {s}: "))
                                if v > 0:
                                    base_price = v; break
                            except ValueError:
                                pass
                        elif ch == "3":
                            break
        else:
            base_price = last_close

        # 4. Expected return
        expected_return = (last_pred / base_price) - 1

        # 5. Weekly volatility → scale to forecast horizon
        df_raw["weekly_return"] = df_raw["close"].pct_change()
        hist_returns = df_raw["weekly_return"].dropna().tail(252)  # ~5 năm
        # std_dev trong forecast_steps tuần
        std_dev = float(hist_returns.std() * np.sqrt(forecast_steps))

        return expected_return, std_dev, hist_returns, last_close, last_pred, base_price

    except Exception as e:
        print(f"❌ Lỗi {symbol}: {e}")
        return None, None, None, None, None, None


# ============================================================
# BACKTEST
# ============================================================

def backtest_model(symbol: str,
                   backtest_start: str = "2023-01-01") -> pd.DataFrame:
    s = symbol.upper()
    model, f_sc, t_sc, meta = load_model_package(s)

    end_fetch = datetime.now().strftime("%Y-%m-%d")
    df_raw    = load_data(s, CONFIG["start_date"], end_fetch, CONFIG["interval"])
    df_raw    = validate_data(df_raw)
    df        = compute_features_DELTA(df_raw,
                                       zlema_period=CONFIG["zlema_period"])
    tick      = detect_tick_size(df)
    bt_idx    = df[df["time"] >= pd.to_datetime(backtest_start)].index.tolist()

    results = []
    for idx in bt_idx:
        past_df = df.loc[:idx - 1].copy()
        if len(past_df) < CONFIG["lookback"] + 30:
            continue
        past_window = past_df.tail(CONFIG["lookback"]).copy()
        actual      = float(df.loc[idx, "close"])
        prev_close  = float(past_window["close"].iloc[-1])

        try:
            pred_raw = predict_next_price(
                past_window  = past_window,
                feature_list = CORE_MOMENTUM_FEATURES_DELTA,
                model        = model,
                f_scaler     = f_sc,
                t_scaler     = t_sc,
                lookback     = CONFIG["lookback"],
            )
            pred = round_to_tick(pred_raw, tick)
            results.append({
                "Ngày"      : df.loc[idx, "time"].strftime("%d/%m/%Y"),
                "Thực tế"   : actual,
                "Dự báo"    : pred,
                "Sai số"    : pred - actual,
                "% Sai số"  : abs(pred - actual) / actual * 100,
                "Đúng hướng": "✔" if ((pred > prev_close) == (actual > prev_close))
                              else "✖",
                "Bước giá"  : tick,
            })
        except Exception:
            continue

    bt_df = pd.DataFrame(results)
    if len(bt_df) == 0:
        print("⚠️ Không có kết quả backtest")
        return bt_df

    mape   = bt_df["% Sai số"].mean()
    mae    = bt_df["Sai số"].abs().mean()
    diracc = (bt_df["Đúng hướng"] == "✔").mean() * 100

    print(f"\n{'='*60}")
    print(f"BACKTEST WEEKLY (PURE RF)  {s}  từ {backtest_start}")
    print(f"RF + ZLEMA({CONFIG['zlema_period']}) | interval=w")
    print(f"{'='*60}")
    print(bt_df.head(30).to_string(index=False, formatters={
        "Thực tế"  : "{:,.2f}".format,
        "Dự báo"   : "{:,.2f}".format,
        "Sai số"   : "{:+.2f}".format,
        "% Sai số" : "{:.3f}".format,
    }))
    print(f"\n📊 SUMMARY  |  N={len(bt_df)}  "
          f"|  MAE={mae:.2f}  |  MAPE={mape:.3f}%  "
          f"|  DirAcc={diracc:.1f}%")
    return bt_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("PURE RF WEEKLY FORECAST")
    print(f"RF + ZLEMA({CONFIG['zlema_period']}) + Core+Momentum (DELTA)")
    print(f"interval=w  |  start={CONFIG['start_date']}")
    print("NO Hybrid | NO Trend | NO Noise")
    print("="*60)

    symbol = input("Nhập mã (VD: VNINDEX, HPG, VNM): ").strip().upper()
    print("="*60)

    ensure_model(symbol)

    while True:
        print(f"\n[{symbol}] Chọn chức năng:")
        print("1. Dự báo giá (PURE RF weekly, 4 tuần)")
        print("2. Backtest (PURE RF weekly)")
        print("3. Retrain model")
        print("0. Thoát")

        choice = input("→ ").strip()

        if choice == "1":
            try:
                steps = int(input(f"Số tuần (1-12, mặc định {FORECAST_STEPS}): ")
                            .strip() or str(FORECAST_STEPS))
                steps = max(1, min(12, steps))
            except Exception:
                steps = FORECAST_STEPS
            forecast_future(symbol, forecast_steps=steps)

        elif choice == "2":
            bt_start = input(
                "Từ ngày (YYYY-MM-DD, mặc định 2023-01-01): "
            ).strip() or "2023-01-01"
            backtest_model(symbol, backtest_start=bt_start)

        elif choice == "3":
            train_model_for_symbol(symbol)

        elif choice == "0":
            print("👋 Thoát.")
            break
        else:
            print("❌ Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()