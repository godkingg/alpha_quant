# ============================================================
# a_ML2.py — OPTIMAL ML CONFIG (RF + ZLEMA5 + Core+Momentum DELTA)
# Optimal config từ notebook:
#   - ZLEMA(5)
#   - Random Forest (tuned hyperparameters)
#   - Core + Momentum features (DELTA format)
#   - Retrain mỗi 5 phiên (trading week)
# ============================================================

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import talib as ta
from talipp.indicators import ZLEMA as TalippZLEMA
from vnstock import Quote

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

# ============================================================
# GLOBAL CONFIG — Optimal từ notebook
# ============================================================

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG = {
    "start_date"    : "2019-06-15",
    "end_date"      : "2025-12-31",
    "interval"      : "d",
    "lookback"      : 22,           # từ notebook
    "zlema_period"  : 5,            # ✅ ZLEMA(5) — optimal
    "retrain_every" : 5,            # retrain mỗi 5 phiên
    # RF hyperparameters (từ Cell 11B tuning)
    "rf_n_estimators"    : 200,
    "rf_max_depth"       : 8,
    "rf_min_samples_leaf": 5,
    "rf_min_samples_split": 10,
    "rf_max_features"    : "sqrt",
    "random_state"       : 42,
}

# ✅ Core + Momentum features — DELTA format (từ notebook Cell 14)
# Best ablation: MAPE=0.801%, TheilU=0.9089
CORE_MOMENTUM_FEATURES_DELTA = [
    # Core: residual deltas
    "residual_delta_1",
    "residual_delta_2",
    "residual_delta_3",
    "close_delta",
    "zlema_delta",
    # Momentum (DELTA)
    "slope_5_close_delta",
    "rsi_delta",
    "macd_hist_delta",
    "stoch_diff_delta",
    "adx_delta",
    "mfi_delta",
]

TARGET_ADJ = "reg_target_adj"   # ✅ Mean-adjusted target từ notebook


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_data(symbol: str, start_date: str, end_date: str,
              interval: str = "d") -> pd.DataFrame:
    quote = Quote(symbol=symbol.upper(), source="VCI")
    df    = quote.history(start=start_date, end=end_date, interval=interval)
    if df is None or df.empty:
        raise ValueError(f"❌ Không có dữ liệu cho {symbol}")
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
# FEATURE ENGINEERING — DELTA VERSION (từ notebook Cell 4)
# ============================================================

def linreg_slope(arr) -> float:
    """Linear regression slope cho rolling.apply(raw=True)"""
    try:
        y = np.asarray(arr, dtype=np.float64)
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        x = np.arange(len(y), dtype=np.float64)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return np.nan


def compute_features_DELTA(df: pd.DataFrame,
                           zlema_period: int = None) -> pd.DataFrame:
    """
    ✅ Feature engineering DELTA format từ notebook Cell 4.
    Target: reg_target_adj (mean-adjusted)
    """
    if zlema_period is None:
        zlema_period = CONFIG["zlema_period"]

    df    = df.copy().sort_values("time").reset_index(drop=True)
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0)

    # ── Base indicators ─────────────────────────────────────
    df["log_return"] = np.log(close / close.shift(1))
    df["hv_14"]      = df["log_return"].rolling(14).std() * np.sqrt(252)
    df["rsi"]        = ta.RSI(close, timeperiod=14)
    _, _, df["macd_hist"] = ta.MACD(close, 12, 26, 9)
    df["stoch_k"], df["stoch_d"] = ta.STOCH(high, low, close, 14, 3, 3)
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]
    df["adx"]        = ta.ADX(high, low, close, timeperiod=14)
    df["mfi"]        = ta.MFI(high, low, close, volume, timeperiod=14)
    df["atr_14"]     = ta.ATR(high, low, close, timeperiod=14)

    # ── ZLEMA ───────────────────────────────────────────────
    try:
        z = TalippZLEMA(period=zlema_period)
        z.update(close.tolist())
        z_vals = [np.nan if v is None else float(v) for v in z]
        df["zlema_val"] = np.nan
        df.iloc[-len(z_vals):, df.columns.get_loc("zlema_val")] = z_vals
    except Exception:
        df["zlema_val"] = ta.EMA(close, timeperiod=zlema_period)

    # ── Target (original + adjusted) ────────────────────────
    df["close_next"] = close.shift(-1)
    df["reg_target"] = df["close_next"] - df["zlema_val"]
    
    # Target expanding mean (no look-ahead)
    df["target_expanding_mean"] = df["reg_target"].expanding(min_periods=20).mean()
    df["reg_target_adj"] = df["reg_target"] - df["target_expanding_mean"].shift(1)

    # ── DELTA features ──────────────────────────────────────
    # Core: residual deltas
    residual = close - df["zlema_val"]
    df["residual_delta_1"] = residual.diff(1)
    df["residual_delta_2"] = residual.diff(2)
    df["residual_delta_3"] = residual.diff(3)
    
    df["close_delta"] = close.diff(1)
    df["zlema_delta"] = df["zlema_val"].diff(1)
    
    # Momentum indicators → DELTA
    df["rsi_delta"]        = df["rsi"].diff(1)
    df["macd_hist_delta"]  = df["macd_hist"].diff(1)
    df["stoch_diff_delta"] = df["stoch_diff"].diff(1)
    df["adx_delta"]        = df["adx"].diff(1)
    df["mfi_delta"]        = df["mfi"].diff(1)
    
    # Slope on deltas
    df["slope_5_close_delta"] = df["close_delta"].rolling(5).apply(linreg_slope, raw=True)

    # ── Clean ───────────────────────────────────────────────
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df.dropna().reset_index(drop=True)


# ============================================================
# TICK SIZE UTILITIES
# ============================================================

def detect_tick_size(df: pd.DataFrame, price_col: str = "close",
                     tail_n: int = 300) -> float:
    s    = df[price_col].dropna().tail(tail_n).astype(float)
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

    common = np.array([0.001,0.005,0.01,0.02,0.05,
                       0.1,0.2,0.5,1.0,2.0,5.0])
    return float(common[np.argmin(np.abs(common - tick))])


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    return round(round(price / tick) * tick, 6)


# ============================================================
# SEQUENCE BUILDER — từ notebook Cell 7
# ============================================================

def create_single_sequence(X_window: np.ndarray) -> np.ndarray:
    """
    1 window (lookback × n_features) → 1 feature vector
    6 statistics per feature: last, mean, std, slope, position, delta
    """
    n_features = X_window.shape[1]
    row = []
    
    for f in range(n_features):
        col = X_window[:, f]
        
        # Clean data
        col_clean = col[np.isfinite(col)]
        if len(col_clean) == 0:
            row.extend([0.0] * 6)
            continue
        
        last_val = float(col[-1]) if np.isfinite(col[-1]) else 0.0
        mean_val = float(np.nanmean(col_clean))
        std_val  = float(np.nanstd(col_clean))
        
        # Slope
        if len(col_clean) >= 2 and np.all(np.isfinite(col)):
            slope_val = float(np.polyfit(range(len(col)), col, 1)[0])
        else:
            slope_val = 0.0
        
        # Position in range
        col_range = np.nanmax(col_clean) - np.nanmin(col_clean)
        if col_range > 1e-10:
            position = float((last_val - np.nanmin(col_clean)) / col_range)
            position = np.clip(position, 0.0, 1.0)
        else:
            position = 0.5
        
        # Delta
        if len(col) >= 6:
            delta = float(col[-1] - col[-6])
        else:
            delta = float(col[-1] - col[0]) if len(col) > 0 else 0.0

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
    """Train Random Forest trên toàn bộ df, trả về (model, f_scaler, t_scaler)."""
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

    # ✅ Random Forest với tuned hyperparameters
    model = RandomForestRegressor(
        n_estimators=CONFIG["rf_n_estimators"],
        max_depth=CONFIG["rf_max_depth"],
        min_samples_leaf=CONFIG["rf_min_samples_leaf"],
        min_samples_split=CONFIG["rf_min_samples_split"],
        max_features=CONFIG["rf_max_features"],
        random_state=CONFIG["random_state"],
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
    Dự báo close[t+1].
    Công thức: close[t+1] = zlema_val[t] + predicted_residual_adj
    """
    if lookback is None:
        lookback = CONFIG["lookback"]

    feats_ok = [f for f in feature_list if f in past_window.columns]
    window   = past_window[feats_ok].values[-lookback:].astype(np.float64)
    window   = np.where(np.isfinite(window), window, 0.0)

    window_s = f_scaler.transform(window)
    feat_vec = create_single_sequence(window_s).reshape(1, -1)

    pred_s       = model.predict(feat_vec)
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
        "model"         : os.path.join(SAVE_DIR, f"rf_z5_cm_{s}.joblib"),
        "f_scaler"      : os.path.join(SAVE_DIR, f"f_scaler_z5_{s}.joblib"),
        "t_scaler"      : os.path.join(SAVE_DIR, f"t_scaler_z5_{s}.joblib"),
        "meta"          : os.path.join(SAVE_DIR, f"meta_z5_{s}.json"),
    }


def model_exists(symbol: str) -> bool:
    return all(os.path.exists(p) for p in get_model_paths(symbol).values())


def needs_retrain(symbol: str) -> bool:
    """True nếu model cũ hơn retrain_every phiên (ngày)."""
    paths = get_model_paths(symbol)
    if not os.path.exists(paths["meta"]):
        return True
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    saved_at = datetime.fromisoformat(meta.get("saved_at", "2000-01-01"))
    days_old = (datetime.now() - saved_at).days
    return days_old >= CONFIG["retrain_every"]


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
            "model"        : "RandomForest",
            "zlema_period" : CONFIG["zlema_period"],
            "feature_list" : feature_list,
            "target"       : TARGET_ADJ,
            "lookback"     : CONFIG["lookback"],
            "rf_params"    : {
                "n_estimators": CONFIG["rf_n_estimators"],
                "max_depth": CONFIG["rf_max_depth"],
                "min_samples_leaf": CONFIG["rf_min_samples_leaf"],
            },
            "saved_at"     : datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved RF model for {s}")


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
    print(f"TRAIN  {s}  |  RF + ZLEMA({CONFIG['zlema_period']}) + Core+Momentum (DELTA)")
    print(f"{'='*60}")

    end_fetch = datetime.now().strftime("%Y-%m-%d")
    df = load_data(s, CONFIG["start_date"], end_fetch, CONFIG["interval"])
    df = validate_data(df)
    df = compute_features_DELTA(df, zlema_period=CONFIG["zlema_period"])

    model, f_sc, t_sc = fit_final_model(
        df=df,
        feature_list=CORE_MOMENTUM_FEATURES_DELTA,
        target=TARGET_ADJ,
        lookback=CONFIG["lookback"],
    )
    save_model_package(s, model, f_sc, t_sc, CORE_MOMENTUM_FEATURES_DELTA)
    print(f"✅ Training done — {len(df)} rows, "
          f"{len(CORE_MOMENTUM_FEATURES_DELTA)} features (DELTA)")
    return model, f_sc, t_sc


def ensure_model(symbol: str) -> tuple:
    """Load nếu model còn mới, retrain nếu đã cũ ≥ retrain_every ngày."""
    s = symbol.upper()
    if not model_exists(s) or needs_retrain(s):
        action = "Chưa có model" if not model_exists(s) else \
                 f"Model cũ ≥ {CONFIG['retrain_every']} ngày"
        print(f"⚙️  {action} → Retrain {s}...")
        train_model_for_symbol(s)
    return load_model_package(s)


# ============================================================
# FORECAST (RECURSIVE with RECOMPUTE)
# ============================================================

def forecast_future(symbol: str,
                    forecast_steps: int = 5) -> pd.DataFrame:
    """
    ✅ Recursive forecast với recompute features mỗi bước
    Blend: 60% ML + 25% Trend + 15% ZLEMA anchor
    """
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
    print(f"FORECAST {forecast_steps} PHIÊN  |  {s}  |  "
          f"RF + ZLEMA({CONFIG['zlema_period']})")
    print(f"{'='*60}")

    last_real_date = df["time"].iloc[-1]
    next_date = last_real_date + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)

    prev_close = float(df["close"].iloc[-1])

    # Pre-compute trend parameters
    working_feat = compute_features_DELTA(
        working_raw.tail(300).copy(),
        zlema_period=CONFIG["zlema_period"],
    )

    atr_now     = float(working_feat["atr_14"].iloc[-1])
    close_delta = float(working_feat["close_delta"].iloc[-1])
    slope_5     = float(working_feat["slope_5_close_delta"].iloc[-1])
    trend_slope = close_delta * 0.7 + slope_5 * 0.3  # Weighted momentum

    for step in range(1, forecast_steps + 1):

        # ===== RECOMPUTE FEATURES (recursive) =====
        working_feat = compute_features_DELTA(
            working_raw.tail(300).copy(),
            zlema_period=CONFIG["zlema_period"],
        )

        past_window = working_feat.tail(CONFIG["lookback"]).copy()

        # ===== ML PREDICTION =====
        pred_ml_raw = predict_next_price(
            past_window  = past_window,
            feature_list = CORE_MOMENTUM_FEATURES_DELTA,
            model        = model,
            f_scaler     = f_sc,
            t_scaler     = t_sc,
            lookback     = CONFIG["lookback"],
        )

        # ===== TREND COMPONENT =====
        pred_trend = prev_close + trend_slope

        # ===== ZLEMA ANCHOR =====
        zlema_now = float(past_window["zlema_val"].iloc[-1])

        # ===== BLEND (60% ML + 25% Trend + 15% ZLEMA) =====
        pred_raw = (
            0.60 * pred_ml_raw +
            0.25 * pred_trend +
            0.15 * zlema_now
        )

        # ===== NOISE DECAY =====
        decay = 1 - step / (forecast_steps + 1)
        noise = atr_now * 0.10 * decay * np.sign(trend_slope)
        pred_raw += noise

        # ===== ROUND TO TICK =====
        pred_price = round_to_tick(pred_raw, tick)
        pct_change = (pred_price / prev_close - 1) * 100

        results.append({
            "Phiên"       : step,
            "Ngày"        : next_date.strftime("%d/%m/%Y"),
            "Giá dự báo"  : pred_price,
            "Thay đổi_%"  : pct_change,
            "Bước giá"    : tick,
            "Loại"        : "ML" if step == 1 else "Hybrid",
        })

        print(f"  Bước {step} [{'ML' if step==1 else 'Hybrid'}] | "
              f"{next_date.strftime('%d/%m/%Y')} | "
              f"Giá: {pred_price:,.2f} | {pct_change:+.2f}%")

        # ===== APPEND SYNTHETIC ROW (recursive) =====
        new_row = working_raw.iloc[-1].copy()
        new_row["time"]  = next_date
        new_row["open"]  = prev_close
        new_row["close"] = pred_price
        new_row["high"]  = max(prev_close, pred_price)
        new_row["low"]   = min(prev_close, pred_price)
        new_row["volume"] = working_raw["volume"].tail(20).mean()

        working_raw = pd.concat(
            [working_raw, pd.DataFrame([new_row])],
            ignore_index=True
        )

        prev_close = pred_price

        next_date = next_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False, formatters={
        "Giá dự báo" : "{:,.2f}".format,
        "Thay đổi_%" : "{:+.2f}".format,
        "Bước giá"   : "{:.4f}".format,
    }))
    return result_df


# ============================================================
# BACKTEST
# ============================================================

def backtest_model(symbol: str,
                   backtest_start: str = "2025-01-01") -> pd.DataFrame:
    s = symbol.upper()
    model, f_sc, t_sc, meta = load_model_package(s)

    end_fetch = datetime.now().strftime("%Y-%m-%d")
    df_raw    = load_data(s, CONFIG["start_date"], end_fetch,
                          CONFIG["interval"])
    df_raw    = validate_data(df_raw)
    df        = compute_features_DELTA(df_raw,
                                      zlema_period=CONFIG["zlema_period"])
    tick      = detect_tick_size(df)
    bt_idx    = df[df["time"] >= pd.to_datetime(backtest_start)].index.tolist()

    results = []
    for idx in bt_idx:
        past_df = df.loc[:idx - 1].copy()
        if len(past_df) < CONFIG["lookback"] + 50:
            continue

        past_window = past_df.tail(CONFIG["lookback"]).copy()
        actual      = float(df.loc[idx, "close"])
        prev_close  = float(past_window["close"].iloc[-1])

        try:
            pred_raw  = predict_next_price(
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
    print(f"BACKTEST  {s}  từ {backtest_start}")
    print(f"RF + ZLEMA({CONFIG['zlema_period']}) + Core+Momentum (DELTA)")
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
    print("RF + ZLEMA(5) + Core+Momentum (DELTA)")
    print("Optimal config from notebook analysis")
    print("="*60)
    
    symbol = input("Nhập mã (VD: VNINDEX, HPG, VNM): ").strip().upper()
    print("="*60)

    ensure_model(symbol)

    while True:
        print(f"\n[{symbol}] Chọn chức năng:")
        print("1. Dự báo giá (recursive forecast)")
        print("2. Backtest")
        print("3. Retrain model")
        print("0. Thoát")

        choice = input("→ ").strip()

        if choice == "1":
            try:
                steps = int(input("Số phiên (1-30, mặc định 5): ").strip() or "5")
                steps = max(1, min(30, steps))
            except Exception:
                steps = 5
            forecast_future(symbol, forecast_steps=steps)

        elif choice == "2":
            bt_start = input("Từ ngày (YYYY-MM-DD, mặc định 2025-01-01): ").strip()
            if not bt_start:
                bt_start = "2025-01-01"
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