# ============================================================
# a_ML2.py — ML THUẦN TÚY (ZLEMA5 + Lasso + Core+Momentum)
# Optimal config từ notebook: ZLEMA(5), Core+Momentum features
# Retrain mỗi 5 phiên (trading week)
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
from sklearn.linear_model import Lasso

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4f}".format

# ============================================================
# GLOBAL CONFIG — Đồng bộ với notebook optimal config
# ============================================================

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG = {
    "start_date"    : "2019-06-15",
    "end_date"      : "2025-12-31",
    "interval"      : "d",
    "lookback"      : 22,           # từ notebook
    "zlema_period"  : 5,            # ZLEMA(5) — optimal từ notebook
    "lasso_alpha"   : 0.01,         # từ notebook Cell 10
    "retrain_every" : 5,            # retrain mỗi 5 phiên (1 tuần giao dịch)
}

# Core + Momentum features — optimal từ notebook (ablation Cell 14)
# MAPE=0.939%, TheilU=0.9627 — tốt nhất trong ablation
CORE_MOMENTUM_FEATURES = [
    # Core: residual lags
    "residual_lag1",
    "residual_lag2",
    "residual_lag3",
    "residual_lag4",
    "residual_lag5",
    # Momentum
    "slope_5",
    "slope_10",
    "macd_hist",
    "stoch_diff",
    "rsi",
]

TARGET = "reg_target"   # close[t+1] - zlema[t] — từ notebook


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
# FEATURE ENGINEERING — Đồng bộ với notebook Cell 4
# ============================================================

def linreg_slope(arr) -> float:
    """
    Linear regression slope robust cho rolling.apply(raw=True)
    """
    try:
        y = np.asarray(arr, dtype=np.float64)

        if len(y) < 2:
            return np.nan

        if np.isnan(y).any():
            return np.nan

        x = np.arange(len(y), dtype=np.float64)

        slope = np.polyfit(x, y, 1)[0]
        mean_price = np.mean(y)

        if abs(mean_price) > 1e-9:
            slope = (slope / mean_price) * 100
        return float(slope)

    except Exception:
        return np.nan


def compute_features(df: pd.DataFrame,
                     zlema_period: int = None) -> pd.DataFrame:
    """
    Feature engineering đồng bộ hoàn toàn với notebook Cell 4.
    Target: close[t+1] - zlema[t]  (reg_target)
    """
    if zlema_period is None:
        zlema_period = CONFIG["zlema_period"]

    df    = df.copy().sort_values("time").reset_index(drop=True)
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0)

    # ── Indicators ───────────────────────────────────────────
    df["log_return"] = np.log(close / close.shift(1))
    df["hv_14"]      = df["log_return"].rolling(14).std() * np.sqrt(252)
    df["rsi"]        = ta.RSI(close, timeperiod=14)

    _, _, df["macd_hist"] = ta.MACD(close, fastperiod=12,
                                     slowperiod=26, signalperiod=9)

    df["stoch_k"], df["stoch_d"] = ta.STOCH(
        high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
    )
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]
    df["adx"]        = ta.ADX(high, low, close, timeperiod=14)
    df["mfi"]        = ta.MFI(high, low, close, volume, timeperiod=14)
    df["atr_14"]     = ta.ATR(high, low, close, timeperiod=14)

    # ── ZLEMA (period từ config) ──────────────────────────────
    try:
        z = TalippZLEMA(period=zlema_period)
        z.update(close.tolist())
        z_vals = [np.nan if v is None else float(v) for v in z]
        df["zlema_val"] = np.nan
        df.iloc[-len(z_vals):, df.columns.get_loc("zlema_val")] = z_vals
    except Exception:
        df["zlema_val"] = ta.EMA(close, timeperiod=zlema_period)

    # ── Target: close[t+1] - zlema[t] ───────────────────────
    df["close_next"] = close.shift(-1)
    df["reg_target"] = df["close_next"] - df["zlema_val"]

    # ── Core features: residual lags ─────────────────────────
    df["residual"] = close - df["zlema_val"]
    for lag in range(1, 6):
        df[f"residual_lag{lag}"] = df["residual"].shift(lag)

    # ── Momentum features ────────────────────────────────────
    df["slope_5"]  = close.rolling(5).apply(linreg_slope, raw=True)
    df["slope_10"] = close.rolling(10).apply(linreg_slope, raw=True)
    # macd_hist, stoch_diff, rsi đã có ở trên

    # ── Clean ────────────────────────────────────────────────
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
# SEQUENCE BUILDER — Đồng bộ với notebook Cell 7 (FIXED v3)
# ============================================================

def create_single_sequence(X_window: np.ndarray) -> np.ndarray:
    """
    1 window (lookback × n_features) → 1 feature vector
    6 statistics per feature: last, mean, std, slope, position, delta
    """
    n_features = X_window.shape[1]
    row = []
    for f in range(n_features):
        col       = X_window[:, f]
        last_val  = col[-1]
        mean_val  = np.nanmean(col)
        std_val   = np.nanstd(col)
        col_range = np.nanmax(col) - np.nanmin(col)
        position  = ((last_val - np.nanmin(col)) / col_range
                     if col_range > 1e-10 else 0.5)
        delta     = col[-1] - col[-6] if len(col) >= 6 else col[-1] - col[0]

        if np.isnan(col).any() or np.isinf(col).any():
            slope_val = 0.0
        else:
            slope_val = float(np.polyfit(range(len(col)), col, 1)[0])

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
    """Train Lasso trên toàn bộ df, trả về (model, f_scaler, t_scaler)."""
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

    model = Lasso(alpha=CONFIG["lasso_alpha"], max_iter=5000)
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
    Công thức: close[t+1] = zlema_val[t] + predicted_residual
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
        "model"         : os.path.join(SAVE_DIR, f"lasso_cm_{s}.joblib"),
        "f_scaler"      : os.path.join(SAVE_DIR, f"f_scaler_{s}.joblib"),
        "t_scaler"      : os.path.join(SAVE_DIR, f"t_scaler_{s}.joblib"),
        "meta"          : os.path.join(SAVE_DIR, f"meta_{s}.json"),
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
            "model"        : "Lasso",
            "zlema_period" : CONFIG["zlema_period"],
            "feature_list" : feature_list,
            "target"       : TARGET,
            "lookback"     : CONFIG["lookback"],
            "saved_at"     : datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved model for {s}")


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
    print(f"TRAIN  {s}  |  ZLEMA({CONFIG['zlema_period']}) + Lasso")
    print(f"{'='*60}")

    end_fetch = datetime.now().strftime("%Y-%m-%d")
    df = load_data(s, CONFIG["start_date"], end_fetch, CONFIG["interval"])
    df = validate_data(df)
    df = compute_features(df, zlema_period=CONFIG["zlema_period"])

    model, f_sc, t_sc = fit_final_model(
        df=df,
        feature_list=CORE_MOMENTUM_FEATURES,
        target=TARGET,
        lookback=CONFIG["lookback"],
    )
    save_model_package(s, model, f_sc, t_sc, CORE_MOMENTUM_FEATURES)
    print(f"✅ Training done — {len(df)} rows, "
          f"{len(CORE_MOMENTUM_FEATURES)} features")
    return model, f_sc, t_sc


def ensure_model(symbol: str) -> tuple:
    """Chỉ load model. Việc auto-retrain đã được chuyển sang auto.py."""
    s = symbol.upper()
    if not model_exists(s):
        print(f"⚙️  Chưa có model → Train {s}...")
        train_model_for_symbol(s)
    return load_model_package(s)


# ============================================================
# FORECAST
# ============================================================

# def forecast_future(symbol: str,
#                     forecast_steps: int = 5) -> pd.DataFrame:
#     s = symbol.upper()
#     model, f_sc, t_sc, meta = ensure_model(s)

#     end_fetch  = datetime.now().strftime("%Y-%m-%d")
#     df_raw     = load_data(s, CONFIG["start_date"], end_fetch,
#                            CONFIG["interval"])
#     df_raw     = validate_data(df_raw)
#     df         = compute_features(df_raw,
#                                   zlema_period=CONFIG["zlema_period"])
#     tick       = detect_tick_size(df)

#     # Giữ raw OHLCV riêng để synthetic update không bị ảnh hưởng bởi dropna
#     working_raw = df_raw.copy()   # raw OHLCV chưa qua compute_features
#     results     = []

#     print(f"\n{'='*60}")
#     print(f"FORECAST {forecast_steps} PHIÊN  |  {s}  |  "
#           f"ZLEMA({CONFIG['zlema_period']})")
#     print(f"{'='*60}")

#     # next_date ban đầu = ngày giao dịch tiếp theo sau ngày cuối dữ liệu
#     last_real_date = df["time"].iloc[-1]
#     next_date = last_real_date + timedelta(days=1)
#     while next_date.weekday() >= 5:
#         next_date += timedelta(days=1)

#     prev_close = float(df["close"].iloc[-1])

#     # ── Bước 1: ML thuần — dự báo thực sự ───────────────────────
#     working_feat = compute_features(
#         working_raw.tail(300).copy(),
#         zlema_period=CONFIG["zlema_period"],
#     )
#     past_window = working_feat.tail(CONFIG["lookback"]).copy()

#     # Lấy thêm thông tin trend từ features hiện tại để dùng cho bước 2+
#     atr_now    = float(working_feat["atr_14"].iloc[-1])
#     slope_5    = float(working_feat["slope_5"].iloc[-1])   # điểm/phiên
#     slope_10   = float(working_feat["slope_10"].iloc[-1])
#     trend_slope = (slope_5 * 0.6 + slope_10 * 0.4)         # weighted slope

#     pred_step1_raw = predict_next_price(
#         past_window  = past_window,
#         feature_list = CORE_MOMENTUM_FEATURES,
#         model        = model,
#         f_scaler     = f_sc,
#         t_scaler     = t_sc,
#         lookback     = CONFIG["lookback"],
#     )
#     pred_step1 = round_to_tick(pred_step1_raw, tick)
#     pct1       = (pred_step1 / prev_close - 1) * 100

#     results.append({
#         "Phiên"       : 1,
#         "Ngày"        : next_date.strftime("%d/%m/%Y"),
#         "Giá dự báo"  : pred_step1,
#         "Thay đổi_%"  : pct1,
#         "Bước giá"    : tick,
#         "Loại"        : "ML",
#     })
#     print(f"  Bước 1 [ML] | {next_date.strftime('%d/%m/%Y')} | "
#           f"Giá: {pred_step1:,.2f} | {pct1:+.2f}%")

#     # ── Bước 2+: Trend extrapolation từ giá bước 1 ──────────────
#     # Dùng momentum thực tế (slope) + ATR để tạo ra dao động có ý nghĩa
#     # Không dùng synthetic → không bị mean-reversion bias
#     ref_price  = pred_step1
#     prev_close = pred_step1

#     next_date = next_date + timedelta(days=1)
#     while next_date.weekday() >= 5:
#         next_date += timedelta(days=1)

#     for step in range(2, forecast_steps + 1):
#         # Trend component: slope × step (tuyến tính theo momentum)
#         trend_move = trend_slope * step * 0.5   # 0.5: dampen factor

#         # Noise component: nhỏ dần theo step (uncertainty tăng)
#         noise_scale = atr_now * 0.15 * (1 - step / (forecast_steps + 2))
#         # Dấu noise theo momentum: nếu slope > 0 → noise dương nhẹ
#         noise = noise_scale * np.sign(trend_slope) if abs(trend_slope) > 0.1 else 0

#         pred_price_raw = ref_price + trend_move + noise
#         pred_price     = round_to_tick(pred_price_raw, tick)
#         pct_change     = (pred_price / prev_close - 1) * 100

#         results.append({
#             "Phiên"       : step,
#             "Ngày"        : next_date.strftime("%d/%m/%Y"),
#             "Giá dự báo"  : pred_price,
#             "Thay đổi_%"  : pct_change,
#             "Bước giá"    : tick,
#             "Loại"        : "Trend",
#         })
#         print(f"  Bước {step} [Trend] | {next_date.strftime('%d/%m/%Y')} | "
#               f"Giá: {pred_price:,.2f} | {pct_change:+.2f}%  "
#               f"(slope={trend_slope:+.2f}, noise={noise:+.2f})")

#         prev_close = pred_price
#         next_date  = next_date + timedelta(days=1)
#         while next_date.weekday() >= 5:
#             next_date += timedelta(days=1)

#     result_df = pd.DataFrame(results)
#     print(result_df.to_string(index=False, formatters={
#         "Giá dự báo" : "{:,.2f}".format,
#         "Thay đổi_%" : "{:+.2f}".format,
#         "Bước giá"   : "{:.4f}".format,
#     }))
#     return result_df

def forecast_future(symbol: str,
                    forecast_steps: int = 5) -> pd.DataFrame:
    s = symbol.upper()
    model, f_sc, t_sc, meta = ensure_model(s)

    end_fetch  = datetime.now().strftime("%Y-%m-%d")
    df_raw     = load_data(s, CONFIG["start_date"], end_fetch,
                           CONFIG["interval"])
    df_raw     = validate_data(df_raw)
    df         = compute_features(df_raw,
                                  zlema_period=CONFIG["zlema_period"])
    tick       = detect_tick_size(df)

    working_raw = df_raw.copy()
    results     = []

    print(f"\n{'='*60}")
    print(f"FORECAST {forecast_steps} PHIÊN  |  {s}  |  "
          f"ZLEMA({CONFIG['zlema_period']})")
    print(f"{'='*60}")

    last_real_date = df["time"].iloc[-1]
    next_date = last_real_date + timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += timedelta(days=1)

    prev_close = float(df["close"].iloc[-1])

    # ===== PRE-COMPUTE =====
    working_feat = compute_features(
        working_raw.tail(300).copy(),
        zlema_period=CONFIG["zlema_period"],
    )

    atr_now    = float(working_feat["atr_14"].iloc[-1])
    slope_5    = float(working_feat["slope_5"].iloc[-1])
    slope_10   = float(working_feat["slope_10"].iloc[-1])
    trend_slope = (slope_5 * 0.6 + slope_10 * 0.4)

    for step in range(1, forecast_steps + 1):

        # ===== RECOMPUTE FEATURES (recursive thật) =====
        working_feat = compute_features(
            working_raw.tail(300).copy(),
            zlema_period=CONFIG["zlema_period"],
        )

        past_window = working_feat.tail(CONFIG["lookback"]).copy()

        # ===== ML PRED =====
        pred_ml_raw = predict_next_price(
            past_window  = past_window,
            feature_list = CORE_MOMENTUM_FEATURES,
            model        = model,
            f_scaler     = f_sc,
            t_scaler     = t_sc,
            lookback     = CONFIG["lookback"],
        )

        # ===== TREND =====
        pred_trend = prev_close + trend_slope

        # ===== ZLEMA ANCHOR =====
        zlema_now = float(past_window["zlema_val"].iloc[-1])

        # ===== BLEND =====
        pred_raw = (
            0.6 * pred_ml_raw +
            0.25 * pred_trend +
            0.15 * zlema_now
        )

        # ===== NOISE DECAY =====
        decay = 1 - step / (forecast_steps + 1)
        noise = atr_now * 0.1 * decay * np.sign(trend_slope)
        pred_raw += noise

        # ===== ROUND =====
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
    df        = compute_features(df_raw,
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
                feature_list = CORE_MOMENTUM_FEATURES,
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
    symbol = input("Nhập mã (VD: VNINDEX, HPG, VNM): ").strip().upper()
    print("="*60)

    ensure_model(symbol)

    while True:
        print(f"\n[{symbol}] Chọn chức năng:")
        print("1. Dự báo giá")
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