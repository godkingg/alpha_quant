# ============================================================
# COMPLETE FORECASTING PIPELINE WITH CLEAN SCALING + TICK SIZE
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
# GLOBAL CONFIG
# ============================================================

SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG = {
    "start_date": "2019-06-15",
    "end_date": "2025-12-31",
    "interval": "d",
    "lookback": 22,
    "retrain_every": 63,
}

CORE_MOMENTUM_FEATURES = [
    "close_zlema20_ratio",
    "residual_lag1",
    "residual_lag2",
    "res_change",
    "slope_5",
    "slope_10",
    "macd_hist",
    "stoch_diff",
    "rsi",
]

TARGET = "residual"

# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_data(symbol: str, start_date: str, end_date: str, interval: str = "d") -> pd.DataFrame:
    quote = Quote(symbol=symbol.upper(), source="VCI")
    df = quote.history(start=start_date, end=end_date, interval=interval)

    if df is None or df.empty:
        raise ValueError(f"❌ Không có dữ liệu cho {symbol}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Thiếu cột: {missing}")

    df["time"] = pd.to_datetime(df["time"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)

    # duplicates
    df = df.drop_duplicates(subset="time", keep="last").reset_index(drop=True)

    # fill missing
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill().bfill()
    df["volume"] = df["volume"].fillna(0).clip(lower=0)

    # OHLC logic
    mask = df["high"] < df["low"]
    if mask.any():
        df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)
    df.loc[df["high"] < max_oc, "high"] = max_oc[df["high"] < max_oc]
    df.loc[df["low"] > min_oc, "low"] = min_oc[df["low"] > min_oc]

    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def linreg_slope(series):
    if len(series) < 5:
        return np.nan
    x = np.arange(len(series))
    y = series.values if hasattr(series, "values") else series
    if np.isnan(y).any():
        return np.nan
    slope, _ = np.polyfit(x, y, 1)
    return slope


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0)

    df["log_return"] = np.log(close / close.shift(1))
    df["hv_14"] = df["log_return"].rolling(14).std() * np.sqrt(252)

    df["rsi"] = ta.RSI(close, timeperiod=14)
    _, _, df["macd_hist"] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    df["stoch_k"], df["stoch_d"] = ta.STOCH(
        high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
    )
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]

    df["adx"] = ta.ADX(high, low, close, timeperiod=14)
    df["mfi"] = ta.MFI(high, low, close, volume, timeperiod=14)
    df["atr_14"] = ta.ATR(high, low, close, timeperiod=14)

    try:
        z = TalippZLEMA(period=20)
        z.update(close.tolist())
        z_vals = list(z)
        df["zlema20"] = np.nan
        df.iloc[-len(z_vals):, df.columns.get_loc("zlema20")] = z_vals
    except:
        df["zlema20"] = ta.EMA(close, timeperiod=20)

    df["residual"] = close - df["zlema20"]
    df["residual_lag1"] = df["residual"].shift(1)
    df["residual_lag2"] = df["residual"].shift(2)
    df["res_change"] = df["residual"].diff()
    df["close_zlema20_ratio"] = close / df["zlema20"]

    df["slope_5"] = close.rolling(5).apply(linreg_slope, raw=True)
    df["slope_10"] = close.rolling(10).apply(linreg_slope, raw=True)

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)
    return df

# ============================================================
# TICK SIZE UTILITIES
# ============================================================

def detect_tick_size_from_data(df: pd.DataFrame, price_col: str = "close", tail_n: int = 300) -> float:
    s = df[price_col].dropna().tail(tail_n).astype(float)
    diffs = s.diff().abs().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 0.01

    diffs_rounded = diffs.round(4)
    value_counts = diffs_rounded.value_counts().sort_index()
    candidates = value_counts[value_counts.index > 0]

    if len(candidates) == 0:
        return 0.01

    candidates = candidates.head(20)
    min_count = max(2, int(len(diffs_rounded) * 0.01))
    candidates = candidates[candidates >= min_count]

    if len(candidates) == 0:
        q = diffs_rounded.quantile(0.05)
        return float(max(round(q, 4), 0.01))

    tick = float(candidates.index.min())

    common_ticks = np.array([
        0.001, 0.005,
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.5,
        1.0, 2.0, 5.0
    ])

    nearest = common_ticks[np.argmin(np.abs(common_ticks - tick))]
    return float(nearest)


def round_to_tick(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return float(price)
    return round(round(price / tick_size) * tick_size, 6)


def apply_tick_rounding(price: float, df_ref: pd.DataFrame, price_col: str = "close") -> Tuple[float, float]:
    tick = detect_tick_size_from_data(df_ref, price_col=price_col)
    rounded_price = round_to_tick(price, tick)
    return rounded_price, tick

# ============================================================
# SEQUENCE + CLEAN SCALING
# ============================================================

def create_sequences_summary_xy(X_data: np.ndarray, y_data: np.ndarray, lookback: int):
    n_features = X_data.shape[1]
    X_out, y_out = [], []

    for i in range(lookback, len(X_data)):
        window = X_data[i-lookback:i, :]
        target = y_data[i]

        row = []
        for f in range(n_features):
            col = window[:, f]

            last_val = col[-1]
            mean_val = np.nanmean(col)
            std_val = np.nanstd(col)

            if not np.any(np.isnan(col)):
                try:
                    slope_val = np.polyfit(range(len(col)), col, 1)[0]
                except:
                    slope_val = 0.0
            else:
                slope_val = 0.0

            col_range = np.nanmax(col) - np.nanmin(col)
            position = (last_val - np.nanmin(col)) / col_range if col_range > 1e-10 else 0.5
            delta = col[-1] - col[-6] if len(col) >= 6 else col[-1] - col[0]

            row.extend([last_val, mean_val, std_val, slope_val, position, delta])

        latest = window[-1, :]
        if len(latest) >= 9:
            row.append(latest[min(6, len(latest)-1)] * latest[min(7, len(latest)-1)] / 10000)
            row.append(latest[min(3, len(latest)-1)] * np.sign(latest[min(4, len(latest)-1)]))
            row.append(latest[min(1, len(latest)-1)] * latest[min(2, len(latest)-1)])
            row.append(latest[min(0, len(latest)-1)] * latest[min(3, len(latest)-1)])
            row.append(latest[min(6, len(latest)-1)] * latest[min(8, len(latest)-1)])
        else:
            row.extend([0, 0, 0, 0, 0])

        X_out.append(row)
        y_out.append(target)

    return np.array(X_out), np.array(y_out)


def fit_final_model(df: pd.DataFrame, feature_list: List[str], target: str, lookback: int = 22):
    X_raw = df[feature_list].values
    y_raw = df[target].values

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(X_raw)
    y_scaled = target_scaler.fit_transform(y_raw.reshape(-1, 1)).ravel()

    X_seq, y_seq = create_sequences_summary_xy(X_scaled, y_scaled, lookback)

    model = Lasso(alpha=0.01, max_iter=5000)
    model.fit(X_seq, y_seq)

    return model, feature_scaler, target_scaler


def predict_next_price(past_window: pd.DataFrame, feature_list, model, feature_scaler, target_scaler, lookback=22):
    X_window_raw = past_window[feature_list].values
    X_window_scaled = feature_scaler.transform(X_window_raw)

    dummy_y = np.zeros(len(X_window_scaled))
    X_pred_seq, _ = create_sequences_summary_xy(
        np.vstack([X_window_scaled, X_window_scaled[-1:]]),
        np.append(dummy_y, 0),
        lookback
    )

    pred_scaled = model.predict(X_pred_seq[:1])[0]
    pred_target = target_scaler.inverse_transform(np.array([[pred_scaled]])).ravel()[0]

    zlema_prev = past_window["zlema20"].iloc[-1]
    pred_price = zlema_prev + pred_target
    return float(pred_price)

# ============================================================
# REGIME + TREND + HYBRID
# ============================================================

def compute_regime_info(window_df):
    latest = window_df.iloc[-1]
    prev = window_df.iloc[-2] if len(window_df) >= 2 else latest

    close = latest["close"]
    prev_close = prev["close"]
    zlema20 = latest["zlema20"]
    adx = latest["adx"]
    rsi = latest["rsi"]
    macd_hist = latest["macd_hist"]
    slope_5 = latest["slope_5"]
    slope_10 = latest["slope_10"]

    price_dev = (close / zlema20 - 1) if zlema20 != 0 else 0
    daily_ret = (close / prev_close - 1) if prev_close != 0 else 0

    trend_score = (
        0.25 * np.sign(slope_5) +
        0.20 * np.sign(slope_10) +
        0.20 * np.sign(macd_hist) +
        0.15 * (1 if rsi > 55 else -1 if rsi < 45 else 0) +
        0.20 * np.sign(price_dev)
    )
    trend_score = float(np.clip(trend_score, -1, 1))

    trend_strength = float(np.clip(
        0.40 * np.clip((adx - 15) / 20, 0, 1) +
        0.30 * np.clip(abs(price_dev) / 0.03, 0, 1) +
        0.30 * np.clip(abs(daily_ret) / 0.015, 0, 1),
        0, 1
    ))

    shock = abs(daily_ret) > 0.03 or abs(price_dev) > 0.055

    if shock:
        regime = "SHOCK"
        w_residual = 1.0
    elif adx < 18 and abs(price_dev) < 0.015:
        regime = "RANGING"
        w_residual = 0.80
    elif adx > 25 and abs(trend_score) > 0.35:
        regime = "TRENDING"
        w_residual = 0.25
    else:
        regime = "NORMAL"
        w_residual = 0.50

    if regime != "SHOCK":
        w_residual = float(np.clip(w_residual - 0.15 * trend_strength, 0.10, 0.90))

    return {
        "regime": regime,
        "trend_score": trend_score,
        "trend_strength": trend_strength,
        "w_residual": w_residual
    }


def trend_forecast(window_df):
    latest = window_df.iloc[-1]
    close = latest["close"]
    atr_14 = latest["atr_14"]
    info = compute_regime_info(window_df)

    if pd.isna(atr_14) or atr_14 <= 0:
        atr_14 = max(window_df["close"].tail(14).std(), close * 0.005)

    step = atr_14 * (0.2 + 0.85 * info["trend_strength"]) * info["trend_score"]
    return float(close + step)

# ============================================================
# SAVE / LOAD MODEL
# ============================================================

def get_model_paths(symbol: str):
    symbol = symbol.upper()
    return {
        "model": os.path.join(SAVE_DIR, f"final_lasso_core_momentum_{symbol}.joblib"),
        "feature_scaler": os.path.join(SAVE_DIR, f"final_feature_scaler_{symbol}.joblib"),
        "target_scaler": os.path.join(SAVE_DIR, f"final_target_scaler_{symbol}.joblib"),
        "meta": os.path.join(SAVE_DIR, f"final_package_clean_scaling_{symbol}.json"),
    }


def model_exists(symbol: str) -> bool:
    paths = get_model_paths(symbol)
    return all(os.path.exists(p) for p in paths.values())


def save_model_package(symbol: str, model, feature_scaler, target_scaler, feature_list: List[str], target: str):
    symbol = symbol.upper()
    paths = get_model_paths(symbol)

    joblib.dump(model, paths["model"])
    joblib.dump(feature_scaler, paths["feature_scaler"])
    joblib.dump(target_scaler, paths["target_scaler"])

    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump({
            "symbol": symbol,
            "model": "Lasso + Summary + Core_Momentum",
            "feature_list": feature_list,
            "target": target,
            "lookback": CONFIG["lookback"],
            "saved_at": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved model package for {symbol}")


def load_model_package(symbol: str):
    symbol = symbol.upper()
    paths = get_model_paths(symbol)

    model = joblib.load(paths["model"])
    feature_scaler = joblib.load(paths["feature_scaler"])
    target_scaler = joblib.load(paths["target_scaler"])

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, feature_scaler, target_scaler, meta

# ============================================================
# TRAIN PIPELINE
# ============================================================

def train_model_for_symbol(symbol: str):
    print(f"\n{'='*80}")
    print(f"TRAIN MODEL FOR {symbol}")
    print(f"{'='*80}")

    df = load_data(symbol, CONFIG["start_date"], CONFIG["end_date"], CONFIG["interval"])
    df = validate_data(df)
    df = compute_features(df)

    model, feature_scaler, target_scaler = fit_final_model(
        df=df,
        feature_list=CORE_MOMENTUM_FEATURES,
        target=TARGET,
        lookback=CONFIG["lookback"]
    )

    save_model_package(
        symbol=symbol,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_list=CORE_MOMENTUM_FEATURES,
        target=TARGET
    )

    print(f"✅ Training completed for {symbol}")
    return model, feature_scaler, target_scaler

# ============================================================
# FORECAST FUTURE PRICE
# ============================================================

def forecast_future(symbol: str, forecast_steps: int = 5):
    symbol = symbol.upper()
    model, feature_scaler, target_scaler, meta = load_model_package(symbol)

    end_fetch = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    df = load_data(symbol, CONFIG["start_date"], end_fetch, CONFIG["interval"])
    df = validate_data(df)
    df = compute_features(df)

    working_df = df.copy()
    results = []

    print(f"\n{'='*80}")
    print(f"FORECAST {forecast_steps} PHIÊN TƯƠNG LAI - {symbol}")
    print(f"{'='*80}")

    for step in range(1, forecast_steps + 1):
        past_window = working_df.tail(CONFIG["lookback"]).copy()
        last_date = working_df["time"].iloc[-1]
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        pred_res = predict_next_price(
            past_window=past_window,
            feature_list=CORE_MOMENTUM_FEATURES,
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            lookback=CONFIG["lookback"]
        )

        pred_trend = trend_forecast(past_window)
        regime_info = compute_regime_info(past_window)

        pred_hybrid_raw = regime_info["w_residual"] * pred_res + (1 - regime_info["w_residual"]) * pred_trend
        pred_hybrid, tick_size = apply_tick_rounding(pred_hybrid_raw, working_df, price_col="close")

        prev_close = past_window["close"].iloc[-1]
        pct_change = (pred_hybrid / prev_close - 1) * 100

        results.append({
            "Phiên": step,
            "Ngày": next_date.strftime("%d/%m/%Y"),
            "Giá dự báo": pred_hybrid,
            "Giá thô": pred_hybrid_raw,
            "Bước giá": tick_size,
            "Thay đổi_%": pct_change,
            "Regime": regime_info["regime"],
            "Weight_residual": regime_info["w_residual"],
        })

        # synthetic update
        new_row = working_df.iloc[-1].copy()
        new_row["time"] = next_date
        new_row["open"] = prev_close
        new_row["close"] = pred_hybrid
        new_row["high"] = max(prev_close, pred_hybrid) * 1.002
        new_row["low"] = min(prev_close, pred_hybrid) * 0.998
        new_row["volume"] = working_df["volume"].tail(10).mean()

        working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)
        working_df = compute_features(working_df.tail(300).copy())

    result_df = pd.DataFrame(results)

    print(result_df.to_string(index=False, formatters={
        "Giá dự báo": "{:,.2f}".format,
        "Giá thô": "{:,.4f}".format,
        "Bước giá": "{:.4f}".format,
        "Thay đổi_%": "{:+.2f}".format,
        "Weight_residual": "{:.2f}".format,
    }))

    return result_df

# ============================================================
# BACKTEST
# ============================================================

def backtest_model(symbol: str, backtest_start: str = "2025-01-01"):
    symbol = symbol.upper()
    model, feature_scaler, target_scaler, meta = load_model_package(symbol)

    end_fetch = datetime.now().strftime("%Y-%m-%d")
    df = load_data(symbol, CONFIG["start_date"], end_fetch, CONFIG["interval"])
    df = validate_data(df)
    df = compute_features(df)

    bt_idx = df[df["time"] >= pd.to_datetime(backtest_start)].index.tolist()

    results = []

    for idx in bt_idx:
        past_df = df.loc[:idx-1].copy()
        if len(past_df) < CONFIG["lookback"] + 120:
            continue

        past_window = past_df.tail(CONFIG["lookback"]).copy()
        actual = df.loc[idx, "close"]
        prev_close = past_window["close"].iloc[-1]
        current_date = df.loc[idx, "time"]

        try:
            pred_res_raw = predict_next_price(
                past_window=past_window,
                feature_list=CORE_MOMENTUM_FEATURES,
                model=model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                lookback=CONFIG["lookback"]
            )

            pred_trend_raw = trend_forecast(past_window)
            regime_info = compute_regime_info(past_window)

            pred_hybrid_raw = regime_info["w_residual"] * pred_res_raw + (1 - regime_info["w_residual"]) * pred_trend_raw

            pred_res, _ = apply_tick_rounding(pred_res_raw, past_df, price_col="close")
            pred_trend, _ = apply_tick_rounding(pred_trend_raw, past_df, price_col="close")
            pred_hybrid, tick_size = apply_tick_rounding(pred_hybrid_raw, past_df, price_col="close")

            results.append({
                "Ngày": current_date.strftime("%d/%m/%Y"),
                "Thực tế": actual,
                "Dự báo Residual": pred_res,
                "Dự báo Trend": pred_trend,
                "Dự báo Hybrid": pred_hybrid,
                "Bước giá": tick_size,
                "Sai số Hybrid": pred_hybrid - actual,
                "% Sai Hybrid": abs(pred_hybrid - actual) / actual * 100,
                "Hướng Hybrid": "✔" if ((pred_hybrid > prev_close) == (actual > prev_close)) else "✖",
                "Regime": regime_info["regime"],
            })
        except:
            continue

    bt_df = pd.DataFrame(results)

    mape = bt_df["% Sai Hybrid"].mean()
    mae = bt_df["Sai số Hybrid"].abs().mean()
    diracc = (bt_df["Hướng Hybrid"] == "✔").mean() * 100

    print(f"\n{'='*80}")
    print(f"BACKTEST - {symbol}")
    print(f"{'='*80}")
    print(bt_df.head(30).to_string(index=False, formatters={
        "Thực tế": "{:,.2f}".format,
        "Dự báo Residual": "{:,.2f}".format,
        "Dự báo Trend": "{:,.2f}".format,
        "Dự báo Hybrid": "{:,.2f}".format,
        "Bước giá": "{:.4f}".format,
        "Sai số Hybrid": "{:+.2f}".format,
        "% Sai Hybrid": "{:.2f}".format,
    }))

    print(f"\n📊 BACKTEST SUMMARY")
    print(f"   Sessions: {len(bt_df)}")
    print(f"   MAE:      {mae:.2f}")
    print(f"   MAPE:     {mape:.3f}%")
    print(f"   DirAcc:   {diracc:.1f}%")

    return bt_df

# ============================================================
# MAIN INTERACTIVE PIPELINE
# ============================================================

def main():
    print("\n" + "="*80)
    symbol = input("Nhập mã cổ phiếu (VD: VNINDEX, HPG, VNM): ").strip().upper()
    print("="*80)

    if not model_exists(symbol):
        print(f"⚠️ Chưa có model cho {symbol}. Bắt đầu training...")
        train_model_for_symbol(symbol)
    else:
        print(f"✅ Đã tìm thấy model cho {symbol}")

    while True:
        print(f"\nChọn chức năng cho {symbol}:")
        print("1. Dự báo giá cổ phiếu tương lai")
        print("2. Backtest model")
        print("0. Thoát")

        choice = input("Nhập lựa chọn: ").strip()

        if choice == "1":
            try:
                steps = int(input("Nhập số phiên muốn dự báo (1-30, mặc định 5): ").strip() or "5")
                steps = max(1, min(30, steps))
            except:
                steps = 5

            forecast_future(symbol, forecast_steps=steps)

        elif choice == "2":
            backtest_start = input("Nhập ngày bắt đầu backtest (YYYY-MM-DD, mặc định 2025-01-01): ").strip()
            if backtest_start == "":
                backtest_start = "2025-01-01"

            backtest_model(symbol, backtest_start=backtest_start)

        elif choice == "0":
            print("👋 Thoát chương trình.")
            break

        else:
            print("❌ Lựa chọn không hợp lệ, vui lòng thử lại.")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()