# ============================================================
# a_dashboard.py — Dashboard terminal-style v3
# Bổ sung: giá trị chỉ báo + delta so hôm qua + trạng thái
# ML thuần túy (không hybrid)
# ============================================================

import os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta

import talib as ta
from talipp.indicators import ZLEMA as TalippZLEMA

from pyecharts import options as opts
from pyecharts.charts import Candlestick, Line, Bar, Grid
from pyecharts.commons.utils import JsCode

from a_chibao2 import _get_data_with_indicators
from a_ML3 import (model_exists, train_model_for_symbol,
                   forecast_future, ensure_model, CONFIG as ML_CONFIG)
from playwright.sync_api import sync_playwright


# ============================================================
# HELPERS
# ============================================================

def get_next_business_day(date):
    nxt = date + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def get_previous_business_day(date):
    prev = date - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def html_to_image3(html_file: str, png_file: str = None,
                  width: int = 1600) -> str:
    if png_file is None:
        png_file = html_file.replace(".html", ".png")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page    = context.new_page()
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(f"file:///{os.path.abspath(html_file)}", wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.screenshot(path=png_file, full_page=True)
        browser.close()
    print(f"✅ Đã lưu ảnh: {png_file}")
    return png_file


def compute_fibonacci_levels(df, window=120) -> dict:
    recent = df.tail(window)
    high_p = float(recent["high"].max())
    low_p  = float(recent["low"].min())
    diff   = high_p - low_p
    return {
        "Fib 100% (Đỉnh)": high_p,
        "Fib 78.6%"       : high_p - 0.214 * diff,
        "Fib 61.8%"       : high_p - 0.382 * diff,
        "Fib 50.0%"       : high_p - 0.500 * diff,
        "Fib 38.2%"       : high_p - 0.618 * diff,
        "Fib 23.6%"       : high_p - 0.764 * diff,
        "Fib 0% (Đáy)"    : low_p,
    }


# ============================================================
# 1. DATA + INDICATORS
# ============================================================

def get_data_with_indicators_and_zlema(symbol: str,
                                       months: int = 6) -> pd.DataFrame:
    data_months = max(months, 14)
    df = _get_data_with_indicators(symbol, data_months)

    if df is None or df.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")

    for c in ("close", "high", "low", "volume"):
        df[c] = df[c].astype(float)

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    if "adx"    not in df.columns:
        df["adx"]  = ta.ADX(high, low, close, timeperiod=14)
    if "mfi"    not in df.columns:
        df["mfi"]  = ta.MFI(high, low, close, volume, timeperiod=14)
    if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
        df["stoch_k"], df["stoch_d"] = ta.STOCH(high, low, close)
    if "sar"    not in df.columns:
        df["sar"]  = ta.SAR(high, low, acceleration=0.02, maximum=0.2)

    # ZLEMA period từ ML_CONFIG (đồng bộ với model)
    zlema_period = ML_CONFIG.get("zlema_period", 5)
    try:
        z = TalippZLEMA(period=zlema_period)
        z.update(close.tolist())
        z_vals = [float(x) if x is not None else np.nan for x in z]
        df["zlema_val"] = pd.Series(z_vals, index=df.index)
    except Exception:
        df["zlema_val"] = ta.EMA(close, timeperiod=zlema_period)

    df["residual"]        = df["close"] - df["zlema_val"]
    df["residual_mean20"] = df["residual"].rolling(20).mean()
    df["residual_std20"]  = df["residual"].rolling(20).std()
    df["residual_z"]      = df["residual"] / (df["residual_std20"] + 1e-9)
    df["tb50"]            = ta.MA(close, timeperiod=50)
    df["tb200"]           = ta.MA(close, timeperiod=200)
    df["vol_ratio"]       = volume / volume.rolling(20).mean()
    # ===== SLOPE =====
    z = df["zlema_val"]
    df["slope_5"] = (
        (z - z.shift(5))
        / z.shift(5)
    ) * 100

    df = df.replace([np.inf, -np.inf], np.nan)
    req = [c for c in ("close", "zlema_val", "ma20", "rsi",
                        "macd", "signal", "volume", "slope_5") if c in df.columns]
    df  = df.dropna(subset=req).copy()

    if len(df) > 0:
        cutoff = df.index[-1] - pd.DateOffset(months=months)
        dfd    = df[df.index >= cutoff].copy()
        if len(dfd) > 20:
            df = dfd

    if df.empty:
        raise ValueError("DataFrame rỗng sau khi xử lý indicators")
    return df


# ============================================================
# 2. SIGNALS — Giá trị + Delta + Trạng thái
# ============================================================

def _delta_str(val: float, prev: float, unit: str = "",
               decimals: int = 2) -> str:
    """Chuỗi delta có màu: +X.XX hoặc -X.XX"""
    d = val - prev
    if abs(d) < 1e-6:
        return f'<span style="color:#8b949e">0{unit}</span>'
    color = "#26a69a" if d > 0 else "#ef5350"
    sign  = "+" if d > 0 else ""
    return f'<span style="color:{color}">{sign}{d:.{decimals}f}{unit}</span>'


def _status_tag(text: str, color: str) -> str:
    return f'<span style="color:{color};font-weight:700">{text}</span>'


def generate_signals_v2(latest: pd.Series,
                        prev: pd.Series) -> list:
    """
    Trả về list of dict:
    {name, value_str, delta_str, delta_raw, status_text, status_color}
    delta_raw dùng để hiển thị mũi tên xu hướng ngắn hạn của chỉ báo
    """
    rows = []
    close = float(latest["close"])

    def _safe(series, key, default=0.0):
        try:
            v = series[key]
            return float(v) if pd.notna(v) else default
        except Exception:
            return default

    # ── RSI ──────────────────────────────────────────────────
    rsi, rsi_p = _safe(latest,"rsi",50), _safe(prev,"rsi",50)
    d = rsi - rsi_p
    if   rsi > 70: st, sc = "QUÁ MUA", "#ff6b6b"
    elif rsi > 55: st, sc = "THIÊN TĂNG", "#26a69a"
    elif rsi < 30: st, sc = "QUÁ BÁN", "#26a69a"
    elif rsi < 45: st, sc = "THIÊN GIẢM", "#ff6b6b"
    else:          st, sc = "TRUNG TÍNH", "#f4d35e"
    rows.append({
        "name": "RSI(14)", "value_str": f"{rsi:.1f}",
        "delta_str": _delta_str(rsi, rsi_p, decimals=1),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── MACD Hist ─────────────────────────────────────────────
    mh  = _safe(latest, "macd_hist",
                _safe(latest,"macd") - _safe(latest,"signal"))
    mh_p = _safe(prev, "macd_hist",
                 _safe(prev,"macd") - _safe(prev,"signal"))
    d = mh - mh_p
    if mh > 0 and d > 0: st, sc = "TĂNG MẠNH", "#26a69a"
    elif mh > 0:          st, sc = "TĂNG", "#26a69a"
    elif mh < 0 and d < 0: st, sc = "GIẢM MẠNH", "#ff6b6b"
    else:                  st, sc = "GIẢM", "#ff6b6b"
    rows.append({
        "name": "MACD Hist", "value_str": f"{mh:.2f}",
        "delta_str": _delta_str(mh, mh_p, decimals=2),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── ADX ───────────────────────────────────────────────────
    adx, adx_p = _safe(latest,"adx",20), _safe(prev,"adx",20)
    d = adx - adx_p
    if   adx > 40: st, sc = "XU HƯỚNG MẠNH", "#ff6b6b"
    elif adx > 25: st, sc = "CÓ XU HƯỚNG", "#26a69a"
    elif adx > 15: st, sc = "YẾU", "#f4d35e"
    else:          st, sc = "SIDEWAY", "#8b949e"
    rows.append({
        "name": "ADX(14)", "value_str": f"{adx:.1f}",
        "delta_str": _delta_str(adx, adx_p, decimals=1),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── MFI ───────────────────────────────────────────────────
    mfi, mfi_p = _safe(latest,"mfi",50), _safe(prev,"mfi",50)
    d = mfi - mfi_p
    if   mfi > 80: st, sc = "QUÁ MUA", "#ff6b6b"
    elif mfi > 55: st, sc = "DÒNG TIỀN VÀO", "#26a69a"
    elif mfi < 20: st, sc = "QUÁ BÁN", "#26a69a"
    elif mfi < 45: st, sc = "DÒNG TIỀN RA", "#ff6b6b"
    else:          st, sc = "TRUNG TÍNH", "#f4d35e"
    rows.append({
        "name": "MFI(14)", "value_str": f"{mfi:.1f}",
        "delta_str": _delta_str(mfi, mfi_p, decimals=1),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── Stoch %K ─────────────────────────────────────────────
    sk, sk_p = _safe(latest,"stoch_k",50), _safe(prev,"stoch_k",50)
    sd        = _safe(latest,"stoch_d",50)
    d = sk - sk_p
    if   sk > 80 and d < 0: st, sc = "QUÁ MUA – ĐẢO CHIỀU", "#ff6b6b"
    elif sk < 20 and d > 0: st, sc = "QUÁ BÁN – ĐẢO CHIỀU", "#26a69a"
    elif sk > sd:            st, sc = "TĂNG", "#26a69a"
    else:                    st, sc = "GIẢM", "#ff6b6b"
    rows.append({
        "name": "Stoch %K", "value_str": f"{sk:.1f}",
        "delta_str": _delta_str(sk, sk_p, decimals=1),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── Parabolic SAR ─────────────────────────────────────────
    sar, sar_p = _safe(latest,"sar", close), _safe(prev,"sar", close)
    d = sar - sar_p
    if close > sar: st, sc = "UPTREND", "#26a69a"
    else:            st, sc = "DOWNTREND", "#ff6b6b"
    rows.append({
        "name": "Parabolic SAR", "value_str": f"{sar:,.2f}",
        "delta_str": _delta_str(sar, sar_p, decimals=2),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── BBANDS ───────────────────────────────────────────────
    upper  = _safe(latest,"upper", close*1.02)
    lower  = _safe(latest,"lower", close*0.98)
    upper_p = _safe(prev,"upper", close*1.02)
    lower_p = _safe(prev,"lower", close*0.98)
    bw, bw_p = upper - lower, upper_p - lower_p
    d = bw - bw_p
    if   close > upper: st, sc = "BREAK TRÊN", "#ff6b6b"
    elif close < lower: st, sc = "BREAK DƯỚI", "#26a69a"
    else:               st, sc = "TRONG DẢI", "#f4d35e"
    rows.append({
        "name": "BB Width", "value_str": f"{bw:.2f}",
        "delta_str": _delta_str(bw, bw_p, decimals=2),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── ATR ───────────────────────────────────────────────────
    atr, atr_p = _safe(latest,"atr", close*0.01), _safe(prev,"atr", close*0.01)
    d = atr - atr_p
    atr_r = atr / close if close > 0 else 0
    if atr_r > 0.03:    st, sc = "BIẾN ĐỘNG CAO", "#ff6b6b"
    elif atr_r > 0.015: st, sc = "BÌNH THƯỜNG", "#f4d35e"
    else:               st, sc = "ỔN ĐỊNH", "#26a69a"
    rows.append({
        "name": "ATR(14)", "value_str": f"{atr:.2f}",
        "delta_str": _delta_str(atr, atr_p, decimals=2),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── ZLEMA ────────────────────────────────────────────────
    zlema_col = "zlema_val" if "zlema_val" in latest.index else "zlema20"
    zlema,  zlema_p = _safe(latest, zlema_col, close), _safe(prev, zlema_col, close)
    d = close - zlema   # khoảng cách giá vs ZLEMA quan trọng hơn delta ZLEMA
    if close > zlema: st, sc = "UPTREND", "#26a69a"
    else:              st, sc = "DOWNTREND", "#ff6b6b"
    rows.append({
        "name": f"ZLEMA({ML_CONFIG.get('zlema_period',5)})",
        "value_str": f"{zlema:,.2f}",
        "delta_str": _delta_str(zlema, zlema_p, decimals=2),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    # ── Slope(5) ─────────────────────────────────────────────
    # Đây là chỉ báo quan trọng từ notebook nhưng bị thiếu trong UI cũ
    sl5, sl5_p = _safe(latest,"slope_5",0), _safe(prev,"slope_5",0)
    d = sl5 - sl5_p
    if   sl5 > 2:  st, sc = "TĂNG MẠNH", "#26a69a"
    elif sl5 > 0:  st, sc = "TĂNG", "#26a69a"
    elif sl5 < -2: st, sc = "GIẢM MẠNH", "#ff6b6b"
    else:          st, sc = "GIẢM", "#ff6b6b"
    rows.append({
        "name": "Slope(5)", "value_str": f"{sl5:.2f}",
        "delta_str": _delta_str(sl5, sl5_p, decimals=2),
        "delta_raw": d, "status_text": st, "status_color": sc,
    })

    return rows


# (giữ lại hàm cũ cho phần score)
def generate_signals(latest: pd.Series) -> dict:
    s = {}
    rsi = latest["rsi"]
    if   rsi > 70: s["RSI"] = {"text":"QUÁ MUA",    "arrow":"↑↑","color":"#ff6b6b"}
    elif rsi > 55: s["RSI"] = {"text":"THIÊN TĂNG",  "arrow":"↗", "color":"#26a69a"}
    elif rsi < 30: s["RSI"] = {"text":"QUÁ BÁN",    "arrow":"↓↓","color":"#26a69a"}
    elif rsi < 45: s["RSI"] = {"text":"THIÊN GIẢM",  "arrow":"↘", "color":"#ff6b6b"}
    else:          s["RSI"] = {"text":"TRUNG TÍNH",  "arrow":"→", "color":"#f4d35e"}

    zlema_col = "zlema_val" if "zlema_val" in latest.index else "zlema20"
    if latest["macd"] > latest["signal"]:
        s["MACD"] = {"text":"TĂNG", "arrow":"↑","color":"#26a69a"}
    else:
        s["MACD"] = {"text":"GIẢM", "arrow":"↓","color":"#ff6b6b"}

    if latest["close"] > latest["sar"]:
        s["SAR"] = {"text":"UPTREND",   "arrow":"↑","color":"#26a69a"}
    else:
        s["SAR"] = {"text":"DOWNTREND", "arrow":"↓","color":"#ff6b6b"}

    if   latest["close"] > latest["upper"]:
        s["BBANDS"] = {"text":"BREAK TRÊN","arrow":"↑↑","color":"#ff6b6b"}
    elif latest["close"] < latest["lower"]:
        s["BBANDS"] = {"text":"BREAK DƯỚI","arrow":"↓↓","color":"#26a69a"}
    else:
        s["BBANDS"] = {"text":"TRONG DẢI", "arrow":"→", "color":"#f4d35e"}

    if latest["atr"] / latest["close"] > 0.03:
        s["ATR"] = {"text":"BIẾN ĐỘNG CAO","arrow":"⚡","color":"#ff6b6b"}
    else:
        s["ATR"] = {"text":"ỔN ĐỊNH",       "arrow":"→","color":"#f4d35e"}

    if latest["close"] > latest[zlema_col]:
        s["ZLEMA"] = {"text":"UPTREND",   "arrow":"↑","color":"#26a69a"}
    else:
        s["ZLEMA"] = {"text":"DOWNTREND", "arrow":"↓","color":"#ff6b6b"}

    return s


# ============================================================
# 3. SCORE ENGINE
# ============================================================

def compute_dashboard_scores(df: pd.DataFrame) -> dict:
    latest    = df.iloc[-1]
    zlema_col = "zlema_val" if "zlema_val" in latest.index else "zlema20"

    trend = 50
    if latest["close"] > latest[zlema_col]:              trend += 15
    if latest["ma9"]   > latest["ma20"]:                 trend += 15
    if pd.notna(latest.get("tb50")) and latest["close"] > latest["tb50"]: trend += 10
    if latest["adx"]   > 25:                             trend += 10
    trend = min(100, max(0, trend))

    momentum = 50
    if latest["macd"]    > latest["signal"]:  momentum += 15
    if latest["rsi"]     > 50:                momentum += 15
    if latest["stoch_k"] > latest["stoch_d"]: momentum += 10
    momentum = min(100, max(0, momentum))

    money = 50
    if latest["mfi"]       > 50: money += 20
    if latest["vol_ratio"] > 1:  money += 15
    money = min(100, max(0, money))

    atr_r     = latest["atr"] / latest["close"]
    stability = min(100, max(0, 100 - atr_r * 2000))

    cycle = 50
    if   latest["rsi"] < 35: cycle += 20
    elif latest["rsi"] > 70: cycle -= 10
    if latest["close"] > latest["sar"]: cycle += 15
    cycle = min(100, max(0, cycle))

    health = int(.30*trend + .22*momentum + .15*money
                + .18*stability + .15*cycle)
    return dict(trend=round(trend), momentum=round(momentum),
                money=round(money), stability=round(stability),
                cycle=round(cycle), health=round(health))


# ============================================================
# 4. FORECAST INTERPRETATION
# ============================================================

def interpret_forecast(forecast_df: pd.DataFrame) -> dict:
    if forecast_df is None or len(forecast_df) == 0:
        return dict(trend_5d="KHÔNG XÁC ĐỊNH", change_5d=0,
                    recommendation="CHỜ", prob_up=50)

    # Tìm tên cột giá linh hoạt
    price_col = next(
        (c for c in forecast_df.columns
         if "giá" in c.lower() or "price" in c.lower() or "dự báo" in c.lower()),
        forecast_df.columns[0]
    )

    fp = float(forecast_df[price_col].iloc[0])
    lp = float(forecast_df[price_col].iloc[-1])
    ch = (lp / fp - 1) * 100

    if   ch >  1: trend, rec, pu = "TĂNG",     "ƯU TIÊN TĂNG",    min(85, 55 + abs(ch)*5)
    elif ch < -1: trend, rec, pu = "GIẢM",     "THẬN TRỌNG",      max(15, 45 - abs(ch)*5)
    else:         trend, rec, pu = "TÍCH LŨY", "CHỜ & QUAN SÁT",  50 + ch * 3

    return dict(trend_5d=trend, change_5d=round(ch, 2),
                recommendation=rec,
                prob_up=int(np.clip(pu, 5, 95)))


# ============================================================
# 5. CHART
# ============================================================

def build_main_chart_fragment(df: pd.DataFrame,
                               forecast_df: pd.DataFrame,
                               symbol: str) -> str:
    dates = df.index.strftime("%Y-%m-%d").tolist()
    ohlc  = [[round(float(r["open"]),2), round(float(r["close"]),2),
               round(float(r["low"]),2),  round(float(r["high"]),2)]
              for _, r in df.iterrows()]
    volumes = df["volume"].tolist()

    fc_dates, fc_prices = [], []
    if forecast_df is not None and len(forecast_df) > 0:
        for _, r in forecast_df.iterrows():
            fd = pd.to_datetime(r["Ngày"], format="%d/%m/%Y").strftime("%Y-%m-%d")
            fc_dates.append(fd)
            fc_prices.append(round(float(r["Giá dự báo"]), 2))

    all_dates = dates + fc_dates
    n_hist, n_fc = len(dates), len(fc_dates)
    ohlc_ext  = ohlc  # Dùng mảng không padding (chuẩn a_chibao2)
    vol_ext   = volumes + [0]*n_fc

    vol_colors = (["#26a69a" if d[1] >= d[0] else "#ef5350" for d in ohlc]
                  + ["#444"]*n_fc)

    total_pts   = len(all_dates)
    show_pts    = min(80, total_pts)
    range_start = max(0, int((1 - show_pts / total_pts) * 100))

    candle = (
        Candlestick()
        .add_xaxis(all_dates)
        .add_yaxis(
            "OHLC", ohlc_ext,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#26a69a", color0="#ef5350",
                border_color="#26a69a", border_color0="#ef5350"),
            markpoint_opts=opts.MarkPointOpts(
                data=[opts.MarkPointItem(type_="max", name="Đỉnh"),
                      opts.MarkPointItem(type_="min", name="Đáy")],
                symbol_size=40,
                label_opts=opts.LabelOpts(font_size=10, color="#fff")),
        )
    )

    zlema_col = "zlema_val" if "zlema_val" in df.columns else "zlema20"

    def _line(name, vals, color, width=1, dash=False):
        ext = vals + [None]*n_fc
        return (
            Line()
            .add_xaxis(all_dates)
            .add_yaxis(name, ext, is_symbol_show=False,
                       linestyle_opts=opts.LineStyleOpts(
                           color=color, width=width,
                           type_="dashed" if dash else "solid"),
                       label_opts=opts.LabelOpts(is_show=False))
        )

    candle = candle.overlap(
        _line(f"ZLEMA({ML_CONFIG.get('zlema_period',5)})",
              df[zlema_col].round(2).tolist(), "#f4d35e", 2, True)
    )

    if n_fc > 0:
        pred_ser = [None]*n_hist + fc_prices
        atr_val  = float(df["atr"].iloc[-1])
        ci_upper = ([None]*(n_hist - 1) + [float(df["close"].iloc[-1])] +
                    [round(p + 0.5*atr_val*(1+i*0.1), 2) for i, p in enumerate(fc_prices)])
        ci_lower = ([None]*(n_hist - 1) + [float(df["close"].iloc[-1])] +
                    [round(p - 0.5*atr_val*(1+i*0.1), 2) for i, p in enumerate(fc_prices)])

        fc_line = (
            Line()
            .add_xaxis(all_dates)
            .add_yaxis("🔮 Dự báo", pred_ser, is_connect_nones=False,
                       is_symbol_show=True, symbol="diamond", symbol_size=7,
                       linestyle_opts=opts.LineStyleOpts(
                           color="#00e5ff", width=2.5, type_="dashed"),
                       itemstyle_opts=opts.ItemStyleOpts(color="#00e5ff"),
                       label_opts=opts.LabelOpts(is_show=False))
        )
        ci_up_l = (Line().add_xaxis(all_dates)
                   .add_yaxis("CI Upper", ci_upper, is_symbol_show=False,
                              linestyle_opts=opts.LineStyleOpts(
                                  color="rgba(0,229,255,0.25)", width=1,
                                  type_="dotted"),
                              label_opts=opts.LabelOpts(is_show=False)))
        ci_lo_l = (Line().add_xaxis(all_dates)
                   .add_yaxis("CI Lower", ci_lower, is_symbol_show=False,
                              linestyle_opts=opts.LineStyleOpts(
                                  color="rgba(0,229,255,0.25)", width=1,
                                  type_="dotted"),
                              label_opts=opts.LabelOpts(is_show=False)))
        candle = candle.overlap(fc_line).overlap(ci_up_l).overlap(ci_lo_l)

    candle.set_global_opts(
        # title_opts=opts.TitleOpts(
        #     title=f"  {symbol}  —  BIỂU ĐỒ GIÁ & DỰ BÁO",
        #     pos_left="center",
        #     title_textstyle_opts=opts.TextStyleOpts(
        #         color="#e6e6e6", font_size=15)),
        legend_opts=opts.LegendOpts(
            pos_top="32px", pos_left="3%",
            textstyle_opts=opts.TextStyleOpts(color="#8b949e", font_size=11)),
        xaxis_opts=opts.AxisOpts(
            type_="category", is_show=False,
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(color="#30363d")),
            splitline_opts=opts.SplitLineOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(
            is_scale=True, position="right",
            axislabel_opts=opts.LabelOpts(
                color="#c9d1d9",
                formatter=JsCode("function(v){return v.toLocaleString('en');}")),
            splitline_opts=opts.SplitLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(
                    color="#21262d", type_="dashed"))),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross",
            background_color="rgba(13,17,23,0.92)",
            border_color="#30363d",
            textstyle_opts=opts.TextStyleOpts(color="#e6e6e6", font_size=12)),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1],
                              range_start=range_start, range_end=100),
            opts.DataZoomOpts(type_="slider", xaxis_index=[0, 1],
                              pos_bottom="2%",
                              range_start=range_start, range_end=100),
        ],
    )

    vol_items = [
        opts.BarItem(name=all_dates[i], value=vol_ext[i],
                     itemstyle_opts=opts.ItemStyleOpts(color=vol_colors[i]))
        for i in range(len(all_dates))
    ]
    vol_bar = (
        Bar()
        .add_xaxis(all_dates)
        .add_yaxis("Volume", vol_items,
                   label_opts=opts.LabelOpts(is_show=False),
                   itemstyle_opts=opts.ItemStyleOpts(opacity=0.6))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                axislabel_opts=opts.LabelOpts(
                    color="#8b949e", rotate=45, font_size=9),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#30363d")),
                splitline_opts=opts.SplitLineOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(
                position="right", is_show=False,
                axislabel_opts=opts.LabelOpts(
                    color="#484f58", font_size=9,
                    formatter=JsCode(
                        "function(v){if(v>=1e6)return (v/1e6).toFixed(1)+'M';"
                        "if(v>=1e3)return (v/1e3).toFixed(0)+'K';return v;}")),
                splitline_opts=opts.SplitLineOpts(is_show=False)),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    grid = (
        Grid(init_opts=opts.InitOpts(
            width="100%", height="560px",
            bg_color="transparent",
            renderer="svg",
            animation_opts=opts.AnimationOpts(animation=False)))
        .add(candle,
             grid_opts=opts.GridOpts(pos_left="6%", pos_right="6%",
                                     pos_top="60px", pos_bottom="185px"),
             is_control_axis_index=True)
        .add(vol_bar,
             grid_opts=opts.GridOpts(pos_left="6%", pos_right="6%",
                                     pos_top="420px", pos_bottom="95px"))
    )
    return grid.render_embed()


# ============================================================
# 6. SIGNAL TABLE HTML — Mới (Giá trị + Delta + Trạng thái)
# ============================================================

def build_signal_table_html(signal_rows: list) -> str:
    header = """
    <table class="sig-tbl">
      <thead>
        <tr>
          <th style="width:28%">Chỉ báo</th>
          <th style="width:18%;text-align:right">Giá trị</th>
          <th style="width:22%;text-align:center">vs Hôm qua</th>
          <th style="width:32%;text-align:right">Trạng thái</th>
        </tr>
      </thead>
      <tbody>"""

    body = ""
    for r in signal_rows:
        # Mũi tên ngắn gọn theo delta_raw
        d = r.get("delta_raw", 0)
        if   d >  0.5: arrow = '<span style="color:#26a69a;font-size:15px">▲</span>'
        elif d < -0.5: arrow = '<span style="color:#ef5350;font-size:15px">▼</span>'
        else:          arrow = '<span style="color:#8b949e;font-size:15px">▬</span>'

        body += f"""
        <tr>
          <td class="sig-name">{r['name']}</td>
          <td style="text-align:right;font-weight:700;color:#f0f6fc;font-size:13px">
            {r['value_str']}</td>
          <td style="text-align:center">
            {arrow}&nbsp;<span style="font-size:12px">{r['delta_str']}</span>
          </td>
          <td style="text-align:right;color:{r['status_color']};
                     font-weight:700;font-size:12px;white-space:nowrap">
            {r['status_text']}
          </td>
        </tr>"""

    return header + body + "</tbody></table>"


# ============================================================
# 7. MAIN DASHBOARD
# ============================================================

def create_terminal_dashboard3(symbol: str,
                               months: int = 6,
                               forecast_steps: int = 5) -> str:
    symbol = symbol.upper()

    # Ensure model (auto-retrain nếu cũ)
    ensure_model(symbol)

    forecast_df = forecast_future(symbol, forecast_steps=forecast_steps)
    df          = get_data_with_indicators_and_zlema(symbol, months)

    latest     = df.iloc[-1]
    prev       = df.iloc[-2]          # hôm qua
    signals    = generate_signals(latest)
    sig_rows   = generate_signals_v2(latest, prev)
    today_dt   = df.index[-1]
    today      = today_dt.strftime("%d/%m/%Y")
    next_bday  = get_next_business_day(today_dt)
    tomorrow_str = next_bday.strftime("%d/%m/%Y")

    scores     = compute_dashboard_scores(df)
    fc_info    = interpret_forecast(forecast_df)
    chart_html = build_main_chart_fragment(df, forecast_df, symbol)
    sig_tbl    = build_signal_table_html(sig_rows)
    latest_atr = float(latest["atr"])

    zlema_col  = "zlema_val" if "zlema_val" in latest.index else "zlema20"

    # TB200
    tb200_val = (float(latest["tb200"])
                 if "tb200" in df.columns and pd.notna(latest["tb200"])
                 else None)
    tb50_val  = (float(latest["tb50"])
                 if "tb50" in df.columns and pd.notna(latest["tb50"])
                 else None)

    fibs         = compute_fibonacci_levels(df, window=120)
    current_price = float(latest["close"])
    fib_sorted   = sorted(fibs.items(), key=lambda x: x[1])
    fib_below    = [(n, v) for n, v in fib_sorted if v <= current_price]
    fib_above    = [(n, v) for n, v in fib_sorted if v >  current_price]
    support_1    = fib_below[-1] if len(fib_below) >= 1 else fib_sorted[0]
    support_2    = fib_below[-2] if len(fib_below) >= 2 else fib_sorted[0]
    resist_1     = fib_above[0]  if len(fib_above) >= 1 else fib_sorted[-1]
    resist_2     = fib_above[1]  if len(fib_above) >= 2 else fib_sorted[-1]

    fib_rows_html = ""
    for name, val in sorted(fibs.items(), key=lambda x: x[1], reverse=True):
        is_nearest = (abs(val - current_price) ==
                      min(abs(v - current_price) for v in fibs.values()))
        is_above   = val > current_price
        color = "#ef5350" if is_above else "#26a69a"
        arrow = "▼" if is_above else "▲"
        if is_nearest:
            fib_rows_html += f"""
            <div class="sr-row sr-cur">
              <span><b style="color:#f4d35e">● {name}</b></span>
              <span><b style="color:#f4d35e">{val:,.2f}</b></span>
            </div>"""
        else:
            fib_rows_html += f"""
            <div class="sr-row">
              <span style="color:{color}">{arrow} {name}</span>
              <span>{val:,.2f}</span>
            </div>"""

    sr_cur_row = f"""
    <div class="sr-row sr-cur">
      <span><b style="color:#58a6ff">★ Giá hiện tại</b></span>
      <span><b style="color:#58a6ff">{current_price:,.2f}</b></span>
    </div>"""

    # Forecast table
    fc_rows = ""
    if forecast_df is not None and len(forecast_df) > 0:
        for i, (_, row) in enumerate(forecast_df.iterrows()):
            price = float(row["Giá dự báo"])
            chg   = float(row["Thay đổi_%"])
            ci_h  = 0.5 * latest_atr * (1 + i * 0.1)
            c_col = "#26a69a" if chg >= 0 else "#ef5350"
            c_arr = "↑" if chg > 0.3 else ("↓" if chg < -0.3 else "→")
            fc_rows += f"""
            <tr>
              <td style="font-size:15px;font-weight:600">{row['Ngày']}</td>
              <td style="font-size:17px;font-weight:800">{price:,.2f}</td>
              <td style="font-size:16px;color:{c_col};font-weight:700">
                {c_arr} {chg:+.2f}%</td>
              <td style="font-size:13px;color:#8b949e">
                {price - ci_h:,.2f} – {price + ci_h:,.2f}</td>
            </tr>"""

    # Trend
    t5_icons  = {"TĂNG":"📈","GIẢM":"📉","TÍCH LŨY":"📊","KHÔNG XÁC ĐỊNH":"❓"}
    t5_colors = {"TĂNG":"#26a69a","GIẢM":"#ef5350",
                 "TÍCH LŨY":"#f4d35e","KHÔNG XÁC ĐỊNH":"#8b949e"}
    t_icon  = t5_icons.get(fc_info["trend_5d"], "❓")
    t_color = t5_colors.get(fc_info["trend_5d"], "#8b949e")
    chg5    = fc_info["change_5d"] if fc_info["change_5d"] else 0
    chg5_cls = "sub-green" if chg5 >= 0 else "sub-red"
    rec = fc_info["recommendation"]
    if   "TĂNG"       in rec: rc, rb = "#26a69a", "#26a69a"
    elif "THẬN TRỌNG" in rec: rc, rb = "#ef5350", "#ef5350"
    else:                     rc, rb = "#f4d35e", "#f4d35e"

    # Score bars (cũ giữ nguyên)
    def bar_col(v):
        return "#26a69a" if v >= 60 else ("#f4d35e" if v >= 40 else "#ef5350")

    # Signal rows cũ (cho card Xu hướng)
    sig_simple_html = ""
    for name, s in signals.items():
        sig_simple_html += f"""
        <div class="drow">
          <span class="drow-k">{name}</span>
          <span style="color:{s['color']};font-weight:700">
            {s['arrow']} {s['text']}</span>
        </div>"""

    # ── HTML ─────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{symbol} Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:#0d1117;color:#c9d1d9;
  font-family:'Segoe UI','Roboto',sans-serif;line-height:1.5;
}}
.wrap{{max-width:1580px;margin:0 auto;padding:14px 20px}}

.hdr{{
  display:flex;flex-wrap:wrap;align-items:center;
  justify-content:center;gap:12px;
  padding:10px 18px;margin-bottom:14px;
  background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
  border:1px solid #21262d;border-radius:10px;
}}
.hdr-sym{{font-size:22px;font-weight:800;color:#58a6ff}}
.hdr-price{{font-size:20px;font-weight:700}}
.hdr-sep{{color:#30363d}}

.g4{{display:grid;grid-template-columns:1.3fr 1fr 1fr 1fr;
     gap:14px;margin-bottom:14px}}
.g-main{{display:grid;grid-template-columns:2.4fr 1.2fr;
         gap:14px;margin-bottom:14px}}
.g3{{display:grid;grid-template-columns:1.5fr 1fr 1fr;
     gap:14px;margin-bottom:14px}}

.card{{
  background:#161b22;border:1px solid #21262d;
  border-radius:10px;padding:16px;
  transition:border-color .2s;
}}
.card:hover{{border-color:#388bfd44}}
.stitle{{
  text-align:center;font-size:14px;font-weight:700;
  color:#8b949e;text-transform:uppercase;letter-spacing:.8px;
  margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #21262d;
}}

.ms{{margin:6px 0 3px;font-size:12.5px;color:#8b949e}}
.bar-bg{{width:100%;background:#21262d;height:16px;
         border-radius:4px;overflow:hidden}}
.bar-fill{{height:16px;border-radius:4px;transition:width .6s ease}}

.vbox{{font-size:44px;font-weight:800;text-align:center;color:#f0f6fc}}
.sub-green{{color:#26a69a;font-weight:700;text-align:center;margin-top:6px}}
.sub-red{{color:#ef5350;font-weight:700;text-align:center;margin-top:6px}}

.rec-badge{{
  display:inline-block;text-align:center;
  font-size:22px;font-weight:800;
  color:{rc};border:2px solid {rb};border-radius:10px;
  padding:6px 18px;margin-top:14px;
}}
.prob-bg{{position:relative;margin-top:8px}}
.prob-label{{display:flex;justify-content:space-between;
             font-size:10px;color:#484f58;margin-top:3px}}

table{{width:100%;border-collapse:collapse;color:#c9d1d9}}
th{{
  padding:10px 6px;border-bottom:2px solid #30363d;
  color:#8b949e;font-size:12px;text-transform:uppercase;
  text-align:center;letter-spacing:.5px;
}}
td{{padding:10px 6px;border-bottom:1px solid #21262d;text-align:center}}
tr:hover td{{background:#21262d55}}

/* ── Signal table (mới) ── */
.sig-tbl{{width:100%;border-collapse:collapse;font-size:13px}}
.sig-tbl th{{
  padding:8px 6px;border-bottom:2px solid #30363d;
  color:#8b949e;font-size:11px;text-transform:uppercase;
  text-align:left;
}}
.sig-tbl td{{
  padding:9px 6px;border-bottom:1px solid #21262d08;text-align:left;
  vertical-align:middle;
}}
.sig-tbl tr:hover td{{background:#21262d55}}
.sig-name{{color:#8b949e;font-size:12px;white-space:nowrap}}
.sig-val{{font-size:14px;font-weight:700;color:#f0f6fc}}
.sig-delta{{font-size:13px;font-weight:600}}

.sr-row{{
  display:flex;justify-content:space-between;
  padding:6px 4px;border-bottom:1px solid #ffffff08;font-size:13.5px;
}}
.sr-row:hover{{background:#21262d55}}
.sr-cur{{background:#21262d;border-radius:5px;
         padding:8px 6px;margin:4px 0}}

.sc{{padding:11px 12px;border-radius:8px;margin-bottom:9px;border-left:3px solid}}
.sc-up{{border-color:#26a69a;background:rgba(38,166,154,.04)}}
.sc-mid{{border-color:#f4d35e;background:rgba(244,211,94,.04)}}
.sc-dn{{border-color:#ef5350;background:rgba(239,83,80,.04)}}

.drow{{display:flex;justify-content:space-between;
       font-size:13px;margin-bottom:5px}}
.drow-k{{color:#8b949e}}

.green{{color:#26a69a}}.red{{color:#ef5350}}
.yellow{{color:#f4d35e}}.cyan{{color:#58a6ff}}

.foot{{text-align:center;color:#484f58;font-size:11px;padding:10px 0}}

@media(max-width:1100px){{
  .g4{{grid-template-columns:1fr 1fr}}
  .g-main{{grid-template-columns:1fr}}
  .g3{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<div class="wrap">

<!-- HEADER -->
<div class="hdr">
  <span class="hdr-sym">{symbol}</span>
  <span class="hdr-price">{latest['close']:,.2f}</span>
  <span style="color:{'#26a69a' if chg5>=0 else '#ef5350'}">{chg5:+.2f}% (5D)</span>
  <span class="hdr-sep">│</span>
  <span style="color:#8b949e">{today}</span>
  <span class="hdr-sep">│</span>
  <span style="color:{rc}">Phiên tới: {rec}</span>
  <span class="hdr-sep">│</span>
  <span style="color:#484f58;font-size:12px">
    ZLEMA({ML_CONFIG.get('zlema_period',5)}) + Lasso ML</span>
</div>

<!-- TOP 4-COL -->
<div class="g4">

  <div class="card">
    <div class="stitle">📊 Xu Hướng Sắp Tới</div>
    <div style="text-align:center;font-size:52px;margin:8px 0">{t_icon}</div>
    <div style="text-align:center;font-size:26px;font-weight:800;color:{t_color}">
      {fc_info['trend_5d']}</div>
    <div style="text-align:center;margin-top:10px;font-size:15px">
      Dự kiến&nbsp;<b style="color:{t_color}">{chg5:+.2f}%</b>&nbsp;trong 5 phiên
    </div>
    <div style="margin-top:14px;padding-top:10px;border-top:1px solid #21262d">
      {sig_simple_html}
    </div>
  </div>

  <div class="card">
    <div class="stitle">📋 5 Tiêu Chí</div>
    <div class="ms">Xu hướng: <b>{scores['trend']}</b></div>
    <div class="bar-bg"><div class="bar-fill"
      style="width:{scores['trend']}%;background:{bar_col(scores['trend'])}"></div></div>
    <div class="ms">Động lực: <b>{scores['momentum']}</b></div>
    <div class="bar-bg"><div class="bar-fill"
      style="width:{scores['momentum']}%;background:{bar_col(scores['momentum'])}"></div></div>
    <div class="ms">Dòng tiền: <b>{scores['money']}</b></div>
    <div class="bar-bg"><div class="bar-fill"
      style="width:{scores['money']}%;background:{bar_col(scores['money'])}"></div></div>
    <div class="ms">Ổn định: <b>{scores['stability']}</b></div>
    <div class="bar-bg"><div class="bar-fill"
      style="width:{scores['stability']}%;background:{bar_col(scores['stability'])}"></div></div>
    <div class="ms">Chu kỳ: <b>{scores['cycle']}</b></div>
    <div class="bar-bg"><div class="bar-fill"
      style="width:{scores['cycle']}%;background:{bar_col(scores['cycle'])}"></div></div>
  </div>

  <div class="card" style="text-align:center">
    <div class="stitle">📅 Phiên Tiếp Theo</div>
    <div style="font-size:19px;color:#58a6ff;margin-bottom:6px">{tomorrow_str}</div>
    <div class="rec-badge">{rec}</div>
    <div style="margin-top:16px;font-size:14px">
      Xác suất tăng: <b style="color:#26a69a">{fc_info['prob_up']}%</b>
    </div>
    <div class="prob-bg">
      <div class="bar-bg" style="margin-top:6px">
        <div class="bar-fill"
          style="width:{fc_info['prob_up']}%;
                 background:linear-gradient(90deg,#ef5350 0%,#f4d35e 50%,#26a69a 100%)">
        </div>
      </div>
      <div class="prob-label">
        <span>Giảm</span><span>Trung tính</span><span>Tăng</span>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="stitle">💰 {symbol}</div>
    <div class="vbox">{latest['close']:,.2f}</div>
    <div class="{chg5_cls}" style="font-size:17px">{chg5:+.2f}% / 5 ngày</div>
    <div style="margin-top:14px;border-top:1px solid #21262d;padding-top:10px">
      <div class="drow"><span class="drow-k">High</span>
        <span class="green">{latest['high']:,.2f}</span></div>
      <div class="drow"><span class="drow-k">Low</span>
        <span class="red">{latest['low']:,.2f}</span></div>
      <div class="drow"><span class="drow-k">Volume</span>
        <span>{latest['volume']:,.0f}</span></div>
      <div class="drow"><span class="drow-k">ATR(14)</span>
        <span class="yellow">{latest_atr:,.2f}</span></div>
    </div>
  </div>
</div>

<!-- MAIN ROW: CHART + FORECAST -->
<div style="display:flex; gap:14px; margin-bottom:14px; align-items:stretch;">
  <!-- CHART -->
  <div class="card" style="flex:2; padding:8px; min-width:0;">
    <div style="
        text-align:center;
        font-size:18px;
        font-weight:800;
        color:#e6e6e6;
        margin:14px 0 12px 0;
        letter-spacing:.5px;
    ">
        📈 {symbol} — BIỂU ĐỒ GIÁ & DỰ BÁO
    </div>
    {chart_html}
  </div>

  <!-- FORECAST -->
  <div class="card" style="flex:1; display:flex; flex-direction:column; min-width:320px;">
    <div class="stitle" style="font-size:16px">
      🔮 Dự Báo {forecast_steps} Phiên
    </div>
    <div style="flex:1;overflow-y:auto">
      <table>
        <thead>
          <tr>
            <th>Ngày</th><th>Giá</th><th>%</th><th>Khoảng tin cậy</th>
          </tr>
        </thead>
        <tbody>{fc_rows}</tbody>
      </table>
    </div>
    <div style="margin-top:12px;font-size:11px;color:#484f58;text-align:center;
                border-top:1px solid #21262d;padding-top:10px">
      * Khoảng tin cậy ≈ Giá ± 0.5 × ATR ({latest_atr:,.2f})
    </div>
  </div>
</div>

<!-- BOTTOM 3-COL -->
<div class="g3">
  <div class="card">
    <div class="stitle">📐 Fibonacci Retracement (120 phiên)</div>
    {fib_rows_html}
    {sr_cur_row}
  </div>

  <div class="card">
    <div class="stitle">🎯 3 Kịch Bản</div>
    <div class="sc sc-up">
      <b class="green">↑ TĂNG</b><br/>
      <span style="font-size:13px">Nếu vượt
        <b>{resist_1[1]:,.2f}</b> ({resist_1[0]})</span><br/>
      <span style="font-size:11.5px;color:#8b949e">
        → Kháng cự tiếp: {resist_2[1]:,.2f} ({resist_2[0]})</span>
    </div>
    <div class="sc sc-mid">
      <b class="yellow">→ TÍCH LŨY</b><br/>
      <span style="font-size:13px">Giữa
        <b>{support_1[1]:,.2f}</b> – <b>{resist_1[1]:,.2f}</b></span><br/>
      <span style="font-size:11.5px;color:#8b949e">→ Chờ breakout</span>
    </div>
    <div class="sc sc-dn">
      <b class="red">↓ GIẢM</b><br/>
      <span style="font-size:13px">Nếu thủng
        <b>{support_1[1]:,.2f}</b> ({support_1[0]})</span><br/>
      <span style="font-size:11.5px;color:#8b949e">
        → Hỗ trợ tiếp: {support_2[1]:,.2f} ({support_2[0]})</span>
    </div>
  </div>

  <!-- ★ TÍN HIỆU CHỈ BÁO MỚI — Giá trị + Delta + Trạng thái -->
  <div class="card">
    <div class="stitle">📡 Tín Hiệu Chỉ Báo</div>
    {sig_tbl}
  </div>
</div>

<!-- FOOTER -->
<div class="foot">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} •
  Model: ZLEMA({ML_CONFIG.get('zlema_period',5)}) + Lasso •
  Retrain mỗi {ML_CONFIG.get('retrain_every',5)} ngày •
  Dữ liệu chỉ mang tính tham khảo, không phải khuyến nghị đầu tư
</div>

</div>
<script>
window.addEventListener('resize', function(){{
  document.querySelectorAll('[_echarts_instance_]').forEach(function(el){{
    var inst = echarts.getInstanceByDom(el);
    if(inst) inst.resize();
  }});
}});
setTimeout(function(){{ window.dispatchEvent(new Event('resize')); }}, 200);
</script>
</body>
</html>"""

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"terminal_dashboard_{symbol}_{ts}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Dashboard: {fname}")
    return fname


# ============================================================
# RUN
# ============================================================
# fname = create_terminal_dashboard3("TCB", months=6, forecast_steps=5)
# html_to_image3(fname)