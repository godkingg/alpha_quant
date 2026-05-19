# ============================================================
# a_dashboard_terminal.py
# Dashboard terminal-style — FIXED VERSION v2
# ============================================================

import os, json, numpy as np, pandas as pd
from datetime import datetime, timedelta

import talib as ta
from talipp.indicators import ZLEMA as TalippZLEMA

from pyecharts import options as opts
from pyecharts.charts import Candlestick, Line, Bar, Grid
from pyecharts.commons.utils import JsCode

from a_chibao2 import _get_data_with_indicators
from a_ML2 import model_exists, train_model_for_symbol, forecast_future
from playwright.sync_api import sync_playwright


# ============================================================
# HELPER — Ngày giao dịch tiếp theo (bỏ qua T7/CN)
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

def compute_fibonacci_levels(df, window=100):
    # Lấy giá cao nhất và thấp nhất trong khoảng 'window' phiên gần nhất
    recent_df = df.tail(window)
    high_p = float(recent_df['high'].max())
    low_p = float(recent_df['low'].min())
    diff = high_p - low_p
    
    # Tính các mức Fibonacci Retracement
    levels = {
        "Fib 100.0% (Đỉnh)": high_p,
        "Fib 78.6%": high_p - 0.214 * diff,
        "Fib 61.8%": high_p - 0.382 * diff,
        "Fib 50.0%": high_p - 0.500 * diff,
        "Fib 38.2%": high_p - 0.618 * diff,
        "Fib 23.6%": high_p - 0.764 * diff,
        "Fib 0.0% (Đáy)": low_p
    }
    return levels

def html_to_image2(html_file: str, png_file: str = None, width: int = 1600):
    if png_file is None:
        png_file = html_file.replace(".html", ".png")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(f"file:///{os.path.abspath(html_file)}", wait_until="networkidle")
        
        # Chờ echarts tải thư viện và render animation xong
        page.wait_for_timeout(2500)
        
        # Chụp toàn trang
        page.screenshot(path=png_file, full_page=True)
        browser.close()
    
    print(f"✅ Đã lưu ảnh: {png_file}")
    return png_file

# ============================================================
# 1.  DATA + ZLEMA
# ============================================================
def get_data_with_indicators_and_zlema(symbol, months=6):
    data_months = max(months, 14)
    df = _get_data_with_indicators(symbol, data_months)

    if df is None or df.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")

    for c in ("close", "high", "low", "volume"):
        df[c] = df[c].astype(float)

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    if "adx" not in df.columns:
        df["adx"] = ta.ADX(high, low, close, timeperiod=14)
    if "mfi" not in df.columns:
        df["mfi"] = ta.MFI(high, low, close, volume, timeperiod=14)
    if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
        k, d = ta.STOCH(high, low, close)
        df["stoch_k"], df["stoch_d"] = k, d
    if "sar" not in df.columns:
        df["sar"] = ta.SAR(high, low, acceleration=0.02, maximum=0.2)

    try:
        z = TalippZLEMA(20)
        for p in close.tolist():
            z.add(float(p))
        z_vals = [float(x) if x is not None else np.nan for x in z]
        df["zlema20"] = pd.Series(z_vals, index=df.index)
    except Exception as e:
        print("Fallback EMA:", e)
        df["zlema20"] = ta.EMA(close, timeperiod=20)

    df["residual"]        = df["close"] - df["zlema20"]
    df["residual_mean20"] = df["residual"].rolling(20).mean()
    df["residual_std20"]  = df["residual"].rolling(20).std()
    df["residual_z"]      = df["residual"] / (df["residual_std20"] + 1e-9)
    df["tb50"]            = ta.MA(close, timeperiod=50)
    df["tb200"]           = ta.MA(close, timeperiod=200)
    df["vol_ratio"]       = volume / volume.rolling(20).mean()

    df = df.replace([np.inf, -np.inf], np.nan)
    req = [c for c in ("close","zlema20","ma20","rsi","macd","signal","volume")
           if c in df.columns]
    df = df.dropna(subset=req).copy()

    if len(df) > 0:
        cutoff = df.index[-1] - pd.DateOffset(months=months)
        dfd = df[df.index >= cutoff].copy()
        if len(dfd) > 20:
            df = dfd

    if df.empty:
        raise ValueError("DataFrame rỗng sau khi xử lý indicators")
    return df


# ============================================================
# Signals
# ============================================================
def generate_signals(latest: pd.Series) -> dict:
    s = {}

    rsi = latest["rsi"]
    if   rsi > 70: s["RSI"] = {"text":"QUÁ MUA",    "arrow":"↑↑","color":"#ff6b6b"}
    elif rsi > 55: s["RSI"] = {"text":"THIÊN TĂNG",  "arrow":"↗", "color":"#00d084"}
    elif rsi < 30: s["RSI"] = {"text":"QUÁ BÁN",    "arrow":"↓↓","color":"#00d084"}
    elif rsi < 45: s["RSI"] = {"text":"THIÊN GIẢM",  "arrow":"↘", "color":"#ff6b6b"}
    else:          s["RSI"] = {"text":"TRUNG TÍNH",  "arrow":"→", "color":"#f4d35e"}

    if latest["macd"] > latest["signal"]:
        s["MACD"] = {"text":"TĂNG",  "arrow":"↑","color":"#00d084"}
    else:
        s["MACD"] = {"text":"GIẢM",  "arrow":"↓","color":"#ff6b6b"}

    if latest["close"] > latest["sar"]:
        s["SAR"] = {"text":"UPTREND",  "arrow":"↑","color":"#00d084"}
    else:
        s["SAR"] = {"text":"DOWNTREND","arrow":"↓","color":"#ff6b6b"}

    if   latest["close"] > latest["upper"]:
        s["BBANDS"] = {"text":"BREAK TRÊN","arrow":"↑↑","color":"#ff6b6b"}
    elif latest["close"] < latest["lower"]:
        s["BBANDS"] = {"text":"BREAK DƯỚI","arrow":"↓↓","color":"#00d084"}
    else:
        s["BBANDS"] = {"text":"TRONG DẢI", "arrow":"→", "color":"#f4d35e"}

    if latest["atr"]/latest["close"] > 0.03:
        s["ATR"] = {"text":"BIẾN ĐỘNG CAO","arrow":"⚡","color":"#ff6b6b"}
    else:
        s["ATR"] = {"text":"ỔN ĐỊNH",      "arrow":"→", "color":"#f4d35e"}

    if latest["close"] > latest["zlema20"]:
        s["ZLEMA"] = {"text":"UPTREND",  "arrow":"↑","color":"#00d084"}
    else:
        s["ZLEMA"] = {"text":"DOWNTREND","arrow":"↓","color":"#ff6b6b"}

    return s


# ============================================================
# 2.  SCORE ENGINE
# ============================================================
def compute_dashboard_scores(df: pd.DataFrame):
    latest = df.iloc[-1]

    trend = 50
    if latest["close"] > latest["zlema20"]:              trend += 15
    if latest["ma9"]  > latest["ma20"]:                  trend += 15
    if pd.notna(latest.get("tb50")) and latest["close"] > latest["tb50"]: trend += 10
    if latest["adx"]  > 25:                              trend += 10
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

    atr_r = latest["atr"] / latest["close"]
    stability = min(100, max(0, 100 - atr_r * 2000))

    cycle = 50
    if   latest["rsi"] < 35: cycle += 20
    elif latest["rsi"] > 70: cycle -= 10
    if latest["close"] > latest["sar"]: cycle += 15
    cycle = min(100, max(0, cycle))

    health = int(.30*trend + .22*momentum + .15*money + .18*stability + .15*cycle)
    return dict(trend=round(trend), momentum=round(momentum),
                money=round(money), stability=round(stability),
                cycle=round(cycle), health=round(health))


# ============================================================
# 3.  FORECAST INTERPRETATION
# ============================================================
def interpret_forecast(forecast_df):
    if forecast_df is None or len(forecast_df) == 0:
        return dict(trend_5d="KHÔNG XÁC ĐỊNH", change_5d=0,
                    recommendation="CHỜ", prob_up=50)

    fp = float(forecast_df["Giá dự báo"].iloc[0])
    lp = float(forecast_df["Giá dự báo"].iloc[-1])
    ch = (lp / fp - 1) * 100

    if   ch >  1: trend,rec,pu = "TĂNG",    "ƯU TIÊN TĂNG", min(85,55+abs(ch)*5)
    elif ch < -1: trend,rec,pu = "GIẢM",    "THẬN TRỌNG",   max(15,45-abs(ch)*5)
    else:         trend,rec,pu = "TÍCH LŨY","CHỜ & QUAN SÁT",50+ch*3

    return dict(trend_5d=trend, change_5d=round(ch,2),
                recommendation=rec, prob_up=int(np.clip(pu,5,95)))


# ============================================================
# 4.  CHART  — FIXED: Tách Volume thành Grid riêng
# ============================================================
def build_main_chart_fragment(df: pd.DataFrame,
                              forecast_df: pd.DataFrame,
                              symbol: str):

    dates = df.index.strftime("%Y-%m-%d").tolist()
    ohlc  = []
    for _, r in df.iterrows():
        ohlc.append([round(float(r["open"]),2), round(float(r["close"]),2),
                      round(float(r["low"]),2),  round(float(r["high"]),2)])

    volumes = df["volume"].tolist()

    # --- forecast dates ---
    fc_dates, fc_prices = [], []
    if forecast_df is not None and len(forecast_df) > 0:
        for _, r in forecast_df.iterrows():
            fd = pd.to_datetime(r["Ngày"], format="%d/%m/%Y").strftime("%Y-%m-%d")
            fc_dates.append(fd)
            fc_prices.append(round(float(r["Giá dự báo"]), 2))

    all_dates = dates + fc_dates
    n_hist    = len(dates)
    n_fc      = len(fc_dates)

    # Giữ nguyên độ dài nến thật (logic học lỏm từ a_chibao2.py)
    ohlc_ext = ohlc
    vol_ext  = volumes + [0]*n_fc

    # Volume colours
    vol_colors = []
    for d in ohlc:
        vol_colors.append("#26a69a" if d[1] >= d[0] else "#ef5350")
    vol_colors += ["#444"]*n_fc

    # ========== DataZoom range ==========
    total_pts   = len(all_dates)
    show_pts    = min(80, total_pts)
    range_start = max(0, int((1 - show_pts/total_pts)*100))

    # ========== CANDLE (grid 0) ==========
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

    # ========== Overlay lines trên candle ==========
    def _line(name, vals, color, width=1, dash=False, yaxis_idx=0):
        ext   = vals + [None]*n_fc
        style = opts.LineStyleOpts(color=color, width=width,
                                    type_="dashed" if dash else "solid")
        
        line_chart = (Line()
                .add_xaxis(all_dates)
                .add_yaxis(name, ext,
                           yaxis_index=yaxis_idx,
                           is_symbol_show=False,
                           linestyle_opts=style,
                           label_opts=opts.LabelOpts(is_show=False)))
        
        # CAN THIỆP VÀO OPTIONS CỦA LINE TRƯỚC KHI TRẢ VỀ
        line_chart.options["legend_selected"] = {name: False} 
        return line_chart

    bb_up   = _line("BB Upper", df["upper"].round(2).tolist(), "rgba(156,136,255,0.5)")
    bb_lo   = _line("BB Lower", df["lower"].round(2).tolist(), "rgba(156,136,255,0.5)")
    zlema_l = _line("ZLEMA20",  df["zlema20"].round(2).tolist(), "#f4d35e", 2, True)
    ma20_l  = _line("MA20",     df["ma20"].round(2).tolist(), "#888888", 1)

    # candle = candle.overlap(bb_up).overlap(bb_lo).overlap(zlema_l).overlap(ma20_l)
    candle = candle.overlap(zlema_l)

    if "tb50" in df.columns:
        tb50_l = _line("TB50", df["tb50"].round(2).tolist(), "#4ea8de", 1)
        #candle = candle.overlap(tb50_l)

    # ========== Forecast line ==========
    if n_fc > 0:
        pred_ser = [None]*(n_hist - 1) + [float(df["close"].iloc[-1])] + fc_prices
        fc_line  = (
            Line()
            .add_xaxis(all_dates)
            .add_yaxis("🔮 Dự báo", pred_ser,
                       is_symbol_show=True, symbol="diamond", symbol_size=7,
                       linestyle_opts=opts.LineStyleOpts(
                           color="#00e5ff", width=2.5, type_="dashed"),
                       itemstyle_opts=opts.ItemStyleOpts(color="#00e5ff"),
                       label_opts=opts.LabelOpts(is_show=False))
        )

        atr_val  = float(df["atr"].iloc[-1])
        ci_upper = ([None]*(n_hist - 1) + [float(df["close"].iloc[-1])] +
                    [round(p + 0.5*atr_val*(1+i*0.1), 2) for i, p in enumerate(fc_prices)])
        ci_lower = ([None]*(n_hist - 1) + [float(df["close"].iloc[-1])] +
                    [round(p - 0.5*atr_val*(1+i*0.1), 2) for i, p in enumerate(fc_prices)])

        ci_up_l = (Line().add_xaxis(all_dates)
                   .add_yaxis("CI Upper", ci_upper, is_symbol_show=False,
                              linestyle_opts=opts.LineStyleOpts(
                                  color="rgba(0,229,255,0.25)", width=1, type_="dotted"),
                              label_opts=opts.LabelOpts(is_show=False)))
        ci_lo_l = (Line().add_xaxis(all_dates)
                   .add_yaxis("CI Lower", ci_lower, is_symbol_show=False,
                              linestyle_opts=opts.LineStyleOpts(
                                  color="rgba(0,229,255,0.25)", width=1, type_="dotted"),
                              label_opts=opts.LabelOpts(is_show=False)))
        candle = candle.overlap(fc_line).overlap(ci_up_l).overlap(ci_lo_l)

    # ========== Global opts cho candle ==========
    candle.set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"  {symbol}  —  BIỂU ĐỒ GIÁ & DỰ BÁO",
            pos_left="center",
            title_textstyle_opts=opts.TextStyleOpts(
                color="#e6e6e6", font_size=15,
                font_family="'Segoe UI','Times New Roman',serif")),
        legend_opts=opts.LegendOpts(
            pos_top="32px", pos_left="3%",
            textstyle_opts=opts.TextStyleOpts(color="#8b949e", font_size=11)),
        # X-axis ẩn trên candle, hiển thị ở volume
        xaxis_opts=opts.AxisOpts(
            type_="category",
            is_show=False,                          # ← ẩn ở panel nến
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(color="#30363d")),
            splitline_opts=opts.SplitLineOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            position="right",
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
            opts.DataZoomOpts(
                type_="inside", xaxis_index=[0, 1],   # ← đồng bộ cả 2 panel
                range_start=range_start, range_end=100),
            opts.DataZoomOpts(
                type_="slider", xaxis_index=[0, 1],   # ← đồng bộ cả 2 panel
                pos_bottom="2%",
                range_start=range_start, range_end=100),
        ],
    )

    # ========== VOLUME BAR (grid 1) — trục Y độc lập ==========
    vol_items = [
        opts.BarItem(
            name=all_dates[i],
            value=vol_ext[i],
            itemstyle_opts=opts.ItemStyleOpts(color=vol_colors[i]))
        for i in range(len(all_dates))
    ]

    vol_bar = (
        Bar()
        .add_xaxis(all_dates)
        .add_yaxis(
            "Volume", vol_items,
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
                position="right",
                is_show=False,
                axislabel_opts=opts.LabelOpts(
                    color="#484f58", font_size=9,
                    formatter=JsCode(
                        "function(v){if(v>=1e6)return (v/1e6).toFixed(1)+'M';"
                        "if(v>=1e3)return (v/1e3).toFixed(0)+'K';return v;}")),
                splitline_opts=opts.SplitLineOpts(is_show=False)),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    # ========== Grid: 75% nến / 25% volume ==========
    grid = (
        Grid(init_opts=opts.InitOpts(
            width="100%", height="560px",
            bg_color="transparent",
            renderer="svg",
            animation_opts=opts.AnimationOpts(animation=False)))
        .add(candle,
             grid_opts=opts.GridOpts(
                 pos_left="6%", pos_right="6%",
                 pos_top="60px", pos_bottom="185px"),   # ← để chỗ cho volume + datazoom
             is_control_axis_index=True)
        .add(vol_bar,
             grid_opts=opts.GridOpts(
                 pos_left="6%", pos_right="6%",
                 pos_top="420px", pos_bottom="95px"),   # ← panel volume bên dưới
             )
    )

    return grid.render_embed()


# ============================================================
# 5.  HTML DASHBOARD
# ============================================================
def create_terminal_dashboard2(symbol: str,
                              months: int = 6,
                              forecast_steps: int = 5):
    symbol = symbol.upper()

    if not model_exists(symbol):
        print(f"⚠️  Chưa có model cho {symbol}, đang train…")
        train_model_for_symbol(symbol)

    forecast_df = forecast_future(symbol, forecast_steps=forecast_steps)
    df          = get_data_with_indicators_and_zlema(symbol, months)

    latest   = df.iloc[-1]
    signals  = generate_signals(latest)
    today_dt = df.index[-1]
    today    = today_dt.strftime("%d/%m/%Y")

    next_bday    = get_next_business_day(today_dt)
    tomorrow_str = next_bday.strftime("%d/%m/%Y")

    scores  = compute_dashboard_scores(df)
    fc_info = interpret_forecast(forecast_df)

    chart_html = build_main_chart_fragment(df, forecast_df, symbol)

    latest_atr = float(latest["atr"])

    # TB200
    tb200_val = None
    if "tb200" in df.columns and pd.notna(latest["tb200"]):
        tb200_val = float(latest["tb200"])
    tb200_display = f"{tb200_val:,.2f}" if tb200_val else "N/A (cần thêm dữ liệu)"

    tb50_val = None
    if "tb50" in df.columns and pd.notna(latest["tb50"]):
        tb50_val = float(latest["tb50"])
    tb50_display = f"{tb50_val:,.2f}" if tb50_val else "N/A"

    fibs = compute_fibonacci_levels(df, window=120)
    current_price = float(latest["close"])
    # Sắp xếp tất cả levels theo giá
    fib_sorted = sorted(fibs.items(), key=lambda x: x[1])  # tăng dần

    # Tìm level ngay trên và ngay dưới giá hiện tại
    fib_below = [(n,v) for n,v in fib_sorted if v <= current_price]
    fib_above = [(n,v) for n,v in fib_sorted if v >  current_price]

    # Hỗ trợ = 2 mức ngay dưới giá, kháng cự = 2 mức ngay trên giá
    support_1  = fib_below[-1] if len(fib_below) >= 1 else fib_sorted[0]
    support_2  = fib_below[-2] if len(fib_below) >= 2 else fib_sorted[0]
    resist_1   = fib_above[0]  if len(fib_above) >= 1 else fib_sorted[-1]
    resist_2   = fib_above[1]  if len(fib_above) >= 2 else fib_sorted[-1]

    fib_rows_html = ""
    fib_items = sorted(fibs.items(), key=lambda x: x[1], reverse=True)

    for name, val in fib_items:
        # Tìm level gần giá hiện tại nhất
        is_nearest = abs(val - current_price) == min(abs(v - current_price) for v in fibs.values())
        is_above   = val > current_price
        color      = "#ef5350" if is_above else "#26a69a"
        arrow      = "▼" if is_above else "▲"
        
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

    # Thêm giá hiện tại vào đúng vị trí
    sr_cur_row = f"""
        <div class="sr-row sr-cur">
        <span><b style="color:#58a6ff">★ Giá hiện tại</b></span>
        <span><b style="color:#58a6ff">{current_price:,.2f}</b></span>
        </div>"""
    # ---------- Bảng dự báo ----------
    fc_rows = ""
    if forecast_df is not None and len(forecast_df) > 0:
        for i, (_, row) in enumerate(forecast_df.iterrows()):
            price = float(row["Giá dự báo"])
            chg   = float(row["Thay đổi_%"])
            ci_h  = 0.5 * latest_atr * (1 + i * 0.1)
            ci_lo = price - ci_h
            ci_hi = price + ci_h
            c_col = "#26a69a" if chg >= 0 else "#ef5350"
            c_arr = "↑" if chg > 0.3 else ("↓" if chg < -0.3 else "→")
            fc_rows += f"""
            <tr>
              <td style="font-size:15px;font-weight:600">{row['Ngày']}</td>
              <td style="font-size:17px;font-weight:800">{price:,.2f}</td>
              <td style="font-size:16px;color:{c_col};font-weight:700">{c_arr} {chg:+.2f}%</td>
              <td style="font-size:13px;color:#8b949e">{ci_lo:,.2f} – {ci_hi:,.2f}</td>
            </tr>"""

    # ---------- Xu hướng ----------
    t5_icons  = {"TĂNG":"📈","GIẢM":"📉","TÍCH LŨY":"📊","KHÔNG XÁC ĐỊNH":"❓"}
    t5_colors = {"TĂNG":"#26a69a","GIẢM":"#ef5350","TÍCH LŨY":"#f4d35e",
                 "KHÔNG XÁC ĐỊNH":"#8b949e"}
    t_icon  = t5_icons.get(fc_info["trend_5d"], "❓")
    t_color = t5_colors.get(fc_info["trend_5d"], "#8b949e")

    chg5     = fc_info["change_5d"] if fc_info["change_5d"] else 0
    chg5_cls = "sub-green" if chg5 >= 0 else "sub-red"

    rec = fc_info["recommendation"]
    if   "TĂNG"       in rec: rc, rb = "#26a69a","#26a69a"
    elif "THẬN TRỌNG" in rec: rc, rb = "#ef5350","#ef5350"
    else:                     rc, rb = "#f4d35e","#f4d35e"

    # ---------- Signal rows ----------
    sig_html = ""
    for name, s in signals.items():
        sig_html += f"""
        <div class="sig-row">
          <span class="sig-name">{name}</span>
          <span style="color:{s['color']};font-weight:700">{s['arrow']} {s['text']}</span>
        </div>"""

    def bar_col(v):
        if v >= 60: return "#26a69a"
        if v >= 40: return "#f4d35e"
        return "#ef5350"

    # ================================================================
    #  HTML
    # ================================================================
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
  font-family:'Segoe UI','Times New Roman',sans-serif;
  line-height:1.5;
}}
.wrap{{max-width:1580px;margin:0 auto;padding:14px 20px}}

.hdr{{
  display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:12px;
  padding:10px 18px;margin-bottom:14px;
  background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
  border:1px solid #21262d;border-radius:10px;
}}
.hdr-sym{{font-size:22px;font-weight:800;color:#58a6ff}}
.hdr-price{{font-size:20px;font-weight:700}}
.hdr-sep{{color:#30363d}}

.g4{{display:grid;grid-template-columns:1.3fr 1fr 1fr 1fr;gap:14px;margin-bottom:14px}}

/* ★ THAY ĐỔI: cột bảng dự báo rộng hơn (1.2fr thay vì 1fr) */
.g-main{{display:grid;grid-template-columns:2.4fr 1.2fr;gap:14px;margin-bottom:14px}}
.g3{{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:14px;margin-bottom:14px}}

.card{{
  background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px;
  transition:border-color .2s;
}}
.card:hover{{border-color:#388bfd44}}
.stitle{{
  text-align:center;font-size:14px;font-weight:700;
  color:#8b949e;text-transform:uppercase;letter-spacing:.8px;
  margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #21262d;
}}

.ms{{margin:6px 0 3px;font-size:12.5px;color:#8b949e}}
.bar-bg{{width:100%;background:#21262d;height:16px;border-radius:4px;overflow:hidden}}
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
.prob-label{{display:flex;justify-content:space-between;font-size:10px;color:#484f58;margin-top:3px}}

/* ★ BẢNG DỰ BÁO — to hơn, dễ đọc hơn */
table{{width:100%;border-collapse:collapse;color:#c9d1d9}}
th{{
  padding:10px 6px;border-bottom:2px solid #30363d;
  color:#8b949e;font-size:12px;text-transform:uppercase;text-align:center;
  letter-spacing:.5px;
}}
td{{padding:12px 6px;border-bottom:1px solid #21262d;text-align:center}}
tr:hover td{{background:#21262d55}}

.sr-row{{
  display:flex;justify-content:space-between;
  padding:6px 4px;border-bottom:1px solid #ffffff08;font-size:13.5px;
}}
.sr-row:hover{{background:#21262d55}}
.sr-cur{{background:#21262d;border-radius:5px;padding:8px 6px;margin:4px 0}}

.sc{{padding:11px 12px;border-radius:8px;margin-bottom:9px;border-left:3px solid}}
.sc-up{{border-color:#26a69a;background:rgba(38,166,154,.04)}}
.sc-mid{{border-color:#f4d35e;background:rgba(244,211,94,.04)}}
.sc-dn{{border-color:#ef5350;background:rgba(239,83,80,.04)}}

.sig-row{{
  display:flex;justify-content:space-between;
  padding:8px 2px;border-bottom:1px solid #ffffff08;
}}
.sig-name{{color:#8b949e;font-size:13px}}

.drow{{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}}
.drow-k{{color:#8b949e}}
.green{{color:#26a69a}}.red{{color:#ef5350}}.yellow{{color:#f4d35e}}.cyan{{color:#58a6ff}}

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
</div>

<!-- TOP 4-COL -->
<div class="g4">

  <div class="card">
    <div class="stitle">📊 Xu Hướng Sắp Tới</div>
    <div style="text-align:center;font-size:52px;margin:8px 0">{t_icon}</div>
    <div style="text-align:center;font-size:26px;font-weight:800;color:{t_color}">{fc_info['trend_5d']}</div>
    <div style="text-align:center;margin-top:10px;font-size:15px">
      Dự kiến&nbsp;<b style="color:{t_color}">{chg5:+.2f}%</b>&nbsp;trong 5 phiên
    </div>
    <div style="margin-top:14px;padding-top:10px;border-top:1px solid #21262d">
      <div class="drow"><span class="drow-k">ZLEMA</span><span style="color:{signals['ZLEMA']['color']}">{signals['ZLEMA']['arrow']} {signals['ZLEMA']['text']}</span></div>
      <div class="drow"><span class="drow-k">MACD</span><span style="color:{signals['MACD']['color']}">{signals['MACD']['arrow']} {signals['MACD']['text']}</span></div>
      <div class="drow"><span class="drow-k">SAR</span><span style="color:{signals['SAR']['color']}">{signals['SAR']['arrow']} {signals['SAR']['text']}</span></div>
    </div>
  </div>

  <div class="card">
    <div class="stitle">📋 5 Tiêu Chí</div>
    <div class="ms">Xu hướng: <b>{scores['trend']}</b></div>
    <div class="bar-bg"><div class="bar-fill" style="width:{scores['trend']}%;background:{bar_col(scores['trend'])}"></div></div>
    <div class="ms">Động lực: <b>{scores['momentum']}</b></div>
    <div class="bar-bg"><div class="bar-fill" style="width:{scores['momentum']}%;background:{bar_col(scores['momentum'])}"></div></div>
    <div class="ms">Dòng tiền: <b>{scores['money']}</b></div>
    <div class="bar-bg"><div class="bar-fill" style="width:{scores['money']}%;background:{bar_col(scores['money'])}"></div></div>
    <div class="ms">Ổn định: <b>{scores['stability']}</b></div>
    <div class="bar-bg"><div class="bar-fill" style="width:{scores['stability']}%;background:{bar_col(scores['stability'])}"></div></div>
    <div class="ms">Chu kỳ: <b>{scores['cycle']}</b></div>
    <div class="bar-bg"><div class="bar-fill" style="width:{scores['cycle']}%;background:{bar_col(scores['cycle'])}"></div></div>
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
      <div class="prob-label"><span>Giảm</span><span>Trung tính</span><span>Tăng</span></div>
    </div>
  </div>

  <div class="card">
    <div class="stitle">💰 {symbol}</div>
    <div class="vbox">{latest['close']:,.2f}</div>
    <div class="{chg5_cls}" style="font-size:17px">{chg5:+.2f}% / 5 ngày</div>
    <div style="margin-top:14px;border-top:1px solid #21262d;padding-top:10px">
      <div class="drow"><span class="drow-k">High</span><span class="green">{latest['high']:,.2f}</span></div>
      <div class="drow"><span class="drow-k">Low</span><span class="red">{latest['low']:,.2f}</span></div>
      <div class="drow"><span class="drow-k">Volume</span><span>{latest['volume']:,.0f}</span></div>
      <div class="drow"><span class="drow-k">ATR</span><span class="yellow">{latest['atr']:,.2f}</span></div>
    </div>
  </div>
</div>

<!-- MAIN: CHART + FORECAST TABLE -->
<div class="g-main">
  <div class="card" style="padding:8px">
    {chart_html}
  </div>

  <!-- ★ Bảng dự báo lớn hơn, padding thoáng hơn -->
  <div class="card" style="display:flex;flex-direction:column">
    <div class="stitle" style="font-size:16px">🔮 Dự Báo {forecast_steps} Phiên</div>
    <div style="flex:1;overflow-y:auto">
      <table>
        <thead>
          <tr>
            <th style="font-size:13px">Ngày</th>
            <th style="font-size:13px">Giá</th>
            <th style="font-size:13px">%</th>
            <th style="font-size:13px">Khoảng tin cậy</th>
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
        <span style="font-size:13px">Nếu vượt <b>{resist_1[1]:,.2f}</b> ({resist_1[0]})</span><br/>
        <span style="font-size:11.5px;color:#8b949e">→ Kháng cự tiếp: {resist_2[1]:,.2f} ({resist_2[0]})</span>
    </div>
    <div class="sc sc-mid">
        <b class="yellow">→ TÍCH LŨY</b><br/>
        <span style="font-size:13px">Giá đang giữa <b>{support_1[1]:,.2f}</b> – <b>{resist_1[1]:,.2f}</b></span><br/>
        <span style="font-size:11.5px;color:#8b949e">→ Chờ breakout khỏi vùng này</span>
    </div>
        <div class="sc sc-dn">
        <b class="red">↓ GIẢM</b><br/>
        <span style="font-size:13px">Nếu thủng <b>{support_1[1]:,.2f}</b> ({support_1[0]})</span><br/>
        <span style="font-size:11.5px;color:#8b949e">→ Hỗ trợ tiếp: {support_2[1]:,.2f} ({support_2[0]})</span>
    </div>
  </div>

  <div class="card">
    <div class="stitle">📡 Tín Hiệu Chỉ Báo</div>
    {sig_html}
    <div style="margin-top:14px;padding-top:10px;border-top:1px solid #21262d;
                font-size:12px;color:#8b949e;text-align:center">
      RSI: {latest['rsi']:.1f} &nbsp;│&nbsp;
      ADX: {latest['adx']:.1f} &nbsp;│&nbsp;
      MFI: {latest['mfi']:.1f}
    </div>
  </div>
</div>

<!-- FOOTER -->
<div class="foot">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} •
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

    print(f"✅ Dashboard đã tạo: {fname}")
    return fname


# # ============================================================
# fname = create_terminal_dashboard2("TCB", months=60, forecast_steps=5)
# html_to_image2(fname)