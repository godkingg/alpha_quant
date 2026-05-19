# =========================================================
# a_portfolio_v5.py — PORTFOLIO RECOMMENDATION SYSTEM
# MPT + Core-Satellite Strategy
#
# Khác biệt so với phiên bản cũ:
#   - Dùng a_ML5_weekly (Pure RF, interval=w) thay a_ML3
#   - FORECAST_HORIZON = 4 tuần (thay vì 30 ngày)
#   - Risk-free rate tính theo tuần (weekly)
#   - Covariance matrix scale × 4 (tuần)
#   - Dashboard cập nhật nhãn "4 tuần / Pure RF Weekly"
# =========================================================

import os
import sys
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import a_ML4_weekly as ml
import warnings
from playwright.sync_api import sync_playwright

warnings.filterwarnings('ignore')

# =========================================================
# CONFIG
# =========================================================
ANNUAL_RF_RATE  = 0.05
# Weekly risk-free: (1+5%)^(1/52) - 1
WEEKLY_RF_RATE  = (1 + ANNUAL_RF_RATE) ** (1 / 52) - 1

DEFAULT_LAMBDA_REG      = 0.001
DEFAULT_MAX_TURNOVER    = 0.50
DEFAULT_DIVERSIFICATION = 'balanced'

MIN_POSITION_SIZE = 0.05
MAX_POSITION_SIZE = 0.35

FORECAST_HORIZON = 4   # tuần

# Core-Satellite thresholds (tính trên 4 tuần)
MIN_CORE_ER      =  0.01   # Core: ER >= 1%
MAX_SATELLITE_LOSS = -0.03 # Satellite: ER >= -3%


# =========================================================
# DATA ACQUISITION — dùng a_ML5_weekly
# =========================================================
def get_data_for_symbol(symbol,
                        steps: int = None,
                        cost_basis: float = None,
                        auto_confirm: bool = False):
    """
    Wrapper quanh a_ML5_weekly.get_expected_return_for_portfolio().

    Returns:
        (expected_return, std_dev, hist_returns_weekly,
         current_price, predicted_price, base_price)
    """
    if steps is None:
        steps = FORECAST_HORIZON

    return ml.get_expected_return_for_portfolio(
        symbol        = symbol,
        cost_basis    = cost_basis,
        forecast_steps= steps,
        auto_confirm  = auto_confirm,
    )


# =========================================================
# CORE-SATELLITE STRATEGY  (giữ nguyên logic)
# =========================================================
def build_portfolio_strategy(all_ers, data,
                              min_core_er=MIN_CORE_ER,
                              max_satellite_loss=MAX_SATELLITE_LOSS,
                              auto_confirm=False):
    """
    Core-Satellite strategy:
    - Core      : ER >= min_core_er
    - Satellite : max_satellite_loss < ER < min_core_er
    - Defensive : fallback nếu không đủ mã
    """
    core      = {t: er for t, er in all_ers.items() if er >= min_core_er}
    satellite = {t: er for t, er in all_ers.items()
                 if max_satellite_loss < er < min_core_er}

    print("\n" + "="*70)
    print("📊 CORE-SATELLITE STRATEGY")
    print("="*70)

    if core:
        print(f"\n🎯 CORE ({len(core)} mã — ER >= {min_core_er*100}%):")
        for t, er in sorted(core.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {er*100:+6.2f}%  (Risk: {data[t]['Risk']*100:5.1f}%)")

    if satellite:
        print(f"\n🔍 SATELLITE ({len(satellite)} mã — thăm dò):")
        for t, er in sorted(satellite.items(), key=lambda x: x[1], reverse=True):
            print(f"  {t}: {er*100:+6.2f}%  (Risk: {data[t]['Risk']*100:5.1f}%)")

    # ── CASE 1: Không có mã nào đủ điều kiện ─────────────
    if not core and not satellite:
        print("\n❌ KHÔNG CÓ MÃ NÀO ĐỦ ĐIỀU KIỆN!")
        print(f"   Tất cả mã đều có ER < {max_satellite_loss*100}%")
        print("\n💡 Khuyến nghị: GIỮ TIỀN MẶT hoặc ĐẦU TƯ TRÁI PHIẾU")

        fallback = 'y' if auto_confirm else \
            input("\nBạn có muốn tạo danh mục defensive (mã ít thua nhất)? (y/n): ")
        if fallback.lower() != 'y':
            return None

        sorted_all = sorted(all_ers.items(), key=lambda x: x[1], reverse=True)
        top_n      = min(3, len(sorted_all))
        print(f"\n📌 Chọn {top_n} mã ít thua nhất (DEFENSIVE):")
        portfolio  = {}
        for t, er in sorted_all[:top_n]:
            portfolio[t] = er
            print(f"  {t}: {er*100:+6.2f}%")
        min_er    = min(portfolio.values())
        portfolio = {t: er - min_er + 0.001 for t, er in portfolio.items()}
        print("\n⚠️ CẢNH BÁO: Danh mục defensive — rủi ro CAO, chỉ dùng ngắn hạn!")
        return portfolio

    # ── CASE 2: Không có Core ─────────────────────────────
    if not core:
        print(f"\n⚠️ KHÔNG CÓ MÃ CORE (ER >= {min_core_er*100:.1f}%)")
        print(f"   Danh mục chỉ có {len(satellite)} Satellite — Rủi ro cao!")
        confirm = 'y' if auto_confirm else \
            input("\nVẫn tiếp tục với Satellite? (y/n): ")
        if confirm.lower() != 'y':
            return None
        portfolio = satellite

    # ── CASE 3: Chỉ 1 Core ────────────────────────────────
    elif len(core) == 1:
        core_ticker = list(core.keys())[0]
        core_er     = list(core.values())[0]
        core_risk   = data[core_ticker]['Risk']

        print(f"\n⚠️ CHỈ CÓ 1 MÃ CORE: {core_ticker}")
        print(f"   ER: {core_er*100:+.2f}%  | Risk: {core_risk*100:.1f}%")

        if satellite:
            print(f"\n💡 Có {len(satellite)} Satellite — Khuyến nghị thêm để đa dạng hóa")
            print(f"\n📋 Lựa chọn:")
            print(f"[1] 100% vào {core_ticker}  (Tập trung — Rủi ro {core_risk*100:.1f}%)")
            print(f"[2] Core + Satellite         (Đa dạng hóa — Khuyến nghị)")
            print(f"[3] Core + Defensive         (Thêm mã ít thua nhất)")

            choice = '2' if auto_confirm else \
                input("\nLựa chọn [1/2/3] (mặc định: 2): ").strip()

            if choice == '1':
                portfolio = core
                print(f"\n⚠️ 100% vào {core_ticker} — RỦI RO TẬP TRUNG!")
            elif choice == '3':
                others = {t: er for t, er in all_ers.items()
                          if t not in core and t not in satellite}
                if others:
                    sorted_others = sorted(others.items(),
                                          key=lambda x: x[1], reverse=True)
                    top_2 = dict(sorted_others[:min(2, len(sorted_others))])
                    print(f"\n📌 Thêm {len(top_2)} mã defensive:")
                    for t, er in top_2.items():
                        print(f"  {t}: {er*100:+6.2f}%")
                    portfolio = {**core, **satellite, **top_2}
                else:
                    print("\n→ Không có mã defensive — Dùng Core + Satellite")
                    portfolio = {**core, **satellite}
            else:
                portfolio = {**core, **satellite}
        else:
            print(f"\n   Không có Satellite")
            print(f"\n📋 Lựa chọn:")
            print(f"[1] 100% vào {core_ticker}  (Rủi ro {core_risk*100:.1f}%)")
            print(f"[2] Thêm 2 mã defensive     (ít thua nhất)")

            choice = '2' if auto_confirm else \
                input("\nLựa chọn [1/2] (mặc định: 2): ").strip()

            if choice == '1':
                portfolio = core
                print(f"\n⚠️ 100% vào {core_ticker}!")
            else:
                others = {t: er for t, er in all_ers.items() if t not in core}
                sorted_others = sorted(others.items(),
                                      key=lambda x: x[1], reverse=True)
                top_2 = dict(sorted_others[:min(2, len(sorted_others))])
                print(f"\n📌 Thêm {len(top_2)} mã defensive:")
                for t, er in top_2.items():
                    print(f"  {t}: {er*100:+6.2f}%")
                portfolio = {**core, **top_2}

    # ── CASE 4: Nhiều Core (>= 2) ─────────────────────────
    else:
        print(f"\n✅ CÓ {len(core)} MÃ CORE — Đa dạng hóa tốt!")
        if satellite:
            print(f"   + {len(satellite)} Satellite")
            portfolio = {**core, **satellite}
        else:
            portfolio = core

    # ── Shift ER nếu có âm ────────────────────────────────
    min_er = min(portfolio.values())
    if min_er < 0:
        print(f"\n📊 Shift ER để tất cả dương:")
        print(f"   Min ER: {min_er*100:.2f}% → 0.1%")
        portfolio = {t: er - min_er + 0.001 for t, er in portfolio.items()}

    print(f"\n✅ Danh mục cuối cùng: {len(portfolio)} mã")
    print("-" * 70)
    for t in sorted(portfolio.keys()):
        orig    = all_ers[t]
        shifted = portfolio[t]
        if orig != shifted:
            print(f"  {t}: {orig*100:+6.2f}% → {shifted*100:+6.2f}% (shifted)")
        else:
            print(f"  {t}: {orig*100:+6.2f}%")

    return portfolio


# =========================================================
# PORTFOLIO OPTIMIZER
# =========================================================
class PortfolioOptimizer:
    """
    MPT optimizer:
    - Ledoit-Wolf shrinkage covariance
    - Weekly risk-free rate adjusted Sharpe
    - L2 regularization
    - Turnover constraint
    """

    def __init__(self, lambda_reg=DEFAULT_LAMBDA_REG,
                 risk_free_rate=WEEKLY_RF_RATE):
        self.lambda_reg = lambda_reg
        self.rf         = risk_free_rate

    def optimize(self, expected_returns, cov_matrix,
                 diversification_mode='balanced',
                 current_weights=None, max_turnover=None):
        tickers  = list(expected_returns.keys())
        n        = len(tickers)
        if n == 0:
            return {}

        er_array  = np.array([expected_returns[t] for t in tickers])
        cov_array = cov_matrix.loc[tickers, tickers].values

        def objective(weights):
            port_return = np.sum(er_array * weights)
            port_std    = np.sqrt(np.dot(weights.T, np.dot(cov_array, weights)))
            if port_std <= 0:
                return 1e10
            sharpe  = (port_return - self.rf) / port_std
            penalty = self.lambda_reg * np.sum(weights ** 2)
            return -sharpe + penalty

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if current_weights is not None and max_turnover is not None:
            cur_arr = np.array([current_weights.get(t, 0.0) for t in tickers])
            constraints.append({
                'type': 'ineq',
                'fun' : lambda x, c=cur_arr: max_turnover - np.sum(np.abs(x - c))
            })

        if diversification_mode == 'balanced':
            min_w = MIN_POSITION_SIZE
            max_w = MAX_POSITION_SIZE
            if min_w * n > 1.0: min_w = 1.0 / n * 0.7
            if max_w * n < 1.0: max_w = 1.0 / n * 1.5
            bounds = tuple((min_w, max_w) for _ in range(n))
        elif diversification_mode == 'strict':
            avg_w  = 1.0 / n
            bounds = tuple((avg_w * 0.8, avg_w * 1.2) for _ in range(n))
        else:
            bounds = tuple((0.0, 1.0) for _ in range(n))

        x0 = (np.array([current_weights.get(t, 1.0/n) for t in tickers])
               if current_weights else np.array([1.0/n] * n))
        x0 /= x0.sum()

        result = minimize(objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 5000, 'ftol': 1e-12})

        if not result.success:
            print(f"   ⚠️ Tối ưu thất bại: {result.message}")
            weights_array = x0
        else:
            weights_array = result.x

        weights_array /= weights_array.sum()

        if current_weights is not None:
            cur_arr       = np.array([current_weights.get(t, 0.0) for t in tickers])
            actual_to     = np.sum(np.abs(weights_array - cur_arr))
            print(f"   ↳ Turnover thực tế: {actual_to*100:.1f}%")

        return dict(zip(tickers, weights_array))


# =========================================================
# HTML DASHBOARD
# =========================================================
def html_to_image(html_file: str, png_file: str = None,
                  width: int = 1200) -> str:
    if png_file is None:
        png_file = html_file.replace(".html", ".png")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            )
            page = context.new_page()
            page.set_viewport_size({"width": width, "height": 900})
            page.goto(f"file:///{os.path.abspath(html_file)}",
                      wait_until="networkidle")
            page.wait_for_timeout(2000)
            body_height = page.evaluate("document.body.scrollHeight")
            page.set_viewport_size({"width": width, "height": body_height + 50})
            page.screenshot(path=png_file, full_page=True)
            browser.close()
        print(f"✅ Đã lưu ảnh: {png_file}")
    except Exception as e:
        print(f"⚠️ Không thể tạo ảnh: {e}")
    return png_file


def create_dashboard_html(market_stats, port_stats, data, weights,
                          config, file_name="danhmuc_dashboard_v5.html"):
    from pyecharts.charts import Pie, Bar
    from pyecharts import options as opts

    # ── Pie chart ─────────────────────────────────────────
    pie_data = [(sym, round(w * 100, 2))
                for sym, w in weights.items() if w > 0.001]

    pie = (
        Pie(init_opts=opts.InitOpts(
            bg_color="transparent", width="100%", height="350px"))
        .add("", pie_data,
             radius=["40%", "70%"],
             itemstyle_opts=opts.ItemStyleOpts(
                 border_color="#161b22", border_width=2),
             label_opts=opts.LabelOpts(
                 formatter="{b}: {c}%", color="#c9d1d9",
                 font_weight="bold", font_size=14))
        .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
    )
    pie_html = pie.render_embed()

    # ── Bar chart trọng số ────────────────────────────────
    bar_syms = [sym for sym, w in sorted(
        weights.items(), key=lambda x: x[1], reverse=True) if w > 0.001]
    bar_vals = [round(weights[s] * 100, 2) for s in bar_syms]
    bar_colors = ["#26a69a" if weights[s] >= 0.15 else "#58a6ff"
                  for s in bar_syms]

    bar_items = [
        opts.BarItem(name=bar_syms[i], value=bar_vals[i],
                     itemstyle_opts=opts.ItemStyleOpts(color=bar_colors[i]))
        for i in range(len(bar_syms))
    ]
    bar = (
        Bar(init_opts=opts.InitOpts(
            bg_color="transparent", width="100%", height="280px"))
        .add_xaxis(bar_syms)
        .add_yaxis("Tỷ trọng (%)", bar_items,
                   label_opts=opts.LabelOpts(
                       is_show=True, color="#c9d1d9",
                       formatter="{c}%", position="top"))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(color="#8b949e", font_size=12)),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(color="#8b949e"),
                splitline_opts=opts.SplitLineOpts(
                    linestyle_opts=opts.LineStyleOpts(
                        color="#21262d", type_="dashed"))),
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(
                formatter="{b}: {c}%",
                background_color="rgba(13,17,23,0.9)",
                border_color="#30363d",
                textstyle_opts=opts.TextStyleOpts(color="#e6e6e6")),
        )
    )
    bar_html = bar.render_embed()

    # ── Table rows ────────────────────────────────────────
    tr_rows = ""
    for sym, d in data.items():
        if sym == "VNINDEX":
            continue

        ret  = d['Expected_Return']
        risk = d['Risk']
        cb   = d['Cost_Basis']
        cp   = d['Current_Price']
        pp   = d['Predicted_Price']
        w    = weights.get(sym, 0)

        ret_color = "#26a69a" if ret >= 0 else "#ef5350"
        ret_str   = f"+{ret*100:.2f}%" if ret >= 0 else f"{ret*100:.2f}%"

        # Role tag
        orig_er = d['Expected_Return']
        if orig_er >= MIN_CORE_ER:
            role_html = '<span style="color:#26a69a;font-size:11px;font-weight:700;' \
                        'background:rgba(38,166,154,.12);padding:2px 6px;' \
                        'border-radius:4px;">CORE</span>'
        elif orig_er >= MAX_SATELLITE_LOSS:
            role_html = '<span style="color:#58a6ff;font-size:11px;font-weight:700;' \
                        'background:rgba(88,166,255,.12);padding:2px 6px;' \
                        'border-radius:4px;">SATELLITE</span>'
        else:
            role_html = '<span style="color:#8b949e;font-size:11px;font-weight:700;' \
                        'background:rgba(139,148,158,.12);padding:2px 6px;' \
                        'border-radius:4px;">DEFENSIVE</span>'

        # Recommendation
        if ret < -0.02:
            rec_html = '<span style="color:#ef5350;font-weight:bold;">🔴 GIẢM</span>'
        elif ret > 0.05:
            rec_html = '<span style="color:#26a69a;font-weight:bold;">🟢 TĂNG</span>'
        else:
            rec_html = '<span style="color:#f4d35e;font-weight:bold;">🟡 TÍCH LŨY</span>'

        if risk > 0.12:
            rec_html += '<br><span style="color:#ff9800;font-size:11px;">⚠️ Rủi ro cao</span>'
        elif risk < 0.06:
            rec_html += '<br><span style="color:#58a6ff;font-size:11px;">🛡️ Ổn định</span>'

        cb_str = f"{cb:,.2f}" if cb else "—"

        # Weight bar mini
        bar_w  = int(w * 200)
        bar_bg = "#26a69a" if w >= 0.15 else "#58a6ff"
        w_bar  = (f'<div style="display:flex;align-items:center;gap:6px">'
                  f'<div style="width:{bar_w}px;height:6px;'
                  f'background:{bar_bg};border-radius:3px;min-width:2px"></div>'
                  f'<span style="font-size:13px;font-weight:700;color:#f0f6fc">'
                  f'{w*100:.1f}%</span></div>')

        tr_rows += f"""
        <tr>
          <td style="padding:13px 14px;font-weight:700;color:#58a6ff;
                     text-align:center">{sym}<br>{role_html}</td>
          <td style="padding:13px 14px;text-align:right;color:#8b949e">
            {cb_str}</td>
          <td style="padding:13px 14px;text-align:right">{cp:,.2f}</td>
          <td style="padding:13px 14px;text-align:right;font-weight:700;
                     color:#e6e6e6">{pp:,.2f}</td>
          <td style="padding:13px 14px;text-align:right;color:{ret_color};
                     font-weight:700">{ret_str}</td>
          <td style="padding:13px 14px;text-align:right;color:#e6e6e6">
            {risk*100:.2f}%</td>
          <td style="padding:13px 14px;text-align:center;
                     line-height:1.6">{rec_html}</td>
          <td style="padding:13px 14px">{w_bar}</td>
        </tr>"""

    # ── Stats ─────────────────────────────────────────────
    m_ret      = market_stats['Expected_Return']
    m_ret_col  = "#26a69a" if m_ret >= 0 else "#ef5350"
    m_sharpe   = market_stats['Sharpe']

    p_ret      = port_stats['Expected_Return']
    p_ret_col  = "#26a69a" if p_ret >= 0 else "#ef5350"
    p_sharpe   = port_stats['Sharpe']

    sharpe_diff = p_sharpe - m_sharpe
    sharpe_note = (
        f'<span style="color:#26a69a">▲ +{sharpe_diff:.4f} so với VNINDEX</span>'
        if sharpe_diff >= 0 else
        f'<span style="color:#ef5350">▼ {sharpe_diff:.4f} so với VNINDEX</span>'
    )

    mode_labels = {
        'none'    : '❌ Không ràng buộc',
        'balanced': '✅ Cân bằng (Khuyến nghị)',
        'strict'  : '🛡️ Nghiêm ngặt',
    }
    mode_label = mode_labels.get(config['diversification'],
                                 config['diversification'])

    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── HTML ──────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<title>Danh Mục v5 — Pure RF Weekly</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:#0d1117;color:#c9d1d9;
  font-family:'Segoe UI','Roboto',sans-serif;
  line-height:1.5;padding:20px;
}}
.wrap{{max-width:1300px;margin:0 auto}}

/* Header */
.hdr{{
  text-align:center;margin-bottom:20px;
  background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);
  border:1px solid #21262d;border-radius:12px;padding:28px 24px;
}}
.hdr-title{{font-size:26px;font-weight:800;color:#58a6ff;margin-bottom:6px}}
.hdr-sub{{font-size:13px;color:#8b949e;margin-bottom:14px}}
.badge{{
  display:inline-block;font-size:11px;font-weight:700;letter-spacing:.5px;
  color:#00e5ff;border:1px solid #00e5ff44;
  background:rgba(0,229,255,.07);border-radius:4px;
  padding:2px 10px;vertical-align:middle;margin:0 4px;
}}
.cfg-box{{
  display:inline-block;text-align:left;
  background:#161b22;border-left:3px solid #58a6ff;
  border-radius:6px;padding:12px 20px;
  font-size:12.5px;color:#c9d1d9;margin-top:12px;
  line-height:1.8;
}}

/* Grid layouts */
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:18px}}
.g-wide{{display:grid;grid-template-columns:1.3fr 1fr;gap:18px;margin-bottom:18px}}

/* Card */
.card{{
  background:#161b22;border:1px solid #21262d;
  border-radius:10px;padding:20px;
}}
.card:hover{{border-color:#388bfd33}}
.ctitle{{
  font-size:15px;font-weight:700;color:#e6e6e6;
  margin-bottom:14px;padding-bottom:10px;
  border-bottom:1px solid #21262d;text-align:center;
}}

/* Stat rows */
.stat{{display:flex;justify-content:space-between;
       margin-bottom:10px;font-size:13.5px}}
.stat-k{{color:#8b949e}}
.stat-v{{font-weight:700}}

/* Comparison boxes */
.cmp-box{{
  background:#0d1117;padding:16px;border-radius:8px;
  border:1px solid #30363d;
}}
.cmp-label{{color:#8b949e;font-size:12px;text-align:center;
            font-weight:700;margin-bottom:12px}}

/* Table */
table{{width:100%;border-collapse:collapse}}
th{{
  background:#21262d;color:#8b949e;padding:12px 14px;
  font-weight:600;font-size:12px;text-transform:uppercase;
  letter-spacing:.4px;text-align:right;border-bottom:2px solid #30363d;
}}
th:first-child{{text-align:center}}
td{{font-size:13px;border-bottom:1px solid #21262d33;text-align:right}}
td:first-child{{text-align:center}}
tr:hover td{{background:#21262d44}}

/* Score bar */
.bar-bg{{width:100%;background:#21262d;height:14px;
         border-radius:4px;overflow:hidden;margin-top:4px}}
.bar-fill{{height:14px;border-radius:4px}}

/* Footer */
.foot{{
  text-align:center;margin-top:18px;padding:14px;
  background:#161b22;border:1px solid #21262d;
  border-radius:8px;font-size:11px;color:#484f58;
}}
@media(max-width:900px){{
  .g2,.g3,.g-wide{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<div class="wrap">

<!-- HEADER -->
<div class="hdr">
  <div class="hdr-title">📈 HỆ THỐNG ĐỀ XUẤT DANH MỤC ĐẦU TƯ</div>
  <div class="hdr-sub">
    Modern Portfolio Theory + Machine Learning
    <span class="badge">PURE RF</span>
    <span class="badge">WEEKLY</span>
    <span class="badge">4 TUẦN</span>
  </div>
  <div class="cfg-box">
    <strong>Cấu hình tối ưu hóa:</strong><br>
    • Chế độ: {mode_label}<br>
    • Mô hình: Pure Random Forest + ZLEMA({ml.CONFIG.get('zlema_period',5)})
      — interval=w — NO HYBRID<br>
    • Risk-free rate: {WEEKLY_RF_RATE*100:.5f}% / tuần
      ({ANNUAL_RF_RATE*100:.1f}% / năm)<br>
    • L2 Regularization λ: {config['lambda_reg']}<br>
    • Max Turnover: {config.get('max_turnover','N/A')
                     if config.get('max_turnover') else 'N/A'}<br>
    • Shrinkage (Ledoit-Wolf): {config.get('shrinkage','N/A')}<br>
    • Strategy: {config.get('strategy','Core-Satellite')}
  </div>
</div>

<!-- ROW 1: So sánh + Pie -->
<div class="g-wide">

  <!-- So sánh hiệu suất -->
  <div class="card">
    <div class="ctitle">📊 So Sánh Hiệu Suất (4 Tuần)</div>
    <div style="display:flex;gap:14px">
      <div class="cmp-box" style="flex:1">
        <div class="cmp-label">🏛 VNINDEX (Benchmark)</div>
        <div class="stat">
          <span class="stat-k">Lợi suất kỳ vọng</span>
          <span class="stat-v" style="color:{m_ret_col}">
            {'+' if m_ret>=0 else ''}{m_ret*100:.2f}%</span>
        </div>
        <div class="stat">
          <span class="stat-k">Rủi ro (Std)</span>
          <span class="stat-v">{market_stats['Risk']*100:.2f}%</span>
        </div>
        <div class="stat" style="margin:0">
          <span class="stat-k">Sharpe Ratio</span>
          <span class="stat-v" style="color:#f4d35e">{m_sharpe:.4f}</span>
        </div>
      </div>
      <div class="cmp-box" style="flex:1;border-color:#388bfd44">
        <div class="cmp-label" style="color:#58a6ff">
          📌 Danh Mục Đề Xuất</div>
        <div class="stat">
          <span class="stat-k">Lợi suất kỳ vọng</span>
          <span class="stat-v" style="color:{p_ret_col}">
            {'+' if p_ret>=0 else ''}{p_ret*100:.2f}%</span>
        </div>
        <div class="stat">
          <span class="stat-k">Rủi ro (Std)</span>
          <span class="stat-v">{port_stats['Risk']*100:.2f}%</span>
        </div>
        <div class="stat" style="margin:0">
          <span class="stat-k">Sharpe Ratio</span>
          <span class="stat-v" style="color:#f4d35e">{p_sharpe:.4f}</span>
        </div>
      </div>
    </div>
    <div style="margin-top:14px;text-align:center;font-size:13px">
      {sharpe_note}
    </div>
    <div style="margin-top:8px;font-size:11px;color:#484f58;text-align:center">
      * Sharpe = (Return − Risk-free) / Risk &nbsp;|&nbsp;
      Risk-free = {WEEKLY_RF_RATE*100:.5f}%/tuần
    </div>

    <!-- Bar chart tỷ trọng -->
    <div style="margin-top:18px;border-top:1px solid #21262d;padding-top:14px">
      <div style="font-size:13px;font-weight:700;color:#8b949e;
                  text-align:center;margin-bottom:8px">
        TỶ TRỌNG ĐỀ XUẤT
      </div>
      {bar_html}
    </div>
  </div>

  <!-- Pie chart -->
  <div class="card">
    <div class="ctitle">💰 Cơ Cấu Danh Mục</div>
    {pie_html}
    <!-- Legend tự làm -->
    <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;
                justify-content:center">
      {''.join(
        f'<div style="font-size:12px;color:#c9d1d9">'
        f'<b style="color:#58a6ff">{sym}</b> {weights[sym]*100:.1f}%</div>'
        for sym, _ in sorted(weights.items(), key=lambda x: x[1], reverse=True)
        if weights[sym] > 0.001
      )}
    </div>
  </div>
</div>

<!-- ROW 2: Chi tiết từng mã -->
<div class="card" style="margin-bottom:18px;padding:0;overflow:hidden">
  <div style="padding:18px 20px;border-bottom:1px solid #21262d">
    <div class="ctitle" style="border:none;margin:0;padding:0">
      📋 Chi Tiết Từng Mã — Dự Báo 4 Tuần
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th style="text-align:center">Mã / Role</th>
        <th>Giá Vốn</th>
        <th>Giá Hiện Tại</th>
        <th>Dự Báo (4W)</th>
        <th>Lợi Suất</th>
        <th>Rủi Ro</th>
        <th style="text-align:center">Khuyến Nghị</th>
        <th>Tỷ Trọng</th>
      </tr>
    </thead>
    <tbody>{tr_rows}</tbody>
  </table>
</div>

<!-- FOOTER -->
<div class="foot">
  ⚠️ <strong>Miễn trừ trách nhiệm:</strong>
  Kết quả chỉ mang tính tham khảo — không phải lời khuyên đầu tư.
  Luôn tham khảo chuyên gia trước khi quyết định.
  <br>
  Generated {gen_time} &nbsp;•&nbsp;
  Model: Pure RF + ZLEMA({ml.CONFIG.get('zlema_period',5)}) &nbsp;•&nbsp;
  interval=w &nbsp;•&nbsp; start={ml.CONFIG['start_date']}
</div>

</div>
<script>
window.addEventListener('resize', function(){{
  document.querySelectorAll('[_echarts_instance_]').forEach(function(el){{
    var inst = echarts.getInstanceByDom(el);
    if(inst) inst.resize();
  }});
}});
setTimeout(function(){{ window.dispatchEvent(new Event('resize')); }}, 150);
</script>
</body>
</html>"""

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Đã lưu HTML: {file_name}")
    return file_name


# =========================================================
# BOT API — run_portfolio_optimization
# =========================================================
def run_portfolio_optimization(tickers,
                                diversification_mode=DEFAULT_DIVERSIFICATION,
                                lambda_reg=DEFAULT_LAMBDA_REG,
                                max_turnover=DEFAULT_MAX_TURNOVER,
                                user_costs=None,
                                current_weights=None):
    """
    Chạy tối ưu hóa danh mục tự động (không có input tương tác).
    Returns: (html_file, png_file)
    """
    if not tickers:
        raise ValueError("Danh sách mã trống!")
    if user_costs is None:
        user_costs = {}

    symbols = tickers + ['VNINDEX']
    data, hist_returns_dict = {}, {}

    for sym in symbols:
        result = get_data_for_symbol(
            sym,
            steps        = FORECAST_HORIZON,
            cost_basis   = user_costs.get(sym),
            auto_confirm = True,
        )
        ret, std, h_ret, last_close, last_pred, base_price = result
        if ret is not None:
            data[sym] = {
                'Expected_Return': ret,
                'Risk'           : std,
                'Cost_Basis'     : base_price,
                'Current_Price'  : last_close,
                'Predicted_Price': last_pred,
            }
            hist_returns_dict[sym] = h_ret

    if 'VNINDEX' not in data:
        raise ValueError("Không lấy được dữ liệu VNINDEX!")
    valid_tickers = [t for t in tickers if t in data]
    if not valid_tickers:
        raise ValueError("Không lấy được dữ liệu cho bất kỳ mã nào!")

    # Market stats
    mkt          = data['VNINDEX']
    market_sharpe = ((mkt['Expected_Return'] - WEEKLY_RF_RATE)
                     / mkt['Risk'] if mkt['Risk'] > 0 else 0)
    market_stats = {
        'Expected_Return': mkt['Expected_Return'],
        'Risk'           : mkt['Risk'],
        'Sharpe'         : market_sharpe,
    }

    # Core-Satellite
    all_ers = {t: data[t]['Expected_Return'] for t in valid_tickers}
    ers     = build_portfolio_strategy(
        all_ers, data,
        min_core_er=MIN_CORE_ER,
        max_satellite_loss=MAX_SATELLITE_LOSS,
        auto_confirm=True,
    )
    if ers is None:
        raise ValueError("Không có mã nào đủ điều kiện đầu tư!")

    # Covariance (weekly returns × 4 tuần)
    returns_df = pd.DataFrame(
        {t: hist_returns_dict[t] for t in ers.keys()}
    ).dropna()
    if returns_df.empty or len(returns_df) < 30:
        raise ValueError("Không đủ dữ liệu lịch sử để tính covariance!")

    lw = LedoitWolf()
    lw.fit(returns_df)
    cov_matrix = pd.DataFrame(
        lw.covariance_ * FORECAST_HORIZON,   # × 4 tuần
        index=returns_df.columns,
        columns=returns_df.columns,
    )

    # Optimize
    optimizer = PortfolioOptimizer(
        lambda_reg=lambda_reg,
        risk_free_rate=WEEKLY_RF_RATE,
    )
    f_cw = None
    if current_weights is not None:
        f_cw   = {t: current_weights.get(t, 0) for t in ers.keys()}
        total  = sum(f_cw.values())
        if total > 0:
            f_cw = {t: w / total for t, w in f_cw.items()}

    weights = optimizer.optimize(
        expected_returns    = ers,
        cov_matrix          = cov_matrix,
        diversification_mode= diversification_mode,
        current_weights     = f_cw,
        max_turnover        = max_turnover,
    )

    # Portfolio stats
    w_arr   = np.array([weights[t] for t in ers.keys()])
    er_arr  = np.array([ers[t]     for t in ers.keys()])
    cov_arr = cov_matrix.loc[list(ers.keys()), list(ers.keys())].values

    p_ret   = float(np.sum(er_arr * w_arr))
    p_risk  = float(np.sqrt(w_arr @ cov_arr @ w_arr))
    p_sharp = (p_ret - WEEKLY_RF_RATE) / p_risk if p_risk > 0 else 0
    port_stats = {'Expected_Return': p_ret, 'Risk': p_risk, 'Sharpe': p_sharp}

    config = {
        'diversification': diversification_mode,
        'lambda_reg'     : lambda_reg,
        'max_turnover'   : max_turnover,
        'shrinkage'      : f"{lw.shrinkage_:.4f}",
        'strategy'       : 'Core-Satellite (Pure RF Weekly)',
    }

    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = f"danhmuc_v5_{ts}.html"
    png_file  = f"danhmuc_v5_{ts}.png"

    create_dashboard_html(market_stats, port_stats, data, weights,
                          config, file_name=html_file)
    html_to_image(html_file, png_file=png_file)
    return html_file, png_file


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    print("\n" + "="*70)
    print("🎯 HỆ THỐNG ĐỀ XUẤT DANH MỤC ĐẦU TƯ v5")
    print("   MPT + Core-Satellite | Pure RF Weekly | 4 Tuần")
    print("="*70)

    # ── BƯỚC 1: Tickers ───────────────────────────────────
    print("\n[BƯỚC 1] Nhập danh sách mã cổ phiếu")
    print("-" * 70)
    tickers_input = input("Ví dụ: HPG,VNM,FPT,TCB,VCB\nNhập mã: ")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    if not tickers:
        print("❌ Danh sách mã trống!")
        return
    print(f"✅ Đã chọn {len(tickers)} mã: {', '.join(tickers)}")

    # ── BƯỚC 2: Diversification ───────────────────────────
    print("\n[BƯỚC 2] Chọn chế độ đa dạng hóa")
    print("-" * 70)
    print("[1] ❌ Không ràng buộc  — 0–100% mỗi mã")
    print("[2] ✅ Cân bằng         — 5–35% mỗi mã (KHUYẾN NGHỊ)")
    print("[3] 🛡️  Nghiêm ngặt   — ±20% quanh trung bình")

    while True:
        mc = input("\nLựa chọn [1/2/3] (mặc định: 2): ").strip()
        if mc in ('', '2'):
            diversification_mode = 'balanced'
            print("✅ Cân bằng")
            break
        elif mc == '1':
            if input("⚠️ Có thể 100% vào 1 mã! Tiếp tục? (y/n): ").lower() == 'y':
                diversification_mode = 'none'
                print("✅ Không ràng buộc")
                break
        elif mc == '3':
            diversification_mode = 'strict'
            print("✅ Nghiêm ngặt")
            break
        else:
            print("❌ Lựa chọn không hợp lệ!")

    # ── BƯỚC 3: Lambda ────────────────────────────────────
    print("\n[BƯỚC 3] Cấu hình nâng cao (Enter = mặc định)")
    print("-" * 70)
    li = input(f"L2 Regularization λ (mặc định {DEFAULT_LAMBDA_REG}): ").strip()
    lambda_reg = float(li) if li else DEFAULT_LAMBDA_REG

    # ── BƯỚC 4: Rebalance ────────────────────────────────
    print("\n[BƯỚC 4] Chế độ tối ưu")
    print("-" * 70)
    print("[1] Danh mục mới")
    print("[2] Rebalance từ danh mục hiện tại")

    current_weights, max_turnover = None, None
    if input("\nLựa chọn [1/2] (mặc định: 1): ").strip() == '2':
        print("\n📊 Nhập trọng số hiện tại:")
        current_weights = {}
        for t in tickers:
            while True:
                ws = input(f"  {t} (% hoặc 0–1, Enter=0): ").strip()
                if not ws:
                    current_weights[t] = 0; break
                try:
                    w = float(ws)
                    if w > 1: w /= 100
                    if 0 <= w <= 1:
                        current_weights[t] = w; break
                    else:
                        print("  ❌ Ngoài khoảng 0–1 / 0–100%")
                except ValueError:
                    print("  ❌ Không hợp lệ!")
        total_w = sum(current_weights.values())
        if total_w > 0:
            current_weights = {t: w/total_w for t, w in current_weights.items()}
        ti = input(f"\nMax Turnover (mặc định {DEFAULT_MAX_TURNOVER*100}%): ").strip()
        max_turnover = float(ti)/100 if ti else DEFAULT_MAX_TURNOVER

    # ── BƯỚC 5: Giá vốn ─────────────────────────────────
    print("\n[BƯỚC 5] Nhập giá vốn (Enter = bỏ qua, dùng giá HT)")
    print("-" * 70)
    user_costs = {}
    for t in tickers:
        while True:
            cs = input(f"Giá vốn {t}: ").strip()
            if cs == "":
                user_costs[t] = None; print(f"  → Bỏ qua {t}"); break
            try:
                c = float(cs)
                if c <= 0:
                    print("  ❌ Giá phải > 0!")
                else:
                    user_costs[t] = c
                    print(f"  ✅ {t}: {c:,.2f}"); break
            except ValueError:
                print("  ❌ Nhập sai định dạng!")

    # ── BƯỚC 6: Lấy dữ liệu ─────────────────────────────
    print("\n[BƯỚC 6] Lấy dữ liệu & chạy mô hình (weekly RF)...")
    print("-" * 70)
    symbols = tickers + ['VNINDEX']
    data, hist_returns_dict = {}, {}

    for sym in symbols:
        print(f"\n⏳ {sym}...", end=" ", flush=True)
        result = get_data_for_symbol(sym, FORECAST_HORIZON,
                                     user_costs.get(sym))
        ret, std, h_ret, last_close, last_pred, base_price = result
        if ret is not None:
            data[sym] = {
                'Expected_Return': ret,
                'Risk'           : std,
                'Cost_Basis'     : base_price,
                'Current_Price'  : last_close,
                'Predicted_Price': last_pred,
            }
            hist_returns_dict[sym] = h_ret
            print("✅")
        else:
            print("❌")

    if 'VNINDEX' not in data:
        print("\n❌ Không lấy được dữ liệu VNINDEX!")
        return
    valid_tickers = [t for t in tickers if t in data]
    if not valid_tickers:
        print("\n❌ Không lấy được dữ liệu cho bất kỳ mã nào!")
        return
    print(f"\n✅ Lấy thành công: {len(valid_tickers)} mã")

    # ── BƯỚC 7: Core-Satellite ────────────────────────────
    print("\n[BƯỚC 7] Phân tích Core-Satellite...")
    print("-" * 70)
    mkt          = data['VNINDEX']
    market_sharpe = ((mkt['Expected_Return'] - WEEKLY_RF_RATE)
                     / mkt['Risk'] if mkt['Risk'] > 0 else 0)
    market_stats = {
        'Expected_Return': mkt['Expected_Return'],
        'Risk'           : mkt['Risk'],
        'Sharpe'         : market_sharpe,
    }
    all_ers = {t: data[t]['Expected_Return'] for t in valid_tickers}
    ers = build_portfolio_strategy(
        all_ers, data,
        min_core_er=MIN_CORE_ER,
        max_satellite_loss=MAX_SATELLITE_LOSS,
    )
    if ers is None:
        print("\n✅ KẾT THÚC — Giữ tiền mặt!")
        return

    # ── BƯỚC 8: Covariance ───────────────────────────────
    print("\n[BƯỚC 8] Tính ma trận hiệp phương sai (Ledoit-Wolf)...")
    print("-" * 70)

    def _fmt_index_val(val):
        """Format giá trị index bất kể kiểu dữ liệu."""
        if val is None:
            return 'N/A'
        if hasattr(val, 'strftime'):
            return val.strftime('%Y-%m-%d')
        return str(val)

    # ── 8.1: Kiểm tra hist_returns_dict ──────────────────
    print("\n  📊 Kiểm tra dữ liệu lịch sử từng mã:")
    print(f"  {'Mã':<8} {'Tổng':>6} {'Valid':>6} {'IndexType':>15}")
    print(f"  {'-'*42}")

    raw_series = {}
    for t in ers.keys():
        if t not in hist_returns_dict:
            print(f"  ❌ {t:<6} — KHÔNG CÓ TRONG hist_returns_dict")
            continue

        s       = hist_returns_dict[t].copy()
        n_total = len(s)
        n_valid = int(s.notna().sum())
        idx_type = type(s.index).__name__
        first   = s.first_valid_index()
        last    = s.index[-1] if n_total > 0 else None
        f_str   = _fmt_index_val(first)
        l_str   = _fmt_index_val(last)

        print(f"  {'✅' if n_valid >= 30 else '⚠️ '} "
            f"{t:<6} {n_total:>6} {n_valid:>6}  "
            f"{idx_type:>15}  {f_str} → {l_str}")

        # Chuẩn hoá: loại inf/-inf, giữ nguyên index
        s = s.replace([np.inf, -np.inf], np.nan)
        raw_series[t] = s

    if len(raw_series) < 2:
        print("\n❌ Không đủ mã để tính covariance (cần ít nhất 2)!")
        return

    # ── 8.2: Align theo POSITION (tail) ──────────────────
    min_len = min(len(s) for s in raw_series.values())

    print(f"\n  🔧 Align theo vị trí:")
    print(f"  ├─ Chiều dài tối thiểu: {min_len} tuần")
    print(f"  ├─ Lấy {min_len} điểm cuối cùng của mỗi mã")

    aligned_series = {
        t: s.iloc[-min_len:].reset_index(drop=True)
        for t, s in raw_series.items()
    }

    returns_df = pd.DataFrame(aligned_series)

    print(f"  ├─ Shape sau align:     {returns_df.shape}")
    print(f"  ├─ Index sau reset:     {type(returns_df.index).__name__}")

    # ── 8.3: Kiểm tra & xử lý NaN ────────────────────────
    print(f"\n  📋 NaN mỗi cột:")
    has_nan_any = False
    for col in returns_df.columns:
        n_nan   = int(returns_df[col].isna().sum())
        n_valid = int(returns_df[col].notna().sum())
        print(f"  │    {col:<6}: {n_valid} valid  {n_nan} NaN")
        if n_nan > 0:
            has_nan_any = True

    if has_nan_any:
        print(f"\n  🔧 Xử lý NaN...")
        # [1] Forward fill
        returns_df = returns_df.ffill(limit=2)
        # [2] Backward fill
        returns_df = returns_df.bfill(limit=2)
        # [3] Fill bằng median từng cột
        for col in returns_df.columns:
            if returns_df[col].isna().any():
                med = returns_df[col].median()
                n   = int(returns_df[col].isna().sum())
                returns_df[col] = returns_df[col].fillna(med)
                print(f"  │  {col}: fill {n} NaN bằng median={med:.6f}")
        # [4] Drop hàng vẫn còn NaN
        before     = len(returns_df)
        returns_df = returns_df.dropna()
        after      = len(returns_df)
        if before != after:
            print(f"  │  Dropped {before - after} hàng còn NaN")

    # ── 8.4: Kiểm tra đủ dữ liệu ─────────────────────────
    n_obs, p = returns_df.shape
    print(f"\n  ├─ Tổng observations: {n_obs} tuần / {p} mã")

    if n_obs < p + 5:
        print(f"\n  ❌ QUÁ ÍT: {n_obs} hàng / {p} mã "
            f"(n/p = {n_obs/p:.1f}) — ma trận sẽ singular!")
        return

    min_recommended = max(30, p * 3)
    if n_obs < min_recommended:
        print(f"  ⚠️  {n_obs} obs < khuyến nghị {min_recommended} "
            f"→ tiếp tục nhưng kết quả kém tin cậy hơn")
    else:
        print(f"  ├─ n/p = {n_obs}/{p} = {n_obs/p:.1f}  ✅")

    # ── 8.5: Fit Ledoit-Wolf ──────────────────────────────
    lw = LedoitWolf()
    lw.fit(returns_df)

    cov_matrix = pd.DataFrame(
        lw.covariance_ * FORECAST_HORIZON,
        index   = returns_df.columns,
        columns = returns_df.columns,
    )

    shrinkage_label = (
        "✅ Tốt"        if lw.shrinkage_ < 0.3 else
        "⚡ Trung bình" if lw.shrinkage_ < 0.5 else
        "⚠️  Cao"
    )
    print(f"  ├─ Shrinkage: {lw.shrinkage_:.4f}  {shrinkage_label}")
    print(f"  └─ Scale × {FORECAST_HORIZON} tuần  ✅")

    # ── BƯỚC 9: Optimize ─────────────────────────────────
    print(f"\n[BƯỚC 9] Tối ưu hóa danh mục...")
    print("-" * 70)
    optimizer = PortfolioOptimizer(
        lambda_reg=lambda_reg, risk_free_rate=WEEKLY_RF_RATE)

    f_cw = None
    if current_weights is not None:
        f_cw  = {t: current_weights.get(t, 0) for t in ers.keys()}
        total = sum(f_cw.values())
        if total > 0:
            f_cw = {t: w/total for t, w in f_cw.items()}

    weights = optimizer.optimize(
        expected_returns    = ers,
        cov_matrix          = cov_matrix,
        diversification_mode= diversification_mode,
        current_weights     = f_cw,
        max_turnover        = max_turnover,
    )

    w_arr   = np.array([weights[t] for t in ers.keys()])
    er_arr  = np.array([ers[t]     for t in ers.keys()])
    cov_arr = cov_matrix.loc[list(ers.keys()), list(ers.keys())].values
    p_ret   = float(np.sum(er_arr * w_arr))
    p_risk  = float(np.sqrt(w_arr @ cov_arr @ w_arr))
    p_sharp = (p_ret - WEEKLY_RF_RATE) / p_risk if p_risk > 0 else 0
    port_stats = {'Expected_Return': p_ret, 'Risk': p_risk, 'Sharpe': p_sharp}

    # ── BƯỚC 10: In kết quả ──────────────────────────────
    print("\n" + "="*70)
    print("🎯 KẾT QUẢ ĐỀ XUẤT DANH MỤC")
    print("="*70)

    print(f"\n📊 [Thị trường — VNINDEX (4 tuần)]")
    print(f"  ├─ Lợi suất kỳ vọng: {mkt['Expected_Return']*100:>8.2f}%")
    print(f"  ├─ Rủi ro (Std):      {mkt['Risk']*100:>8.2f}%")
    print(f"  └─ Sharpe Ratio:      {market_sharpe:>8.4f}")

    print(f"\n📌 [Danh mục đề xuất]")
    print(f"  ├─ Lợi suất kỳ vọng: {p_ret*100:>8.2f}%")
    print(f"  ├─ Rủi ro (Std):      {p_risk*100:>8.2f}%")
    print(f"  ├─ Sharpe Ratio:      {p_sharp:>8.4f}")
    print(f"  └─ L2 norm:           {float(np.sum(w_arr**2)):>8.4f}")

    if p_sharp > market_sharpe:
        impr = (p_sharp - market_sharpe) / abs(market_sharpe) * 100 \
               if market_sharpe != 0 else float('inf')
        print(f"\n  ✅ THÀNH CÔNG: Sharpe cao hơn {impr:.1f}%")
    else:
        print(f"\n  ⚠️  Sharpe thấp hơn thị trường")

    print(f"\n{'Mã':<6} {'Giá Vốn':>12} {'Giá HT':>12} "
          f"{'Dự Báo 4W':>12} {'Lợi Suất':>12} {'Rủi Ro':>10}")
    print("-"*70)
    for t in valid_tickers:
        d    = data[t]
        cb_s = f"{d['Cost_Basis']:,.0f}" if d['Cost_Basis'] else "—"
        print(f"{t:<6} {cb_s:>12} {d['Current_Price']:>12,.0f} "
              f"{d['Predicted_Price']:>12,.0f} "
              f"{d['Expected_Return']*100:>11.2f}% {d['Risk']*100:>9.2f}%")

    print("\n💰 TỶ TRỌNG ĐỀ XUẤT:")
    for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        if w > 0.001:
            bar = "█" * int(w * 50)
            print(f"  {t:<6} {w*100:>6.2f}% │{bar}")

    print(f"\n✅ Tổng trọng số: {sum(weights.values()):.6f}")
    print(f"   Min: {min(weights.values())*100:.2f}%  "
          f"Max: {max(weights.values())*100:.2f}%  "
          f"Số mã: {len(weights)}/{len(valid_tickers)}")

    # ── BƯỚC 11: Dashboard ───────────────────────────────
    print("\n[BƯỚC 11] Tạo dashboard HTML...")
    print("-" * 70)
    config = {
        'diversification': diversification_mode,
        'lambda_reg'     : lambda_reg,
        'max_turnover'   : max_turnover,
        'shrinkage'      : f"{lw.shrinkage_:.4f}",
        'strategy'       : 'Core-Satellite (Pure RF Weekly)',
    }
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = f"danhmuc_v5_{ts}.html"
    create_dashboard_html(market_stats, port_stats, data,
                          weights, config, file_name=html_file)
    html_to_image(html_file)

    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print(f"   📄 HTML: {html_file}")
    print(f"   🖼️  PNG:  {html_file.replace('.html', '.png')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()