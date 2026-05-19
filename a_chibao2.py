from pyecharts import options as opts
from pyecharts.charts import Candlestick, Line, Bar, Grid
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode 
import pandas as pd
from datetime import datetime, timedelta
import talib as ta
from vnstock import Quote
import os
import numpy as np


# --- HÀM HỖ TRỢ: Tải data và tính Chỉ báo ---
def _get_data_with_indicators(symbol, months):
    """Tải dữ liệu và tính toán tất cả chỉ báo, bao gồm PAR và ATR."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=months * 40)).strftime('%Y-%m-%d')
    
    df = Quote(symbol=symbol, source='VCI').history(start=start_date, end=end_date, interval='d')
    
    if df.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")
    
    df = df.set_index('time').sort_index()
    df.columns = df.columns.str.lower()
    
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    
    # Tính toán Chỉ báo
    df['ma9'] = ta.MA(close, timeperiod=9)
    df['ma20'] = ta.MA(close, timeperiod=20)
    df['upper'], df['middle'], df['lower'] = ta.BBANDS(close, timeperiod=20)
    df['rsi'] = ta.RSI(close, timeperiod=14)
    macd, signal, hist = ta.MACD(close)
    df['macd'], df['signal'], df['hist'] = macd, signal, hist
    # Bổ sung PAR (Parabolic SAR)
    df['sar'] = ta.SAR(high, low, acceleration=0.02, maximum=0.2)
    # Bổ sung ATR (Average True Range)
    df['atr'] = ta.ATR(high, low, close, timeperiod=14)
    
    # Lọc NaN sau khi tính toán
    df = df.dropna()
    
    return df


def create_candlestick_chart(symbol, months=60):
    """
    Tạo biểu đồ nến kỹ thuật chuyên nghiệp
    
    Args:
        symbol: Mã cổ phiếu (vd: VCB)
        months: Số tháng dữ liệu (mặc định 3)
    
    Returns:
        filename: Tên file HTML được tạo, hoặc None nếu lỗi
    """
    try:
        # Lấy dữ liệu với các chỉ báo đã tính sẵn
        df = _get_data_with_indicators(symbol, months)
        
        # Chuẩn bị dữ liệu
        dates = df.index.strftime('%Y-%m-%d').tolist()
        data_ohlc = df[['open', 'close', 'low', 'high']].values.tolist()
        volumes = df['volume'].tolist()
        
        # Màu cho volume
        volume_colors = ["#26a69a" if c >= o else "#ef5350" 
                        for c, o in zip(df['close'], df['open'])]
        
        # ===== BIỂU ĐỒ NẾN (Viền đen) =====
        candlestick = (
            Candlestick(init_opts=opts.InitOpts(width="1400px", height="1000px"))
            .add_xaxis(xaxis_data=dates)
            .add_yaxis(
                series_name="Nến Giá",
                y_axis=data_ohlc,
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#26a69a",           # Nến tăng: trắng
                    color0="#ef5350",          # Nến giảm: đen
                    border_color="#000000",    # Viền: đen
                    border_color0="#000000",   # Viền: đen
                    border_width=0.5           
                ),
            )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        )
        
        # ===== ĐƯỜNG MA9 (Tắt mặc định) =====
        line_ma9 = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "MA9",
                df['ma9'].round(2).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#2196F3", width=2),
                is_hover_animation=False,
                is_connect_nones=True,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",
            )
        )
        
        # ===== ĐƯỜNG MA20 (Tắt mặc định) =====
        line_ma20 = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "MA20",
                df['ma20'].round(2).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#FF9800", width=2),
                is_hover_animation=False,
                is_connect_nones=True,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",
            )
        )
        
        # ===== PARABOLIC SAR - CHỈ DẤU CHẤM (Tắt mặc định) =====
        line_sar = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "PAR",
                df['sar'].round(2).tolist(),
                symbol="circle",
                symbol_size=6,
                itemstyle_opts=opts.ItemStyleOpts(color="#E91E63"),
                linestyle_opts=opts.LineStyleOpts(width=0),  # Không có đường nối
                is_hover_animation=False,
                is_connect_nones=False,
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        
        # ===== BOLLINGER BANDS (Tắt mặc định) =====
        line_bb_upper = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "BB Upper",
                df['upper'].round(2).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#FF12D7", width=1, type_="dashed"),
                #  areastyle_opts=opts.AreaStyleOpts(opacity=0.1, color="#F965C5"),
                is_hover_animation=False,
                is_connect_nones=True,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",
            )
        )
        
        line_bb_middle = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "BB Middle",
                df['middle'].round(2).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#BA00A8", width=0.8, type_="dashed"),
                is_hover_animation=False,
                is_connect_nones=True,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",
            )
        )
        
        line_bb_lower = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "BB Lower",
                df['lower'].round(2).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#FF12D7", width=1, type_="dashed"),
                # areastyle_opts=opts.AreaStyleOpts(
                # opacity=0.15, 
                # color="#F8F8F8"
                # ), 
                is_hover_animation=False,
                is_connect_nones=True,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",
            )
        )
        
        # ===== KHỐI LƯỢNG =====
        volume_items = [
            {"value": v, "itemStyle": {"color": c}} 
            for v, c in zip(volumes, volume_colors)
        ]
        bar_volume = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis(
                "Volume",
                volume_items,
                label_opts=opts.LabelOpts(is_show=False),
            )
        )
        
        # ===== RSI (Tắt mặc định) =====
        line_rsi = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "RSI(14)",
                df['rsi'].round(2).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#9C27B0", width=2.5),
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",

                markline_opts=opts.MarkLineOpts(
                    data=[
                        opts.MarkLineItem(y=70, name="Quá mua"),
                        opts.MarkLineItem(y=30, name="Quá bán")
                    ],
                    linestyle_opts=opts.LineStyleOpts(
                        type_="dashed",
                        color="#FF5722",
                        width=1
                    )
                ),
            )
        )
        
        # ===== MACD (Tắt mặc định) =====
        # MACD Histogram
        hist_colors = ["#26a69a" if x > 0 else "#ef5350" for x in df['hist'].tolist()]
        bar_macd = (
            Bar()
            .add_xaxis(dates)
            .add_yaxis(
                "MACD Hist",
                df['hist'].round(4).tolist(),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode("""
                        function(params) {
                            var colorList = %s;
                            return colorList[params.dataIndex];
                        }
                    """ % hist_colors)
                ),
            )
        )
        
        # MACD Line
        line_macd = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "MACD",
                df['macd'].round(4).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#1976D2", width=2),
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",
            )
        )
        
        # Signal Line
        line_signal = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "Signal",
                df['signal'].round(4).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#FF6F00", width=2),
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False),
                symbol ="none",
            )
        )
        
        # Kết hợp MACD
        macd_chart = bar_macd.overlap(line_macd).overlap(line_signal)
        
        # ===== ATR (Tắt mặc định) =====
        line_atr = (
            Line()
            .add_xaxis(dates)
            .add_yaxis(
                "ATR(14)",
                df['atr'].round(2).tolist(),
                linestyle_opts=opts.LineStyleOpts(color="#FF5252", width=2.5),
                symbol ="none",
                is_hover_animation=False,
                label_opts=opts.LabelOpts(is_show=False),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#FF5252"),
            )
        )
        
        # ===== KẾT HỢP CÁC CHỈ BÁO VÀO BIỂU ĐỒ CHÍNH =====
        main_chart = (candlestick
                     .overlap(line_ma9)
                     .overlap(line_ma20)
                     .overlap(line_sar)
                     .overlap(line_bb_upper)
                     .overlap(line_bb_lower)
                     .overlap(line_bb_middle)
                    )                     
        
        # ===== CẤU HÌNH BIỂU ĐỒ CHÍNH =====
        main_chart.set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"📊 {symbol.upper()} - BIỂU ĐỒ KỸ THUẬT",
                subtitle=f"📅 {months} tháng • Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=22, font_weight="bold", color="#2c3e50"),
                subtitle_textstyle_opts=opts.TextStyleOpts(font_size=13, color="#7f8c8d")
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(255, 255, 255, 0.98)",
                border_width=1,
                border_color="#ddd",
                textstyle_opts=opts.TextStyleOpts(color="#000", font_size=12)
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    type_="inside",
                    xaxis_index=[0, 1, 2, 3, 4],
                    range_start=50,
                    range_end=100
                ),
                # opts.DataZoomOpts(
                #     type_="slider",
                #     xaxis_index=[0, 1, 2, 3, 4],
                #     pos_bottom="0.5%",
                #     height=25,
                #     range_start=0,
                #     range_end=100
                # )
            ],
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_show=False,
                grid_index=0,
                boundary_gap=False,
                splitline_opts=opts.SplitLineOpts(is_show=False)
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                grid_index=0,
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.2)),
                position="right",
                axislabel_opts=opts.LabelOpts(font_size=11),
                is_show = True
            ),
            legend_opts=opts.LegendOpts(
                type_="scroll",
                pos_left="8%",
                pos_top="5.5%",
                orient="horizontal",
                selected_mode="multiple",
                textstyle_opts=opts.TextStyleOpts(font_size=12),
                item_gap=15,
                padding=8
            )
        )
        
        # ===== CẤU HÌNH VOLUME =====
        bar_volume.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=1,
                is_show=False,
                splitline_opts=opts.SplitLineOpts(is_show=False)
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                split_number=2,
                is_show=False,
                position="right",
                axislabel_opts=opts.LabelOpts(font_size=10),
                splitline_opts=opts.SplitLineOpts(is_show=False)
            ),
            legend_opts=opts.LegendOpts(is_show=False)
        )
        
        # ===== CẤU HÌNH RSI =====
        line_rsi.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=2,
                is_show=False,
                splitline_opts=opts.SplitLineOpts(is_show=False)
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=2,
                is_show=False,
                position="right",
                min_=0,
                max_=100,
                axislabel_opts=opts.LabelOpts(font_size=10),
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.2))
            ),
            legend_opts=opts.LegendOpts(
                pos_top="59.5%",
                pos_left="8%",
                orient="horizontal",
                textstyle_opts=opts.TextStyleOpts(font_size=11)
            )
        )
        
        # ===== CẤU HÌNH MACD =====
        macd_chart.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=3,
                is_show=False,
                splitline_opts=opts.SplitLineOpts(is_show=False)
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=3,
                is_show=True,
                position="right",
                axislabel_opts=opts.LabelOpts(font_size=10),
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.2))
            ),
            legend_opts=opts.LegendOpts(
                pos_top="75.5%",
                pos_left="8%",
                orient="horizontal",
                textstyle_opts=opts.TextStyleOpts(font_size=11)
            )
        )
        
        # ===== CẤU HÌNH ATR =====
        line_atr.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                grid_index=4,
                is_show=False,
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(font_size=10)
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=4,
                is_show=True,
                position="right",
                axislabel_opts=opts.LabelOpts(font_size=10),
                splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.2))
            ),
            legend_opts=opts.LegendOpts(
                pos_top="91.5%",
                pos_left="8%",
                orient="horizontal",
                textstyle_opts=opts.TextStyleOpts(font_size=11)
            )
        )
        
        # ===== KẾT HỢP TẤT CẢ VÀO GRID =====
        grid = (
            Grid(init_opts=opts.InitOpts(width="1400px", height="1400px"))
            .add(
                main_chart,
                grid_opts=opts.GridOpts(
                    pos_left="3%",
                    pos_right="8%",
                    pos_top="10%",
                    height="38%"
                )
            )
            .add(
                bar_volume,
                grid_opts=opts.GridOpts(
                    pos_left="3%",
                    pos_right="8%",
                    pos_top="51%",
                    height="7%"
                )
            )
            .add(
                line_rsi,
                grid_opts=opts.GridOpts(
                    pos_left="3%",
                    pos_right="8%",
                    pos_top="62%",
                    height="10%"
                )
            )
            .add(
                macd_chart,
                grid_opts=opts.GridOpts(
                    pos_left="3%",
                    pos_right="8%",
                    pos_top="78%",
                    height="10%"
                )
            )
            .add(
                line_atr,
                grid_opts=opts.GridOpts(
                    pos_left="3%",
                    pos_right="8%",
                    pos_top="93%",
                    height="10%"
                )
            )
        )
        
        # Lưu file
        filename = f"chart_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
        grid.render(filename)
        
        print(f"✅ Đã tạo biểu đồ: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo biểu đồ cho {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


# # Ví dụ chạy:
# if __name__ == "__main__":
#     create_candlestick_chart("HPG", 6)