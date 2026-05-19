import sys
import codecs
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

import discord
from discord.ext import commands
from discord import *
import os
from datetime import datetime
from vnstock import Listing
import asyncio
from dotenv import load_dotenv
import json
import re
import requests
# =====================================================
# Import hàm đã tạo
from a_ML import *
from a_ML2 import *
from a_dash2 import create_terminal_dashboard2, html_to_image2
from a_ML3 import model_exists as model_exists3, backtest_model as backtest_model3, train_model_for_symbol as train_model_for_symbol3
from a_dash3 import create_terminal_dashboard3, html_to_image3
import a_port4 as danhmuc
from a_ML4_daily import model_exists as model_exists4, train_model_for_symbol as train_model_for_symbol4, backtest_model as backtest_model4
from a_dash4 import create_terminal_dashboard4, html_to_image4

# =====================================================
# CẤU HÌNH BOT
# =====================================================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Khởi tạo listing để tìm kiếm mã
listing = Listing(source='VCI')

@bot.event
async def on_ready():
    """Khi bot khởi động thành công"""
    try:
        synced = await bot.tree.sync()
        print(f'✅ Bot {bot.user} đã sẵn sàng!')
        print(f'✅ Đã đồng bộ {len(synced)} lệnh slash')
        
        # Gửi tin nhắn chào mừng vào mọi server
        welcome_message = (
            "👋 **Chào mừng bạn đến với Bot Phân tích Chứng khoán!**\n\n"
            "Mình là trợ lý AI chuyên phân tích cổ phiếu Việt Nam, giúp bạn xem biểu đồ, báo cáo tài chính, dự báo giá và hơn thế nữa!\n\n"
            "🔥 **Cách sử dụng:**\n"
            "• Gõ `/tuvan` để hỏi mình về cách dùng lệnh bot...\n"
            "• Ví dụ: `/tuvan <cau_hoi>: Lệnh /nen dùng để làm gì?`\n\n"
            "⚠️ **Lưu ý:** Tất cả phân tích chỉ mang tính tham khảo, không phải lời khuyên đầu tư. Hãy tự chịu trách nhiệm với quyết định của bạn.\n\n"
            "Chúc bạn đầu tư thành công! 📈\n"
        )
        
        for guild in bot.guilds:
            # Tìm kênh text đầu tiên bot có quyền gửi tin nhắn
            for channel in guild.text_channels:
                if channel.permissions_for(guild.me).send_messages:
                    try:
                        await channel.send(welcome_message)
                        print(f"Đã gửi chào mừng vào {guild.name} - kênh {channel.name}")
                    except:
                        print(f"Không gửi được vào {guild.name} - {channel.name}")
                    break  # Chỉ gửi vào 1 kênh mỗi server

    except Exception as e:
        print(f'❌ Lỗi khi đồng bộ lệnh hoặc gửi chào mừng: {e}')

# =====================================================
async def _render_html_to_png(html_filename, png_filename):
    try:
        from pyppeteer import launch
        browser = await launch(headless=True, args=['--no-sandbox'])
        page = await browser.newPage()
        await page.setViewport({'width': 1600, 'height': 1000})
        await page.goto(f"file://{os.path.abspath(html_filename)}")
        await page.waitForSelector('canvas', timeout=10000)
        await page.screenshot({'path': png_filename, 'fullPage': False})
        await browser.close()
        print(f"Đã xuất PNG: {png_filename}")
    except Exception as e:
        print(f"Lỗi render PNG: {e}")
        raise e

# =====================================================
# COMMAND: /dash – Tạo dashboard phân tích kỹ thuật
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)",
    thang="Số tháng dữ liệu lịch sử (1-60, mặc định 60)",
    so_phien_du_bao="Số phiên dự báo (1-30, mặc định 5)"
)
@bot.tree.command(name="dash2", description="Chi tiết cách tính mô hình 2 (Lasso)")
async def dash_cmd(interaction: discord.Interaction, ma_cp: str,so_phien_du_bao: int = 5,  thang: int = 60):
    await interaction.response.defer()

    symbol = ma_cp.strip().upper()

    # Validate
    if not (1 <= thang <= 60):
        await interaction.followup.send("❌ Số tháng phải từ 1 đến 60!", ephemeral=True)
        return
    if not (1 <= so_phien_du_bao <= 30):
        await interaction.followup.send("❌ Số phiên dự báo phải từ 1 đến 30!", ephemeral=True)
        return

    # Kiểm tra model
    if not model_exists(symbol):   # dùng hàm có sẵn trong a_ML2.py
        await interaction.followup.send(
            f"❌ Chưa có model cho **{symbol}**!\n"
            f"Gõ `/train2 {symbol}` để train model trước.",
            ephemeral=True
        )
        return

    msg = await interaction.followup.send(
        f"📊 Đang tạo dashboard cho **{symbol}**...\n"
        f"⏳ Đang tải dữ liệu & render biểu đồ..."
    )

    html_file = None
    png_file  = None

    try:
        # 1. Tạo HTML dashboard trong luồng phụ để tránh block event loop
        html_file = await asyncio.get_event_loop().run_in_executor(
            None, 
            create_terminal_dashboard,
            symbol,
            thang,
            so_phien_du_bao
        )

        # 2. Chuyển HTML → PNG
        png_file = await asyncio.get_event_loop().run_in_executor(
            None, html_to_image, html_file
        )

        # 3. Embed
        embed = discord.Embed(
            title=f"📊 Dashboard • {symbol}",
            description=f"Dữ liệu {thang} tháng • Dự báo {so_phien_du_bao} phiên tới",
            color=0x58a6ff
        )
        embed.set_image(url=f"attachment://{os.path.basename(png_file)}")
        embed.set_footer(text=f"Generated {datetime.now().strftime('%d/%m/%Y %H:%M')}")

        # 4. Gửi cả PNG + HTML
        await interaction.followup.send(
            embed=embed,
            files=[
                discord.File(png_file, filename=f"dashboard_{symbol}.png"),
                discord.File(html_file, filename=f"dashboard_{symbol}.html"),
            ]
        )

        try:
            await msg.delete()
        except:
            pass

        print(f"✅ Dashboard {symbol} hoàn tất")

    except ValueError as e:
        await interaction.followup.send(f"❌ Lỗi dữ liệu: `{str(e)}`", ephemeral=True)
        print(f"[Lỗi /dash ValueError] {symbol}: {e}")

    except Exception as e:
        await interaction.followup.send(
            f"❌ Lỗi khi tạo dashboard cho **{symbol}**:\n`{str(e)}`",
            ephemeral=True
        )
        print(f"[Lỗi /dash] {symbol}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        for f in [html_file, png_file]:
            if f and os.path.exists(f):
                for _ in range(5):
                    try:
                        os.remove(f)
                        print(f"Đã xóa file tạm: {f}")
                        break
                    except PermissionError:
                        await asyncio.sleep(0.5)
                    except Exception as del_err:
                        print(f"Không xóa được {f}: {del_err}")
                        break

# ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
# COMMAND: /backtest2
# ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số",
    ngay_bat_dau="Ngày bắt đầu backtest (YYYY-MM-DD, mặc định 2025-01-01)"
)
@bot.tree.command(name="backtest2", description="Backtest model dự báo")
async def backtest_cmd(interaction: discord.Interaction, ma_cp: str, ngay_bat_dau: str = "2025-01-01"):
    await interaction.response.defer()
    symbol = ma_cp.strip().upper()

    if not model_exists(symbol):
        await interaction.followup.send(
            f"❌ Chưa có model cho **{symbol}**!\n"
            f"Gõ `/train1 {symbol}` để train trước.",
            ephemeral=True
        )
        return

    # Validate ngày
    try:
        datetime.strptime(ngay_bat_dau, "%Y-%m-%d")
    except ValueError:
        await interaction.followup.send(
            "❌ Định dạng ngày không hợp lệ! Dùng YYYY-MM-DD (VD: 2025-01-01)",
            ephemeral=True
        )
        return

    msg = await interaction.followup.send(
        f"⚙️ Đang backtest **{symbol}** từ {ngay_bat_dau}...\n"
        f"⏳ Quá trình này có thể mất vài phút..."
    )

    try:
        # Chạy trong executor để không block event loop
        bt_df = await asyncio.get_event_loop().run_in_executor(
            None, backtest_model, symbol, ngay_bat_dau
        )

        if bt_df is None or bt_df.empty:
            await interaction.followup.send("❌ Không có kết quả backtest!", ephemeral=True)
            return

        # Tính metrics
        mape    = bt_df["% Sai Hybrid"].mean()
        mae     = bt_df["Sai số Hybrid"].abs().mean()
        diracc  = (bt_df["Hướng Hybrid"] == "✔").mean() * 100
        sessions = len(bt_df)

        embed = discord.Embed(
            title=f"⚙️ Backtest • {symbol}",
            description=f"Từ {ngay_bat_dau} • {sessions} phiên",
            color=0xf4d35e
        )
        embed.add_field(name="MAE",         value=f"`{mae:.2f}`",      inline=True)
        embed.add_field(name="MAPE",        value=f"`{mape:.3f}%`",    inline=True)
        embed.add_field(name="Độ chính xác hướng", value=f"`{diracc:.1f}%`", inline=True)

        # Gửi CSV kết quả
        csv_file = f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        bt_df.to_csv(csv_file, index=False, encoding="utf-8-sig")

        await interaction.followup.send(
            embed=embed,
            files=[discord.File(csv_file, filename=f"backtest_{symbol}.csv")]
        )

        try:
            await msg.delete()
        except:
            pass

    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi backtest: `{str(e)}`", ephemeral=True)
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(csv_file):
            try:
                os.remove(csv_file)
            except:
                pass

# ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
# COMMAND: /train2 – Train model Lasso + Momentum
# ==========================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số cần train (VD: VNINDEX, HPG, VNM...)"
)
@bot.tree.command(name="train2", description="Giải thích mô hình 2 (Lasso + Momentum)")
async def train2_cmd(interaction: discord.Interaction, ma_cp: str):
    await interaction.response.defer()

    symbol = ma_cp.strip().upper()

    try:
        company_info = get_company_info(symbol)
        company_name = company_info.get('company_name', symbol)
    except:
        company_name = symbol

    # Kiểm tra model đã tồn tại chưa
    if model_exists(symbol):
        await interaction.followup.send(
            f"⚠️ Model cho **{symbol}** đã tồn tại! Model cũ sẽ bị ghi đè khi train lại.",
            ephemeral=True
        )

    msg = await interaction.followup.send(
        f"🤖 Đang train model cho **{symbol}** ({company_name})...\n"
        f"⏳ Bước 1/4: Lấy dữ liệu lịch sử ({CONFIG['start_date']} → {CONFIG['end_date']})..."
    )

    try:
        import time

        # === BƯỚC 1: LẤY & VALIDATE DỮ LIỆU ===
        df = load_data(symbol, CONFIG["start_date"], CONFIG["end_date"], CONFIG["interval"])
        df = validate_data(df)

        await msg.edit(content=
            f"🤖 Train model **{symbol}**\n"
            f"⏳ Bước 2/4: Tính toán features & indicators..."
        )

        # === BƯỚC 2: FEATURE ENGINEERING ===
        df = compute_features(df)

        await msg.edit(content=
            f"🤖 Train model **{symbol}**\n"
            f"⏳ Bước 3/4: Training Lasso model (có thể mất vài phút)..."
        )

        # === BƯỚC 3: TRAIN ===
        start_time = time.time()

        model, feature_scaler, target_scaler = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: fit_final_model(
                df=df,
                feature_list=CORE_MOMENTUM_FEATURES,
                target=TARGET,
                lookback=CONFIG["lookback"]
            )
        )

        train_time = time.time() - start_time

        await msg.edit(content=
            f"🤖 Train model **{symbol}**\n"
            f"⏳ Bước 4/4: Đánh giá & lưu model..."
        )

        # === BƯỚC 4: ĐÁNH GIÁ NHANH trên 20% cuối ===
        split = int(len(df) * 0.8)
        df_test = df.iloc[split:].copy()

        preds, actuals = [], []
        for i in range(CONFIG["lookback"], len(df_test)):
            past = df_test.iloc[i - CONFIG["lookback"]:i]
            try:
                p = predict_next_price(
                    past_window=past,
                    feature_list=CORE_MOMENTUM_FEATURES,
                    model=model,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                    lookback=CONFIG["lookback"]
                )
                preds.append(p)
                actuals.append(float(df_test["close"].iloc[i]))
            except:
                continue

        preds   = np.array(preds)
        actuals = np.array(actuals)

        mae     = float(np.mean(np.abs(preds - actuals)))
        mape    = float(np.mean(np.abs((preds - actuals) / actuals)) * 100)
        diracc  = float(np.mean(
            np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
        ) * 100) if len(preds) > 1 else 0.0

        # === LƯU MODEL ===
        save_model_package(
            symbol=symbol,
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            feature_list=CORE_MOMENTUM_FEATURES,
            target=TARGET
        )

        # === EMBED KẾT QUẢ ===
        embed = discord.Embed(
            title=f"✅ Train Model Thành Công • {symbol}",
            description=f"**{company_name}**\n"
                        f"Dữ liệu: {CONFIG['start_date']} → {CONFIG['end_date']}",
            color=0x00ff00
        )

        embed.add_field(
            name="📊 Hiệu suất Model (20% test cuối)",
            value=f"```\n"
                  f"MAE:              {mae:.2f} điểm\n"
                  f"MAPE:             {mape:.3f}%\n"
                  f"Directional Acc:  {diracc:.1f}%\n"
                  f"Test samples:     {len(preds)}\n"
                  f"```",
            inline=False
        )

        embed.add_field(
            name="⚙️ Thông tin Training",
            value=f"```\n"
                  f"Algorithm:   Lasso + Summary Sequences\n"
                  f"Features:    {len(CORE_MOMENTUM_FEATURES)} core momentum\n"
                  f"Lookback:    {CONFIG['lookback']} phiên\n"
                  f"Train rows:  {split} samples\n"
                  f"Time:        {train_time:.1f}s\n"
                  f"```",
            inline=False
        )

        embed.add_field(
            name="🎯 Sử dụng Model",
            value=f"• `/dash2 {symbol}` - Chi tiết cách tính\n"
                  f"• `/backtest2 {symbol}` - Kiểm tra độ chính xác",
            inline=False
        )

        paths = get_model_paths(symbol)
        embed.set_footer(text=f"Saved: {paths['model']} • {datetime.now().strftime('%d/%m/%Y %H:%M')}")

        await msg.edit(content=None, embed=embed)
        print(f"✅ Train2 {symbol} hoàn tất — MAE: {mae:.2f}, DirAcc: {diracc:.1f}%")

    except Exception as e:
        await msg.edit(content=f"❌ Lỗi khi train model: `{str(e)}`")
        print(f"[Lỗi /train2] {symbol}: {e}")
        import traceback
        traceback.print_exc()

# =====================================================
# COMMAND: /backtest – Kiểm tra hiệu suất model
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)",
    ngay_bat_dau="Ngày bắt đầu backtest (DD/MM/YYYY, mặc định 16/12/2025)"
)
@bot.tree.command(name="backtest1", description="Backtest mô hình 1 (Random Forest)")
async def backtest_cmd(interaction: discord.Interaction, ma_cp: str, ngay_bat_dau: str = "16/12/2025"):
    await interaction.response.defer()
    
    symbol = ma_cp.strip().upper()
    
    # Validate và convert ngày
    try:
        backtest_start = datetime.strptime(ngay_bat_dau, '%d/%m/%Y').strftime('%Y-%m-%d')
    except:
        await interaction.followup.send("❌ Định dạng ngày không đúng! Dùng DD/MM/YYYY (VD: 16/12/2025)", ephemeral=True)
        return
    
    # Kiểm tra model tồn tại
    rf_model_path = f'saved_model/best_randomforest_default_{symbol}.joblib'
    scaler_path = f'saved_model/scaler_randomforest_{symbol}.joblib'
    
    if not (os.path.exists(rf_model_path) and os.path.exists(scaler_path)):
        await interaction.followup.send(
            f"❌ Chưa có model cho **{symbol}**!\n"
            f"Gõ `/train1 {symbol}` để train model trước.",
            ephemeral=True
        )
        return
    
    # Lấy tên công ty
    try:
        company_info = get_company_info(symbol)
        company_name = company_info.get('company_name', symbol)
    except:
        company_name = symbol
    
    msg = await interaction.followup.send(
        f"🔄 Đang backtest model **{symbol}** từ {ngay_bat_dau}...\n"
        "⏳ Đang tính toán độ chính xác..."
    )
    
    png_filename = None  # Để finally xóa an toàn
    csv_filename = None
    
    try:
        # Chạy backtest
        result = backtest_vnindex_adaptive(
            symbol=symbol,
            backtest_start=backtest_start,
            rf_model_path=rf_model_path,
            scaler_path=scaler_path,
            learning_window=120
        )
        
        if result is None or not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("Backtest thất bại: Kết quả không hợp lệ")
        
        table, mae_recent, acc = result
        
        if table is None or table.empty:
            raise ValueError("Không có dữ liệu backtest hợp lệ")
        
        print(f"Table shape: {table.shape}, Columns: {list(table.columns)}")  # debug
        
        # Tạo CSV
        csv_filename = f'backtest_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        table.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        # Tính MAE an toàn
        try:
            sai_so_clean = table['Sai số'].astype(str).str.replace(r'[\+\s]', '', regex=True)
            mae_all = sai_so_clean.astype(float).abs().mean()
        except Exception as e:
            print(f"Lỗi tính MAE: {e}")
            mae_all = 0.0
        
        # Embed
        embed = discord.Embed(
            title=f"📊 Kết quả Backtest • {symbol}",
            description=f"**{company_name}**\nTừ {ngay_bat_dau} • {len(table)} phiên",
            color=0x00ff88
        )
        
        embed.add_field(
            name="📈 Độ chính xác",
            value=f"```\nMAE toàn bộ:  {mae_all:.2f} điểm\nMAE 5 phiên:  {mae_recent:.2f} điểm\nAccuracy:     {acc:.2f}%\n```",
            inline=False
        )
        
        if acc >= 70:
            rating = "🟢 Xuất sắc"
        elif acc >= 60:
            rating = "🟡 Tốt"
        elif acc >= 50:
            rating = "🟠 Trung bình"
        else:
            rating = "🔴 Cần cải thiện"
        
        embed.add_field(name="⭐ Đánh giá", value=f"{rating} - Accuracy **{acc:.1f}%**", inline=False)
        embed.set_footer(text=f"Gõ /dubao {symbol} để dự báo • {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        # Ảnh bảng
        display_table = table.tail(30)
        num_cols = len(display_table.columns)
        col_widths = [0.9 / num_cols] * num_cols
        
        table_buf = create_table_image(
            display_table,
            title=f"Backtest {symbol} ({len(table)} phiên)",
            col_widths=col_widths
        )
        
        # Gửi message mới
        await interaction.followup.send(
            content="**Kết quả backtest đã sẵn sàng!** 📈",
            embed=embed,
            files=[
                discord.File(table_buf, filename="bang_backtest.png"),
                discord.File(csv_filename, filename=f"backtest_{symbol}.csv")
            ]
        )
        
        # Xóa loading message
        try:
            await msg.delete()
        except:
            pass
        
        print(f"✅ Backtest {symbol} hoàn tất - Accuracy: {acc:.1f}%")
        
    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi backtest: `{str(e)}`")
        print(f"[Lỗi /backtest] {symbol}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Đảm bảo xóa file dù có lỗi hay không
        table_buf.close() if 'table_buf' in locals() else None
        
        for f in [csv_filename]:
            if f and os.path.exists(f):
                for _ in range(5):  # Thử 5 lần
                    try:
                        os.remove(f)
                        print(f"Đã xóa file tạm: {f}")
                        break
                    except PermissionError:
                        await asyncio.sleep(0.5)
                    except Exception as delete_err:
                        print(f"Không xóa được {f}: {delete_err}")
                        break


# =====================================================
# COMMAND: /dubao – Dự báo giá tương lai
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)",
    so_phien="Số phiên dự báo (1-30, mặc định 5)"
)
@bot.tree.command(name="dubao", description="Dự báo giá mô hình 1 (Random Forest)")
async def dubao_cmd(interaction: discord.Interaction, ma_cp: str, so_phien: int = 5):
    await interaction.response.defer()
    
    symbol = ma_cp.strip().upper()
    
    if not (1 <= so_phien <= 30):
        await interaction.followup.send("❌ Số phiên phải từ 1 đến 30!", ephemeral=True)
        return
    
    rf_model_path = f'saved_model/best_randomforest_default_{symbol}.joblib'
    scaler_path = f'saved_model/scaler_randomforest_{symbol}.joblib'
    
    if not (os.path.exists(rf_model_path) and os.path.exists(scaler_path)):
        await interaction.followup.send(
            f"❌ Chưa có model cho **{symbol}**!\n"
            f"Gõ `/train1 {symbol}` để train model trước.",
            ephemeral=True
        )
        return
    
    try:
        company_info = get_company_info(symbol)
        company_name = company_info.get('company_name', symbol)
    except:
        company_name = symbol
    
    msg = await interaction.followup.send(
        f"🔮 Đang dự báo **{symbol}** cho {so_phien} phiên tương lai...\n"
        "⏳ Đang tính toán & vẽ biểu đồ..."
    )
    
    png_filename = None
    csv_filename = None
    table_buf = None
    
    try:
        yesterday = datetime.now() - timedelta(days=1)
        end_fetch = yesterday.strftime('%Y-%m-%d')
        
        quote = Quote(symbol=symbol, source='VCI')
        df_full_for_plot = quote.history(start='2020-01-01', end=end_fetch, interval='d')
        df_full_for_plot = df_full_for_plot.sort_values('time').reset_index(drop=True)
        df_full_for_plot['time'] = pd.to_datetime(df_full_for_plot['time'])
        df_full_for_plot = compute_indicators_single(df_full_for_plot)
        
        forecast_df = forecast_future_prices(
            symbol=symbol,
            forecast_steps=so_phien,
            rf_model_path=rf_model_path,
            scaler_path=scaler_path,
            learning_window=120,
            show_details=False,
            use_yesterday=True
        )
        
        if forecast_df is None:
            raise ValueError("Không thể dự báo cho symbol này")
        
        csv_filename = f'dubao_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        forecast_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        table_buf = create_table_image(
            forecast_df,
            title=f"Dự báo {so_phien} phiên • {symbol}",
            col_widths=[0.08, 0.15, 0.15, 0.12, 0.25]
        )
        
        png_filename = plot_forecast_with_history(
            symbol=symbol,
            forecast_df=forecast_df,
            historical_data=df_full_for_plot,
            lookback=22
        )
        
        try:
            last_price = float(df_full_for_plot['close'].iloc[-1])
            first_forecast = float(forecast_df['Giá dự báo'].iloc[0].replace(',', ''))
            last_forecast = float(forecast_df['Giá dự báo'].iloc[-1].replace(',', ''))
            total_change = (last_forecast / last_price - 1) * 100
        except:
            last_price = 0
            total_change = 0
        
        embed = discord.Embed(
            title=f"🔮 Dự báo {so_phien} phiên • {symbol}",
            color=0xff9500
        )
        
        max_change = forecast_df['Thay đổi'].str.replace('+', '').str.replace('%', '').astype(float).abs().max()
        if max_change > 3:
            embed.add_field(
                name="⚠️ Lưu ý",
                value=f"Có phiên biến động **{max_change:.2f}%** - Cần theo dõi sát thị trường",
                inline=False
            )
        
        embed.set_image(url=f"attachment://{os.path.basename(png_filename)}")
        embed.set_footer(text=f"Model: Hybrid RF + Momentum • {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        attachments = [
            discord.File(table_buf, filename="bang_du_bao.png"),
            discord.File(png_filename, filename="du_bao_bieu_do.png"),
            discord.File(csv_filename, filename=f"du_bao_{symbol}.csv")
        ]
        
        await interaction.followup.send(
            content="**Kết quả dự báo đã sẵn sàng!** 📊",
            embed=embed,
            files=attachments
        )
        
        try:
            await msg.delete()
        except:
            pass
        
        print(f"✅ Dự báo {symbol} hoàn tất")
        
    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi dự báo: `{str(e)}`")
        print(f"[Lỗi /dubao] {symbol}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Đảm bảo dọn dẹp
        if table_buf:
            table_buf.close()
        
        for f in [png_filename, csv_filename]:
            if f and os.path.exists(f):
                for _ in range(5):
                    try:
                        os.remove(f)
                        print(f"Đã xóa file tạm: {f}")
                        break
                    except PermissionError:
                        await asyncio.sleep(0.5)
                    except Exception as delete_err:
                        print(f"Không xóa được {f}: {delete_err}")
                        break

# =====================================================
# COMMAND: /train – Train model cho mã cổ phiếu
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số cần train (VD: VNINDEX, HPG, VNM...)"
)
@bot.tree.command(name="train1", description="Giải thích mô hình 1 (Random Forest)")
async def train_cmd(interaction: discord.Interaction, ma_cp: str):
    await interaction.response.defer()
    
    symbol = ma_cp.strip().upper()
    
    # Lấy thông tin công ty
    try:
        company_info = get_company_info(symbol)
        company_name = company_info.get('company_name', symbol)
    except:
        company_name = symbol
    
    # Kiểm tra model đã tồn tại chưa
    rf_model_path = f'saved_model/best_randomforest_default_{symbol}.joblib'
    scaler_path = f'saved_model/scaler_randomforest_{symbol}.joblib'
    
    model_exists = os.path.exists(rf_model_path) and os.path.exists(scaler_path)
    
    if model_exists:
        # Hỏi user có muốn retrain không
        msg = await interaction.followup.send(
            f"⚠️ Model cho **{symbol}** đã tồn tại!\n"
            f"Bạn có muốn train lại không? (Model cũ sẽ bị ghi đè)\n"
            f"React ✅ để tiếp tục, ❌ để hủy",
            ephemeral=True
        )
        # TODO: Thêm logic confirm với reactions nếu cần
        # Tạm thời cho phép train lại luôn
    
    msg = await interaction.followup.send(
        f"🤖 Đang train model cho **{symbol}** ({company_name})...\n"
        f"⏳ Bước 1/5: Lấy dữ liệu lịch sử (2019-2025)..."
    )
    
    try:
        # === BƯỚC 1: LẤY DỮ LIỆU ===
        START_DATE = '2019-06-15'
        END_DATE = '2025-12-15'
        INTERVAL = 'd'
        
        all_dataframes = get_stock_historical_data(
            symbols=symbol,
            start_date=START_DATE,
            end_date=END_DATE,
            interval=INTERVAL
        )
        
        if symbol not in all_dataframes:
            await msg.edit(content=f"❌ Không thể lấy dữ liệu cho **{symbol}**!\nVui lòng kiểm tra lại mã.")
            return
        
        await msg.edit(content=f"🤖 Train model **{symbol}**\n⏳ Bước 2/5: Tính toán chỉ báo kỹ thuật...")
        
        # === BƯỚC 2: TÍNH INDICATORS ===
        all_dataframes = compute_indicators_for_all(all_dataframes)
        
        await msg.edit(content=f"🤖 Train model **{symbol}**\n⏳ Bước 3/5: Chuẩn bị dữ liệu training...")
        
        # === BƯỚC 3: CHUẨN BỊ DỮ LIỆU ===
        hybrid_data = prepare_hybrid_residual(all_dataframes)
        
        hr = hybrid_data[symbol]['hybrid_residual']
        scaler = hybrid_data[symbol]['scalers']['hybrid_residual']
        
        X_train_3d = hr['X_train']
        y_train = hr['y_train']
        X_test_3d = hr['X_test']
        y_test = hr['y_test']
        actual_prices_test = hr['raw_close_test']
        zlema20_test = hr.get('zlema20_test')
        
        # Flatten cho Random Forest
        X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
        X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)
        
        await msg.edit(content=f"🤖 Train model **{symbol}**\n⏳ Bước 4/5: Training Random Forest (có thể mất vài phút)...")
        
        # === BƯỚC 4: TRAIN MODEL ===
        import time
        start_time = time.time()
        
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_2d, y_train)
        
        train_time = time.time() - start_time
        
        await msg.edit(content=f"🤖 Train model **{symbol}**\n⏳ Bước 5/5: Đánh giá model...")
        
        # === BƯỚC 5: ĐÁNH GIÁ MODEL ===
        def inverse_residual_local(scaled_residual, scaler):
            dummy = np.zeros((len(scaled_residual), scaler.n_features_in_))
            dummy[:, -1] = scaled_residual
            return scaler.inverse_transform(dummy)[:, -1]
        
        pred_residual_rf = inverse_residual_local(rf_model.predict(X_test_2d), scaler)
        pred_prices = zlema20_test + pred_residual_rf
        
        rmse = np.sqrt(mean_squared_error(actual_prices_test, pred_prices))
        mae = mean_absolute_error(actual_prices_test, pred_prices)
        mape_val = np.mean(np.abs((actual_prices_test - pred_prices) / actual_prices_test)) * 100
        r2 = r2_score(actual_prices_test, pred_prices)
        dir_acc = np.mean(np.sign(np.diff(actual_prices_test)) == np.sign(np.diff(pred_prices))) * 100
        
        # === BƯỚC 6: LƯU MODEL ===
        os.makedirs("saved_model", exist_ok=True)
        joblib.dump(rf_model, rf_model_path)
        joblib.dump(scaler, scaler_path)
        
        # === TẠO EMBED KẾT QUẢ ===
        embed = discord.Embed(
            title=f"✅ Train Model Thành Công • {symbol}",
            description=f"**{company_name}**\n"
                       f"Dữ liệu: {START_DATE} → {END_DATE}",
            color=0x00ff00
        )
        
        embed.add_field(
            name="📊 Hiệu suất Model (Thuần)",
            value=f"```\n"
                  f"MAE:              {mae:.2f} điểm\n"
                  f"RMSE:             {rmse:.2f} điểm\n"
                  f"MAPE:             {mape_val:.2f}%\n"
                  f"R² Score:         {r2:.3f}\n"
                  f"Directional Acc:  {dir_acc:.1f}%\n"
                  f"```",
            inline=False
        )
        
        embed.add_field(
            name="⚙️ Thông tin Training",
            value=f"```\n"
                  f"Algorithm:   Random Forest\n"
                  f"Features:    {len(SELECTED_FEATURES)} indicators\n"
                  f"Train Size:  {len(X_train_2d)} samples\n"
                  f"Test Size:   {len(X_test_2d)} samples\n"
                  f"Time:        {train_time:.1f}s\n"
                  f"```",
            inline=False
        )
        
        embed.add_field(
            name="🎯 Sử dụng Model",
            value=f"• `/dubao {symbol}` - Dự báo giá tương lai\n"
                  f"• `/backtest1 {symbol}` - Kiểm tra độ chính xác",
            inline=False
        )
        
        embed.set_footer(text=f"Model saved: {rf_model_path} • {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        await msg.edit(
            content=None,
            embed=embed
        )
        
        print(f"✅ Train model {symbol} hoàn tất - MAE: {mae:.2f}, Acc: {dir_acc:.1f}%")
        
    except Exception as e:
        await msg.edit(content=f"❌ Lỗi khi train model: `{str(e)}`")
        print(f"[Lỗi /train] {symbol}: {e}")
        import traceback
        traceback.print_exc()


# =====================================================
# COMMAND: /modelinfo – Xem thông tin model đã train
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)"
)
@bot.tree.command(name="modelinfo", description="Xem thông tin model đã train")
async def modelinfo_cmd(interaction: discord.Interaction, ma_cp: str):
    await interaction.response.defer(ephemeral=True)
    
    symbol = ma_cp.strip().upper()
    
    rf_model_path = f'saved_model/best_randomforest_default_{symbol}.joblib'
    scaler_path = f'saved_model/scaler_randomforest_{symbol}.joblib'
    
    if not (os.path.exists(rf_model_path) and os.path.exists(scaler_path)):
        await interaction.followup.send(
            f"❌ Chưa có model cho **{symbol}**!\n"
            f"Gõ `/train1 {symbol}` để train model.",
            ephemeral=True
        )
        return
    
    try:
        # Load model để lấy thông tin
        rf_model = joblib.load(rf_model_path)
        scaler = joblib.load(scaler_path)
        
        # Lấy file info
        model_size = os.path.getsize(rf_model_path) / 1024  # KB
        scaler_size = os.path.getsize(scaler_path) / 1024  # KB
        model_time = datetime.fromtimestamp(os.path.getmtime(rf_model_path))
        
        # Lấy tên công ty
        try:
            company_info = get_company_info(symbol)
            company_name = company_info.get('company_name', symbol)
        except:
            company_name = symbol
        
        embed = discord.Embed(
            title=f"ℹ️ Thông tin Model • {symbol}",
            description=f"**{company_name}**",
            color=0x3498db
        )
        
        embed.add_field(
            name="📁 File Info",
            value=f"```\n"
                  f"Model:   {model_size:.1f} KB\n"
                  f"Scaler:  {scaler_size:.1f} KB\n"
                  f"Trained: {model_time.strftime('%d/%m/%Y %H:%M')}\n"
                  f"```",
            inline=False
        )
        
        embed.add_field(
            name="🔧 Model Config",
            value=f"```\n"
                  f"Type:        Random Forest\n"
                  f"Estimators:  {rf_model.n_estimators}\n"
                  f"Features:    {len(SELECTED_FEATURES)}\n"
                  f"Lookback:    22 phiên\n"
                  f"```",
            inline=False
        )
        
        embed.add_field(
            name="📊 Features Used",
            value=f"```\n{', '.join(SELECTED_FEATURES[:6])}\n"
                  f"... và {len(SELECTED_FEATURES)-6} features khác```",
            inline=False
        )
        
        embed.set_footer(text="Gõ /dubao để dự báo • /backtest1 để kiểm tra độ chính xác")
        
        await interaction.followup.send(embed=embed, ephemeral=True)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi: `{str(e)}`", ephemeral=True)


# =====================================================
# COMMAND: /listmodels – Liệt kê tất cả models đã train
# =====================================================
@bot.tree.command(name="listmodels", description="Xem danh sách các model đã train")
async def listmodels_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    try:
        model_dir = 'saved_model'
        if not os.path.exists(model_dir):
            await interaction.followup.send("❌ Chưa có model nào được train!", ephemeral=True)
            return
        
        # Tìm tất cả file model
        model_files = [f for f in os.listdir(model_dir) if f.startswith('best_randomforest_default_') and f.endswith('.joblib')]
        
        if not model_files:
            await interaction.followup.send("❌ Chưa có model nào được train!", ephemeral=True)
            return
        
        # Extract symbols
        models = []
        for filename in model_files:
            symbol = filename.replace('best_randomforest_default_', '').replace('.joblib', '')
            scaler_file = f'scaler_randomforest_{symbol}.joblib'
            
            # Kiểm tra scaler có tồn tại không
            if os.path.exists(os.path.join(model_dir, scaler_file)):
                model_path = os.path.join(model_dir, filename)
                file_size = os.path.getsize(model_path) / 1024  # KB
                mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                # Lấy tên công ty
                try:
                    company_info = get_company_info(symbol)
                    name = company_info.get('company_name', symbol)
                except:
                    name = symbol
                
                models.append({
                    'symbol': symbol,
                    'name': name,
                    'size': file_size,
                    'time': mod_time
                })
        
        # Sort by time (mới nhất trước)
        models.sort(key=lambda x: x['time'], reverse=True)
        
        embed = discord.Embed(
            title=f"📋 Danh sách Models ({len(models)})",
            description="Các model dự báo đã được train",
            color=0x9b59b6
        )
        
        # Hiển thị tối đa 10 models
        display_count = min(10, len(models))
        model_list = ""
        
        for i, m in enumerate(models[:display_count], 1):
            model_list += f"**{i}. {m['symbol']}** - {m['name'][:30]}\n"
            model_list += f"   ↳ {m['size']:.1f}KB • {m['time'].strftime('%d/%m/%Y %H:%M')}\n"
        
        if len(models) > 10:
            model_list += f"\n_... và {len(models)-10} models khác_"
        
        embed.add_field(
            name="🤖 Models",
            value=model_list,
            inline=False
        )
        
        embed.set_footer(text="Gõ /modelinfo [mã] để xem chi tiết • /train [mã] để train mới")
        
        await interaction.followup.send(embed=embed, ephemeral=True)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi: `{str(e)}`", ephemeral=True)


# =====================================================
# HÀM PHỤ TRỢ: Lấy thông tin công ty
# =====================================================
def get_company_info(symbol):
    """Lấy tên công ty từ mã CK"""
    try:
        all_symbols = listing.all_symbols()
        mask = all_symbols['symbol'] == symbol
        if mask.any():
            return all_symbols[mask].iloc[0].to_dict()
    except:
        pass
    return {'symbol': symbol, 'company_name': symbol}

def get_all_slash_commands():
    """Trả về danh sách tất cả lệnh slash đã đăng ký với mô tả và tham số"""
    commands_list = []
    
    for command in bot.tree.get_commands():
        cmd_info = {
            "name": command.name,
            "description": command.description or "Không có mô tả",
            "parameters": []
        }
        
        # Lấy tham số nếu có
        if hasattr(command, 'options'):
            for opt in command.options:
                param_desc = f"{opt.name}: {opt.description or 'Không có mô tả'}"
                if hasattr(opt, 'choices') and opt.choices:
                    param_desc += f" (các lựa chọn: {', '.join([c.name for c in opt.choices])})"
                cmd_info["parameters"].append(param_desc)
        
        commands_list.append(cmd_info)
    
    return commands_list

# =====================================================
# COMMAND: /cmd – Liệt kê tất cả lệnh slash trong bot
# =====================================================
@bot.tree.command(name="cmd", description="Liệt kê tất cả lệnh slash có trong bot kèm mô tả và tham số")
async def cmd(interaction: discord.Interaction):
    await interaction.response.defer()
    
    # Gọi hàm lấy danh sách lệnh
    commands = get_all_slash_commands()
    
    if not commands:
        await interaction.followup.send("❌ Không tìm thấy lệnh slash nào!")
        return
    
    # Tạo embed đẹp
    embed = discord.Embed(
        title="📋 Danh sách lệnh slash của bot",
        description=f"Tổng cộng **{len(commands)}** lệnh có sẵn",
        color=0x00d4ff,
        timestamp=datetime.now()
    )
    
    # Sắp xếp lệnh theo tên (A-Z)
    commands.sort(key=lambda x: x['name'])
    
    # Giới hạn mỗi field để tránh vượt 1024 ký tự
    current_field_value = ""
    current_field_name = ""
    
    for cmd in commands:
        # Tạo nội dung cho 1 lệnh
        param_text = ""
        if cmd['parameters']:
            param_text = "\n".join([f"   • {p}" for p in cmd['parameters']])
            param_text = "\n" + param_text
        
        cmd_text = f"**/{cmd['name']}**\n{cmd['description']}{param_text}\n"
        
        # Nếu thêm lệnh này vượt quá giới hạn → tạo field mới
        if len(current_field_value + cmd_text) > 1000:  # Để an toàn dưới 1024
            if current_field_value:
                embed.add_field(name=current_field_name, value=current_field_value, inline=False)
            current_field_value = cmd_text
            current_field_name = f"Lệnh ({len(embed.fields)+1}-{min(len(embed.fields)+5, len(commands))})"
        else:
            current_field_value += cmd_text
    
    # Thêm field cuối cùng
    if current_field_value:
        embed.add_field(name=current_field_name or "Các lệnh", value=current_field_value, inline=False)
    
    embed.set_footer(text="Gõ /ai để hỏi trợ lý về cách dùng lệnh • Cập nhật tự động")
    
    await interaction.followup.send(embed=embed)
    
    print(f"[{interaction.user}] Đã dùng lệnh /cmd")

def get_ai_system_prompt():
    commands = get_all_slash_commands()
    
    cmd_text = "Danh sách lệnh slash có sẵn trong bot:\n"
    for cmd in commands:
        cmd_text += f"/{cmd['name']} - {cmd['description']}\n"
        if cmd['parameters']:
            cmd_text += "   Tham số:\n" + "\n".join([f"   - {p}" for p in cmd['parameters']]) + "\n"
        cmd_text += "\n"
    
    system_prompt = f"""
Bạn là Nasjuro, trợ lý chứng khoán thông minh của bot Stock Comparision, cũng đóng vai trò là 1 chuyên gia trong lĩnh vực công nghệ tài chính.
Bạn chỉ thực hiện tư vấn cho cổ phiếu thuộc Việt Nam, ko tư vấn về cổ phiếu nước ngoài, tiền ảo, forex, hay các lĩnh vực khác.
Bạn hiểu rõ về các lệnh slash có trong bot và cách sử dụng chúng để hỗ trợ người dùng.
Khi người dùng hỏi về cách dùng lệnh, cách gọi lệnh slash, cú pháp, ví dụ, hoặc chức năng của lệnh nào đó trong bot, bạn PHẢI trả lời dựa trên danh sách lệnh dưới đây.

Danh sách lệnh:
{cmd_text}

Quy tắc trả lời:
- Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng, dễ hiểu.
- Luôn đưa ví dụ cụ thể nếu người dùng hỏi về lệnh.
- Format câu trả lời theo dạng: -<Lệnh>: <mô tả> \n Ví dụ: <ví dụ sử dụng>. Câu trả lời được đưa ra dưới dạng danh sách / markdown, ko được đưa các định dạng khác.
- Không bịa thêm lệnh không có trong danh sách.
- Không đưa khuyến nghị mua/bán cổ phiếu.
- Bạn cần nắm rõ quy trình, đặc biệt là việc dự đoán cổ phiếu, khi người dùng có nhu cầu, bạn phải tư vấn người dùng sử dụng lệnh /train để huấn luyện mô hình cho mã cổ phiếu/chỉ số họ quan tâm, sau đó sử dụng lệnh /dubao để dự báo giá trong tương lai, và lệnh /backtest để kiểm tra độ chính xác của mô hình.
- Bạn có thể tư vấn thêm cách dùng các chỉ báo nếu người dùng hỏi về phân tích kỹ thuật.
Nếu câu hỏi không liên quan đến bot và cổ phiếu, bạn phải từ chối trả lời, hoặc giới thiệu người dùng về bot.
"""
    return system_prompt
def ask_ai(user_message):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    system_prompt = get_ai_system_prompt()  # Lấy prompt chứa danh sách lệnh
    
    payload = {
        "model": "openai/gpt-oss-120b:free",  # hoặc model nhanh khác
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 800,
        "stream": False
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if 'choices' in data and len(data['choices']) > 0:
            return data['choices'][0]['message']['content'].strip()
        else:
            return "Lỗi: Không nhận được phản hồi hợp lệ."
    except Exception as e:
        return f"Lỗi: {str(e)}"
    

# =====================================================
# COMMAND: /train3
# =====================================================
@app_commands.describe(ma_cp="Mã cổ phiếu/chỉ số cần train (VD: VNINDEX, HPG, VNM...)")
@bot.tree.command(name="train3", description="Giải thích mô hình 3 (ZLEMA5 + Lasso)")
async def train3_cmd(interaction: discord.Interaction, ma_cp: str):
    await interaction.response.defer()
    symbol = ma_cp.strip().upper()
    try:
        company_info = get_company_info(symbol)
        company_name = company_info.get('company_name', symbol)
    except:
        company_name = symbol

    msg = await interaction.followup.send(
        f"🤖 Đang train mô hình 3 cho **{symbol}** ({company_name})...\n"
        f"⏳ Vui lòng đợi trong giây lát..."
    )
    try:
        model, f_sc, t_sc = await asyncio.get_event_loop().run_in_executor(
            None, train_model_for_symbol3, symbol
        )
        embed = discord.Embed(
            title=f"✅ Train Mô Hình 3 Thành Công • {symbol}",
            description=f"**{company_name}**\n"
                        f"Thuật toán: ZLEMA(5) + Lasso + Core Momentum",
            color=0x00ff00
        )
        embed.add_field(
            name="🎯 Sử dụng Model",
            value=f"• `/dash3 {symbol}` - Chi tiết cách tính & Dự báo\n"
                  f"• `/backtest3 {symbol}` - Kiểm tra độ chính xác",
            inline=False
        )
        await msg.edit(content=None, embed=embed)
    except Exception as e:
        await msg.edit(content=f"❌ Lỗi khi train model 3: `{str(e)}`")
        print(f"[Lỗi /train3] {symbol}: {e}")

# =====================================================
# COMMAND: /backtest3
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)",
    ngay_bat_dau="Ngày bắt đầu backtest (YYYY-MM-DD, mặc định 2025-01-01)"
)
@bot.tree.command(name="backtest3", description="Backtest mô hình 3 (ZLEMA5 + Lasso)")
async def backtest3_cmd(interaction: discord.Interaction, ma_cp: str, ngay_bat_dau: str = "2025-01-01"):
    await interaction.response.defer()
    symbol = ma_cp.strip().upper()

    if not model_exists3(symbol):
        await interaction.followup.send(
            f"❌ Chưa có mô hình 3 cho **{symbol}**!\nGõ `/train3 {symbol}` để train trước.",
            ephemeral=True
        )
        return

    msg = await interaction.followup.send(f"🔄 Đang backtest mô hình 3 **{symbol}** từ {ngay_bat_dau}...")
    try:
        bt_df = await asyncio.get_event_loop().run_in_executor(
            None, backtest_model3, symbol, ngay_bat_dau
        )
        if bt_df is None or bt_df.empty:
            await interaction.followup.send("❌ Không có kết quả backtest hợp lệ!")
            return
            
        mape = bt_df["% Sai số"].mean()
        mae = bt_df["Sai số"].abs().mean()
        diracc = (bt_df["Đúng hướng"] == "✔").mean() * 100
        
        embed = discord.Embed(
            title=f"📊 Kết quả Backtest 3 • {symbol}",
            description=f"Từ {ngay_bat_dau} • {len(bt_df)} phiên",
            color=0x00ff88
        )
        embed.add_field(
            name="📈 Độ chính xác",
            value=f"```\nMAE:      {mae:.2f} điểm\nMAPE:     {mape:.3f}%\nAccuracy: {diracc:.1f}%\n```",
            inline=False
        )
        
        csv_file = f"backtest3_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        bt_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        await interaction.followup.send(
            embed=embed,
            files=[discord.File(csv_file, filename=f"backtest3_{symbol}.csv")]
        )
        try:
            await msg.delete()
            os.remove(csv_file)
        except:
            pass
    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi backtest 3: `{str(e)}`")

# =====================================================
# COMMAND: /dash3
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)",
    thang="Số tháng dữ liệu lịch sử (1-60, mặc định 6)",
    so_phien_du_bao="Số phiên dự báo (1-30, mặc định 5)"
)
@bot.tree.command(name="dash3", description="Chi tiết cách tính mô hình 3")
async def dash3_cmd(interaction: discord.Interaction, ma_cp: str, so_phien_du_bao: int = 5, thang: int = 6):
    await interaction.response.defer()
    symbol = ma_cp.strip().upper()

    if not model_exists3(symbol):
        await interaction.followup.send(
            f"❌ Chưa có mô hình 3 cho **{symbol}**!\nGõ `/train3 {symbol}` để train trước.",
            ephemeral=True
        )
        return

    msg = await interaction.followup.send(
        f"📊 Đang tạo dashboard 3 cho **{symbol}**...\n⏳ Đang tải dữ liệu & render biểu đồ..."
    )
    
    html_file = None
    png_file = None
    try:
        html_file = await asyncio.get_event_loop().run_in_executor(
            None,
            create_terminal_dashboard3,
            symbol,
            thang,
            so_phien_du_bao
        )
        png_file = await asyncio.get_event_loop().run_in_executor(
            None, html_to_image3, html_file, None, 1600
        )
        
        embed = discord.Embed(
            title=f"📊 Dashboard Mô hình 3 • {symbol}",
            description=f"Dữ liệu {thang} tháng • Dự báo {so_phien_du_bao} phiên tới",
            color=0x58a6ff
        )
        embed.set_image(url=f"attachment://{os.path.basename(png_file)}")
        
        await interaction.followup.send(
            embed=embed,
            files=[
                discord.File(png_file, filename=f"dash3_{symbol}.png"),
                discord.File(html_file, filename=f"dash3_{symbol}.html"),
            ]
        )
        try:
            await msg.delete()
        except:
            pass
    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi khi tạo dashboard 3 cho **{symbol}**: `{str(e)}`")
    finally:
        for f in [html_file, png_file]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass

# =====================================================
# COMMAND: /train4
# =====================================================
@app_commands.describe(ma_cp="Mã cổ phiếu/chỉ số cần train (VD: VNINDEX, HPG, VNM...)")
@bot.tree.command(name="train4", description="Train mô hình 4 (Pure RF + ZLEMA5 + DELTA)")
async def train4_cmd(interaction: discord.Interaction, ma_cp: str):
    await interaction.response.defer()
    symbol = ma_cp.strip().upper()

    try:
        company_info = get_company_info(symbol)
        company_name = company_info.get("company_name", symbol)
    except:
        company_name = symbol

    msg = await interaction.followup.send(
        f"🤖 Đang train mô hình 4 cho **{symbol}** ({company_name})...\n"
        f"⏳ Pure RF + ZLEMA(5) + DELTA features..."
    )

    try:
        model, f_sc, t_sc = await asyncio.get_event_loop().run_in_executor(
            None,
            train_model_for_symbol4,
            symbol
        )

        embed = discord.Embed(
            title=f"✅ Train Mô Hình 4 Thành Công • {symbol}",
            description=(
                f"**{company_name}**\n"
                f"Thuật toán: Pure Random Forest + ZLEMA(5)\n"
                f"Feature: Core + Momentum DELTA\n"
                f"Forecast: Recursive Pure ML"
            ),
            color=0x00ff00
        )

        embed.add_field(
            name="🎯 Sử dụng Model",
            value=(
                f"• `/dash4 {symbol}` - Dashboard & Dự báo\n"
                f"• `/backtest4 {symbol}` - Kiểm tra độ chính xác"
            ),
            inline=False
        )

        await msg.edit(content=None, embed=embed)

    except Exception as e:
        await msg.edit(content=f"❌ Lỗi khi train model 4: `{str(e)}`")
        print(f"[Lỗi /train4] {symbol}: {e}")

# =====================================================
# COMMAND: /backtest4
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)",
    ngay_bat_dau="Ngày bắt đầu backtest (YYYY-MM-DD, mặc định 2025-01-01)"
)
@bot.tree.command(name="backtest4", description="Backtest mô hình 4 (Pure RF + ZLEMA5 + DELTA)")
async def backtest4_cmd(
    interaction: discord.Interaction,
    ma_cp: str,
    ngay_bat_dau: str = "2025-01-01"
):
    await interaction.response.defer()
    symbol = ma_cp.strip().upper()

    if not model_exists4(symbol):
        await interaction.followup.send(
            f"❌ Chưa có mô hình 4 cho **{symbol}**!\n"
            f"Gõ `/train4 {symbol}` để train trước.",
            ephemeral=True
        )
        return

    msg = await interaction.followup.send(
        f"🔄 Đang backtest mô hình 4 **{symbol}** từ {ngay_bat_dau}..."
    )

    csv_file = None

    try:
        bt_df = await asyncio.get_event_loop().run_in_executor(
            None,
            backtest_model4,
            symbol,
            ngay_bat_dau
        )

        if bt_df is None or bt_df.empty:
            await interaction.followup.send("❌ Không có kết quả backtest hợp lệ!")
            return

        mape = bt_df["% Sai số"].mean()
        mae = bt_df["Sai số"].abs().mean()
        diracc = (bt_df["Đúng hướng"] == "✔").mean() * 100

        embed = discord.Embed(
            title=f"📊 Kết quả Backtest 4 • {symbol}",
            description=f"Từ {ngay_bat_dau} • {len(bt_df)} phiên",
            color=0x00ff88
        )

        embed.add_field(
            name="📈 Độ chính xác",
            value=(
                f"```\n"
                f"MAE:      {mae:.2f} điểm\n"
                f"MAPE:     {mape:.3f}%\n"
                f"Accuracy: {diracc:.1f}%\n"
                f"```"
            ),
            inline=False
        )

        embed.add_field(
            name="🧠 Model",
            value="Pure Random Forest + ZLEMA(5) + Core Momentum DELTA",
            inline=False
        )

        csv_file = f"backtest4_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        bt_df.to_csv(csv_file, index=False, encoding="utf-8-sig")

        await interaction.followup.send(
            embed=embed,
            files=[discord.File(csv_file, filename=f"backtest4_{symbol}.csv")]
        )

        try:
            await msg.delete()
        except:
            pass

    except Exception as e:
        await interaction.followup.send(f"❌ Lỗi backtest 4: `{str(e)}`")

    finally:
        if csv_file and os.path.exists(csv_file):
            try:
                os.remove(csv_file)
            except:
                pass

# =====================================================
# COMMAND: /dash4
# =====================================================
@app_commands.describe(
    ma_cp="Mã cổ phiếu/chỉ số (VD: VNINDEX, HPG, VNM...)",
    thang="Số tháng dữ liệu lịch sử (1-60, mặc định 6)",
    so_phien_du_bao="Số phiên dự báo (1-30, mặc định 5)"
)
@bot.tree.command(name="dash4", description="Dashboard mô hình 4 (Pure RF + ZLEMA5 + DELTA)")
async def dash4_cmd(
    interaction: discord.Interaction,
    ma_cp: str,
    so_phien_du_bao: int = 5,
    thang: int = 6
):
    await interaction.response.defer()
    symbol = ma_cp.strip().upper()

    if not model_exists4(symbol):
        await interaction.followup.send(
            f"❌ Chưa có mô hình 4 cho **{symbol}**!\n"
            f"Gõ `/train4 {symbol}` để train trước.",
            ephemeral=True
        )
        return

    thang = max(1, min(60, thang))
    so_phien_du_bao = max(1, min(30, so_phien_du_bao))

    msg = await interaction.followup.send(
        f"📊 Đang tạo dashboard 4 cho **{symbol}**...\n"
        f"⏳ Pure RF recursive forecast + render biểu đồ..."
    )

    html_file = None
    png_file = None

    try:
        html_file = await asyncio.get_event_loop().run_in_executor(
            None,
            create_terminal_dashboard4,
            symbol,
            thang,
            so_phien_du_bao
        )

        png_file = await asyncio.get_event_loop().run_in_executor(
            None,
            html_to_image4,
            html_file,
            None,
            1600
        )

        embed = discord.Embed(
            title=f"📊 Dashboard Mô hình 4 • {symbol}",
            description=(
                f"Dữ liệu {thang} tháng • Dự báo {so_phien_du_bao} phiên tới\n"
                f"Pure RF + ZLEMA(5) + DELTA features"
            ),
            color=0x58a6ff
        )

        embed.set_image(url=f"attachment://{os.path.basename(png_file)}")

        await interaction.followup.send(
            embed=embed,
            files=[
                discord.File(png_file, filename=f"dash4_{symbol}.png"),
                discord.File(html_file, filename=f"dash4_{symbol}.html"),
            ]
        )

        try:
            await msg.delete()
        except:
            pass

    except Exception as e:
        await interaction.followup.send(
            f"❌ Lỗi khi tạo dashboard 4 cho **{symbol}**: `{str(e)}`"
        )

    finally:
        for f in [html_file, png_file]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

# =====================================================
# COMMAND: /cocau
# =====================================================
@app_commands.describe(
    ma_cp="Danh sách mã cổ phiếu cách nhau bằng dấu phẩy (VD: HPG,VNM,FPT)",
    che_do="Chế độ đa dạng hóa: none, balanced, strict. Mặc định: balanced"
)
@bot.tree.command(
    name="cocau",
    description="Đề xuất cơ cấu danh mục tối ưu 4 tuần (MPT + Core-Satellite + Pure RF Weekly)"
)
async def cocau_cmd(
    interaction: discord.Interaction,
    ma_cp: str,
    che_do: str = "balanced"
):
    await interaction.response.defer()

    # ===== Validate tickers =====
    tickers = []
    for t in ma_cp.split(","):
        t = t.strip().upper()
        if t and t not in tickers:
            tickers.append(t)

    if not tickers:
        await interaction.followup.send(
            "❌ Danh sách mã không hợp lệ!\nVí dụ: `/cocau HPG,VNM,FPT`",
            ephemeral=True
        )
        return

    if len(tickers) < 2:
        await interaction.followup.send(
            "❌ Cần ít nhất **2 mã cổ phiếu** để tối ưu danh mục.",
            ephemeral=True
        )
        return

    if len(tickers) > 15:
        await interaction.followup.send(
            "⚠️ Bạn nhập quá nhiều mã. Vui lòng dùng tối đa **15 mã** để tối ưu ổn định hơn.",
            ephemeral=True
        )
        return

    # ===== Validate diversification mode =====
    che_do = (che_do or "balanced").strip().lower()
    valid_modes = ["none", "balanced", "strict"]

    if che_do not in valid_modes:
        che_do = "balanced"

    mode_label = {
        "none": "Không ràng buộc",
        "balanced": "Cân bằng",
        "strict": "Nghiêm ngặt"
    }.get(che_do, "Cân bằng")

    msg = await interaction.followup.send(
        f"📊 Đang phân tích danh mục **{len(tickers)} mã**: "
        f"**{', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''}**\n"
        f"🧠 Model: Pure RF Weekly + ZLEMA + Core-Satellite\n"
        f"⚙️ Chế độ đa dạng hóa: **{mode_label}**"
    )

    html_file = None
    png_file = None

    try:
        loop = asyncio.get_event_loop()

        html_file, png_file = await loop.run_in_executor(
            None,
            danhmuc.run_portfolio_optimization,
            tickers,
            che_do
        )

        if not html_file or not os.path.exists(html_file):
            raise FileNotFoundError("Không tạo được file HTML dashboard.")

        if not png_file or not os.path.exists(png_file):
            raise FileNotFoundError("Không tạo được file PNG dashboard.")

        embed = discord.Embed(
            title="📊 Đề xuất Cơ cấu Danh mục",
            description=(
                f"**Số mã:** {len(tickers)}\n"
                f"**Chế độ:** {mode_label}\n"
                f"**Mô hình:** Pure RF Weekly + Core-Satellite\n"
                f"**Horizon:** 4 tuần"
            ),
            color=0x58a6ff
        )

        embed.set_image(url=f"attachment://{os.path.basename(png_file)}")
        embed.set_footer(text="Kết quả chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.")

        await interaction.followup.send(
            embed=embed,
            files=[
                discord.File(png_file, filename="cocau_dashboard.png"),
                discord.File(html_file, filename="cocau_dashboard.html"),
            ]
        )

        try:
            await msg.delete()
        except:
            pass

    except Exception as e:
        err = str(e)

        await interaction.followup.send(
            f"❌ Lỗi khi phân tích danh mục:\n```{err[:1800]}```"
        )

        print(f"[Lỗi /cocau] {e}")
        import traceback
        traceback.print_exc()

    finally:
        for f in [html_file, png_file]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
#===========================================
# CHẠY BOT
# =====================================================
if __name__ == "__main__":
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    print("🚀 Đang khởi động bot...")
    print("⏳ Vui lòng đợi...")
    
    try:
        bot.run(TOKEN)
    except Exception as e:
        print(f"❌ Lỗi khi chạy bot: {e}")