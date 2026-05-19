import io
from vnstock import Listing, Quote
import pandas as pd
from typing import Dict, List, Union

import warnings
warnings.filterwarnings('ignore')

# Hoặc cụ thể hơn:
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

import numpy as np
import pandas as pd
import talib as ta
import pandas_ta as pta
from vnstock import Quote
import joblib
from datetime import datetime, timedelta
from talipp.indicators import ZLEMA as TalippZLEMA
import os
import warnings
import hashlib
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.2f}'.format

import numpy as np
import pandas as pd
import talib as ta
from talipp.indicators import ZLEMA as TalippZLEMA
import joblib

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

selected_features = [
    # Core (bắt buộc)
    'close_zlema20_ratio', 'residual_lag1', 'residual_lag2',      
    # Momentum
    'slope_5', 'macd_hist','stoch_diff', 'rsi',                  
    # Trend
    'adx', 'zlema_slope_5',
    # Volume
    'mfi', 
    # Volatility
    'hv_14', 'atr_14',
    # Optional
    'res_change',
]      

symbol = 'VNINDEX'

def get_stock_historical_data(
    symbols: Union[str, List[str]] = 'VN30',
    start_date: str = None,
    end_date: str = None,
    interval: str = 'd'
) -> Dict[str, pd.DataFrame]:
    """
    Lấy dữ liệu lịch sử linh hoạt cho 1 hoặc nhiều mã cổ phiếu.
    
    Parameters:
    -----------
    symbols : str hoặc List[str]
        - Một mã cụ thể: "HPG", "VNM", etc.
        - Danh sách mã: ["HPG", "VNM", "VIC"]
        - "VN30": Lấy toàn bộ 30 mã trong VN30 + VNINDEX
        - "VNINDEX": Chỉ lấy chỉ số VNINDEX
    start_date : str
        Ngày bắt đầu (format: 'YYYY-MM-DD')
    end_date : str
        Ngày kết thúc (format: 'YYYY-MM-DD')
    interval : str
        Chu kỳ dữ liệu ('d', '1D', '1W', '1M', etc.)
    
    Returns:
    --------
    Dict[str, pd.DataFrame]: Dictionary với key là mã CK, value là DataFrame
    
    Examples:
    ---------
    # Lấy 1 mã
    data = get_stock_historical_data("HPG", "2024-01-01", "2024-12-31")
    
    # Lấy nhiều mã
    data = get_stock_historical_data(["HPG", "VNM", "VIC"], "2024-01-01", "2024-12-31")
    
    # Lấy toàn bộ VN30
    data = get_stock_historical_data("VN30", "2024-01-01", "2024-12-31")
    """
    
    INDEXES = ['VNINDEX', 'HNXINDEX', 'UPCOMINDEX', 'VN30', 'HNX30']
    all_symbols: List[str] = []
    
    # --- 1. Xử lý input symbols ---
    if isinstance(symbols, str):
        symbols_upper = symbols.upper()
        
        # Trường hợp đặc biệt: Lấy toàn bộ VN30
        if symbols_upper == 'VN30':
            try:
                print("Đang lấy danh sách mã VN30 từ nguồn VCI...")
                listing = Listing(source='VCI')
                vn30_data: Union[pd.DataFrame, pd.Series] = listing.symbols_by_group('VN30')
                
                vn30_symbols = []
                
                if isinstance(vn30_data, pd.Series):
                    vn30_symbols = vn30_data.tolist()
                elif isinstance(vn30_data, pd.DataFrame):
                    if 'symbol' in vn30_data.columns:
                        vn30_symbols = vn30_data['symbol'].tolist()
                    elif 'ticker' in vn30_data.columns:
                        vn30_symbols = vn30_data['ticker'].tolist()
                    elif 0 in vn30_data.columns:
                        vn30_symbols = vn30_data[0].tolist()
                    else:
                        raise ValueError("Không tìm thấy cột chứa mã chứng khoán.")
                
                # Lọc và chuẩn hóa
                vn30_symbols = [s.upper() for s in vn30_symbols if s is not None and s != '']
                
                # Thêm VNINDEX + các mã VN30
                all_symbols = ['VNINDEX'] + vn30_symbols
                print(f"✅ Đã lấy được {len(vn30_symbols)} mã VN30 + VNINDEX")
                
            except Exception as e:
                print(f"❌ Lỗi khi lấy danh sách VN30: {e}")
                return {}
        else:
            # Trường hợp: Lấy 1 mã cụ thể hoặc 1 chỉ số
            all_symbols = [symbols_upper]
            
    elif isinstance(symbols, list):
        # Trường hợp: Danh sách nhiều mã
        all_symbols = [s.upper() for s in symbols if s]
    else:
        raise ValueError("symbols phải là string hoặc list of strings")
    
    if not all_symbols:
        print("⚠️ Không có mã nào để lấy dữ liệu")
        return {}
    
    # --- 2. Lấy dữ liệu cho từng mã ---
    dataframes: Dict[str, pd.DataFrame] = {}
    
    # print(f"\n--- Bắt đầu lấy dữ liệu {len(all_symbols)} mã từ {start_date} đến {end_date} (interval: {interval}) ---")
    
    for symbol in all_symbols:
        try:
            quote = Quote(symbol=symbol, source='VCI')
            df = quote.history(start=start_date, end=end_date, interval=interval)
            
            if not df.empty:
                dataframes[symbol] = df
                print(f"✅ {symbol}: {len(df)} dòng")
            else:
                print(f"⚠️ {symbol}: Dữ liệu trống")
                
        except Exception as e:
            print(f"❌ {symbol}: Lỗi - {e}")
    
    # print(f"\n--- Hoàn thành: {len(dataframes)}/{len(all_symbols)} mã thành công ---")
    return dataframes
    
# --- THỰC THI CHƯƠNG TRÌNH ---
START_DATE = '2019-06-15' #lấy thêm 6 tháng để tính các đường chỉ báo dài như MA100, MA200
END_DATE = '2025-12-15'
INTERVAL = 'd'

all_dataframes = get_stock_historical_data(
    symbols='VNINDEX',
    start_date=START_DATE,
    end_date=END_DATE,
    interval=INTERVAL
)



def linreg_slope(series):
    """
    Tính slope của linear regression cho series
    """
    if len(series) < 5:
        return np.nan
    x = np.arange(len(series))
    y = series.values if hasattr(series, 'values') else series
    if np.isnan(y).any():
        return np.nan
    slope, _ = np.polyfit(x, y, 1)
    return slope


def compute_indicators_for_all(data_dict):
    """
    Tính toán CHỈ các chỉ báo cần thiết cho selected_features mới.
    Tối ưu tốc độ và bộ nhớ bằng cách loại bỏ hoàn toàn các indicators thừa.
    """
    print("--- Đang bắt đầu tính toán chỉ báo cần thiết cho model ---")
    
    for symbol, df in data_dict.items():
        try:
            df = df.sort_values('time').reset_index(drop=True)
            df = df.copy()
            
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            volume = df['volume'].clip(lower=0).astype(float)

            # === CẦN THIẾT CHO CÁC FEATURES ===
            
            # 1. Log return & Historical Volatility
            df['log_return'] = np.log(close / close.shift(1))
            df['hv_14'] = df['log_return'].rolling(14).std() * np.sqrt(252)

            # 2. RSI
            df['rsi'] = ta.RSI(close, timeperiod=14)

            # 3. MACD Histogram
            _, _, df['macd_hist'] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

            # 4. Stochastic Diff
            df['stoch_k'], df['stoch_d'] = ta.STOCH(high, low, close, 
                                                   fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

            # 5. ADX
            df['adx'] = ta.ADX(high, low, close, timeperiod=14)

            # 6. MFI (Money Flow Index - thay vol_osc)
            df['mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)

            # 7. ZLEMA20 & ZLEMA Slope
            try:
                zlema20_indicator = TalippZLEMA(20)
                for p in close.tolist():
                    zlema20_indicator.add(float(p))
                zlema20_values = list(zlema20_indicator)
                
                df['zlema20'] = np.nan
                df.iloc[-len(zlema20_values):, df.columns.get_loc('zlema20')] = zlema20_values
            except:
                df['zlema20'] = ta.EMA(close, timeperiod=20)

            # 8. Residual và các lag/change
            df['residual'] = close - df['zlema20']
            df['residual_lag1'] = df['residual'].shift(1)
            df['residual_lag2'] = df['residual'].shift(2)
            df['res_change'] = df['residual'].diff()

            # 9. Ratio chính
            df['close_zlema20_ratio'] = close / df['zlema20']

            # 10. Slope_5 cho close và zlema_slope_5
            df['slope_5'] = df['close'].rolling(window=5).apply(linreg_slope, raw=True)
            df['zlema_slope_5'] = df['zlema20'].rolling(window=5).apply(linreg_slope, raw=True)

            # 11. ATR
            df['atr_14'] = ta.ATR(high, low, close, timeperiod=14)
            
            # === XÓA NaN ===
            df = df.dropna().reset_index(drop=True)

            # Kiểm tra số dòng còn lại
            if len(df) < 300:
                print(f"⚠️ {symbol}: Chỉ còn {len(df)} dòng sau drop NaN → có thể không đủ để train")

            data_dict[symbol] = df
            print(f"✅ {symbol}: Hoàn tất tính toán ({len(df)} dòng dữ liệu hợp lệ)")

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng với {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print("--- Tất cả mã đã được xử lý xong ---")
    return data_dict

all_dataframes = compute_indicators_for_all(all_dataframes)

# Lưu dữ liệu mới (overwrite hoặc lưu tên mới)
# joblib.dump(all_dataframes, f'processed_{symbol}_data_updated.joblib')  # Lưu tên mới để an toàn



from sklearn.preprocessing import StandardScaler
def create_sequences(data, lookback=22):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :-1])  # Features (không có target)
        y.append(data[i, -1])              # Residual ở cột cuối
    return np.array(X), np.array(y)

def prepare_hybrid_residual(data_dict, lookback=22, test_size=0.2):
    prepared_data = {}

    for symbol, df in data_dict.items():
        print(f"--- Đang xử lý hybrid residual cho: {symbol} ---")
        df_clean = df.sort_values('time').copy().reset_index(drop=True)
        df_clean = df_clean.dropna().reset_index(drop=True)

        # Tính residual = close - ma20
        f_list = selected_features + ['residual']  # Features + residual ở cuối
        
        missing = [f for f in f_list if f not in df_clean.columns]
        if missing:
            print(f"Warning: Missing columns: {missing}")
            continue

        data_subset = df_clean[f_list].values
        
        # Chia Train/Test
        train_len = int(len(data_subset) * (1 - test_size))
        train_data = data_subset[:train_len]
        test_data = data_subset[train_len:]
        
        # Chuẩn hóa (StandardScaler cho residual)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Tạo sequence
        X_train, y_train = create_sequences(train_scaled, lookback)
        X_test, y_test = create_sequences(test_scaled, lookback)
        
        test_start_idx = train_len + lookback
        test_len = len(y_test)
        
        prepared_data[symbol] = {
            'hybrid_residual': {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'raw_close_test': df_clean['close'].iloc[test_start_idx : test_start_idx + test_len].values,
                'zlema20_test': df_clean['zlema20'].iloc[test_start_idx : test_start_idx + test_len].values,  # ← THAY ĐỔI NÀY
                'test_dates': df_clean['time'].iloc[test_start_idx : test_start_idx + test_len].values,
                'feature_names': selected_features,   # 👈 THÊM DÒNG NÀY
                'target_name': 'residual',
            },
            'scalers': {'hybrid_residual': scaler}
        }
        
    return prepared_data

# # --- THỰC THI ---
# hybrid_data = prepare_hybrid_residual(all_dataframes)
# # joblib.dump(hybrid_data, f'lstm_hybrid_residual_{symbol}.joblib')
# # print("\n✅ Đã chuẩn bị xong scenario Hybrid Residual.")
# # print(f"   File lưu: lstm_hybrid_residual_{symbol}.joblib")

# hr = hybrid_data[symbol]['hybrid_residual']


# # --- LOAD DATA HYBRID RESIDUAL ---
# # symbol = 'VNINDEX'
# # hybrid_data = joblib.load(f'lstm_hybrid_residual_{symbol}.joblib')  # Hoặc file ZLEMA mới nhất
# scenario = 'hybrid_residual'

# # s_data = hybrid_data[symbol][scenario]
# # scaler = hybrid_data[symbol]['scalers'][scenario]
# s_data = hr
# scaler = hybrid_data[symbol]['scalers'][scenario]

# X_train_3d = s_data['X_train']
# y_train = s_data['y_train']
# X_test_3d = s_data['X_test']
# y_test = s_data['y_test']
# actual_prices_test = s_data['raw_close_test']


# zlema20_test = s_data.get('zlema20_test')

# # Flatten cho tree-based models
# X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
# X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)

# Inverse residual function
def inverse_residual(scaled_residual, scaler):
    dummy = np.zeros((len(scaled_residual), scaler.n_features_in_))
    dummy[:, -1] = scaled_residual
    return scaler.inverse_transform(dummy)[:, -1]

# Hàm đánh giá chung
def evaluate_model(pred_residual):
    pred_prices = zlema20_test + pred_residual
    rmse = np.sqrt(mean_squared_error(actual_prices_test, pred_prices))
    mae = mean_absolute_error(actual_prices_test, pred_prices)
    mape_val = np.mean(np.abs((actual_prices_test - pred_prices) / actual_prices_test)) * 100
    r2 = r2_score(actual_prices_test, pred_prices)
    dir_acc = np.mean(np.sign(np.diff(actual_prices_test)) == np.sign(np.diff(pred_prices))) * 100
    return rmse, mae, mape_val, r2, dir_acc

# # --- 1. TRAIN & LƯU RANDOMFOREST (default - đã chứng minh tốt nhất về sai số) ---
# print("=== TRAIN RANDOMFOREST (DEFAULT) ===")
# start_time = time.time()

# rf_model = RandomForestRegressor(
#     n_estimators=100,      # Default tốt nhất từ bảng cũ
#     random_state=42,
#     n_jobs=-1
# )
# rf_model.fit(X_train_2d, y_train)
# train_time_rf = time.time() - start_time

# pred_residual_rf = inverse_residual(rf_model.predict(X_test_2d), scaler)
# rmse_rf, mae_rf, mape_rf, r2_rf, dir_acc_rf = evaluate_model(pred_residual_rf)

# print(f"RandomForest - RMSE: {rmse_rf:.2f} | MAE: {mae_rf:.2f} | MAPE: {mape_rf:.2f}% | R²: {r2_rf:.3f} | Dir Acc: {dir_acc_rf:.1f}%")
# print(f"Thời gian train: {train_time_rf:.2f}s")

# # Lưu model và scaler
# joblib.dump(rf_model, f'best_randomforest_default_{symbol}.joblib')
# joblib.dump(scaler, f'scaler_randomforest_{symbol}.joblib')
# # print(f"Đã lưu: best_randomforest_default_{symbol}.joblib + scaler_randomforest_{symbol}.joblib")


def linreg_slope(series):
    """
    Tính slope của linear regression cho series
    """
    if len(series) < 5:
        return np.nan
    x = np.arange(len(series))
    y = series.values if hasattr(series, 'values') else series
    if np.isnan(y).any():
        return np.nan
    slope, _ = np.polyfit(x, y, 1)
    return slope

def compute_indicators_single(df):
    """
    Tính toán CHỈ các chỉ báo cần cho SELECTED_FEATURES
    """
    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)
    
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].clip(lower=0).astype(float)

    # 1. Log return & HV14
    df['log_return'] = np.log(close / close.shift(1))
    df['hv_14'] = df['log_return'].rolling(14).std() * np.sqrt(252)

    # 2. RSI
    df['rsi'] = ta.RSI(close, timeperiod=14)

    # 3. MACD Hist
    _, _, df['macd_hist'] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # 4. Stochastic Diff
    df['stoch_k'], df['stoch_d'] = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

    # 5. ADX
    df['adx'] = ta.ADX(high, low, close, timeperiod=14)

    # 6. MFI
    df['mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)

    # 7. ZLEMA20
    try:
        zlema20_indicator = TalippZLEMA(20)
        for p in close.tolist():
            zlema20_indicator.add(float(p))
        zlema20_values = list(zlema20_indicator)
        df['zlema20'] = np.nan
        df.iloc[-len(zlema20_values):, df.columns.get_loc('zlema20')] = zlema20_values
    except:
        df['zlema20'] = ta.EMA(close, timeperiod=20)  # fallback

    # 8. Residual & related
    df['residual'] = close - df['zlema20']
    df['residual_lag1'] = df['residual'].shift(1)
    df['residual_lag2'] = df['residual'].shift(2)
    df['res_change'] = df['residual'].diff()

    # 9. Ratios
    df['close_zlema20_ratio'] = close / df['zlema20']

    # 10. Slopes
    df['slope_5'] = df['close'].rolling(5).apply(linreg_slope, raw=True)
    df['zlema_slope_5'] = df['zlema20'].rolling(5).apply(linreg_slope, raw=True)

    # 11. ATR
    df['atr_14'] = ta.ATR(high, low, close, timeperiod=14)

    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    return df


def load_models_safely(symbol, rf_path=None, scaler_path=None):
    if rf_path is None:
        rf_path = f'saved_model/best_randomforest_default_{symbol}.joblib'
    if scaler_path is None:
        scaler_path = f'saved_model/scaler_randomforest_{symbol}.joblib'
    
    for path in [rf_path, scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ File not found: {path}")
    
    rf_model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded RF model and scaler for {symbol}")
    print(f"   Features expected: {len(SELECTED_FEATURES)}")
    return rf_model, scaler


def prepare_input_safely(past_data, selected_features, scaler):
    feature_cols = selected_features + ['residual']
    missing = [f for f in feature_cols if f not in past_data.columns]
    if missing:
        raise ValueError(f"❌ Missing features: {missing}")
    
    data_with_residual = past_data[feature_cols].values
    data_scaled = scaler.transform(data_with_residual)
    X_seq_scaled = data_scaled[:, :-1]  # Bỏ residual đã scale
    return X_seq_scaled


def update_sequence_with_prediction(current_data, price_pred, lookback):
    """
    Cập nhật window chỉ với các features cần thiết (tối giản)
    """
    prev_close = current_data['close'].iloc[-1]
    recent_volatility = (current_data['high'] - current_data['low']).mean()
    avg_volume = current_data['volume'].tail(10).mean()

    # Tạo nến mới
    new_row = pd.Series({
        'open': prev_close,
        'close': price_pred,
        'volume': avg_volume
    })
    max_oc = max(prev_close, price_pred)
    min_oc = min(prev_close, price_pred)
    new_row['high'] = max_oc + recent_volatility * 0.2
    new_row['low'] = min_oc - recent_volatility * 0.2

    close_new = price_pred
    high_new = new_row['high']
    low_new = new_row['low']

    # Chỉ tính các indicators cần cho selected_features
    new_row['log_return'] = np.log(close_new / prev_close)

    temp_close = pd.concat([current_data['close'].tail(14), pd.Series([close_new])])
    new_row['rsi'] = ta.RSI(temp_close.values, 14)[-1]

    temp_macd = pd.concat([current_data['close'].tail(35), pd.Series([close_new])])
    _, _, hist = ta.MACD(temp_macd.values, 12, 26, 9)
    new_row['macd_hist'] = hist[-1]

    temp_stoch = pd.concat([current_data[['high','low','close']].tail(14),
                           pd.DataFrame([{'high':high_new,'low':low_new,'close':close_new}])])
    k, d = ta.STOCH(temp_stoch['high'].values, temp_stoch['low'].values, temp_stoch['close'].values)
    new_row['stoch_diff'] = k[-1] - d[-1]

    temp_adx = pd.concat([current_data[['high','low','close']].tail(14),
                         pd.DataFrame([{'high':high_new,'low':low_new,'close':close_new}])])
    new_row['adx'] = ta.ADX(temp_adx['high'].values, temp_adx['low'].values, temp_adx['close'].values, 14)[-1]

    temp_mfi = pd.concat([current_data[['high','low','close','volume']].tail(14),
                         pd.DataFrame([{'high':high_new,'low':low_new,'close':close_new,'volume':avg_volume}])])
    new_row['mfi'] = ta.MFI(temp_mfi['high'].values, temp_mfi['low'].values,
                           temp_mfi['close'].values, temp_mfi['volume'].values, 14)[-1]

    temp_hv = pd.concat([current_data['log_return'].tail(13), pd.Series([new_row['log_return']])])
    new_row['hv_14'] = temp_hv.std() * np.sqrt(252)

    # ZLEMA20 incremental
    try:
        zlema20_prev = current_data['zlema20'].iloc[-1]
        lag = 9  # (20-1)/2
        ema_data = 2 * close_new - current_data['close'].iloc[-(lag+1)] if len(current_data) > lag else close_new
        alpha = 2 / 21
        new_row['zlema20'] = alpha * ema_data + (1 - alpha) * zlema20_prev
    except:
        new_row['zlema20'] = close_new  # fallback

    new_row['residual'] = close_new - new_row['zlema20']
    new_row['residual_lag1'] = current_data['residual'].iloc[-1]
    new_row['residual_lag2'] = current_data['residual'].iloc[-2] if len(current_data) > 1 else new_row['residual_lag1']
    new_row['res_change'] = new_row['residual'] - current_data['residual'].iloc[-1]

    new_row['close_zlema20_ratio'] = close_new / new_row['zlema20']

    temp_slope = pd.concat([current_data['close'].tail(4), pd.Series([close_new])])
    new_row['slope_5'] = linreg_slope(temp_slope)

    temp_zlema_slope = pd.concat([current_data['zlema20'].tail(4), pd.Series([new_row['zlema20']])])
    temp_zlema_slope = temp_zlema_slope.reset_index(drop=True)
    new_row['zlema_slope_5'] = linreg_slope(temp_zlema_slope)

    # Sliding window
    updated_df = pd.concat([current_data.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
    return updated_df.tail(lookback).reset_index(drop=True)


# ==================== BACKTEST FUNCTION ====================

def backtest_vnindex(
    symbol='HPG',
    backtest_start='2025-12-16',
    rf_model_path=None,
    scaler_path=None,
    start_fetch='2020-01-01',
    end_fetch=None
):
    if end_fetch is None:
        end_fetch = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n{'='*80}")
    print(f"BACKTEST: {symbol} từ {backtest_start} đến {end_fetch}")
    print(f"{'='*80}")
    
    rf_model, scaler = load_models_safely(symbol, rf_model_path, scaler_path)
    
    quote = Quote(symbol=symbol, source='VCI')
    df_full = quote.history(start=start_fetch, end=end_fetch, interval='d')
    
    if df_full.empty:
        print("❌ Không có dữ liệu")
        return None
    
    df_full = df_full.sort_values('time').reset_index(drop=True)
    df_full['time'] = pd.to_datetime(df_full['time'])
    
    print("🔄 Tính toán chỉ báo cần thiết...")
    df_full = compute_indicators_single(df_full)
    
    backtest_df = df_full[df_full['time'] >= pd.to_datetime(backtest_start)].copy()
    results = []
    correct_dir = 0
    
    print(f"🔄 Backtest {len(backtest_df)} phiên...")
    
    for i in range(len(backtest_df)):
        current_date = backtest_df.iloc[i]['time']
        past_data = df_full[df_full['time'] < current_date].tail(LOOKBACK)
        
        if len(past_data) != LOOKBACK:
            continue
        
        try:
            X_scaled = prepare_input_safely(past_data, SELECTED_FEATURES, scaler)
            X_2d = X_scaled.reshape(1, -1)
            
            pred_residual = rf_model.predict(X_2d)[0]
            
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, -1] = pred_residual
            pred_residual_inv = scaler.inverse_transform(dummy)[0, -1]
            
            zlema20_prev = past_data['zlema20'].iloc[-1]
            price_pred = zlema20_prev + pred_residual_inv
            
            actual = backtest_df.iloc[i]['close']
            prev_close = past_data['close'].iloc[-1]
            
            pred_change = (price_pred / prev_close - 1) * 100
            actual_change = (actual / prev_close - 1) * 100
            
            is_correct = (price_pred > prev_close) == (actual > prev_close)
            if is_correct:
                correct_dir += 1
            
            results.append({
                'Ngày': current_date.strftime('%d/%m/%Y'),
                'Thực tế': f"{actual:,.2f}",
                'Dự báo': f"{price_pred:,.2f}",
                'Sai số': f"{price_pred - actual:+.2f}",
                '% Dự': f"{pred_change:+.2f}%",
                '% Thực': f"{actual_change:+.2f}%",
                'Hướng': "✅" if is_correct else "❌"
            })
            
        except Exception as e:
            print(f"❌ Lỗi {current_date}: {e}")
            continue
    
    if not results:
        print("❌ Không có kết quả backtest")
        return None
    
    table = pd.DataFrame(results)
    mae = table['Sai số'].str.replace('+','').astype(float).abs().mean()
    mae_recent = table['Sai số'].str.replace('+','').astype(float).abs().tail(5).mean()
    acc = (correct_dir / len(results)) * 100
    
    print(f"\n{'='*80}")
    print(f"KẾT QUẢ BACKTEST - {symbol}")
    print(f"{'='*80}")
    print(table.to_string(index=False))
    print(f"\n📊 MAE toàn bộ: {mae:.2f} điểm")
    print(f"📊 MAE 5 phiên gần: {mae_recent:.2f} điểm")
    print(f"📊 Directional Accuracy: {acc:.2f}%")
    print(f"{'='*80}")
    
    return table, mae_recent, acc

SELECTED_FEATURES = [
    # Core (bắt buộc)
    'close_zlema20_ratio', 'residual_lag1', 'residual_lag2',     
    
    # Momentum
    'slope_5', 'macd_hist','stoch_diff', 'rsi',                
    
    # Trend
    'adx', 'zlema_slope_5',
    
    # Volume
    'mfi', 
    
    # Volatility
    'hv_14', 'atr_14',
    
    # Optional
    'res_change',
]      
LOOKBACK = 22

# backtest_vnindex(
#         symbol='VNINDEX',
#         backtest_start='2025-12-16',
#     );

import numpy as np
import pandas as pd
import talib as ta
import pandas_ta as pta
from vnstock import Quote
import joblib
from datetime import datetime, timedelta
from talipp.indicators import ZLEMA as TalippZLEMA
import os
import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.2f}'.format

# ==================== GLOBAL CONSTANTS ====================
LOOKBACK = 22
SELECTED_FEATURES = [
    'close_zlema20_ratio',
    'residual_lag1',
    'residual_lag2',
    'slope_5',
    'macd_hist',
    'stoch_diff',
    'rsi',
    'adx',
    'zlema_slope_5',
    'mfi',
    'hv_14', 'atr_14',
    'res_change'
]
# ==================== ADAPTIVE LEARNING REGIME ====================

def learn_optimal_thresholds(historical_data, lookback_period=120):
    """
    Học thresholds tối ưu từ lịch sử:
    - Tìm các ngày có trend movement mạnh nhất
    - Lấy percentile của các chỉ báo trong những ngày đó
    - Tự động adapt cho từng mã
    
    IMPORTANT: Sử dụng TẤT CẢ historical data thay vì chỉ lookback_period
    → Học từ toàn bộ lịch sử để có thresholds ổn định hơn
    """
    # Sử dụng MIN(lookback_period, toàn bộ data) 
    # Nếu lookback_period >= len(data) → dùng hết data
    if lookback_period >= len(historical_data) * 0.8:
        recent = historical_data.copy()  # Dùng hết
        actual_window = len(historical_data)
    else:
        recent = historical_data.tail(lookback_period).copy()
        actual_window = lookback_period
    
    # 1. Tính actual momentum (next day return)
    recent['next_return'] = recent['close'].shift(-1) / recent['close'] - 1
    recent['abs_next_return'] = recent['next_return'].abs()
    
    # 2. Phân loại ngày theo momentum thực tế
    # Sử dụng ADAPTIVE PERCENTILES dựa trên data size
    if actual_window >= 500:
        # Nhiều data → strict hơn (top 20% = strong)
        strong_percentile = 0.80
        weak_percentile = 0.55
    elif actual_window >= 200:
        # Medium data → balanced
        strong_percentile = 0.75
        weak_percentile = 0.50
    else:
        # Ít data → lenient hơn
        strong_percentile = 0.70
        weak_percentile = 0.45
    
    strong_threshold = recent['abs_next_return'].quantile(strong_percentile)
    weak_threshold = recent['abs_next_return'].quantile(weak_percentile)
    
    strong_days = recent[recent['abs_next_return'] >= strong_threshold]
    weak_days = recent[(recent['abs_next_return'] >= weak_threshold) & 
                       (recent['abs_next_return'] < strong_threshold)]
    ranging_days = recent[recent['abs_next_return'] < weak_threshold]
    
    # 3. Tìm thresholds từ các ngày đó (dùng percentile thấp hơn cho stability)
    thresholds = {}
    
    if len(strong_days) >= 10:  # Cần ít nhất 10 samples
        # Lấy percentile 35% thay vì 40% → stricter criteria
        thresholds['strong'] = {
            'adx_min': strong_days['adx'].quantile(0.35),
            'slope_min': strong_days['slope_5'].abs().quantile(0.35),
            'price_chg_min': strong_days['close'].pct_change(5).abs().quantile(0.35) * 100,
            'sample_size': len(strong_days)
        }
    else:
        # Fallback: conservative defaults
        thresholds['strong'] = {
            'adx_min': 28, 
            'slope_min': 2.5, 
            'price_chg_min': 1.8,
            'sample_size': 0
        }
    
    if len(ranging_days) >= 10:
        # Lấy percentile 65% cho ranging → stricter
        thresholds['ranging'] = {
            'adx_max': ranging_days['adx'].quantile(0.65),
            'slope_max': ranging_days['slope_5'].abs().quantile(0.65),
            'price_chg_max': ranging_days['close'].pct_change(5).abs().quantile(0.65) * 100,
            'sample_size': len(ranging_days)
        }
    else:
        thresholds['ranging'] = {
            'adx_max': 16, 
            'slope_max': 0.4, 
            'price_chg_max': 0.4,
            'sample_size': 0
        }
    
    # Metadata để debug
    thresholds['meta'] = {
        'actual_window': actual_window,
        'strong_percentile': strong_percentile,
        'strong_days_count': len(strong_days),
        'ranging_days_count': len(ranging_days)
    }
    
    return thresholds


def detect_adaptive_regime(past_data, learned_thresholds):
    """
    Sử dụng learned thresholds để detect regime
    """
    latest = past_data.iloc[-1]
    recent_5 = past_data.tail(5)
    
    # Metrics
    adx = latest['adx']
    slope_5 = abs(latest['slope_5'])
    price_change_5d = abs((recent_5['close'].iloc[-1] / recent_5['close'].iloc[0] - 1) * 100)
    macd_hist = abs(latest['macd_hist'])
    rsi = latest['rsi']
    
    # Thresholds
    strong = learned_thresholds['strong']
    ranging = learned_thresholds['ranging']
    
    # Classification logic
    strong_score = sum([
        adx >= strong['adx_min'],
        slope_5 >= strong['slope_min'],
        price_change_5d >= strong['price_chg_min']
    ])
    
    ranging_score = sum([
        adx <= ranging['adx_max'],
        slope_5 <= ranging['slope_max'],
        price_change_5d <= ranging['price_chg_max']
    ])
    
    # Regime + confidence
    if strong_score >= 2:
        regime_type = 'STRONG_TREND'
        confidence = 0.70 + (strong_score / 3) * 0.25
    elif ranging_score >= 2:
        regime_type = 'RANGING'
        confidence = 0.45 + (ranging_score / 3) * 0.20
    else:
        regime_type = 'WEAK_TREND'
        confidence = 0.55
    
    # Direction (multi-indicator vote)
    direction_votes = [
        np.sign(latest['slope_5']) * 0.35,
        np.sign(latest['zlema_slope_5']) * 0.25,
        np.sign(latest['macd_hist']) * 0.20,
        (1 if rsi > 55 else -1 if rsi < 45 else 0) * 0.20
    ]
    direction_score = sum(direction_votes)
    
    if direction_score > 0.3:
        direction = 'UPTREND'
    elif direction_score < -0.3:
        direction = 'DOWNTREND'
    else:
        direction = 'NEUTRAL'
    
    # Trend exhaustion
    price_changes = past_data['close'].pct_change().tail(10)
    current_dir = np.sign(latest['slope_5']) if latest['slope_5'] != 0 else 0
    trend_age = sum(1 for pc in price_changes if np.sign(pc) == current_dir and not np.isnan(pc))
    
    if trend_age > 8:
        confidence *= 0.75
    
    # RSI extreme penalty
    if (rsi > 75 and direction == 'UPTREND') or (rsi < 25 and direction == 'DOWNTREND'):
        confidence *= 0.80
    
    return {
        'type': regime_type,
        'direction': direction,
        'confidence': np.clip(confidence, 0.30, 0.95),
        'adx': adx,
        'slope_5': latest['slope_5'],
        'trend_age': trend_age,
        'rsi': rsi,
        'learned_thresholds': learned_thresholds
    }


def get_adaptive_momentum_weight(regime):
    """
    Weight dựa trên regime + confidence
    """
    # base_weights = {
    #     'RANGING': 0.05,       # Gần như không dùng momentum
    #     'WEAK_TREND': 0.45,
    #     'STRONG_TREND': 0.75
    # }
    base_weights = {
        'RANGING': 0.05,       # Gần như không dùng momentum
        'WEAK_TREND': 0.55,
        'STRONG_TREND': 0.85
    }
    
    weight = base_weights.get(regime['type'], 0.35)
    
    # Confidence adjustment
    confidence_adj = (regime['confidence'] - 0.55) * 0.5
    weight += confidence_adj
    
    # Trend age penalty
    if regime.get('trend_age', 0) > 8:
        weight *= 0.70
    elif regime.get('trend_age', 0) > 6:
        weight *= 0.85
    
    # RSI extreme check
    rsi = regime.get('rsi', 50)
    if (rsi > 75 and regime['direction'] == 'UPTREND') or \
       (rsi < 25 and regime['direction'] == 'DOWNTREND'):
        weight *= 0.65
    
    return np.clip(weight, 0.0, 0.90)


def predict_from_momentum_adaptive(past_data, regime):
    """
    Momentum prediction với adaptive damping
    """
    latest = past_data.iloc[-1]
    current_price = latest['close']
    
    # Adaptive damping
    trend_age = regime.get('trend_age', 0)
    if trend_age > 9:
        damping = 0.55
    elif trend_age > 7:
        damping = 0.75
    elif trend_age > 5:
        damping = 0.90
    else:
        damping = 1.05  # Slight boost khi trend còn tươi
    
    # RSI-based damping
    rsi = regime.get('rsi', 50)
    if rsi > 70 or rsi < 30:
        damping *= 0.85
    
    # 1. Slope extrapolation
    slope_pred = current_price + latest['slope_5'] * damping * 1.4
    
    # 2. Multi-timeframe momentum
    ret_3d = past_data['close'].tail(3).pct_change().mean()
    ret_5d = past_data['close'].tail(5).pct_change().mean()
    ret_10d = past_data['close'].tail(10).pct_change().mean()
    
    weighted_return = 0.5 * ret_3d + 0.3 * ret_5d + 0.2 * ret_10d
    ema_pred = current_price * (1 + weighted_return * damping * 2.5)
    
    # 3. ZLEMA trend
    zlema_pred = latest['zlema20'] + latest['zlema_slope_5'] * damping * 1.2
    
    # 4. ATR directional
    direction = 1 if regime['direction'] == 'UPTREND' else (-1 if regime['direction'] == 'DOWNTREND' else 0)
    atr_val = latest.get('atr', (past_data['high'] - past_data['low']).tail(14).mean())
    atr_pred = current_price + direction * atr_val * 0.45 * damping
    
    # 5. MACD momentum
    macd_pred = current_price + np.sign(latest['macd_hist']) * abs(latest['macd_hist']) * 0.15
    
    # Blending
    final_pred = (
        0.35 * slope_pred + #0.4
        0.14 * ema_pred + # 0.25
        0.35 * zlema_pred + # 0.15
        0.14 * atr_pred + #0.12
        0.02 * macd_pred #0.08
    )
    
    return final_pred


# ==================== ADAPTIVE HYBRID BACKTEST ====================
def backtest_vnindex_adaptive(
    symbol='VNINDEX',
    backtest_start='2025-12-16',
    rf_model_path=None,
    scaler_path=None,
    start_fetch='2020-01-01',
    end_fetch=None,
    learning_window=120
):
    if end_fetch is None:
        end_fetch = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n{'='*100}")
    print(f"ADAPTIVE LEARNING BACKTEST: {symbol}")
    print(f"Learning from past {learning_window} days to find optimal thresholds")
    print(f"{'='*100}")
    
    rf_model, scaler = load_models_safely(symbol, rf_model_path, scaler_path)
    
    quote = Quote(symbol=symbol, source='VCI')
    df_full = quote.history(start=start_fetch, end=end_fetch, interval='d')
    
    if df_full.empty:
        print("❌ Không có dữ liệu")
        return None
    
    df_full = df_full.sort_values('time').reset_index(drop=True)
    df_full['time'] = pd.to_datetime(df_full['time'])
    
    print("🔄 Tính toán chỉ báo...")
    df_full = compute_indicators_single(df_full)
    
    backtest_df = df_full[df_full['time'] >= pd.to_datetime(backtest_start)].copy()
    results = []
    correct_dir = 0
    
    print(f"🔄 Backtest {len(backtest_df)} phiên...")
    
    for i in range(len(backtest_df)):
        current_date = backtest_df.iloc[i]['time']
        past_data = df_full[df_full['time'] < current_date].tail(LOOKBACK)
        
        if len(past_data) != LOOKBACK:
            continue
        
        # Lấy historical data để learn thresholds
        historical_data = df_full[df_full['time'] < current_date]
        
        try:
            # 1. Learn optimal thresholds từ lịch sử
            learned_thresholds = learn_optimal_thresholds(historical_data, learning_window)
            
            # 2. Model prediction
            X_scaled = prepare_input_safely(past_data, SELECTED_FEATURES, scaler)
            X_2d = X_scaled.reshape(1, -1)
            pred_residual = rf_model.predict(X_2d)[0]
            
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, -1] = pred_residual
            pred_residual_inv = scaler.inverse_transform(dummy)[0, -1]
            
            zlema20_prev = past_data['zlema20'].iloc[-1]
            price_pred_model = zlema20_prev + pred_residual_inv
            
            # 3. Adaptive regime detection
            regime = detect_adaptive_regime(past_data, learned_thresholds)
            momentum_weight = get_adaptive_momentum_weight(regime)
            
            if momentum_weight > 0.05:
                momentum_pred = predict_from_momentum_adaptive(past_data, regime)
                price_pred_final = (1 - momentum_weight) * price_pred_model + momentum_weight * momentum_pred
            else:
                price_pred_final = price_pred_model
            
            # 4. Evaluate
            actual = backtest_df.iloc[i]['close']
            prev_close = past_data['close'].iloc[-1]
            
            pred_change = (price_pred_final / prev_close - 1) * 100
            actual_change = (actual / prev_close - 1) * 100
            
            is_correct = (price_pred_final > prev_close) == (actual > prev_close)
            if is_correct:
                correct_dir += 1

            # Tính độ lệch %
            error_pct = abs(price_pred_final - actual) / actual * 100

            # Rút gọn regime
            regime_short = {
                'STRONG_TREND': 'Strong',
                'WEAK_TREND': 'Weak', 
                'RANGING': 'Ranging'
            }.get(regime['type'], regime['type'][:2])

            # Direction symbol
            dir_symbol = '↑' if regime['direction'] == 'UPTREND' else '↓' if regime['direction'] == 'DOWNTREND' else '→'
            
            # results.append({
            #     'Ngày': current_date.strftime('%d/%m/%Y'),
            #     'Thực tế': f"{actual:,.2f}",
            #     'Dự báo': f"{price_pred_final:,.2f}",
            #     # 'Model': f"{price_pred_model:,.2f}",
            #     'Sai số': f"{price_pred_final - actual:+.2f}",
            #     '% Dự báo': f"{pred_change:+.2f}%",
            #     '% Thực tế': f"{actual_change:+.2f}%",
            #     # 'Regime': regime['type'][:6],
            #     'Dir': regime['direction'][:2],
            #     # 'MomWt': f"{momentum_weight:.2f}",
            #     'OK': "✅" if is_correct else "❌"
            # })
            results.append({
                'Ngày': current_date.strftime('%d/%m/%Y'),
                'Thực tế': f"{actual:,.2f}",
                'Dự báo': f"{price_pred_final:,.2f}",
                'Sai số': f"{price_pred_final - actual:+.2f}",
                '% Sai lệch': f"{error_pct:.2f}%",  # MỚI
                '% Dự báo': f"{pred_change:+.2f}%",
                '% Thực tế': f"{actual_change:+.2f}%",
                'Xu hướng': f"{regime_short} {dir_symbol}", 
                'Kết quả': "✔" if is_correct else "✖"
            })
            
        except Exception as e:
            print(f"❌ Lỗi {current_date}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("❌ Không có kết quả")
        return None
    
    table = pd.DataFrame(results)
    mae = table['Sai số'].str.replace('+','').astype(float).abs().mean()
    mae_recent = table['Sai số'].str.replace('+','').astype(float).abs().tail(5).mean()
    err_mean = table['% Sai lệch'].str.replace('%','').astype(float).mean()
    acc = (correct_dir / len(results)) * 100
    
    # Stats by regime
    regime_stats = table.groupby('Xu hướng').agg({
        'Sai số': lambda x: x.str.replace('+','').astype(float).abs().mean(),
        'Kết quả': lambda x: (x == '✅').sum() / len(x) * 100
    }).round(2)
    regime_stats.columns = ['MAE', 'Accuracy%']
    
    # Show learned thresholds (last iteration)
    print(f"\n📚 LEARNED THRESHOLDS (từ {learning_window} ngày gần nhất):")
    if 'learned_thresholds' in regime:
        print(f"   STRONG: ADX≥{regime['learned_thresholds']['strong']['adx_min']:.1f}, "
              f"Slope≥{regime['learned_thresholds']['strong']['slope_min']:.2f}, "
              f"PriceChg≥{regime['learned_thresholds']['strong']['price_chg_min']:.2f}%")
        print(f"   RANGING: ADX≤{regime['learned_thresholds']['ranging']['adx_max']:.1f}, "
              f"Slope≤{regime['learned_thresholds']['ranging']['slope_max']:.2f}, "
              f"PriceChg≤{regime['learned_thresholds']['ranging']['price_chg_max']:.2f}%")
    
    print(f"\n{'='*100}")
    print(f"KẾT QUẢ ADAPTIVE LEARNING BACKTEST - {symbol}")
    print(f"{'='*100}")
    print(table.to_string(index=False))
    print(f"\n📊 TỔNG QUAN:")
    print(f"   MAE toàn bộ: {mae:.2f} điểm")
    print(f"   MAE 5 phiên gần: {mae_recent:.2f} điểm")
    print(f"   % Sai lệch trung bình: {err_mean:.2f}%")
    print(f"   Directional Accuracy: {acc:.2f}%")
    print(f"\n📊 THEO REGIME:")
    print(regime_stats.to_string())
    print(f"{'='*100}")
    
    return table, mae_recent, acc






def generate_realistic_noise(symbol, step_number, base_date, regime_type, 
                             current_volatility, rsi_value):
    """
    Enhanced noise với:
    1. Fat-tailed distribution (Student's t)
    2. Volatility clustering
    3. Mean reversion khi RSI extreme
    """
    # Deterministic seed
    seed_string = f"{symbol}_{step_number}_{base_date}_{regime_type}"
    seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
    np.random.seed(seed_hash % (2**32))
    
    # Base noise scale từ regime
    base_scales = {
        'RANGING': 0.002,
        'WEAK_TREND': 0.004,
        'STRONG_TREND': 0.006
    }
    base_scale = base_scales.get(regime_type, 0.004)
    
    # 1. Fat-tailed noise (Student's t với df=7)
    # → Thỉnh thoảng có extreme moves
    t_noise = np.random.standard_t(df=7) * base_scale
    
    # 2. Volatility clustering
    # Nếu current_volatility cao → noise cũng cao
    vol_factor = 1.0 + (current_volatility - 0.02) * 5  # Normalize around 2%
    vol_factor = np.clip(vol_factor, 0.5, 2.0)
    
    # 3. Mean reversion từ RSI
    # RSI > 75 → bias giảm, RSI < 25 → bias tăng
    mean_reversion_bias = 0
    if rsi_value > 75:
        mean_reversion_bias = -0.001 * (rsi_value - 75) / 25  # Max -0.2%
    elif rsi_value < 25:
        mean_reversion_bias = 0.001 * (25 - rsi_value) / 25   # Max +0.2%
    
    final_noise = t_noise * vol_factor + mean_reversion_bias
    
    return np.clip(final_noise, -0.10, 0.10)  # Max ±3% per day


def generate_realistic_volume(current_data, price_change_pct, regime, step_number, symbol):
    """
    Volume phụ thuộc vào:
    1. Price volatility (vol tăng → volume tăng)
    2. Trend strength (strong trend → volume cao)
    3. Day-of-week pattern
    """
    avg_volume = current_data['volume'].tail(10).mean()
    
    # Deterministic seed
    seed = int(hashlib.md5(f"{symbol}_{step_number}_vol".encode()).hexdigest()[:8], 16)
    np.random.seed(seed % (2**32))
    
    # 1. Base random walk
    base_noise = np.random.normal(0, 0.15)
    
    # 2. Volatility correlation
    vol_factor = 1.0 + abs(price_change_pct) * 10  # 1% price change → +10% volume
    
    # 3. Regime factor
    regime_multipliers = {
        'RANGING': 0.8,
        'WEAK_TREND': 1.0,
        'STRONG_TREND': 1.3
    }
    regime_mult = regime_multipliers.get(regime['type'], 1.0)
    
    # 4. Day-of-week pattern (simplified - assume 5 days cycle)
    day_pattern = [0.9, 1.0, 1.05, 1.1, 1.15]  # Thứ 2-6
    day_mult = day_pattern[step_number % 5]
    
    final_volume = avg_volume * (1 + base_noise) * vol_factor * regime_mult * day_mult
    
    return max(1, final_volume)


def generate_realistic_ohlc(open_price, close_pred, current_data, regime, step_number, symbol):
    """
    OHLC realistic hơn với:
    1. High/Low từ intraday volatility
    2. Asymmetric wicks (uptrend → lower wick, downtrend → upper wick)
    3. Support/resistance awareness
    """
    # Deterministic seed
    seed = int(hashlib.md5(f"{symbol}_{step_number}_ohlc".encode()).hexdigest()[:8], 16)
    np.random.seed(seed % (2**32))
    
    # 1. Tính ATR cho intraday range
    recent_atr = current_data['atr_14'].tail(5).mean()
    if pd.isna(recent_atr) or recent_atr <= 0:
        recent_atr = abs(close_pred - open_price) * 1.5
    
    # 2. Direction của nến
    is_bullish = close_pred > open_price
    body_size = abs(close_pred - open_price)
    
    # 3. Generate high/low với asymmetric wicks
    if is_bullish:
        # Uptrend: Lower wick dài hơn (test support rồi bounce)
        upper_wick = np.random.uniform(0.2, 0.5) * recent_atr
        lower_wick = np.random.uniform(0.4, 0.8) * recent_atr
        
        high = close_pred + upper_wick
        low = open_price - lower_wick
    else:
        # Downtrend: Upper wick dài hơn (test resistance rồi reject)
        upper_wick = np.random.uniform(0.4, 0.8) * recent_atr
        lower_wick = np.random.uniform(0.2, 0.5) * recent_atr
        
        high = open_price + upper_wick
        low = close_pred - lower_wick
    
    # 4. Adjust nếu trend mạnh → wick ngắn hơn
    if regime['type'] == 'STRONG_TREND':
        wick_reduction = 0.7
        if is_bullish:
            low = open_price - lower_wick * wick_reduction
        else:
            high = open_price + upper_wick * wick_reduction
    
    # 5. Ensure OHLC logic
    high = max(high, open_price, close_pred)
    low = min(low, open_price, close_pred)
    
    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_pred
    }


def calculate_support_resistance_bias(current_data, price_pred):
    """
    Tính bias từ support/resistance levels
    - Gần resistance → khó vượt
    - Gần support → khó thủng
    """
    # Tìm recent high/low (20 phiên)
    recent_high = current_data['high'].tail(20).max()
    recent_low = current_data['low'].tail(20).min()
    
    price_range = recent_high - recent_low
    if price_range == 0:
        return 0
    
    # Distance to resistance/support
    dist_to_resistance = (recent_high - price_pred) / price_range
    dist_to_support = (price_pred - recent_low) / price_range
    
    bias = 0
    
    # Gần resistance (within 5%)
    if dist_to_resistance < 0.05:
        bias = -0.003  # Slight downward pressure
    
    # Gần support (within 5%)
    if dist_to_support < 0.05:
        bias = 0.003  # Slight upward bounce
    
    return bias


def update_synthetic_candle_enhanced(current_data, price_pred, step_number, 
                                     symbol, base_date, regime):
    """
    ENHANCED VERSION với realistic market behavior
    """
    from datetime import timedelta
    
    # Validate
    if len(current_data) < 22:
        raise ValueError(f"❌ current_data cần ≥22 hàng, hiện tại: {len(current_data)}")
    
    # ✅ THÊM ĐOẠN NÀY NGAY SAU VALIDATE
    # Đảm bảo current_data có cột 'atr' (alias của 'atr_14')
    if 'atr_14' in current_data.columns and 'atr' not in current_data.columns:
        current_data = current_data.copy()
        current_data['atr'] = current_data['atr_14']
    elif 'atr' not in current_data.columns and 'atr_14' not in current_data.columns:
        # Tính ATR nếu cả 2 đều không có
        current_data = current_data.copy()
        high = current_data['high']
        low = current_data['low']
        close = current_data['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        current_data['atr_14'] = tr.rolling(14).mean()
        current_data['atr'] = current_data['atr_14']
    
    prev_close = current_data['close'].iloc[-1]
    current_volatility = current_data['hv_14'].iloc[-1]
    rsi_value = current_data['rsi'].iloc[-1]
    
    # === 1. REALISTIC NOISE ===
    noise = generate_realistic_noise(
        symbol, step_number, base_date, regime['type'],
        current_volatility, rsi_value
    )
    
    # === 2. SUPPORT/RESISTANCE BIAS ===
    sr_bias = calculate_support_resistance_bias(current_data, price_pred)
    
    # === 3. FINAL PRICE ===
    price_pred_final = price_pred * (1 + noise + sr_bias)
    price_change_pct = (price_pred_final / prev_close - 1) * 100
    
    # === 4. REALISTIC OHLC ===
    ohlc = generate_realistic_ohlc(
        prev_close, price_pred_final, current_data, 
        regime, step_number, symbol
    )
    
    # === 5. REALISTIC VOLUME ===
    volume_new = generate_realistic_volume(
        current_data, price_change_pct, regime, step_number, symbol
    )
    
    # === 6. TIME ===
    if 'time' in current_data.columns:
        last_time = current_data['time'].iloc[-1]
        next_time = last_time + timedelta(days=1)
        while next_time.weekday() >= 5:
            next_time += timedelta(days=1)
    else:
        next_time = pd.NaT
    
    # === 7. CREATE NEW ROW ===
    new_row = pd.Series({
        'time': next_time,
        'open': ohlc['open'],
        'high': ohlc['high'],
        'low': ohlc['low'],
        'close': ohlc['close'],
        'volume': volume_new
    })
    
    # === 8. CALCULATE INDICATORS (giống code gốc) ===
    close_new = new_row['close']
    high_new = new_row['high']
    low_new = new_row['low']
    
    # Log return
    new_row['log_return'] = np.log(close_new / prev_close) if prev_close > 0 else 0
    
    # RSI
    try:
        temp_close = pd.concat([current_data['close'].tail(15), pd.Series([close_new])])
        rsi_values = ta.RSI(temp_close.values, 14)
        new_row['rsi'] = rsi_values[-1] if not pd.isna(rsi_values[-1]) else current_data['rsi'].iloc[-1]
    except:
        new_row['rsi'] = current_data['rsi'].iloc[-1]
    
    # MACD
    try:
        temp_macd = pd.concat([current_data['close'].tail(36), pd.Series([close_new])])
        _, _, hist = ta.MACD(temp_macd.values, 12, 26, 9)
        new_row['macd_hist'] = hist[-1] if not pd.isna(hist[-1]) else current_data['macd_hist'].iloc[-1]
    except:
        new_row['macd_hist'] = current_data['macd_hist'].iloc[-1]
    
    # Stochastic
    try:
        temp_stoch = pd.concat([
            current_data[['high','low','close']].tail(15),
            pd.DataFrame([{'high':high_new,'low':low_new,'close':close_new}])
        ])
        k, d = ta.STOCH(temp_stoch['high'].values, temp_stoch['low'].values, 
                       temp_stoch['close'].values, fastk_period=14, slowk_period=3, slowd_period=3)
        if not pd.isna(k[-1]) and not pd.isna(d[-1]):
            new_row['stoch_k'] = k[-1]
            new_row['stoch_d'] = d[-1]
            new_row['stoch_diff'] = k[-1] - d[-1]
        else:
            new_row['stoch_k'] = current_data['stoch_k'].iloc[-1]
            new_row['stoch_d'] = current_data['stoch_d'].iloc[-1]
            new_row['stoch_diff'] = current_data['stoch_diff'].iloc[-1]
    except:
        new_row['stoch_k'] = current_data['stoch_k'].iloc[-1]
        new_row['stoch_d'] = current_data['stoch_d'].iloc[-1]
        new_row['stoch_diff'] = current_data['stoch_diff'].iloc[-1]
    
    # ADX
    try:
        temp_adx = pd.concat([
            current_data[['high','low','close']].tail(15),
            pd.DataFrame([{'high':high_new,'low':low_new,'close':close_new}])
        ])
        adx_values = ta.ADX(temp_adx['high'].values, temp_adx['low'].values, 
                           temp_adx['close'].values, 14)
        new_row['adx'] = adx_values[-1] if not pd.isna(adx_values[-1]) else current_data['adx'].iloc[-1]
    except:
        new_row['adx'] = current_data['adx'].iloc[-1]
    
    # MFI
    try:
        temp_mfi = pd.concat([
            current_data[['high','low','close','volume']].tail(15),
            pd.DataFrame([{'high':high_new,'low':low_new,'close':close_new,'volume':volume_new}])
        ])
        mfi_values = ta.MFI(temp_mfi['high'].values, temp_mfi['low'].values,
                           temp_mfi['close'].values, temp_mfi['volume'].values, 14)
        new_row['mfi'] = mfi_values[-1] if not pd.isna(mfi_values[-1]) else current_data['mfi'].iloc[-1]
    except:
        new_row['mfi'] = current_data['mfi'].iloc[-1]
    
    # HV
    try:
        temp_hv = pd.concat([current_data['log_return'].tail(14), pd.Series([new_row['log_return']])])
        hv = temp_hv.std() * np.sqrt(252)
        new_row['hv_14'] = hv if not pd.isna(hv) else current_data['hv_14'].iloc[-1]
    except:
        new_row['hv_14'] = current_data['hv_14'].iloc[-1]
    
    # ZLEMA
    try:
        zlema20_prev = current_data['zlema20'].iloc[-1]
        if pd.isna(zlema20_prev):
            zlema20_prev = prev_close
        
        lag = 9
        if len(current_data) > lag:
            close_lag = current_data['close'].iloc[-(lag+1)]
            if pd.isna(close_lag):
                close_lag = prev_close
            ema_data = 2 * close_new - close_lag
        else:
            ema_data = close_new
        
        alpha = 2 / 21
        new_zlema = alpha * ema_data + (1 - alpha) * zlema20_prev
        new_row['zlema20'] = new_zlema if not pd.isna(new_zlema) and new_zlema > 0 else close_new
    except:
        new_row['zlema20'] = close_new
    
    # Residual
    new_row['residual'] = close_new - new_row['zlema20']
    new_row['residual_lag1'] = current_data['residual'].iloc[-1]
    new_row['residual_lag2'] = current_data['residual'].iloc[-2] if len(current_data) > 1 else new_row['residual_lag1']
    new_row['res_change'] = new_row['residual'] - current_data['residual'].iloc[-1]
    
    # Ratio
    new_row['close_zlema20_ratio'] = close_new / new_row['zlema20'] if new_row['zlema20'] > 0 else 1.0
    
    # Slopes
    try:
        temp_slope = pd.concat([current_data['close'].tail(5), pd.Series([close_new])])
        slope = linreg_slope(temp_slope)
        new_row['slope_5'] = slope if not pd.isna(slope) else current_data['slope_5'].iloc[-1]
    except:
        new_row['slope_5'] = current_data['slope_5'].iloc[-1]
    
    try:
        temp_zlema_slope = pd.concat([current_data['zlema20'].tail(5), pd.Series([new_row['zlema20']])])
        zlema_slope = linreg_slope(temp_zlema_slope)
        new_row['zlema_slope_5'] = zlema_slope if not pd.isna(zlema_slope) else current_data['zlema_slope_5'].iloc[-1]
    except:
        new_row['zlema_slope_5'] = current_data['zlema_slope_5'].iloc[-1]

    #ATR_14 
    try:
        temp_atr = pd.concat([
            current_data[['high','low','close']].tail(14),  # 14 phiên trước
            pd.DataFrame([{'high':high_new,'low':low_new,'close':close_new}])
        ])
        
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        tr = pd.concat([
            temp_atr['high'] - temp_atr['low'],
            abs(temp_atr['high'] - temp_atr['close'].shift(1)),
            abs(temp_atr['low'] - temp_atr['close'].shift(1))
        ], axis=1).max(axis=1)
        
        # ATR = SMA(TR, 14)
        atr = tr.tail(14).mean()
        new_row['atr_14'] = atr if not pd.isna(atr) else (high_new - low_new)
        new_row['atr'] = new_row['atr_14']  # Duplicate nếu cần
    except Exception as e:
        # Fallback: Dùng high-low range
        new_row['atr_14'] = high_new - low_new
        new_row['atr'] = new_row['atr_14']
    
    # === 9. SLIDING WINDOW ===
    updated_df = pd.concat([
        current_data.iloc[1:].reset_index(drop=True),
        pd.DataFrame([new_row]).reset_index(drop=True)
    ], ignore_index=True)

    result_df = updated_df.tail(22).reset_index(drop=True)
    
    # ✅ THÊM DÒNG NÀY: Tạo alias cho backward compatibility
    if 'atr_14' in result_df.columns and 'atr' not in result_df.columns:
        result_df['atr'] = result_df['atr_14']
    
    return result_df


# ==================== FORECAST FUNCTION ====================

def forecast_future_prices(
    symbol='KBC',
    forecast_steps=14,
    rf_model_path=None,
    scaler_path=None,
    start_fetch='2020-01-01',
    end_fetch=None,
    learning_window=120,  # CHANGED: 1000 → 120 (consistent với backtest)
    show_details=True,
    use_yesterday=True  # NEW: Chỉ dùng data đến hôm qua
):
    """
    Dự báo giá trong tương lai cho N phiên
    ✅ UPDATED: Thêm hybrid blending (model + momentum) như backtest
    ✅ FIXED: Chỉ lấy data đến hôm qua để tránh data intraday không ổn định
    
    Parameters:
    -----------
    symbol : str
        Mã cổ phiếu (VNINDEX, HPG, ...)
    forecast_steps : int
        Số phiên muốn dự báo (1-30)
    learning_window : int
        Số ngày dùng để học thresholds (khuyến nghị: 120, consistent với backtest)
    show_details : bool
        Hiển thị chi tiết từng bước hay chỉ kết quả cuối
    use_yesterday : bool
        Nếu True, chỉ lấy data đến hôm qua (tránh intraday data)
        Nếu False, lấy đến hôm nay (có thể không ổn định nếu phiên chưa đóng)
    
    Returns:
    --------
    forecast_df : DataFrame
        Bảng dự báo
    """
    
    if forecast_steps < 1 or forecast_steps > 30:
        raise ValueError("❌ forecast_steps phải từ 1 đến 30")
    
    # ✅ FIX: Determine end_fetch date
    if end_fetch is None:
        if use_yesterday:
            # Lấy đến HÔM QUA để tránh intraday data
            yesterday = datetime.now() - timedelta(days=1)
            end_fetch = yesterday.strftime('%Y-%m-%d')
        else:
            # Lấy đến HÔM NAY (có thể không ổn định)
            end_fetch = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n{'='*100}")
    print(f"🔮 FORECAST {forecast_steps} PHIÊN TƯƠNG LAI - {symbol}")
    print(f"✨ HYBRID MODE: Model + Adaptive Momentum Blending")
    if use_yesterday:
        print(f"📅 Sử dụng data đến: {end_fetch} (hôm qua - dữ liệu ổn định)")
    else:
        print(f"📅 Sử dụng data đến: {end_fetch} (hôm nay - có thể chưa đóng cửa)")
    print(f"{'='*100}")
    
    # Load models
    rf_model, scaler = load_models_safely(symbol, rf_model_path, scaler_path)
    
    # Fetch data
    quote = Quote(symbol=symbol, source='VCI')
    df_full = quote.history(start=start_fetch, end=end_fetch, interval='d')
    
    if df_full.empty:
        print("❌ Không có dữ liệu")
        return None
    
    df_full = df_full.sort_values('time').reset_index(drop=True)
    df_full['time'] = pd.to_datetime(df_full['time'])
    
    print("🔄 Tính toán chỉ báo...")
    df_full = compute_indicators_single(df_full)
    
    # Lấy LOOKBACK ngày gần nhất
    current_data = df_full.tail(LOOKBACK).copy()
    last_date = df_full['time'].iloc[-1]
    last_price = df_full['close'].iloc[-1]
    
    # ✅ FIX: Hiển thị thông tin về data date
    today = datetime.now().date()
    last_data_date = last_date.date()
    
    print(f"📅 Ngày dữ liệu gần nhất: {last_date.strftime('%d/%m/%Y')}")
    print(f"💰 Giá đóng cửa: {last_price:,.2f}")
    
    if last_data_date == today:
        print(f"⚠️  WARNING: Dữ liệu từ HÔM NAY - Có thể chưa đóng cửa!")
        print(f"   Khuyến nghị: Dùng use_yesterday=True để forecast ổn định hơn")
    elif last_data_date == today - timedelta(days=1):
        print(f"✅ Dữ liệu từ HÔM QUA - Đã đóng cửa, ổn định")
    else:
        days_ago = (today - last_data_date).days
        print(f"⚠️  Dữ liệu cũ ({days_ago} ngày trước) - Có thể cần update")
    
    # ✅ THÊM: Learn optimal thresholds từ historical data
    print(f"\n📚 Learning thresholds từ {learning_window} phiên gần nhất...")
    learned_thresholds = learn_optimal_thresholds(df_full, learning_window)
    
    print(f"   STRONG: ADX≥{learned_thresholds['strong']['adx_min']:.1f}, "
          f"Slope≥{learned_thresholds['strong']['slope_min']:.2f}")
    print(f"   RANGING: ADX≤{learned_thresholds['ranging']['adx_max']:.1f}, "
          f"Slope≤{learned_thresholds['ranging']['slope_max']:.2f}")
    
    # Multi-step forecasting
    forecasts = []
    working_data = current_data.copy()
    
    # ✅ FIX: Track next business day incrementally
    current_forecast_date = last_date
    previous_extreme_move = False  # Track nếu phiên trước có extreme move

    previous_price_predict = last_price  # Khởi tạo = giá cuối cùng

    
    print(f"\n🔄 Bắt đầu forecast {forecast_steps} phiên...\n")
    
    for step in range(1, forecast_steps + 1):
        try:
            # ✅ FIX: Calculate next business day incrementally
            current_forecast_date = current_forecast_date + timedelta(days=1)
            while current_forecast_date.weekday() >= 5:  # Skip weekends
                current_forecast_date = current_forecast_date + timedelta(days=1)
            
            next_date = current_forecast_date  # Use for display
            # ✅ STEP 1: Model prediction (residual-based)
            X_scaled = prepare_input_safely(working_data, SELECTED_FEATURES, scaler)
            X_2d = X_scaled.reshape(1, -1)
            pred_residual = rf_model.predict(X_2d)[0]
            
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, -1] = pred_residual
            pred_residual_inv = scaler.inverse_transform(dummy)[0, -1]
            
            zlema20_prev = working_data['zlema20'].iloc[-1]
            price_pred_model = zlema20_prev + pred_residual_inv
            
            # ✅ STEP 2: Adaptive regime detection
            regime = detect_adaptive_regime(working_data, learned_thresholds)
            
            # ✅ STEP 3: Get momentum weight
            momentum_weight = get_adaptive_momentum_weight(regime)
            if previous_extreme_move:
                # Giảm 50% momentum trong phiên recovery
                momentum_weight *= 0.5
                if show_details:
                    print(f"   🔄 Recovery mode: Momentum weight reduced to {momentum_weight:.2f}")
            
            # ✅ STEP 4: Conditional blending (GIỐNG BACKTEST)
            if momentum_weight > 0.05:
                momentum_pred = predict_from_momentum_adaptive(working_data, regime)
                price_pred_final = (1 - momentum_weight) * price_pred_model + momentum_weight * momentum_pred
                blend_used = True
            else:
                price_pred_final = price_pred_model
                blend_used = False
            
            # Validate prediction
            if pd.isna(price_pred_final) or price_pred_final <= 0:
                print(f"⚠️ Phiên {step}: prediction invalid ({price_pred_final}), dùng giá trước")
                price_pred_final = working_data['close'].iloc[-1] * 1.001
            
            # ✅ NEW: Validate working_data BEFORE using it
            if working_data.isnull().any().any():
                print(f"⚠️ WARNING: working_data có NaN TRƯỚC khi predict step {step}!")
                nan_cols = working_data.columns[working_data.isnull().any()].tolist()
                print(f"   Columns with NaN: {nan_cols}")
                print(f"   Filling NaN với forward fill...")
                working_data = working_data.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate changes
            prev_close = working_data['close'].iloc[-1]
            price_change_pct = (price_pred_final / prev_close - 1) * 100

            # print(f"DEBUG: prev_close={prev_close:.2f}, pred={price_pred_final:.2f}, "
            #     f"change={(price_pred_final/prev_close-1)*100:.2f}%")
            CIRCUIT_BREAKER_LIMIT = 7.0  # ±7% giống sàn VN

            if abs(price_change_pct) > CIRCUIT_BREAKER_LIMIT:
                original_change = price_change_pct
                price_change_pct = np.sign(price_change_pct) * CIRCUIT_BREAKER_LIMIT
                price_pred_final = prev_close * (1 + price_change_pct / 100)
                
                if show_details:
                    print(f"   🚨 Circuit Breaker! {original_change:+.2f}% → {price_change_pct:+.2f}%")
            
            display_change_pct = (price_pred_final / previous_price_predict - 1) * 100

            # Update flag cho phiên tiếp theo
            if abs(price_change_pct) > 2.0:  # Extreme move threshold
                previous_extreme_move = True
                if show_details:
                    print(f"   ⚠️  Extreme move detected: {price_change_pct:+.2f}%")
            else:
                previous_extreme_move = False
            
            # ✅ FIX: Validate prediction magnitude (detect anomalies)
            if abs(price_change_pct) > 10:  # Thay đổi > 10% là bất thường
                print(f"⚠️ ANOMALY DETECTED at step {step}:")
                print(f"   Predicted change: {price_change_pct:+.2f}% (> 10%!)")
                print(f"   Model: {price_pred_model:.2f}, Momentum: {momentum_pred if momentum_weight > 0.05 else 'N/A':.2f}")
                print(f"   Capping to ±5% to prevent cascade errors...")
                
                # Cap change to ±5%
                if price_change_pct > 5:
                    price_pred_final = prev_close * 1.05
                    price_change_pct = 5.0
                elif price_change_pct < -5:
                    price_pred_final = prev_close * 0.95
                    price_change_pct = -5.0

            # forecasts.append({
            #     'Phiên': step,
            #     'Ngày': next_date.strftime('%d/%m/%Y'),
            #     'Giá dự báo': f"{price_pred_final:,.2f}",
            #     # 'Model': f"{price_pred_model:,.2f}",
            #     'Thay đổi': f"{display_change_pct:+.2f}%",
            #     # 'Regime': regime['type'][:6],
            #     # 'Direction': regime['direction'][:2],
            #     # 'MomWt': f"{momentum_weight:.2f}",
            #     # 'Conf': f"{regime['confidence']:.2f}",
            #     # 'Blend': "✓" if blend_used else "✗",
            # })
            
            if show_details:
                blend_symbol = "🔀" if blend_used else "🤖"
                print(f"{blend_symbol} Phiên {step:2d} | {next_date.strftime('%d/%m/%Y')} | "
                      f"{price_pred_final:8,.2f} ({price_change_pct:+5.2f}%) | "
                      f"{regime['type']:12s} | {regime['direction']:8s} | "
                      f"Weight: {momentum_weight:.2f}")
            
            # Update working data
            working_data = update_synthetic_candle_enhanced(
                working_data, 
                price_pred_final, 
                step, 
                symbol, 
                last_date.strftime('%Y-%m-%d'),
                regime
            )
            previous_price_predict = price_pred_final
            
            # ✅ Bây giờ ATR đã được update từ giá mới
            forecast_range = working_data['atr_14'].iloc[-1] * 0.5  # ← Dùng ATR MỚI
            forecast_lower = price_pred_final - forecast_range
            forecast_upper = price_pred_final + forecast_range
            
            forecasts.append({
                'Phiên': step,
                'Ngày': next_date.strftime('%d/%m/%Y'),
                'Giá dự báo': f"{price_pred_final:,.2f}",
                'Thay đổi': f"{display_change_pct:+.2f}%",
                'Khoảng dự báo': f"{forecast_lower:,.2f} - {forecast_upper:,.2f}",  # ← Dùng ATR động
                #'ATR': f"{working_data['atr_14'].iloc[-1]:.2f}"  # ← Hiển thị ATR mới
            })

            # ✅ FIX: Validate working_data AFTER update
            if working_data.isnull().any().any():
                print(f"⚠️ WARNING: NaN detected after step {step}")
                nan_cols = working_data.columns[working_data.isnull().any()].tolist()
                print(f"   Columns with NaN: {nan_cols}")
                print(f"   Filling NaN để tiếp tục...")
                working_data = working_data.fillna(method='ffill').fillna(method='bfill')
            
            # ✅ FIX: Validate critical features
            required_features = SELECTED_FEATURES + ['zlema20', 'residual']
            for feat in required_features:
                if pd.isna(working_data[feat].iloc[-1]):
                    print(f"❌ CRITICAL: {feat} is NaN at step {step}! Breaking...")
                    raise ValueError(f"Cannot continue with NaN in {feat}")
            
        except Exception as e:
            print(f"❌ Lỗi tại phiên {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if not forecasts:
        print("❌ Không có forecast nào")
        return None
    
    forecast_df = pd.DataFrame(forecasts)
    
    # Summary
    print(f"\n{'='*100}")
    print(f"📊 TỔNG KẾT FORECAST - {symbol}")
    print(f"{'='*100}")
    print(forecast_df.to_string(index=False))
    
    # Calculate total change
    try:
        first_price = float(forecast_df['Giá dự báo'].iloc[0].replace(',', ''))
        last_forecast_price = float(forecast_df['Giá dự báo'].iloc[-1].replace(',', ''))
        total_change = (last_forecast_price / last_price - 1) * 100
        print(f"\n📈 Từ {last_price:,.2f} → {last_forecast_price:,.2f} ({total_change:+.2f}%)")
    except:
        print("\n⚠️ Không thể tính total change (có NaN)")
    
    # # ✅ THÊM: Regime distribution (giống backtest)
    # regime_counts = forecast_df['Regime'].value_counts()
    # print(f"\n📊 Phân bố Regime:")
    # for regime, count in regime_counts.items():
    #     pct = (count / len(forecast_df)) * 100
    #     print(f"   {regime}: {count} phiên ({pct:.1f}%)")
    
    # # ✅ THÊM: Blending stats
    # blend_count = (forecast_df['Blend'] == '✓').sum()
    # blend_pct = (blend_count / len(forecast_df)) * 100
    # print(f"\n🔀 Hybrid Blending:")
    # print(f"   Sử dụng momentum: {blend_count}/{len(forecast_df)} phiên ({blend_pct:.1f}%)")
    # print(f"   Chỉ dùng model: {len(forecast_df) - blend_count}/{len(forecast_df)} phiên ({100-blend_pct:.1f}%)")
    
    print(f"{'='*100}\n")
    
    return forecast_df


def plot_forecast_with_history(symbol, forecast_df, historical_data, lookback=22):
    """
    Vẽ biểu đồ kết hợp dữ liệu quá khứ và dự báo
    ✅ Dùng trục số thứ tự để ẩn ngày nghỉ
    
    Parameters:
    -----------
    symbol : str
        Mã cổ phiếu
    forecast_df : DataFrame
        Bảng dự báo từ forecast_future_prices
    historical_data : DataFrame
        Dữ liệu lịch sử (df_full từ forecast_future_prices)
    lookback : int
        Số phiên quá khứ muốn hiển thị (mặc định 22)
    """
    # Lấy dữ liệu quá khứ
    hist_data = historical_data.tail(lookback).copy()
    hist_dates = hist_data['time'].dt.strftime('%d/%m/%y').tolist()  # Format ngắn
    hist_prices = hist_data['close'].values
    
    # Parse dữ liệu dự báo
    forecast_dates = forecast_df['Ngày'].tolist()
    forecast_prices = forecast_df['Giá dự báo'].str.replace(',', '').astype(float).values
    
    # Parse khoảng dự báo (nếu có)
    if 'Khoảng dự báo' in forecast_df.columns:
        ranges = forecast_df['Khoảng dự báo'].str.split(' - ', expand=True)
        forecast_lower = ranges[0].str.replace(',', '').astype(float).values
        forecast_upper = ranges[1].str.replace(',', '').astype(float).values
    else:
        forecast_lower = None
        forecast_upper = None
    
    # ✅ TẠO TRỤC SỐ THỨ TỰ (bỏ qua ngày nghỉ)
    hist_x = list(range(len(hist_prices)))
    forecast_x = list(range(len(hist_prices), len(hist_prices) + len(forecast_prices)))
    
    # Kết hợp labels
    all_labels = hist_dates + forecast_dates
    all_x = hist_x + forecast_x
    
    # Tạo figure
    plt.figure(figsize=(14, 7))
    
    # Plot dữ liệu quá khứ
    plt.plot(hist_x, hist_prices, 
             marker='o', linewidth=2, markersize=6,
             color='#2E86AB', label='Dữ liệu quá khứ',
             zorder=3)
    
    # Nối điểm cuối quá khứ với điểm đầu dự báo
    connection_x = [hist_x[-1], forecast_x[0]]
    connection_prices = [hist_prices[-1], forecast_prices[0]]
    plt.plot(connection_x, connection_prices, 
             linestyle='--', color='gray', alpha=0.5, linewidth=1.5)
    
    # Plot dự báo
    plt.plot(forecast_x, forecast_prices,
             marker='s', linewidth=2, markersize=7,
             color='#F77F00', label='Dự báo',
             zorder=3)
    
    # Plot khoảng dự báo (nếu có)
    if forecast_lower is not None and forecast_upper is not None:
        plt.fill_between(forecast_x, forecast_lower, forecast_upper,
                         alpha=0.2, color='#F77F00', 
                         label='Khoảng biến động dự báo')
    
    # ✅ TÙY CHỈNH TRỤC X - Chỉ hiển thị một số labels
    # Hiển thị mỗi N labels để tránh chồng chéo
    step = max(1, len(all_labels) // 15)  # Hiển thị ~15 labels
    selected_indices = list(range(0, len(all_labels), step))
    
    # Đảm bảo điểm đầu, điểm cuối quá khứ, và điểm cuối dự báo luôn hiển thị
    if 0 not in selected_indices:
        selected_indices.insert(0, 0)
    if len(hist_x) - 1 not in selected_indices:
        selected_indices.append(len(hist_x) - 1)
    if len(all_x) - 1 not in selected_indices:
        selected_indices.append(len(all_x) - 1)
    
    selected_indices = sorted(set(selected_indices))
    
    plt.xticks(
        [all_x[i] for i in selected_indices],
        [all_labels[i] for i in selected_indices],
        rotation=90, ha='center'
    )
    
    # Labels và title
    plt.xlabel('Phiên giao dịch', fontsize=12, fontweight='bold')
    plt.ylabel('Giá (VNĐ)', fontsize=12, fontweight='bold')
    plt.title(f'Dự báo giá {symbol} - {lookback} phiên quá khứ + {len(forecast_df)} phiên tương lai',
              fontsize=14, fontweight='bold', pad=20)
    
    # Grid và legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Lưu file
    filename = f'forecast_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Đã lưu biểu đồ: {filename}")
    
    # Hiển thị
    # plt.show()
    
    return filename

def create_table_image(df, title="Bảng dữ liệu", col_widths=None):
    """
    Tạo ảnh PNG của bảng từ DataFrame
    """
    if col_widths is None:
        col_widths = [0.1] * len(df.columns)  # mặc định đều

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.4 + 1.5))
    ax.axis('tight')
    ax.axis('off')

    # Tạo table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 1.6)  # điều chỉnh kích thước ô cho đẹp

    # Thêm tiêu đề
    plt.title(title, fontsize=14, pad=20)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

# forecast_future_prices(symbol='KBC', forecast_steps=14, show_details=True)

# ==================== CHẠY THỬ ====================
# if __name__ == "__main__":
#     backtest_vnindex_adaptive(
#         symbol='VNINDEX',
#         backtest_start='2025-12-16',
#         learning_window=120  # Học từ 120 ngày gần nhất
#     )
# # ==================== MAIN EXECUTION ====================
# if __name__ == "__main__":
#     # Nhập mã cổ phiếu từ người dùng
#     symbol = input("Nhập mã cổ phiếu (VD: VNINDEX, HPG, VNM): ").strip().upper()
    
#     # Đường dẫn file model và scaler
#     rf_model_path = f'saved_model/best_randomforest_default_{symbol}.joblib'
#     scaler_path = f'saved_model/scaler_randomforest_{symbol}.joblib'
    
#     # Kiểm tra xem đã có model chưa
#     model_exists = os.path.exists(rf_model_path) and os.path.exists(scaler_path)
    
#     if not model_exists:
#         print(f"\n⚠️  Chưa có model cho {symbol}. Bắt đầu training...")
#         print("="*100)
        
#         # 1. Lấy dữ liệu
#         print(f"📊 Đang lấy dữ liệu {symbol}...")
#         all_dataframes = get_stock_historical_data(
#             symbols=symbol,
#             start_date=START_DATE,
#             end_date=END_DATE,
#             interval=INTERVAL
#         )
        
#         if symbol not in all_dataframes:
#             print(f"❌ Không thể lấy dữ liệu cho {symbol}")
#             exit(1)
        
#         # 2. Tính toán indicators
#         print(f"🔄 Đang tính toán chỉ báo kỹ thuật...")
#         all_dataframes = compute_indicators_for_all(all_dataframes)
        
#         # 3. Chuẩn bị dữ liệu hybrid residual
#         print(f"🔧 Đang chuẩn bị dữ liệu training...")
#         hybrid_data = prepare_hybrid_residual(all_dataframes)
        
#         hr = hybrid_data[symbol]['hybrid_residual']
#         scaler = hybrid_data[symbol]['scalers']['hybrid_residual']
        
#         X_train_3d = hr['X_train']
#         y_train = hr['y_train']
#         X_test_3d = hr['X_test']
#         y_test = hr['y_test']
#         actual_prices_test = hr['raw_close_test']
#         zlema20_test = hr.get('zlema20_test')
        
#         # Flatten cho tree-based models
#         X_train_2d = X_train_3d.reshape(X_train_3d.shape[0], -1)
#         X_test_2d = X_test_3d.reshape(X_test_3d.shape[0], -1)
        
#         # 4. Train model
#         print(f"🤖 Đang train Random Forest model...")
#         rf_model = RandomForestRegressor(
#             n_estimators=100,
#             random_state=42,
#             n_jobs=-1
#         )
#         rf_model.fit(X_train_2d, y_train)
        
#         # 5. Đánh giá model
#         def inverse_residual(scaled_residual, scaler):
#             dummy = np.zeros((len(scaled_residual), scaler.n_features_in_))
#             dummy[:, -1] = scaled_residual
#             return scaler.inverse_transform(dummy)[:, -1]
        
#         pred_residual_rf = inverse_residual(rf_model.predict(X_test_2d), scaler)
#         pred_prices = zlema20_test + pred_residual_rf
        
#         rmse = np.sqrt(mean_squared_error(actual_prices_test, pred_prices))
#         mae = mean_absolute_error(actual_prices_test, pred_prices)
#         mape_val = np.mean(np.abs((actual_prices_test - pred_prices) / actual_prices_test)) * 100
#         r2 = r2_score(actual_prices_test, pred_prices)
#         dir_acc = np.mean(np.sign(np.diff(actual_prices_test)) == np.sign(np.diff(pred_prices))) * 100
        
#         print(f"\n✅ Kết quả training:")
#         print(f"   RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape_val:.2f}% | R²: {r2:.3f} | Dir Acc: {dir_acc:.1f}%")
        
#         # 6. Lưu model
#         os.makedirs("saved_model", exist_ok=True)
#         joblib.dump(rf_model, f'saved_model/best_randomforest_default_{symbol}.joblib') 
#         joblib.dump(scaler, f'saved_model/scaler_randomforest_{symbol}.joblib')
#         print(f"\n💾 Đã lưu model: {rf_model_path}")
#         print(f"💾 Đã lưu scaler: {scaler_path}")
#         print("="*100)
#     else:
#         print(f"\n✅ Đã tìm thấy model cho {symbol}")
#         print(f"   📁 {rf_model_path}")
#         print(f"   📁 {scaler_path}")
    
    # # ==================== DỰ BÁO GIÁ ====================
    # print(f"\n🔮 BẮT ĐẦU DỰ BÁO GIÁ CHO {symbol}")
    # print("="*100)
    
    # # Nhập số phiên muốn dự báo (mặc định 5)
    # try:
    #     forecast_steps = int(input("Nhập số phiên muốn dự báo (1-30, mặc định 5): ").strip() or "5")
    #     forecast_steps = max(1, min(30, forecast_steps))  # Giới hạn 1-30
    # except:
    #     forecast_steps = 5
    
    # if forecast_steps > 0:
    #     yesterday = datetime.now() - timedelta(days=1)
    #     end_fetch = yesterday.strftime('%Y-%m-%d')
        
    #     quote = Quote(symbol=symbol, source='VCI')
    #     df_full_for_plot = quote.history(start='2020-01-01', end=end_fetch, interval='d')
    #     df_full_for_plot = df_full_for_plot.sort_values('time').reset_index(drop=True)
    #     df_full_for_plot['time'] = pd.to_datetime(df_full_for_plot['time'])
    #     df_full_for_plot = compute_indicators_single(df_full_for_plot)
    
    # # Chạy dự báo
    # forecast_df = forecast_future_prices(
    #     symbol=symbol,
    #     forecast_steps=forecast_steps,
    #     rf_model_path=rf_model_path,
    #     scaler_path=scaler_path,
    #     learning_window=120,
    #     show_details=False,
    #     use_yesterday=True  # Dùng data đến hôm qua để ổn định
    # )
    
    # if forecast_df is not None:
    #     print(f"\n✅ Hoàn tất dự báo {forecast_steps} phiên cho {symbol}!")
        
    #     # ✅ VẼ BIỂU ĐỒ
    #     try:
    #         plot_forecast_with_history(
    #             symbol=symbol,
    #             forecast_df=forecast_df,
    #             historical_data=df_full_for_plot,
    #             lookback=22
    #         )
    #     except Exception as e:
    #         print(f"⚠️ Không thể vẽ biểu đồ: {e}")
    # else:
    #     print(f"\n❌ Có lỗi trong quá trình dự báo")

