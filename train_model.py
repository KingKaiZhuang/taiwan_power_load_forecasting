import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib 

# 設定隨機種子以確保結果可重現 (Set random seed for reproducibility)
torch.manual_seed(42)
np.random.seed(42)

# 檢查是否有 GPU 可用 (Check for GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PowerConsumptionLSTM(nn.Module):
    """
    電力負載預測用的 LSTM 模型
    架構: LSTM 層 -> Fully Connected (全連接) 層
    """
    def __init__(self, input_size=3, hidden_size=64, output_size=1):
        super(PowerConsumptionLSTM, self).__init__()
        self.hidden_size = hidden_size
        # LSTM 層: 接收輸入特徵，輸出隱藏狀態
        # input_size=3 (耗電量, 月份, 星期)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 全連接層: 將 LSTM 的最後一個時間點輸出轉為預測值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 形狀: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # 取最後一個時間點的輸出作為預測依據
        # out 形狀: (batch_size, seq_len, hidden_size) -> 取 [:, -1, :]
        out = self.fc(out[:, -1, :])
        return out

def load_and_process_data(filepath):
    """
    讀取並處理電力數據
    1. 轉換日期格式
    2. 填補缺漏值
    3. 合併工業與民生用電
    4. 產生時間特徵 (月份, 星期)
    """
    print("Loading data... (載入數據中...)")
    df = pd.read_csv(filepath)
    
    # 處理日期格式
    df['date'] = df['date'].astype(str).str.strip()
    df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['ds'])
    
    # 處理數值 (轉為數字並填補 NaN 為 0)
    df['industrial'] = pd.to_numeric(df['industrial'], errors='coerce').fillna(0)
    df['residential'] = pd.to_numeric(df['residential'], errors='coerce').fillna(0)
    df['peak_load'] = pd.to_numeric(df['peak_load'], errors='coerce').fillna(0)
    
    # 重新命名欄位以便識別
    # Columns are already in English

    
    # 計算總耗電量
    df['total'] = df['industrial'] + df['residential']
    
    # 轉為每日數據 (加總，尖峰負載取最大值 - 雖然資料已是每日，依邏輯應為 Max)
    df_daily = df.groupby('ds').agg({
        'industrial': 'sum',
        'residential': 'sum',
        'total': 'sum',
        'peak_load': 'max'
    }).reset_index()
    # 移除異常值 (總耗電 <= 0 的資料)
    df_daily = df_daily[df_daily['total'] > 0]
    df_daily = df_daily.sort_values('ds')
    
    # 特徵工程 (Feature Engineering): 加入月份與星期
    df_daily['month'] = df_daily['ds'].dt.month
    df_daily['day_of_week'] = df_daily['ds'].dt.dayofweek
    
    print(f"Data processed. Daily records: {len(df_daily)} (資料處理完成，共 {len(df_daily)} 筆)")
    return df_daily

def create_sequences(data, seq_length):
    """
    將時間序列資料轉換為 LSTM 訓練用的樣本
    X: 過去 N 天的數據 (包含耗電量與時間特徵)
    y: 第 N+1 天的耗電量 (Target)
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0] # 目標值永遠是第一欄 (耗電量)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_single_model(df, target_col, epochs=300, lr=0.005, seq_length=30):
    """
    訓練單一類別的 LSTM 模型 (例如只訓練工業用電)
    輸入特徵: [目標用電量, 月份, 星期]
    """
    # 自動調整 Sequence Length (防止數據過少導致當機)
    min_samples = 300
    if len(df) - seq_length < min_samples:
        print(f"Warning: Sequence length {seq_length} leaves too few samples for training (Data len: {len(df)}).")
        raw_target_seq = len(df) - min_samples
        
        if raw_target_seq < 30:
             seq_length = max(7, int(len(df) * 0.3))
        else:
             seq_length = raw_target_seq
             
        print(f"Auto-adjusted sequence length to: {seq_length} (to ensure ~{len(df)-seq_length} training samples)")

    # 準備訓練資料矩陣: [耗電量, 月份, 星期]
    data_values = df[[target_col, 'month', 'day_of_week']].values
    
    # 資料標準化 (歸一化到 0~1 之間)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)
    
    # 建立序列數據 (X, y)
    X, y = create_sequences(scaled_data, seq_length)
    
    if len(X) == 0:
         print("Error: No sequences created. Data too short.")
         return None, scaler, seq_length
    
    # 轉為 PyTorch Tensor 並移至裝置 (GPU/CPU)
    X_train = torch.from_numpy(X).float().to(device)
    y_train = torch.from_numpy(y).float().to(device)
    
    # 初始化模型 (Input=3: 耗電, 月, 星)
    model = PowerConsumptionLSTM(input_size=3, hidden_size=64, output_size=1).to(device)
    criterion = nn.MSELoss() # 損失函數: 均方誤差
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 優化器: Adam
    
    model.train() # 設定為訓練模式
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
    return model, scaler, seq_length

def save_checkpoint(name, model, scaler):
    """
    儲存模型權重與 Standardizer 到 models 資料夾
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = os.path.join("models", f"model_{name}.pth")
    scaler_path = os.path.join("models", f"scaler_{name}.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved {model_path} and {scaler_path}")

def load_checkpoint(name):
    """
    從 models 資料夾讀取模型與 Standardizer
    """
    model_path = os.path.join("models", f"model_{name}.pth")
    scaler_path = os.path.join("models", f"scaler_{name}.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = PowerConsumptionLSTM(input_size=3, hidden_size=64, output_size=1)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model {name}: {e}")
        print("Model architecture mismatch. Please retrain the model.")
        return None, None
        
    model.to(device)
    scaler = joblib.load(scaler_path)
    return model, scaler

def train_model(df, epochs=300, lr=0.005, seq_length=720):
    """
    主訓練函式：分別訓練工業與民生用電模型
    """
    print(f"Training PyTorch LSTM models on {device}... Epochs: {epochs}, LR: {lr}, SeqLen: {seq_length}")
    
    # 訓練工業用電模型
    print("Training Industrial Model... (訓練工業用電模型)")
    model_ind, scaler_ind, seq_len_ind = train_single_model(df, 'industrial', epochs, lr, seq_length)
    if model_ind:
        save_checkpoint("industrial", model_ind, scaler_ind)
    
    # 訓練民生用電模型
    print("Training Residential Model... (訓練民生用電模型)")
    model_res, scaler_res, seq_len_res = train_single_model(df, 'residential', epochs, lr, seq_length)
    if model_res:
        save_checkpoint("residential", model_res, scaler_res)
    
    # 訓練尖峰負載模型
    print("Training Peak Load Model... (訓練尖峰負載模型)")
    model_peak, scaler_peak, seq_len_peak = train_single_model(df, 'peak_load', epochs, lr, seq_length)
    if model_peak:
        save_checkpoint("peak_load", model_peak, scaler_peak)

    if model_ind is None or model_res is None or model_peak is None:
        raise ValueError("Training failed due to insufficient data for the chosen Sequence Length.")

    return {
        'industrial': (model_ind, scaler_ind, seq_len_ind),
        'residential': (model_res, scaler_res, seq_len_res),
        'peak_load': (model_peak, scaler_peak, seq_len_peak)
    }

def predict_future(model_tuple, df, target_col, future_dates):
    """
    使用訓練好的模型預測未來數據 (Rolling Forecast)
    """
    model, scaler, seq_length = model_tuple
    if model is None: return np.zeros(len(future_dates))
    
    model.eval() # 設定為評估模式
    model.to(device)
    
    # 準備歷史數據作為初始輸入
    hist_values = df[[target_col, 'month', 'day_of_week']].values
    hist_scaled = scaler.transform(hist_values)
    
    # 確保歷史數據長度足夠
    if len(hist_scaled) < seq_length:
         seq_length = len(hist_scaled)
    
    # 取最後一段序列
    current_seq = hist_scaled[-seq_length:].reshape(1, seq_length, 3) 
    current_seq_tensor = torch.from_numpy(current_seq).float().to(device)
    
    predictions = []
    
    # 處理未來日期的特徵 (如果是 DatetimeIndex 或 Series)
    if hasattr(future_dates, 'dt'):
        future_months = future_dates.dt.month.values
        future_dows = future_dates.dt.dayofweek.values
    else:
        future_months = future_dates.month.values if hasattr(future_dates.month, 'values') else future_dates.month
        future_dows = future_dates.dayofweek.values if hasattr(future_dates.dayofweek, 'values') else future_dates.dayofweek
    
    with torch.no_grad():
        for i in range(len(future_dates)):
            # 1. 預測下一天的耗電量
            pred_consumption = model(current_seq_tensor).item() # 用電量預測值 (Scaled)
            predictions.append(pred_consumption)
            
            # 2. 建構下一步的輸入向量 (包含已知的未來日期特徵)
            next_month = future_months[i]
            next_dow = future_dows[i]
            
            # 標準化輔助特徵 (公式: scaled = (raw * scale) + min)
            cons_scaled = pred_consumption # 已經是 Scaled
            mon_scaled = next_month * scaler.scale_[1] + scaler.min_[1]
            dow_scaled = next_dow * scaler.scale_[2] + scaler.min_[2]
            
            new_row = np.array([[[cons_scaled, mon_scaled, dow_scaled]]]) # 形狀 (1, 1, 3)
            new_row_tensor = torch.from_numpy(new_row).float().to(device)
            
            # 3. 更新序列: 移除最舊的一天，加入新預測的一天
            current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], new_row_tensor), dim=1)
            
    # 反標準化 (Inverse Transform) 轉回真實用電量
    pred_array = np.array(predictions)
    # 我們只需要反標準化第一欄 (耗電量): (val - min) / scale
    pred_raw = (pred_array - scaler.min_[0]) / scaler.scale_[0]
    
    return pred_raw

def make_prediction(train_results, df):
    """
    產生 2026 年的完整預測報告
    """
    print("Generating daily forecast for 2026... (正在產生 2026 每日預測結果...)")
    
    last_date = df['ds'].iloc[-1]
    target_date = pd.Timestamp('2026-12-31')
    days_to_predict = (target_date - last_date).days
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
    
    # 預測工業用電
    pred_ind = predict_future(train_results['industrial'], df, 'industrial', future_dates)
    
    # 預測民生用電
    pred_res = predict_future(train_results['residential'], df, 'residential', future_dates)
    
    # 預測尖峰負載
    pred_peak = predict_future(train_results['peak_load'], df, 'peak_load', future_dates)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'industrial': pred_ind,
        'residential': pred_res,
        'peak_load': pred_peak,
        'data_type': 'Predicted'
    })
    # 計算總和
    forecast_df['total'] = forecast_df['industrial'] + forecast_df['residential']
    
    history_df = df[['ds', 'industrial', 'residential', 'peak_load', 'total']].copy()
    history_df['data_type'] = 'Actual'
    
    full_df = pd.concat([history_df, forecast_df], ignore_index=True)
    return full_df

def evaluate_model(df, test_days=60, seq_length=720):
    """
    模型評估：使用最後 test_days 天的數據進行回測 (Backtesting)
    """
    print(f"Evaluating using SAVED MODELS on last {test_days} days. SeqLen: {seq_length}")
    
    # 載入已存檔的模型
    model_ind, scaler_ind = load_checkpoint("industrial")
    model_res, scaler_res = load_checkpoint("residential")
    model_peak, scaler_peak = load_checkpoint("peak_load")
    
    if model_ind is None or model_res is None or model_peak is None:
        return None, None, None

    cutoff_index = len(df) - test_days
    
    train_df = df.iloc[:cutoff_index]
    test_df = df.iloc[cutoff_index:]
    test_dates = test_df['ds']
    
    # 進行回測預測
    pred_ind = predict_future((model_ind, scaler_ind, seq_length), train_df, 'industrial', test_dates)
    pred_res = predict_future((model_res, scaler_res, seq_length), train_df, 'residential', test_dates)
    pred_peak = predict_future((model_peak, scaler_peak, seq_length), train_df, 'peak_load', test_dates)
    
    metrics = {}
    
    # 計算工業用電指標
    rmse_ind = np.sqrt(mean_squared_error(test_df['industrial'], pred_ind))
    mae_ind = mean_absolute_error(test_df['industrial'], pred_ind)
    metrics['industrial'] = {'rmse': rmse_ind, 'mae': mae_ind}
    
    # 計算民生用電指標
    rmse_res = np.sqrt(mean_squared_error(test_df['residential'], pred_res))
    mae_res = mean_absolute_error(test_df['residential'], pred_res)
    metrics['residential'] = {'rmse': rmse_res, 'mae': mae_res}

    # 計算尖峰負載指標
    rmse_peak = np.sqrt(mean_squared_error(test_df['peak_load'], pred_peak))
    mae_peak = mean_absolute_error(test_df['peak_load'], pred_peak)
    metrics['peak_load'] = {'rmse': rmse_peak, 'mae': mae_peak}
    
    # 計算總用電指標
    pred_total = pred_ind + pred_res
    rmse_total = np.sqrt(mean_squared_error(test_df['total'], pred_total))
    mae_total = mean_absolute_error(test_df['total'], pred_total)
    metrics['total'] = {'rmse': rmse_total, 'mae': mae_total}
    
    # 整理評估結果 DataFrame
    eval_df = test_df[['ds', 'industrial', 'residential', 'peak_load', 'total']].copy()
    eval_df = eval_df.rename(columns={'industrial': 'Actual_Ind', 'residential': 'Actual_Res', 'peak_load': 'Actual_Peak', 'total': 'Actual_Total'})
    eval_df['Predicted_Ind'] = pred_ind
    eval_df['Predicted_Res'] = pred_res
    eval_df['Predicted_Peak'] = pred_peak
    eval_df['Predicted_Total'] = pred_total
    
    return eval_df, metrics, train_df

if __name__ == "__main__":
    file_path = os.path.join("data", "power_data.csv")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        exit(1)
        
    df = load_and_process_data(file_path)
    try:
        train_results = train_model(df)
        forecast = make_prediction(train_results, df)
        output_file = os.path.join("data", "forecast_results.csv")
        forecast.to_csv(output_file, index=False)
        print(f"Forecast saved to {output_file}")
    except ValueError as e:
        print(f"Training Error: {e}")
