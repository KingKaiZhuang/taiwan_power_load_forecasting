import os
import pandas as pd
from train_model import load_and_process_data, train_model, make_prediction

# 資料路徑設定
DATA_PATH = os.path.join("data", "power_data.csv")
RESULTS_PATH = os.path.join("data", "forecast_results.csv")

def generate_forecast_file(epochs, lr, seq_length):
    """
    執行模型訓練並產生預測檔案
    """
    if os.path.exists(DATA_PATH):
        print(f"Starting training process with Epochs={epochs}, LR={lr}, SeqLen={seq_length}...")
        df = load_and_process_data(DATA_PATH)
        # 呼叫 train_model 進行訓練
        train_results = train_model(df, epochs=int(epochs), lr=lr, seq_length=int(seq_length))
        # 產生預測結果
        forecast = make_prediction(train_results, df)
        forecast.to_csv(RESULTS_PATH, index=False)
        print("Training complete and file saved.")
        return True
    return False

def get_forecast_data():
    """
    讀取預測結果 CSV，若檔案不存在則執行初次訓練
    """
    if not os.path.exists(RESULTS_PATH):
        print("Forecast data not found, running initial training...")
        # 預設參數進行初次訓練
        success = generate_forecast_file(300, 0.005, 30)
        if not success:
            return None

    forecast = pd.read_csv(RESULTS_PATH)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    return forecast
