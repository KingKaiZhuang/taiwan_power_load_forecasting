import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib 

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PowerConsumptionLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1):
        super(PowerConsumptionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # out shape: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

def load_and_process_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    df['日期'] = df['日期'].astype(str).str.strip()
    df['ds'] = pd.to_datetime(df['日期'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['ds'])
    
    df['工業用電(百萬度)'] = pd.to_numeric(df['工業用電(百萬度)'], errors='coerce').fillna(0)
    df['民生用電(百萬度)'] = pd.to_numeric(df['民生用電(百萬度)'], errors='coerce').fillna(0)
    
    # Rename for clarity
    df = df.rename(columns={
        '工業用電(百萬度)': 'industrial',
        '民生用電(百萬度)': 'residential'
    })
    
    df['total'] = df['industrial'] + df['residential']
    
    # Use daily data
    df_daily = df.groupby('ds')[['industrial', 'residential', 'total']].sum().reset_index()
    df_daily = df_daily[df_daily['total'] > 0]
    df_daily = df_daily.sort_values('ds')
    
    # Feature Engineering
    df_daily['month'] = df_daily['ds'].dt.month
    df_daily['day_of_week'] = df_daily['ds'].dt.dayofweek
    
    print(f"Data processed. Daily records: {len(df_daily)}")
    return df_daily

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0] # Target is always the first column (consumption)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_single_model(df, target_col, epochs=300, lr=0.005, seq_length=30):
    """
    Trains a model on a dataframe.
    Features: [target_col, 'month', 'day_of_week']
    """
    # Auto-adjust sequence length if it's too large (Protection against exploding gradients)
    min_samples = 300
    if len(df) - seq_length < min_samples:
        print(f"Warning: Sequence length {seq_length} leaves too few samples for training (Data len: {len(df)}).")
        raw_target_seq = len(df) - min_samples
        
        if raw_target_seq < 30:
             seq_length = max(7, int(len(df) * 0.3))
        else:
             seq_length = raw_target_seq
             
        print(f"Auto-adjusted sequence length to: {seq_length} (to ensure ~{len(df)-seq_length} training samples)")

    # Prepare data matrix: [consumption, month, day_of_week]
    data_values = df[[target_col, 'month', 'day_of_week']].values
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)
    
    X, y = create_sequences(scaled_data, seq_length)
    
    if len(X) == 0:
         print("Error: No sequences created. Data too short.")
         return None, scaler, seq_length
    
    X_train = torch.from_numpy(X).float().to(device)
    y_train = torch.from_numpy(y).float().to(device)
    
    # input_size=3 (consumption, month, day_of_week)
    model = PowerConsumptionLSTM(input_size=3, hidden_size=64, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    
    model.train() 
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
    """Saves model weights and scaler to disk."""
    if not os.path.exists("models"):
        os.makedirs("models")
    model_path = os.path.join("models", f"model_{name}.pth")
    scaler_path = os.path.join("models", f"scaler_{name}.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved {model_path} and {scaler_path}")

def load_checkpoint(name):
    """Loads model weights and scaler from disk."""
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
    print(f"Training PyTorch LSTM models on {device}... Epochs: {epochs}, LR: {lr}, SeqLen: {seq_length}")
    
    # Train Industrial Model
    print("Training Industrial Model...")
    model_ind, scaler_ind, seq_len_ind = train_single_model(df, 'industrial', epochs, lr, seq_length)
    if model_ind:
        save_checkpoint("industrial", model_ind, scaler_ind)
    
    # Train Residential Model
    print("Training Residential Model...")
    model_res, scaler_res, seq_len_res = train_single_model(df, 'residential', epochs, lr, seq_length)
    if model_res:
        save_checkpoint("residential", model_res, scaler_res)
    
    if model_ind is None or model_res is None:
        raise ValueError("Training failed due to insufficient data for the chosen Sequence Length.")

    return {
        'industrial': (model_ind, scaler_ind, seq_len_ind),
        'residential': (model_res, scaler_res, seq_len_res)
    }

def predict_future(model_tuple, df, target_col, future_dates):
    model, scaler, seq_length = model_tuple
    if model is None: return np.zeros(len(future_dates))
    
    model.eval()
    model.to(device)
    
    hist_values = df[[target_col, 'month', 'day_of_week']].values
    hist_scaled = scaler.transform(hist_values)
    
    # Check if history is enough
    if len(hist_scaled) < seq_length:
         seq_length = len(hist_scaled)
    
    current_seq = hist_scaled[-seq_length:].reshape(1, seq_length, 3) 
    current_seq_tensor = torch.from_numpy(current_seq).float().to(device)
    
    predictions = []
    
    # Handle both DatetimeIndex (from make_prediction) and Series (from evaluate_model)
    if hasattr(future_dates, 'dt'):
        future_months = future_dates.dt.month.values
        future_dows = future_dates.dt.dayofweek.values
    else:
        future_months = future_dates.month.values if hasattr(future_dates.month, 'values') else future_dates.month
        future_dows = future_dates.dayofweek.values if hasattr(future_dates.dayofweek, 'values') else future_dates.dayofweek
    
    with torch.no_grad():
        for i in range(len(future_dates)):
            # Predict next consumption
            pred_consumption = model(current_seq_tensor).item() # scalar
            predictions.append(pred_consumption)
            
            # Construct next input vector
            next_month = future_months[i]
            next_dow = future_dows[i]
            
            # Scale auxiliary features
            # formula: scaled = (raw * scale) + min
            cons_scaled = pred_consumption # already scaled output from model
            mon_scaled = next_month * scaler.scale_[1] + scaler.min_[1]
            dow_scaled = next_dow * scaler.scale_[2] + scaler.min_[2]
            
            new_row = np.array([[[cons_scaled, mon_scaled, dow_scaled]]]) # (1, 1, 3)
            new_row_tensor = torch.from_numpy(new_row).float().to(device)
            
            # Update sequence
            current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], new_row_tensor), dim=1)
            
    # Inverse transform predictions (only first column)
    pred_array = np.array(predictions)
    pred_raw = (pred_array - scaler.min_[0]) / scaler.scale_[0]
    
    return pred_raw

def make_prediction(train_results, df):
    print("Generating daily forecast for 2026...")
    
    last_date = df['ds'].iloc[-1]
    target_date = pd.Timestamp('2026-12-31')
    days_to_predict = (target_date - last_date).days
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
    
    # Predict Industrial
    pred_ind = predict_future(train_results['industrial'], df, 'industrial', future_dates)
    
    # Predict Residential
    pred_res = predict_future(train_results['residential'], df, 'residential', future_dates)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'industrial': pred_ind,
        'residential': pred_res,
        'data_type': 'Predicted'
    })
    forecast_df['total'] = forecast_df['industrial'] + forecast_df['residential']
    
    history_df = df[['ds', 'industrial', 'residential', 'total']].copy()
    history_df['data_type'] = 'Actual'
    
    full_df = pd.concat([history_df, forecast_df], ignore_index=True)
    return full_df

def evaluate_model(df, test_days=60, seq_length=720):
    print(f"Evaluating using SAVED MODELS on last {test_days} days. SeqLen: {seq_length}")
    
    # Load Models
    model_ind, scaler_ind = load_checkpoint("industrial")
    model_res, scaler_res = load_checkpoint("residential")
    
    if model_ind is None or model_res is None:
        return None, None, None

    cutoff_index = len(df) - test_days
    
    # Dynamic adjust seq_length if needed for evaluation
    # We need cutoff_index >= seq_length ideally
    # But predict_future handles shorter history if implemented carefully.
    # Let's verify predict_future logic. It takes `model_tuple` which has (model, scaler, train_seq_length).
    # But inside predict_future we slice `hist_scaled[-seq_length:]`.
    # We should override the seq_length in tuple if needed, or pass it explicitly.
    # predict_future uses `seq_length` from tuple.
    
    # Fix: ensure passed seq_length is safe
    # But tuple is (model, scaler, original_train_seq_len).
    # We might need to override it.
    # Let's construct a temp tuple or modify predict_future to check.
    # predict_future HAS a check: if len(hist_scaled) < seq_length: seq_length = len(hist_scaled)
    # So it should be safe.
    
    train_df = df.iloc[:cutoff_index]
    test_df = df.iloc[cutoff_index:]
    test_dates = test_df['ds']
    
    # Industrial Eval
    pred_ind = predict_future((model_ind, scaler_ind, seq_length), train_df, 'industrial', test_dates)
    
    # Residential Eval
    pred_res = predict_future((model_res, scaler_res, seq_length), train_df, 'residential', test_dates)
    
    # Metrics
    metrics = {}
    
    # Industrial Metrics
    rmse_ind = np.sqrt(mean_squared_error(test_df['industrial'], pred_ind))
    mae_ind = mean_absolute_error(test_df['industrial'], pred_ind)
    metrics['industrial'] = {'rmse': rmse_ind, 'mae': mae_ind}
    
    # Residential Metrics
    rmse_res = np.sqrt(mean_squared_error(test_df['residential'], pred_res))
    mae_res = mean_absolute_error(test_df['residential'], pred_res)
    metrics['residential'] = {'rmse': rmse_res, 'mae': mae_res}
    
    # Total Metrics
    pred_total = pred_ind + pred_res
    rmse_total = np.sqrt(mean_squared_error(test_df['total'], pred_total))
    mae_total = mean_absolute_error(test_df['total'], pred_total)
    metrics['total'] = {'rmse': rmse_total, 'mae': mae_total}
    
    # Dataframe construction
    eval_df = test_df[['ds', 'industrial', 'residential', 'total']].copy()
    eval_df = eval_df.rename(columns={'industrial': 'Actual_Ind', 'residential': 'Actual_Res', 'total': 'Actual_Total'})
    eval_df['Predicted_Ind'] = pred_ind
    eval_df['Predicted_Res'] = pred_res
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
