import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import os
from train_model import load_and_process_data, train_model, make_prediction, evaluate_model

DATA_PATH = os.path.join("data", "power_data.csv")
RESULTS_PATH = os.path.join("data", "forecast_results.csv")

def generate_forecast_file(epochs, lr, seq_length):
    if os.path.exists(DATA_PATH):
        print(f"Starting training process with Epochs={epochs}, LR={lr}, SeqLen={seq_length}...")
        df = load_and_process_data(DATA_PATH)
        train_results = train_model(df, epochs=int(epochs), lr=lr, seq_length=int(seq_length))
        forecast = make_prediction(train_results, df)
        forecast.to_csv(RESULTS_PATH, index=False)
        print("Training complete and file saved.")
        return True
    return False

def get_forecast_data():
    if not os.path.exists(RESULTS_PATH):
        print("Forecast data not found, running initial training...")
        # Default initial training
        success = generate_forecast_file(300, 0.005, 30)
        if not success:
            return None

    forecast = pd.read_csv(RESULTS_PATH)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    return forecast

def create_dashboard(view_type, selected_month="全部 (All)"):
    forecast = get_forecast_data()
    
    if forecast is None:
        return go.Figure(), pd.DataFrame()

    actual_data = forecast[forecast['data_type'] == 'Actual']
    predicted_data = forecast[forecast['data_type'] == 'Predicted']
    
    forecast_2026 = predicted_data[predicted_data['ds'].dt.year == 2026].copy()
    
    fig = go.Figure()
    
    def add_traces(col_name, label, color_act, color_pred):
        fig.add_trace(go.Scatter(
            x=actual_data['ds'], 
            y=actual_data[col_name], 
            mode='lines', 
            name=f'實際-{label}',
            line=dict(color=color_act, width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=predicted_data['ds'], 
            y=predicted_data[col_name], 
            mode='lines', 
            name=f'預測-{label}',
            line=dict(color=color_pred, width=1.5)
        ))

    if view_type == "總耗電量 (Total)":
        add_traces('total', '總耗電', 'black', 'blue')
    elif view_type == "工業用電 (Industrial)":
        add_traces('industrial', '工業', 'darkred', 'red')
    elif view_type == "民生用電 (Residential)":
        add_traces('residential', '民生', 'darkgreen', 'green')
    elif view_type == "全部顯示 (All)":
        add_traces('total', '總耗電', 'black', 'blue')
        add_traces('industrial', '工業', 'darkred', 'red')
        add_traces('residential', '民生', 'darkgreen', 'green')
    
    fig.update_layout(
        title=f'台灣電力負載：實際 vs 預測 ({view_type})',
        xaxis_title='日期',
        yaxis_title='耗電量 (百萬度)',
        hovermode='x unified',
        template='plotly_white'
    )

    # Filter table data based on month
    if selected_month != "全部 (All)":
        month_map = {
            "1月": 1, "2月": 2, "3月": 3, "4月": 4, "5月": 5, "6月": 6,
            "7月": 7, "8月": 8, "9月": 9, "10月": 10, "11月": 11, "12月": 12
        }
        # Extract month string (e.g., "1月" from "1月") or handle if formats differ
        # Assuming input is like "1月", "2月"
        m = month_map.get(selected_month)
        if m:
            forecast_2026 = forecast_2026[forecast_2026['ds'].dt.month == m]

    table_cols = ['ds', 'total', 'industrial', 'residential']
    table_data = forecast_2026[table_cols].copy()
    table_data['ds'] = table_data['ds'].dt.strftime('%Y-%m-%d')
    table_data.columns = ['日期', '總耗電量预测', '工業用電預測', '民生用電預測']
    table_data = table_data.round(2)

    return fig, table_data

def train_and_update(view_type, epochs, lr, seq_length, selected_month):
    gr.Info(f"訓練開始... (Epochs={epochs}, LR={lr}, Window={seq_length})")
    generate_forecast_file(epochs, lr, seq_length)
    gr.Info("訓練完成！正在更新儀表板...")
    return create_dashboard(view_type, selected_month)

def run_evaluation(seq_length, target_type):
    gr.Info(f"正在執行模型評估 (類別: {target_type})...")
    if os.path.exists(DATA_PATH):
        df = load_and_process_data(DATA_PATH)
        eval_result = evaluate_model(df, test_days=90, seq_length=int(seq_length))
        
        if eval_result[0] is None:
             return go.Figure(), "錯誤：數據不足或模型未訓練。"
             
        eval_df, metrics, train_df = eval_result
        
        fig = go.Figure()
        plot_train_df = train_df.iloc[-180:]
        
        # Determine columns based on selection
        if "總耗電" in target_type:
            col_act = 'Actual_Total'
            col_pred = 'Predicted_Total'
            col_train = 'total'
            metric_key = 'total'
        elif "工業" in target_type:
            col_act = 'Actual_Ind'
            col_pred = 'Predicted_Ind'
            col_train = 'industrial'
            metric_key = 'industrial'
        elif "民生" in target_type:
            col_act = 'Actual_Res'
            col_pred = 'Predicted_Res'
            col_train = 'residential'
            metric_key = 'residential'
            
        fig.add_trace(go.Scatter(
            x=plot_train_df['ds'],
            y=plot_train_df[col_train], 
            mode='lines',
            name='訓練數據',
            line=dict(color='black', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=eval_df['ds'],
            y=eval_df[col_act],
            mode='lines',
            name='測試數據 (實際)',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=eval_df['ds'],
            y=eval_df[col_pred],
            mode='lines',
            name='測試數據 (預測)',
            line=dict(color='orange', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=f'模型評估 ({target_type})',
            xaxis_title='日期',
            yaxis_title='耗電量',
            template='plotly_white'
        )
        
        m = metrics[metric_key]
        metrics_text = f"""
        ### 評估指標 ({target_type})
        *   **RMSE**: {m['rmse']:.2f}
        *   **MAE**: {m['mae']:.2f}
        """
        
        return fig, metrics_text
        
    return go.Figure(), "錯誤：找不到數據。"

with gr.Blocks(title="2026 電力預測") as demo:
    gr.Markdown("# 2026 台灣電力負載預測 🇹🇼⚡")
    
    with gr.Tabs():
        # ... (Prediction Tab omitted) ...
        with gr.TabItem("預測儀表板"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 訓練參數設定")
                    epochs_slider = gr.Slider(minimum=50, maximum=100000, value=300, step=50, label="訓練次數 (Epochs)")
                    lr_number = gr.Number(value=0.005, label="學習率 (Learning Rate)", step=0.001)
                    seq_slider = gr.Slider(minimum=7, maximum=730, value=720, step=1, label="回看天數 (Sequence Length)")
                    train_btn = gr.Button("🔄 重新訓練模型並預測", variant="primary")
                    gr.Markdown("---")
                    view_radio = gr.Radio(
                        ["總耗電量 (Total)", "工業用電 (Industrial)", "民生用電 (Residential)", "全部顯示 (All)"], 
                        label="顯示類別", 
                        value="全部顯示 (All)"
                    )
                with gr.Column(scale=3):
                    plot_output = gr.Plot(label="預測圖表")
            with gr.Row():
                month_filter = gr.Dropdown(
                    choices=["全部 (All)", "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"],
                    value="全部 (All)",
                    label="篩選月份 (Filter Month)"
                )
            with gr.Row():
                table_output = gr.Dataframe(label="2026 每日預測結果 details")
            
            train_btn.click(fn=train_and_update, inputs=[view_radio, epochs_slider, lr_number, seq_slider, month_filter], outputs=[plot_output, table_output])
            view_radio.change(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])
            month_filter.change(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])

        with gr.TabItem("模型評估"):
            gr.Markdown("選擇要評估的對象 (工業/民生/總和)。")
            
            with gr.Row():
                eval_target = gr.Radio(
                    ["總耗電量 (Total)", "工業用電 (Industrial)", "民生用電 (Residential)"],
                    label="評估對象",
                    value="總耗電量 (Total)"
                )
                eval_btn = gr.Button("📊 執行模型評估", variant="secondary")
            
            with gr.Row():
                eval_plot = gr.Plot(label="評估圖表")
                eval_metrics = gr.Markdown(label="指標數據")
                
            eval_btn.click(fn=run_evaluation, inputs=[seq_slider, eval_target], outputs=[eval_plot, eval_metrics])

    demo.load(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])

if __name__ == "__main__":
    demo.launch()
