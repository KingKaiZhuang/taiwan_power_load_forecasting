import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from utils import get_forecast_data, generate_forecast_file

def create_dashboard(view_type, selected_month="全部 (All)"):
    """
    建立 Plotly 預測儀表板 (圖表與表格)
    """
    forecast = get_forecast_data()
    
    if forecast is None:
        return go.Figure(), pd.DataFrame()

    # 1. 根據月份篩選資料 (同時影響圖表與表格)
    if selected_month != "全部 (All)":
        month_map = {
            "1月": 1, "2月": 2, "3月": 3, "4月": 4, "5月": 5, "6月": 6,
            "7月": 7, "8月": 8, "9月": 9, "10月": 10, "11月": 11, "12月": 12
        }
        m = month_map.get(selected_month)
        if m:
            forecast = forecast[forecast['ds'].dt.month == m]

    actual_data = forecast[forecast['data_type'] == 'Actual']
    predicted_data = forecast[forecast['data_type'] == 'Predicted']
    
    # 產生 2026 年預測資料表 (用於 UI 表格顯示)
    forecast_2026 = predicted_data[predicted_data['ds'].dt.year == 2026].copy()
    
    fig = go.Figure()
    
    # 輔助函式: 如果資料有斷層 (跨年份)，插入 None 以中斷連線
    # 單位轉換: 百萬度(Energy) -> 萬瓩(Avg Power)
    # 1 百萬度 = 10^6 kWh. / 24h = 41666 kW = 4.1666 萬瓩
    CONV_FACTOR = 100 / 24

    def get_plotting_data(df, col):
        if selected_month == "全部 (All)":
             return df['ds'], df[col] * CONV_FACTOR
        
        x_vals, y_vals = [], []
        if df.empty: return [], []
        
        df = df.sort_values('ds')
        dates = df['ds'].tolist()
        vals = (df[col] * CONV_FACTOR).tolist()
        
        last_date = None
        for d, v in zip(dates, vals):
            # 如果兩點之間超過 2 天，視為斷層
            if last_date is not None and (d - last_date).days > 2:
                x_vals.append(None)
                y_vals.append(None)
            x_vals.append(d)
            y_vals.append(v)
            last_date = d
        return x_vals, y_vals

    def add_traces(col_name, label, color_act, color_pred):
        # 繪製實際數據 (如果你選了特定月份，會顯示點點 Markers 方便觀察)
        x_act, y_act = get_plotting_data(actual_data, col_name)
        fig.add_trace(go.Scatter(
            x=x_act, 
            y=y_act, 
            mode='lines+markers' if selected_month != "全部 (All)" else 'lines', 
            name=f'實際-{label}',
            line=dict(color=color_act, width=1.5),
            marker=dict(size=4)
        ))
        
        # 繪製預測數據
        x_pred, y_pred = get_plotting_data(predicted_data, col_name)
        fig.add_trace(go.Scatter(
            x=x_pred, 
            y=y_pred, 
            mode='lines+markers' if selected_month != "全部 (All)" else 'lines', 
            name=f'預測-{label}',
            line=dict(color=color_pred, width=1.5),
            marker=dict(size=4)
        ))

    # 3. 根據使用者選擇 (工業/民生/總和) 決定畫哪幾條線
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
        title=f'台灣電力負載：實際 vs 預測 ({view_type}) - {selected_month}',
        xaxis_title='日期',
        yaxis_title='負載及用電 (萬瓩)',
        hovermode='x unified',
        template='plotly_white'
    )

    table_cols = ['ds', 'total', 'industrial', 'residential']
    table_data = forecast_2026[table_cols].copy()
    # Apply conversion to table data
    table_data[['total', 'industrial', 'residential']] = table_data[['total', 'industrial', 'residential']] * CONV_FACTOR
    table_data['ds'] = table_data['ds'].dt.strftime('%Y-%m-%d')
    table_data.columns = ['日期', '總負載預測', '工業用電預測', '民生用電預測']
    table_data = table_data.round(2)

    return fig, table_data

def train_and_update(view_type, epochs, lr, seq_length, selected_month):
    """
    處理訓練按鈕事件: 訓練模型 -> 更新介面
    """
    gr.Info(f"訓練開始... (Epochs={epochs}, LR={lr}, Window={seq_length})")
    generate_forecast_file(epochs, lr, seq_length)
    gr.Info("訓練完成！正在更新儀表板...")
    return create_dashboard(view_type, selected_month)

def create_forecast_tab():
    with gr.TabItem("預測儀表板"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 訓練參數設定")
                epochs_slider = gr.Slider(minimum=50, maximum=100000, value=300, step=50, label="訓練次數 (Epochs)")
                lr_number = gr.Number(value=0.005, label="學習率 (Learning Rate)", step=0.001)
                seq_slider = gr.Slider(minimum=7, maximum=730, value=720, step=1, label="回看天數 (Sequence Length)")
                train_btn = gr.Button("訓練模型並預測", variant="primary")
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
        
        # 設定事件監聽
        train_btn.click(fn=train_and_update, inputs=[view_radio, epochs_slider, lr_number, seq_slider, month_filter], outputs=[plot_output, table_output])
        view_radio.change(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])
        month_filter.change(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])

        return view_radio, month_filter, plot_output, table_output, seq_slider
