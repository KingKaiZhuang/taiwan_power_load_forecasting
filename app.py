import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import os
from train_model import load_and_process_data, train_model, make_prediction, evaluate_model

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
    def get_plotting_data(df, col):
        if selected_month == "全部 (All)":
             return df['ds'], df[col]
        
        x_vals, y_vals = [], []
        if df.empty: return [], []
        
        df = df.sort_values('ds')
        dates = df['ds'].tolist()
        vals = df[col].tolist()
        
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
        yaxis_title='耗電量 (百萬度)',
        hovermode='x unified',
        template='plotly_white'
    )

    table_cols = ['ds', 'total', 'industrial', 'residential']
    table_data = forecast_2026[table_cols].copy()
    table_data['ds'] = table_data['ds'].dt.strftime('%Y-%m-%d')
    table_data.columns = ['日期', '總耗電量预测', '工業用電預測', '民生用電預測']
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

def run_evaluation(seq_length, target_type):
    """
    處理模型評估事件: 讀取模型 -> 執行回測 -> 顯示結果
    """
    gr.Info(f"正在執行模型評估 (類別: {target_type})...")
    if os.path.exists(DATA_PATH):
        df = load_and_process_data(DATA_PATH)
        eval_result = evaluate_model(df, test_days=90, seq_length=int(seq_length))
        
        if eval_result[0] is None:
             return go.Figure(), "錯誤：數據不足或模型未訓練。"
             
        eval_df, metrics, train_df = eval_result
        
        fig = go.Figure()
        plot_train_df = train_df.iloc[-180:] # 只畫最後 180 天的訓練資料以免圖太擠
        
        # 決定要評估哪個欄位
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

def create_light_dashboard(supply_capacity):
    """
    建立供電燈號儀表板
    """
    forecast = get_forecast_data()
    if forecast is None:
        return go.Figure(), pd.DataFrame(), "無數據"

    # 只看 2026 年預測值
    df_2026 = forecast[(forecast['data_type'] == 'Predicted') & (forecast['ds'].dt.year == 2026)].copy()
    
    if df_2026.empty:
        return go.Figure(), pd.DataFrame(), "尚無 2026 預測數據"

    # 計算備轉容量 (Reserve Margin)
    # 公式: 容量 = 供给 - 需求
    # 備轉率 = 容量 / 需求 * 100% (這裡用需求作為分母是簡化估算，台電是用尖峰負載計算)
    df_2026['margin'] = (supply_capacity - df_2026['total'])
    df_2026['margin_percent'] = (df_2026['margin'] / df_2026['total']) * 100
    
    # 定義燈號邏輯
    def get_light(row):
        mp = row['margin_percent']
        m_val = row['margin']
        
        if m_val < 0: return 'Red', '🔴 限電警戒 (不足)'
        if mp < 6: return 'Orange', '🟠 供電警戒 (<6%)'
        if mp < 10: return 'Yellow', '🟡 供電吃緊 (6-10%)'
        return 'Green', '🟢 供電充裕 (>10%)'

    df_2026[['light_color', 'light_status']] = df_2026.apply(get_light, axis=1, result_type='expand')
    
    # 統計各種燈號的天數
    status_counts = df_2026['light_status'].value_counts().reset_index()
    status_counts.columns = ['狀態', '天數']
    
    # 建立視覺化圖表
    colors = {'Red': '#FF0000', 'Orange': '#FFA500', 'Yellow': '#FFD700', 'Green': '#008000'}
    
    fig = go.Figure()
    
    for status, color_code in [('Red', '#FF0000'), ('Orange', '#FFA500'), ('Yellow', '#FFD700'), ('Green', '#008000')]:
        mask = df_2026['light_color'] == status
        if mask.any():
            subset = df_2026[mask]
            fig.add_trace(go.Bar(
                x=subset['ds'],
                y=subset['total'],
                name=subset['light_status'].iloc[0],
                marker_color=color_code,
                customdata=subset['margin_percent'],
                hovertemplate='%{x}<br>耗電: %{y:.1f}<br>備轉率: %{customdata:.2f}%'
            ))

    fig.update_layout(
        title=f'2026 供電燈號模擬 (假設每日供給上限: {supply_capacity} 百萬度)',
        xaxis_title='日期',
        yaxis_title='每日耗電量 (百萬度)',
        template='plotly_white',
        barmode='overlay' 
    )
    
    summary_text = "### 📊 2026 燈號統計\n"
    for index, row in status_counts.iterrows():
        summary_text += f"* **{row['狀態']}**: {row['天數']} 天\n"
        
    # 產生「非綠燈」的警戒清單表格
    warning_days = df_2026[df_2026['light_color'] != 'Green'][['ds', 'total', 'margin_percent', 'light_status']].sort_values('margin_percent')
    warning_days['ds'] = warning_days['ds'].dt.strftime('%Y-%m-%d')
    warning_days['total'] = warning_days['total'].round(1)
    warning_days['margin_percent'] = warning_days['margin_percent'].round(2)
    warning_days.columns = ['日期', '耗電量', '備轉率(%)', '燈號狀態']

    return fig, warning_days, summary_text

def create_carbon_dashboard(coef):
    """
    建立碳排放估算儀表板
    coef: 碳排係數 (kg CO2e/度)
    """
    forecast = get_forecast_data()
    if forecast is None:
        return "無數據", go.Figure(), go.Figure(), pd.DataFrame()

    # 鎖定 2026 預測數據
    df = forecast[(forecast['data_type'] == 'Predicted') & (forecast['ds'].dt.year == 2026)].copy()
    
    if df.empty:
        return "尚無 2026 預測數據", go.Figure(), go.Figure(), pd.DataFrame()

    # 計算碳排放 (單位: 公噸)
    # 耗電量單位: 百萬度 (10^6 kWh)
    # 碳排 = 耗電量 * 10^6 * coef (kg) / 1000 (kg->ton)
    #      = 耗電量 * coef * 1000
    df['carbon_emissions_tons'] = df['total'] * coef * 1000
    
    # 1. 關鍵指標 (KPI)
    total_emission = df['carbon_emissions_tons'].sum()
    avg_emission = df['carbon_emissions_tons'].mean()
    
    kpi_md = f"""
    ### 🌍 2026 碳排放預估摘要
    *   **年度總碳排量**: {total_emission:,.0f} 公噸 (Tons)
    *   **平均每日碳排**: {avg_emission:,.0f} 公噸 (Tons)
    *   **計算基準係數**: {coef} kg CO2e/度
    """
    
    # 2. 每日碳排趨勢圖
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=df['ds'],
        y=df['carbon_emissions_tons'],
        mode='lines',
        name='每日碳排',
        line=dict(color='#8B4513', width=1.5),
        fill='tozeroy'
    ))
    fig_daily.update_layout(
        title='2026 每日碳排放趨勢預測',
        xaxis_title='日期',
        yaxis_title='碳排放量 (公噸)',
        template='plotly_white'
    )
    
    # 3. 月度統計圖
    df['month'] = df['ds'].dt.month
    monthly_data = df.groupby('month')['carbon_emissions_tons'].sum().reset_index()
    monthly_data['month_str'] = monthly_data['month'].apply(lambda x: f"{x}月")
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_data['month_str'],
        y=monthly_data['carbon_emissions_tons'],
        marker_color='#2E8B57',
        text=monthly_data['carbon_emissions_tons'].apply(lambda x: f'{x:,.0f}'),
        textposition='auto'
    ))
    fig_monthly.update_layout(
        title='2026 月度總碳排放量',
        xaxis_title='月份',
        yaxis_title='總碳排放量 (公噸)',
        template='plotly_white'
    )
    
    # 4. 詳細資料表
    table_df = df[['ds', 'total', 'carbon_emissions_tons']].copy()
    table_df['ds'] = table_df['ds'].dt.strftime('%Y-%m-%d')
    table_df['total'] = table_df['total'].round(2)
    table_df['carbon_emissions_tons'] = table_df['carbon_emissions_tons'].round(2)
    table_df.columns = ['日期', '預測耗電(百萬度)', '預估碳排(公噸)']
    
    return kpi_md, fig_daily, fig_monthly, table_df

# 建立 Gradio 介面
with gr.Blocks(title="2026 電力預測") as demo:
    gr.Markdown("# 2026 台灣電力負載預測 🇹🇼⚡")
    
    with gr.Tabs():
        # 分頁 1: 預測儀表板
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
            
            # 設定事件監聽
            train_btn.click(fn=train_and_update, inputs=[view_radio, epochs_slider, lr_number, seq_slider, month_filter], outputs=[plot_output, table_output])
            view_radio.change(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])
            month_filter.change(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])

        # 分頁 2: 模型評估
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
        
        # 分頁 3: 供電燈號儀表板
        with gr.TabItem("供電燈號儀表板"):
            gr.Markdown("### 🚦 2026 台灣供電燈號模擬器")
            gr.Markdown("輸入假設的「每日最大供電能力」，系統將計算每日備轉容量率並模擬燈號。")
            
            with gr.Row():
                # 預設供給值設為 1000 (經驗值，僅供參考)
                supply_input = gr.Number(value=1000, label="每日最大供電能力 (百萬度)", precision=0)
                sim_btn = gr.Button("🚦 執行模擬", variant="primary")
            
            with gr.Row():
                light_plot = gr.Plot(label="燈號分佈圖")
            
            with gr.Row():
                with gr.Column(scale=1):
                    light_summary = gr.Markdown("### 統計摘要")
                with gr.Column(scale=2):
                    light_table = gr.Dataframe(label="警戒天數清單 (非綠燈)")
            
            sim_btn.click(fn=create_light_dashboard, inputs=[supply_input], outputs=[light_plot, light_table, light_summary])

        # 分頁 4: 碳排放估算
        with gr.TabItem("碳排放估算"):
            gr.Markdown("### 🌍 2026 碳排放估算")
            gr.Markdown("輸入碳排放係數，估算 2026 年全台電力消費產生的總碳排放量。")
            
            with gr.Row():
                carbon_coef_input = gr.Number(value=0.495, label="碳排放係數 (kg CO2e/度)", step=0.001)
                carbon_btn = gr.Button("🌍 開始估算", variant="primary")
            
            carbon_kpi = gr.Markdown()
            
            with gr.Row():
                carbon_plot = gr.Plot(label="每日趨勢")
            
            with gr.Row():
                carbon_monthly_plot = gr.Plot(label="月度統計")
            
            with gr.Row():
                carbon_table = gr.Dataframe(label="詳細預測數據")
            
            carbon_btn.click(fn=create_carbon_dashboard, inputs=[carbon_coef_input], outputs=[carbon_kpi, carbon_plot, carbon_monthly_plot, carbon_table])

    # 啟動時的初始化載入
    demo.load(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])
    demo.load(fn=create_light_dashboard, inputs=[supply_input], outputs=[light_plot, light_table, light_summary])
    demo.load(fn=create_carbon_dashboard, inputs=[carbon_coef_input], outputs=[carbon_kpi, carbon_plot, carbon_monthly_plot, carbon_table])

if __name__ == "__main__":
    demo.launch()
