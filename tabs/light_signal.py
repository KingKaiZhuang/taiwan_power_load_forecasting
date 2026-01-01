import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from utils import get_forecast_data

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
    # 備轉率 = 容量 / 需求 * 100% (台電是用尖峰負載計算)
    # User requested values to be ~3 digits (matching original magnitude)
    # CONV_FACTOR = 100 / 24
    df_2026['total_kw'] = df_2026['total']
    df_2026['peak_kw'] = df_2026['peak_load'] 
    
    # 公式: 容量 = 供给 - 需求
    df_2026['margin'] = (supply_capacity - df_2026['peak_kw'])
    df_2026['margin_percent'] = (df_2026['margin'] / df_2026['peak_kw']) * 100
    
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
                y=subset['total_kw'],
                name=subset['light_status'].iloc[0],
                marker_color=color_code,
                customdata=subset['margin_percent'],
                hovertemplate='%{x}<br>平均負載: %{y:.1f}<br>備轉率: %{customdata:.2f}%'
            ))

    fig.update_layout(
        title=f'2026 供電燈號模擬 (假設每日供給上限: {supply_capacity} 萬瓩)',
        xaxis_title='日期',
        yaxis_title='每日平均負載 (萬瓩)',
        template='plotly_white',
        barmode='overlay' 
    )
    
    summary_text = "### 📊 2026 燈號統計\n"
    for index, row in status_counts.iterrows():
        summary_text += f"* **{row['狀態']}**: {row['天數']} 天\n"
        
    # 產生「非綠燈」的警戒清單表格
    warning_days = df_2026[df_2026['light_color'] != 'Green'][['ds', 'total_kw', 'margin_percent', 'light_status']].sort_values('margin_percent')
    warning_days['ds'] = warning_days['ds'].dt.strftime('%Y-%m-%d')
    warning_days['total_kw'] = warning_days['total_kw'].round(1)
    warning_days['margin_percent'] = warning_days['margin_percent'].round(2)
    warning_days.columns = ['日期', '平均負載(萬瓩)', '備轉率(%)', '燈號狀態']

    return fig, warning_days, summary_text

def create_light_signal_tab():
    with gr.TabItem("供電燈號儀表板"):
        gr.Markdown("### 🚦 2026 台灣供電燈號模擬器")
        gr.Markdown("輸入假設的「每日最大供電能力」，系統將計算每日備轉容量率並模擬燈號。")
        
        with gr.Row():
            # Revert default to 1000 as values are ~500
            supply_input = gr.Number(value=1000, label="每日最大供電能力 (萬瓩)", precision=0)
            sim_btn = gr.Button("🚦 執行模擬", variant="primary")
        
        with gr.Row():
            light_plot = gr.Plot(label="燈號分佈圖")
        
        with gr.Row():
            with gr.Column(scale=1):
                light_summary = gr.Markdown("### 統計摘要")
            with gr.Column(scale=2):
                light_table = gr.Dataframe(label="警戒天數清單 (非綠燈)")
        
        sim_btn.click(fn=create_light_dashboard, inputs=[supply_input], outputs=[light_plot, light_table, light_summary])

        return supply_input, light_plot, light_table, light_summary
