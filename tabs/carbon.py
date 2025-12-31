import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from utils import get_forecast_data

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
    # 4. 詳細資料表
    # Revert conversion to match user expectation
    # CONV_FACTOR = 100 / 24
    table_df = df[['ds', 'total', 'carbon_emissions_tons']].copy()
    # table_df['total'] = table_df['total'] * CONV_FACTOR 
    
    table_df['ds'] = table_df['ds'].dt.strftime('%Y-%m-%d')
    table_df['total'] = table_df['total'].round(2)
    table_df['carbon_emissions_tons'] = table_df['carbon_emissions_tons'].round(2)
    table_df.columns = ['日期', '預測負載(萬瓩)', '預估碳排(公噸)']
    
    return kpi_md, fig_daily, fig_monthly, table_df

def create_carbon_tab():
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

        return carbon_coef_input, carbon_kpi, carbon_plot, carbon_monthly_plot, carbon_table
