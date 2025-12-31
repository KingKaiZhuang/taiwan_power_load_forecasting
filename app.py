import gradio as gr
from tabs.forecast import create_forecast_tab, create_dashboard
from tabs.evaluation import create_evaluation_tab
from tabs.light_signal import create_light_signal_tab, create_light_dashboard
from tabs.carbon import create_carbon_tab, create_carbon_dashboard

# 建立 Gradio 介面
with gr.Blocks(title="基於機器學習之台灣電力負載預測系統") as demo:
    gr.Markdown("# 基於機器學習之台灣電力負載預測系統")
    
    with gr.Tabs():
        # 分頁 1: 預測儀表板
        view_radio, month_filter, plot_output, table_output, seq_slider = create_forecast_tab()

        # 分頁 2: 模型評估
        create_evaluation_tab(seq_slider)
        
        # 分頁 3: 供電燈號儀表板
        supply_input, light_plot, light_table, light_summary = create_light_signal_tab()
        
        # 分頁 4: 碳排放估算
        carbon_coef_input, carbon_kpi, carbon_plot, carbon_monthly_plot, carbon_table = create_carbon_tab()

    # 啟動時的初始化載入
    demo.load(fn=create_dashboard, inputs=[view_radio, month_filter], outputs=[plot_output, table_output])
    demo.load(fn=create_light_dashboard, inputs=[supply_input], outputs=[light_plot, light_table, light_summary])
    demo.load(fn=create_carbon_dashboard, inputs=[carbon_coef_input], outputs=[carbon_kpi, carbon_plot, carbon_monthly_plot, carbon_table])

if __name__ == "__main__":
    demo.launch()
