import gradio as gr
import plotly.graph_objects as go
import os
from train_model import load_and_process_data, evaluate_model
from utils import DATA_PATH

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
        # 評估指標 ({target_type})
        ## RMSE: {m['rmse']:.2f}
        ## MAE: {m['mae']:.2f}
        """
        
        return fig, metrics_text
        
    return go.Figure(), "錯誤：找不到數據。"

def create_evaluation_tab(seq_slider):
    with gr.TabItem("模型評估"):
        gr.Markdown("選擇要評估的對象 (工業/民生/總和)。")
        
        with gr.Row():
            eval_target = gr.Radio(
                ["總耗電量 (Total)", "工業用電 (Industrial)", "民生用電 (Residential)"],
                label="評估對象",
                value="總耗電量 (Total)"
            )
            eval_btn = gr.Button("執行模型評估", variant="secondary")
        
        with gr.Row():
            eval_plot = gr.Plot(label="評估圖表")
            eval_metrics = gr.Markdown(label="指標數據")
            
        eval_btn.click(fn=run_evaluation, inputs=[seq_slider, eval_target], outputs=[eval_plot, eval_metrics])
