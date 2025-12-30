# 台灣電力負載預測系統 (Taiwan Power Load Forecasting) 🇹🇼⚡

這是一個基於 PyTorch LSTM 模型開發的電力負載預測應用程式，旨在預測 2026 年台灣每日的電力消費，並細分為「工業用電」與「民生用電」。

## 資料來源 (Data Source)
本專案使用的資料集來自 **台灣電力公司 (Taiwan Power Company)** 之過去電力供需資訊。

## 功能特色 (Features)
*   **雙模型架構**：分別針對「工業用電」與「民生用電」訓練獨立的 LSTM 模型。
*   **特徵工程**：納入「月份」與「星期」特徵，精準捕捉季節性與週間/週末的用電模式。
*   **互動式訓練**：透過 Gradio 介面，使用者可即時調整訓練參數 (Epochs, Learning Rate, Sequence Length)。
*   **模型評估**：提供 In-sample 回測功能，計算 RMSE 與 MAE 指標。
*   **視覺化儀表板**：使用 Plotly 繪製互動式圖表，清楚呈現歷史數據與預測結果。

## 安裝與執行 (Installation & Usage)

1.  **安裝依賴套件**：
    ```bash
    pip install -r requirements.txt
    ```

2.  **啟動應用程式**：
    ```bash
    python app.py
    ```

3.  **使用介面**：
    開啟瀏覽器訪問 `http://127.0.0.1:7860`。
    *   **預測儀表板**：查看預測結果，或調整參數重新訓練。
    *   **模型評估**：檢視模型在過去數據上的擬合表現。

4.  **開發模式 (Development Mode)**：
    若要啟用「熱重載 (Hot Reload)」，即修改程式碼後自動重新整理，請使用以下指令啟動：
    ```bash
    gradio app.py
    ```

## 專案結構 (Project Structure)
*   `app.py`: Gradio 網頁應用程式主程式。
*   `train_model.py`: PyTorch 模型定義、訓練與預測邏輯。
*   `data/`: 存放電力數據 (`power_data.csv`) 與預測結果。
*   `models/`: 存放訓練好的模型權重 (`.pth`) 與標準化參數 (`.pkl`)。
