# CNC 表面粗糙度 AI 預測系統

本專案是一套以深度學習分析 CNC 加工表面影像的系統，透過影像特徵與加工參數預測工件表面粗糙度（Ra）。系統也能自動辨識銑削方式，並提供圖形化操作介面與 API 服務。

## 主要功能

- 上傳工件表面影像，預測表面粗糙度 Ra
- 自動辨識立銑、直銑（周邊銑削）與其他影像
- 可輸入主軸轉速，輔助提升預測結果
- 顯示 Grad-CAM 熱力圖，協助觀察模型判斷的影像區域
- 支援單張檢測、批次驗證與模型管理
- 使用 FastAPI 提供預測 API，使用 Streamlit 提供操作介面

## 專案結構

```text
5000 - BETTER/
├── data/                       # 訓練資料與資料清單
├── results/                    # 訓練完成的模型與預測結果
├── scripts/
│   ├── dataset_prepare.py      # 建立訓練資料清單
│   ├── train_classifier.py     # 訓練銑削方式分類器
│   └── train_model.py          # 訓練表面粗糙度預測模型
├── webapp/
│   ├── app/main.py             # FastAPI 後端
│   └── streamlit_app.py        # Streamlit 使用者介面
├── project_root.py             # 專案路徑工具
├── requirements.txt            # Python 套件需求
└── start_app.cmd               # Windows 快速啟動腳本
```

## 執行方式

請先安裝 Python 3.10 以上版本，並在專案根目錄建立虛擬環境：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

若已完成環境設定，在 Windows 上可直接執行：

```text
start_app.cmd
```

也可以分別啟動後端與前端：

```powershell
python -m uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578
python -m streamlit run webapp/streamlit_app.py
```

啟動後可使用：

- Streamlit 介面：<http://localhost:8501>
- FastAPI 文件：<http://localhost:2578/docs>

## 訓練模型

模型訓練請從 Streamlit 網頁介面操作，不需要另外執行訓練指令。

1. 先啟動系統並登入管理員帳號。
2. 開啟「系統管理與模型控制台」。
3. 在「訓練與終端機」分頁選擇要訓練的模型：
	- 立銑回歸模型
	- 直銑回歸模型
	- 銑法分類器模型
4. 點擊對應的訓練按鈕，並在頁面中查看訓練日誌與進度。

訓練完成的模型與紀錄會自動儲存在 `results/`，之後的預測功能會使用該資料夾中的模型檔案。

## 備註

若要進行模型訓練，建議使用具備 CUDA 的 NVIDIA GPU；僅使用 CPU 也可以執行系統，但訓練時間會較長。
