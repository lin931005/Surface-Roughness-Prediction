# CNC 加工表面粗糙度 AI 預測系統

> 國立虎尾科技大學 (NFU) - 機械與電腦輔助工程系 2026 畢業專題

## 📌 專案概述

本專案建立一套「非接觸式 CNC 表面粗糙度 (Ra) 預測系統」，結合影像與加工參數進行深度學習推論。系統提供訓練、推論、Web API 與 Streamlit 前端顯示，並支援 Docker 一鍵部署。

核心模型架構：
- 影像分支：ResNet-18 提取表面刀痕紋理特徵
- 參數分支：處理主軸轉速與加工條件
- 特徵融合：整合影像與參數輸出 Ra 預測值

## 📁 專案目錄

```text
├── data/                      # 原始資料與訓練清單輸入
│   ├── 5000-0/                # 依轉速與條件分類的影像資料夾
│   ├── 7000-1/
│   └── final_training_manifest.csv  # 影像路徑、參數、Ra 標籤
├── results/                   # 訓練與推論輸出結果
│   ├── best_surface_model.pth
│   ├── current_model.pth      # 當前部署模型（若存在）
│   ├── loss_record.csv
│   ├── predictions.csv
│   └── models/                # 歷史模型版本
├── scripts/                   # 核心訓練與推論腳本
│   ├── dataset_prepare.py
│   ├── train_model.py
│   ├── predict_roughness.py
│   └── test_all_images.py
├── webapp/                    # Web API 與 Streamlit 前端
│   ├── app/
│   │   ├── main.py
│   │   └── auth.py
│   ├── requirements.txt
│   └── streamlit_app.py
├── docker-compose.yml
├── project_root.py
├── requirements-dev.txt
└── README.md
```

## ⚙️ 環境需求

建議使用 Python 3.10+。

主要依賴：
- torch
- torchvision
- pandas
- numpy
- opencv-python
- pillow
- matplotlib
- fastapi
- uvicorn
- python-multipart
- pyjwt
- psutil
- requests
- bcrypt
- python-dotenv

> `webapp/requirements.txt` 已包含 Web API 與 Streamlit 所需套件。

## 🚀 快速安裝

### 1. 建立虛擬環境

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. 安裝套件

```bash
pip install -r webapp/requirements.txt
pip install torch torchvision pandas numpy opencv-python pillow matplotlib requests bcrypt python-dotenv
```

### 3. 生成訓練清單

```bash
python scripts/dataset_prepare.py
```

此腳本會掃描 `data/<speed>-<condition>/pc` 或 `data/<speed>-<condition>/照片`，並建立 `data/final_training_manifest.csv`。

## 🧠 模型訓練

```bash
python scripts/train_model.py
```

可選參數：
- `--output`: 指定模型輸出路徑
- `--log`: 指定訓練日誌輸出路徑

訓練完成後，會儲存最佳模型與訓練損失紀錄至 `results/`。

## 🔍 單張影像推論

```bash
python scripts/predict_roughness.py --image data/5000-0/pc/your_image.jpg --speed 5000 --cond 0
```

若未提供參數，預設會使用內建值進行推論。

## 📊 批次推論

```bash
python scripts/test_all_images.py
```

批次推論結果預設輸出至 `results/results.csv`。

## 🌐 Web 服務

### 啟動 FastAPI

```bash
uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578
```

### 啟動 Streamlit

```bash
streamlit run webapp/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Streamlit 介面支援上傳影像推論、模型切換、訓練日誌與系統監控。

## 🔗 API 端點

- `POST /predict`：影像進行 Ra 預測
- `POST /login`：管理員登入取得 JWT
- `GET /models`：列出可部署模型
- `POST /retrain`：啟動背景訓練
- `GET /train_logs`：列出訓練日誌
- `GET /train_logs/{name}`：讀取訓練日誌內容
- `POST /admin/set_active_model`：切換上線模型
- `POST /report_true`：回報真實 Ra

## 🐳 Docker 部署

```bash
docker-compose up -d --build
```

- `api`: FastAPI 後端
- `streamlit`: Streamlit 前端

> Docker Compose 會掛載 `./data` 與 `./results` 到容器內。

## 🔐 管理員設定

預設帳號：
- username: `admin`
- password: `adminpass`

建議建立 `.env`，並設定：

```bash
ADMIN_PASSWORD=your_secure_password
JWT_SECRET=your_secure_secret
```

## 📌 資料規範

資料夾命名需符合 `<speed>-<condition>`，例如：`5000-0`、`7000-1`。

`dataset_prepare.py` 會讀取這些資料夾下的影像並建立訓練 CSV。

## ✅ 注意事項

- 請勿將原始影像直接提交至 GitHub。
- 若使用 GPU，請確認 PyTorch 與 CUDA 版本相容。
- 後端優先載入 `results/current_model.pth`，若不存在則載入 `results/best_surface_model.pth`。
- Streamlit 預設會透過 `http://127.0.0.1:2578` 呼叫 API。

## 📂 推薦流程

1. 準備並整理 `data/` 影像資料
2. 執行 `python scripts/dataset_prepare.py`
3. 執行 `python scripts/train_model.py`
4. 執行 `python scripts/predict_roughness.py` 測試單張推論
5. 啟動 `uvicorn` 和 `streamlit` 展示系統
