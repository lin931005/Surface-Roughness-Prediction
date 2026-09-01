# CNC 表面粗糙度 AI 預測系統

> **國立虎尾科技大學 (NFU) - 機械與電腦輔助工程系**  
> **2026 畢業專題**

---

## 📖 目錄

- [專案概述](#專案概述)
- [系統架構](#系統架構)
- [環境要求](#環境要求)
- [安裝步驟](#安裝步驟)
- [快速開始](#快速開始)
- [專案結構](#專案結構)
- [API 文檔](#api-文檔)

- [故障排除](#故障排除)
- [貢獻指南](#貢獻指南)

---

## 專案概述

本系統是一套**非接觸式 CNC 表面粗糙度 (Ra) AI 預測系統**，透過深度學習結合影像與加工參數進行表面粗糙度預測。

### 核心特性

✅ **雙分支神經網絡**
- 影像分支：ResNet-18 萃取表面刀痕紋理特徵
- 參數分支：整合主軸轉速、加工條件等參數
- 特徵融合：輸出精確的 Ra (µm) 預測值

✅ **完整的生命週期管理**
- 資料準備與驗證
- 模型訓練與評估
- 線上推論服務
- 模型版本控制

✅ **多種部署方式**
- 命令行工具 (CLI)
- Web API (FastAPI)
- 可視化界面 (Streamlit)

✅ **管理員功能**
- JWT 身份驗證
- 模型動態切換
- 訓練日誌查看
- 真實值反饋機制

---

## 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                    用戶界面層                             │
│  ┌──────────────────┐        ┌──────────────────┐       │
│  │   Streamlit App  │        │   FastAPI 文檔   │       │
│  │   (8501 端口)    │        │   (2578 端口)    │       │
│  └────────┬─────────┘        └────────┬─────────┘       │
└───────────┼──────────────────────────┼─────────────────┘
            │                          │
┌───────────┼──────────────────────────┼─────────────────┐
│           │       API 層 (FastAPI)   │                 │
│           └──────────────┬───────────┘                 │
│                          │                             │
│        ┌─────────────────┼─────────────────┐          │
│        │  認證  │  推論  │  訓練  │  管理  │          │
└────────┼───────┼────────┼───────┼────────┼──────────┘
         │       │        │       │        │
┌────────┼───────┼────────┼───────┼────────┼──────────┐
│        │    模型層 (PyTorch)     │       │          │
│        └────────┬────────────────┘       │          │
│                 │                        │          │
│     ┌───────────┼────────────┐           │          │
│     │   ResNet-18   │ 參數模塊 │           │          │
│     └───────────┼────────────┘           │          │
└────────────────┼──────────────────────────────────────┘
                 │
┌────────────────┼──────────────────────────────────────┐
│                │    數據層                            │
│        ┌───────┼────────┐                             │
│        │  訓練數據  │  模型存儲  │                      │
│        └───────┼────────┘                             │
│         data/  │  results/                            │
└────────────────┼──────────────────────────────────────┘
```

---

## 環境要求

| 項目 | 版本/配置 |
|------|---------|
| **Python** | 3.10+ |
| **PyTorch** | 2.0+ (建議 GPU 版本) |
| **CUDA** | 11.8+ (選用，推薦用於訓練) |
| **系統** | Windows/Linux/macOS |
| **記憶體** | 最少 8GB (推薦 16GB) |
| **磁碟** | 訓練數據至少 10GB 可用空間 |

### 主要依賴

```
核心框架：torch, torchvision
資料處理：pandas, numpy, opencv-python, pillow
Web 服務：fastapi, uvicorn, streamlit
安全性：pyjwt, bcrypt
其他：requests, python-dotenv, psutil
```

---

## 安裝步驟

### 1️⃣ 克隆/下載項目

```bash
# 假設已下載到本地
cd "5000 - BETTER"
```

### 2️⃣ 建立虛擬環境

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ 安裝依賴

```bash
# 安裝 Web 服務依賴
pip install -r webapp/requirements.txt

# 安裝開發依賴 (可選)
pip install -r requirements-dev.txt

# 安裝 PyTorch (選擇適合的版本)
# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU 版本 (CUDA 11.8)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4️⃣ 配置環境變數 (可選)

創建 `.env` 文件於項目根目錄：

```env
# API 設定
API_HOST=0.0.0.0
API_PORT=2578

# 安全設定
ADMIN_PASSWORD=your_secure_password_here
JWT_SECRET=your_secure_secret_key_here

# 模型設定
DEFAULT_MODEL=results/best_classifier.pth
GPU_ENABLED=true
```

---

## 快速開始

### 🎯 方案 1：完整流程 (推薦初學者)

```bash
# 1. 準備資料
#    確保 data/ 下有按 <speed>-<condition> 命名的資料夾
#    例如: 5000-0, 5000-1, 7000-0, 等等
#    每個資料夾內應包含 pc/ 或 照片/ 子目錄和影像檔案

# 2. 生成訓練清單
python scripts/dataset_prepare.py
# ✓ 生成 data/final_training_manifest.csv

# 3. 訓練模型
python scripts/train_model.py
# ✓ 訓練完成後儲存至 results/

# 4. 測試單張推論
python scripts/predict_roughness.py --image data/5000-0/pc/sample.jpg --speed 5000 --cond 0

# 5. 批次推論
python scripts/test_all_images.py
# ✓ 結果儲存至 results/predictions.csv
```

### 🌐 方案 2：Web 服務 (開發環境)

**終端 1 - 啟動 API 服務:**
```bash
uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578 --reload
```

**終端 2 - 啟動 Web 界面:**
```bash
streamlit run webapp/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

然後訪問：
- 🔗 Streamlit 界面：http://localhost:8501
- 📚 API 文檔：http://localhost:2578/docs
- 🔄 API 替代文檔：http://localhost:2578/redoc



## 專案結構

```
5000 - BETTER/
│
├── 📄 project_root.py              # 項目路徑管理工具
├── 📄 requirements-dev.txt         # 開發依賴
├── 📄 start_app.cmd               # Windows 快速啟動腳本
│
├── 📁 data/                       # 訓練數據
│   ├── final_training_manifest.csv
│   ├── build_dataset_csv.py
│   ├── End_Milling/              # 端銑數據
│   │   ├── 5000/
│   │   ├── 7000/
│   │   └── ...
│   ├── Peripheral_Milling/       # 周邊銑削數據
│   │   ├── 5000/
│   │   ├── 7000/
│   │   └── 9000/
│   └── Other/
│
├── 📁 results/                    # 訓練和推論結果
│   ├── best_classifier.pth        # 最佳分類模型
│   ├── best_model_End_Milling.pth
│   ├── best_model_Peripheral_Milling.pth
│   ├── loss_record_*.csv          # 訓練損失記錄
│   ├── predictions.csv            # 推論結果
│   ├── models/                    # 歷史模型檔案庫
│   └── train_logs/                # 訓練日誌
│
├── 📁 scripts/                    # 核心腳本
│   ├── dataset_prepare.py         # 數據準備和驗證
│   ├── train_classifier.py        # 分類模型訓練
│   ├── train_model.py             # 粗糙度模型訓練
│   └── predict_roughness.py       # 單張推論
│
└── 📁 webapp/                     # Web 服務
    ├── streamlit_app.py           # Streamlit 前端
    ├── requirements.txt           # Web 依賴
    └── app/
        ├── main.py                # FastAPI 主應用
        └── auth.py                # 身份驗證模塊
```

---

## API 文檔

### 認證端點

#### `POST /login`
登入並獲取 JWT Token

**請求：**
```json
{
  "username": "admin",
  "password": "adminpass"
}
```

**回應：**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

---

### 推論端點

#### `POST /predict`
對上傳的影像進行粗糙度預測

**參數：**
- `file` (multipart): 影像檔案 (JPG/PNG)
- `speed` (int, 可選): 主軸轉速 (RPM)
- `condition` (int, 可選): 加工條件代碼

**cURL 示例：**
```bash
curl -X POST "http://localhost:2578/predict" \
  -F "file=@image.jpg" \
  -F "speed=5000" \
  -F "condition=0"
```

**回應：**
```json
{
  "predicted_ra": 1.25,
  "confidence": 0.89,
  "model_version": "best_classifier.pth",
  "processing_time_ms": 145
}
```

---

### 模型管理端點

#### `GET /models`
列出所有可用模型

**回應：**
```json
{
  "available_models": [
    "best_classifier.pth",
    "best_model_End_Milling.pth",
    "best_model_Peripheral_Milling.pth"
  ],
  "active_model": "best_classifier.pth"
}
```

#### `POST /admin/set_active_model`
切換活動模型 (需要認證)

**請求：**
```json
{
  "model_name": "best_model_End_Milling.pth"
}
```

**Header：**
```
Authorization: Bearer {token}
```

---

### 訓練端點

#### `POST /retrain`
啟動背景訓練任務 (需要認證)

**請求：**
```json
{
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

#### `GET /train_logs`
列出所有訓練日誌

#### `GET /train_logs/{log_name}`
查看特定訓練日誌內容

---

### 反饋端點

#### `POST /report_true`
回報真實的粗糙度值用於模型改進

**請求：**
```json
{
  "prediction_id": "abc123",
  "true_ra": 1.35,
  "feedback": "optional feedback"
}
```

---



## 故障排除

### 問題 1：模型載入失敗

**症狀：** `FileNotFoundError: No such file or directory: 'results/best_classifier.pth'`

**解決方案：**
```bash
# 1. 確認模型檔案存在
dir results/

# 2. 若不存在，進行訓練
python scripts/train_model.py

# 3. 檢查路徑配置
python project_root.py  # 應該輸出正確的根目錄
```

---

### 問題 2：PyTorch CUDA 不可用

**症狀：** `RuntimeError: CUDA out of memory` 或 CUDA 設備未找到

**解決方案：**
```bash
# 1. 檢查 PyTorch 安裝
python -c "import torch; print(torch.cuda.is_available())"

# 2. 重新安裝 CPU 版本
pip uninstall torch torchvision -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 修改 train_model.py 中的 device 設定
# device = torch.device("cpu")  # 強制使用 CPU
```

---

### 問題 3：Streamlit 無法連接 API

**症狀：** `Connection refused` 或 `Unable to reach server`

**解決方案：**
```bash
# 1. 確認 API 正在運行
curl http://localhost:2578/docs

# 2. 檢查防火牆設定
# Windows: 開啟 2578 和 8501 端口
# Linux: sudo ufw allow 2578/tcp

# 3. 修改 Streamlit 配置中的 API 地址
# streamlit_app.py 中更改: api_url = "http://localhost:2578"
```

---

### 問題 4：資料集準備失敗

**症狀：** `ValueError: No images found in data directory`

**解決方案：**
```bash
# 1. 檢查資料夾結構
#    應該是: data/5000-0/pc/*.jpg 或 data/5000-0/照片/*.jpg

# 2. 驗證影像格式
#    支援格式: jpg, jpeg, png, bmp, tiff

# 3. 手動指定路徑
python scripts/dataset_prepare.py --data-dir ./data --output ./data/final_training_manifest.csv
```

---

### 問題 5：GPU 記憶體溢出

**症狀：** `RuntimeError: CUDA out of memory`

**解決方案：**
```python
# 在 train_model.py 中修改批次大小:
batch_size = 8  # 從 32 減少到 8

# 或清空 GPU 快取
import torch
torch.cuda.empty_cache()
```

---

## 訓練建議

### 資料準備

1. **影像要求**
   - 解析度：建議 640×480 或以上
   - 格式：JPG/PNG
   - 大小：平均 1-5MB

2. **資料夾組織**
   ```
   data/
   ├── 5000-0/pc/*.jpg      (5000 RPM, 條件 0)
   ├── 5000-1/pc/*.jpg      (5000 RPM, 條件 1)
   ├── 7000-0/pc/*.jpg      (7000 RPM, 條件 0)
   └── ...
   ```

3. **最小資料量**
   - 每個配置至少 50 張影像
   - 總計建議 1000+ 張影像用於高精度

### 訓練參數調整

```bash
# 編輯 scripts/train_model.py 中的參數
EPOCHS = 100              # 訓練輪數
BATCH_SIZE = 32           # 批次大小
LEARNING_RATE = 0.001     # 學習率
VALIDATION_SPLIT = 0.2    # 驗證集比例
```

### 監控訓練進度

```bash
# 在 results/train_logs 中查看即時日誌
tail -f results/train_logs/latest.log

# 繪製損失曲線
python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results/loss_record_Classifier.csv')
plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
"
```

---

## 帳號設定

### 預設帳號

| 項目 | 值 |
|------|-----|
| **用戶名** | `admin` |
| **密碼** | `adminpass` |

### 更改密碼

編輯 `.env` 文件或環境變數：
```env
ADMIN_PASSWORD=your_new_secure_password
JWT_SECRET=your_new_secret_key_here
```

---

## 常見問題 (FAQ)

**Q: 我應該用 GPU 還是 CPU？**  
A: 如果有 NVIDIA GPU (建議 6GB+ VRAM)，使用 GPU 會快 10-50 倍。CPU 訓練會很緩慢但可行。

**Q: 訓練需要多長時間？**  
A: 取決於數據量和硬件。約 1000 張影像，GPU 約 2-4 小時，CPU 約 12-24 小時。

**Q: 如何部署到生產環境？**  
A: 使用 WSGI 服務器如 Gunicorn 或 uWSGI 部署，或使用 Nginx 反向代理進行端口轉發。

**Q: 模型精度不夠高怎麼辦？**  
A: 增加訓練數據、調整超參數、或使用更複雜的模型架構。

**Q: 支援模型 A/B 測試嗎？**  
A: 支援！使用 `GET /models` 查看所有模型，用 `POST /admin/set_active_model` 切換。

---

## 貢獻指南

我們歡迎改進！請遵循以下步驟：

1. 創建特性分支：`git checkout -b feature/amazing-feature`
2. 提交更改：`git commit -m 'Add amazing feature'`
3. 推送分支：`git push origin feature/amazing-feature`
4. 開啟 Pull Request

---

## 授權

此專案為國立虎尾科技大學 2026 年畢業專題。  
詳見 LICENSE 檔案。

---

## 聯繫與支援

- 📧 Email: your.email@nfu.edu.tw
- 💬 Issues: 在 GitHub 提交 Issue
- 📚 Documentation: 見上述各章節

---

**最後更新：2026 年 9 月**  
**版本：1.0.0**
