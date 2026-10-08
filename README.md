# CNC 表面粗糙度 AI 預測系統

本專案是一套以深度學習分析 CNC 加工表面影像的系統，透過影像特徵與加工參數預測工件表面粗糙度（Ra）。系統也能自動辨識銑削方式，並提供圖形化操作介面與 API 服務。

## 主要功能

- 上傳工件表面影像，預測表面粗糙度 Ra
- 自動辨識立銑、直銑（周邊銑削）與其他影像
- 可輸入主軸轉速，輔助提升預測結果
- 顯示 Grad-CAM 熱力圖，協助觀察模型判斷的影像區域
- 支援單張檢測、批次驗證與模型管理（含版本切換與回復）
- 使用 FastAPI 提供預測 API，使用 Streamlit 提供操作介面

## 專案結構

```text
5000 - BETTER/
├── data/                       # 訓練資料與資料清單
├── results/                    # 訓練完成的模型與預測結果
│   └── archive/                # 舊版本模型備份（可從管理介面切換回來）
├── scripts/
│   ├── dataset_prepare.py      # 建立訓練資料清單，並依刀切分訓練/驗證集
│   ├── train_classifier.py     # 訓練銑削方式分類器
│   ├── train_model.py          # 訓練表面粗糙度預測模型
│   └── train_traditional_ml.py # 訓練傳統視覺 (OpenCV + Random Forest) 對照模型
├── webapp/
│   ├── app/main.py             # FastAPI 後端
│   └── streamlit_app.py        # Streamlit 使用者介面
├── model_versions.py           # 模型版本備份與切換工具
├── preprocessing.py            # 影像前處理（訓練、驗證與線上推論共用）
├── project_root.py             # 專案路徑工具
├── .env.example                # 密碼與 API 金鑰設定範本
├── requirements.txt            # Python 套件需求
└── start_app.cmd               # Windows 快速啟動腳本
```

## 執行方式

以下步驟適用於已將專案複製到新電腦的 Windows 環境。重點是不要寫死本機路徑，請在專案根目錄中執行。

### 1. 安裝 Python

請先安裝 Python 3.10 以上版本，並確認已加入 PATH。

### 2. 建立虛擬環境並安裝套件

在專案根目錄開啟 PowerShell 後執行：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

有 NVIDIA GPU 時，先安裝 CUDA 版的 PyTorch（沒有 GPU 可跳過這一步）。目前開發環境使用 CUDA 13.2，若顯示卡驅動支援的 CUDA 版本不同，請到 PyTorch 官網取得對應的安裝指令：

```powershell
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
```

接著安裝其他套件：

```powershell
pip install -r requirements.txt
```

### 3. 複製資料與模型

`data/` 底下的照片和 `results/` 裡的模型檔太大，沒有放進 git。換到新電腦時，請從原本的電腦把這兩個資料夾複製過來。

### 4. 設定密碼與 API 金鑰

密碼與 API 金鑰不寫在程式碼裡，而是放在專案根目錄的 `.env`（不會進 git）。第一次使用時，複製範本後填入實際的值：

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # 產生一組 API 金鑰
notepad .env
```

- `CNC_API_KEY`：前端呼叫後端用的金鑰，貼上剛才產生的值即可
- `CNC_ADMIN_PASSWORD`：管理員登入密碼
- `CNC_USER_PASSWORD`：現場作業員登入密碼

任何一個欄位空白，系統都會拒絕啟動。修改 `.env` 後需要重新啟動系統才會生效。

### 5. 啟動系統

若已完成環境設定，可直接在專案根目錄執行：

```text
start_app.cmd
```

這個腳本會自動以目前專案資料夾為根目錄啟動後端與前端，適合在不同電腦上重複使用。

若要手動啟動，可使用：

```powershell
python -m uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578
python -m streamlit run webapp/streamlit_app.py
```

啟動後可使用：

- Streamlit 介面：<http://localhost:8501>
- FastAPI 文件：<http://localhost:2578/docs>（所有 API 都需要在 `X-API-Key` 標頭帶上 `.env` 裡的 `CNC_API_KEY`，可在文件頁右上角的 Authorize 輸入）

## 訓練模型

模型訓練請從 Streamlit 網頁介面操作，不需要另外執行訓練指令。

1. 先啟動系統並登入管理員帳號。
2. 開啟「系統管理與模型控制台」。
3. 在「訓練與終端機」分頁選擇要訓練的模型：
	- 立銑回歸模型
	- 直銑回歸模型
	- 銑法分類器模型
	- 傳統視覺 (Random Forest) 模型
4. 點擊對應的訓練按鈕，並在頁面中查看訓練日誌與進度。

訓練完成的模型與紀錄會自動儲存在 `results/`，之後的預測功能會使用該資料夾中的模型檔案。

### 訓練集與驗證集的切分

每次訓練前，`scripts/dataset_prepare.py` 會重新產生資料清單，並以「刀」為單位切出約 20% 當驗證集，結果記錄在清單的 `split` 欄位：

- 同一刀（同一個條件資料夾）的照片一定在同一邊，驗證集的刀在訓練時完全看不到，驗證成績才能反映模型對沒看過的刀的表現。
- Ra 實測值相同的條件，以及內容完全相同的照片，也會分在同一組。若發現同一張照片放在 Ra 不同的資料夾，會在日誌中顯示警告。
- 四個模型使用相同的切分，訓練時會把保留的驗證刀和驗證成績記錄在模型旁的 `.meta.json`。批量驗證頁會依此標出「訓練時沒看過的刀」，並只用這些影像計算 KPI。

### 影像前處理

所有照片都在同一個倍率（4.0X）下拍攝，原始解析度下固定的像素數就代表固定的實際尺寸。因此深度學習模型一律從原始照片切出 672×672 像素的正方形方塊，再縮成 224×224 輸入模型，不同照片、不同裁切尺寸下，刀痕的放大倍率都相同：

- 訓練：每次隨機取一個方塊，並加入位置、±10% 大小、±15° 旋轉、翻轉與亮度對比的變化。
- 驗證與銑法分類：在照片上均勻取 3×3 個方塊，預測結果取平均。
- Ra 預測：隨機取 32 個方塊，去掉最高與最低各 4 個後取平均；取樣位置由照片內容決定，同一張照片每次的結果都相同。

這部分的設定集中在 `preprocessing.py`，訓練與線上推論共用同一份程式。前處理版本會記錄在模型的 `.meta.json`，用舊版前處理訓練的模型會在介面上顯示警告，需要重新訓練。

### 模型版本切換

重新訓練前，系統會先把當時的線上版本備份到 `results/archive/`。在「模型熱切換」分頁可以選擇任一備份版本設為上線模型，切換前同樣會先備份目前的版本，隨時可以換回來。

## 備註

若要進行模型訓練，建議使用具備 CUDA 的 NVIDIA GPU；僅使用 CPU 也可以執行系統，但訓練時間會較長。
