# CNC 加工表面粗糙度 AI 預測系統 (CNC Surface Roughness AI Prediction)

> 國立虎尾科技大學 (NFU) - 機械與電腦輔助工程系 2026 畢業專題

## 📖 專案簡介

本專案旨在結合深度學習與電腦視覺技術，開發一套非接觸式的 CNC 加工表面粗糙度預測系統。
有別於傳統依賴人工設計特徵的機器學習，本專案採用**雙輸入類神經網路架構 (Dual-Input Neural Network)**：
1. **影像分支 (Image Branch):** 導入遷移學習技術，使用在 ImageNet 預訓練過的 **ResNet-18** 模型，精準萃取金屬表面的刀痕紋理特徵。
2. **參數分支 (Parameter Branch):** 建立深度神經網路 (DNN) 處理加工機台參數 (如主軸轉速、切削條件)。

最後透過特徵匯聚層 (Fusion Layer) 進行綜合推論，輸出高精度的表面粗糙度 (Ra) 預測值。期望能以此系統輔助傳統觸針式粗糙度儀的繁瑣量測，並具備導入智慧製造與自動化檢測的潛力。

## 🛠️ 開發技術與環境

* **核心語言:** Python 3.10+
* **深度學習框架:** PyTorch, Torchvision (支援 CUDA GPU 顯卡加速)
* **影像處理:** OpenCV, PIL, NumPy
* **資料分析與視覺化:** Pandas, Matplotlib

## 📂 專案目錄結構

```text
├── data/                  # 資料集 (已設定 .gitignore，請勿上傳原始圖片)
│   ├── 5000-0/            # 原始 CNC 表面影像資料夾 (依切削條件分類)
│   ├── 7000-1/            
│   └── final_training_manifest.csv # 訓練用影像與參數標籤清單
├── scripts/               # 核心 Python 執行腳本
│   ├── dataset_prepare.py # 訓練清單生成與資料夾整合
│   ├── train_model.py     # 模型訓練 (包含全局微調與資料擴增)
│   ├── predict_roughness.py # 單張影像即時推論與測試
│   └── test_all_images.py # 批次推論與預測結果匯出
├── results/               # 模型產出與測試報告 (由系統自動生成)
│   ├── best_surface_model.pth # 訓練完成的 AI 模型權重檔
│   ├── loss_record.csv    # 訓練過程 Loss 收斂紀錄
│   └── results.csv        # 批次預測結果 (真實 Ra vs 預測 Ra)
├── .gitignore             # Git 忽略清單 (阻擋超大檔案上傳)
├── Git Update.cmd         # 一鍵自動同步 GitHub 腳本
└── README.md              # 專案說明文件