@echo off
chcp 65001 >nul
echo 🚀 準備啟動 CNC MLOps 左右分割戰情室...

:: 1. 安全地切換到目前腳本所在的資料夾 (破解空格報錯的關鍵)
cd /d "%~dp0"

:: 2. 呼叫 Windows Terminal (wt)
:: -d . 代表使用當前目錄
:: -V 代表垂直切割 (Vertical)，也就是左半與右半
wt -d . cmd /k "call venv\Scripts\activate.bat && uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578" ; split-pane -V -d . cmd /k "call venv\Scripts\activate.bat && python -m streamlit run webapp/streamlit_app.py"