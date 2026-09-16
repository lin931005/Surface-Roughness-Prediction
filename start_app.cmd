@echo off
setlocal

:: 以腳本所在資料夾作為專案根目錄，避免寫死本機絕對路徑
cd /d "%~dp0"

:: 若尚未建立 venv，先建立並安裝套件
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:: 使用 Windows Terminal 分割視窗，且左右兩側都啟動 venv
wt -d "%~dp0" cmd /k "call venv\Scripts\activate.bat && python -m uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578" ; split-pane -d "%~dp0" cmd /k "call venv\Scripts\activate.bat && python -m streamlit run webapp/streamlit_app.py"