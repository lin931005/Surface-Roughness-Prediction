@echo off
cd /d "C:\Users\user\Desktop\5000 - BETTER"

:: 使用 Windows Terminal 分割視窗，且左右兩側都啟動 venv
wt -d . cmd /k "venv\Scripts\activate && python -m uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578" ; split-pane -d . cmd /k "venv\Scripts\activate && python -m streamlit run webapp/streamlit_app.py"