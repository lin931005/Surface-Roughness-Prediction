@echo off
:: 切換編碼為 UTF-8，避免命令提示字元出現中文亂碼
chcp 65001 >nul

echo ========================================
echo 🚀 準備開始同步至 GitHub...
echo ========================================

:: 1. 將所有變更加入追蹤清單
git add .

:: 2. 詢問使用者要輸入的註解 (設定變數為空，等待輸入)
set commit_msg=
set /p commit_msg="請輸入這次的修改註解 (直接按 ENTER 會自動使用目前時間)："

:: 3. 判斷使用者有沒有打字。如果沒打字(按ENTER)，就套用預設時間
if "%commit_msg%"=="" (
    set commit_msg=Auto Backup: %date% %time%
)

:: 4. 執行 Commit
git commit -m "%commit_msg%"

:: 5. 推送上雲端
git push

echo ========================================
echo ✨ GitHub 同步完成！你可以關閉這個視窗了。
echo ========================================
pause