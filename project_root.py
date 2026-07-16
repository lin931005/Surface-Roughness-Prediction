from pathlib import Path
import os

# 1. 自動定位到 project_root.py 所在的資料夾 (即 5000 - BETTER 根目錄)
# 這樣無論你從哪裡執行 python 指令，它都會鎖定這裡
ROOT = Path(__file__).resolve().parent

def path(*parts):
    """回傳一個 Path 物件，適合用來進行各種檔案操作 (如 exists(), mkdir())"""
    return ROOT.joinpath(*parts)

def str_path(*parts):
    """回傳一個字串路徑，適合餵給 pandas, cv2, torch 等舊版或標準函式"""
    return str(path(*parts))

# 方便 Debug 的小功能：印出當前根目錄，確認沒跑錯地方
if __name__ == "__main__":
    print(f"📍 專案根目錄已鎖定: {ROOT}")
