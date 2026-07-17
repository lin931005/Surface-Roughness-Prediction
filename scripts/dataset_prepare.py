import os
import glob
import pandas as pd
import sys

# 確保能讀取到上一層的 project_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path

def generate_manifest():
    BASE_DIR = str_path('data')
    CSV_PATH = os.path.join(BASE_DIR, "final_training_manifest.csv")

    # 這裡存放真實的 Ra 量測值
    ra_lookup = {
        "5000-0": 1.354,
        "5000-1": 1.582,
        "5000-2": 2.624,
        "5000-3": 2.345,
        "7000-0": 3.037,
        "7000-1": 2.650,
        "9000-1": 0.9086,
        "9000-2": 1.2042,
        "9000-3": 0.7202,
        "9000-4": 0.9443,
    }

    # 移除了表情符號，避免 cp950 編碼錯誤
    print("開始動態掃描資料夾並生成最新 CSV...")
    all_data_rows = []

    if not os.path.exists(BASE_DIR):
        print(f"找不到 data 資料夾：{BASE_DIR}")
        return

    # 自動掃描 data 資料夾底下的所有東西
    for folder_name in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder_name)

        # 只處理名稱裡面有 '-' 的資料夾
        if not os.path.isdir(folder_path) or '-' not in folder_name:
            continue

        try:
            speed, cond_id = folder_name.split('-')
        except ValueError:
            continue

        # 支援 'pc' 或 '照片' 兩種命名
        img_dir = os.path.join(folder_path, "pc")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(folder_path, "照片")

        if os.path.exists(img_dir):
            search_path = os.path.join(img_dir, "*.jpg")
            image_files = glob.glob(search_path)

            ra_value = ra_lookup.get(folder_name, 0.0)

            for img_path in image_files:
                all_data_rows.append({
                    "image_path": img_path,  # 自動抓取當前電腦的絕對路徑
                    "speed": float(speed),
                    "condition_id": float(cond_id),
                    "ra_target": float(ra_value)
                })

    dataset_df = pd.DataFrame(all_data_rows)
    dataset_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # 移除了表情符號
    print(f"CSV 自動生成完畢！共包含 {len(dataset_df)} 張圖片。")
    print(f"檔案已儲存至: {CSV_PATH}")

if __name__ == '__main__':
    generate_manifest()
