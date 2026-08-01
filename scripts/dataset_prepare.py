import os
import pandas as pd
import sys

# 確保能讀取到上一層的 project_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path

def generate_manifest():
    # 💡 將資料來源指向我們整理好的 Dataset_Cleaned
    # 假設 Dataset_Cleaned 放在專案根目錄下
    DATASET_DIR = str_path('data')

    # 輸出的 CSV 依然存放到 data 資料夾下，保持與後端相容
    CSV_PATH = str_path('data', 'final_training_manifest.csv')

    print("🚀 開始動態掃描 Dataset_Cleaned 資料夾並生成最新 CSV...")

    # ==========================================
    # 💡 最新的雙層 Ra 數值字典 (立銑與直銑徹底分離)
    # ==========================================
    ra_dict = {
        "End_Milling": {
            "5000-1": 1.4786, "5000-2": 1.515, "5000-3": 1.5086,
            "5000-4": 1.19, "5000-5": 1.5132, "5000-6": 1.5084,
            "5000-7": 1.4602, "5000-7.5": 1.4602, "5000-8": 2.1068,
            "5000-9": 2.0038, "5000-10": 2.0156,
            "7000-6": 1.8972, "7000-7": 1.6492, "7000-8": 1.462,
            "7000-9": 1.752, "7000-10": 1.662,
        },
        "Peripheral_Milling": {
            "5000-0": 2.0276, "5000-1": 1.6932, "5000-2": 1.7772, "5000-3": 2.0786,
            "7000-0": 3.1566, "7000-0.5": 2.3746, "7000-1": 1.0238, "7000-2": 1.321,
            "7000-3": 1.565, "7000-4": 1.0698, "7000-5": 1.0558,
            "9000-1": 0.9086, "9000-2": 1.2042, "9000-3": 0.7202,
            "9000-4": 1.392, "9000-5": 1.473, "9000-6": 1.29,
            "9000-7": 1.021, "9000-8": 1.2878, "9000-9": 1.1274,
        }
    }

    all_data_rows = []
    valid_exts = ('.png', '.jpg', '.jpeg')

    if not os.path.exists(DATASET_DIR):
        print(f"❌ 找不到資料夾：{DATASET_DIR}，請確認 Dataset_Cleaned 是否在正確位置。")
        return

    # 自動掃描 Dataset_Cleaned 底下的所有東西
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(valid_exts):
                # 取得絕對路徑給 PyTorch Dataset 讀取
                abs_path = os.path.abspath(os.path.join(root, file))
                rel_path = os.path.relpath(abs_path, start=DATASET_DIR)
                path_parts = rel_path.split(os.sep)

                # 期待結構: Machining_Type / Speed / Condition / img.jpg
                # 注意：Other 資料夾可能沒有這麼深，所以要另外寫判斷
                if len(path_parts) >= 1:
                    machining_type = path_parts[0]  # End_Milling, Peripheral_Milling 或 Other

                    # 💡 解法：如果資料夾是 Other，就無條件收編，不需要查字典！
                    if machining_type == "Other":
                        all_data_rows.append({
                            "image_path": abs_path,
                            "machining_type": "Other",
                            "speed": 0.0,
                            "condition_id": "N/A",
                            "ra_target": 0.0
                        })

                    # 正常的立銑與直銑，才去檢查深層資料夾並查字典
                    elif len(path_parts) >= 4:
                        speed_str = path_parts[1]
                        condition_id = path_parts[2]

                        if machining_type in ra_dict and condition_id in ra_dict[machining_type]:
                            ra_val = ra_dict[machining_type][condition_id]

                            all_data_rows.append({
                                "image_path": abs_path,
                                "machining_type": machining_type,
                                "speed": float(speed_str),
                                "condition_id": condition_id,
                                "ra_target": float(ra_val)
                            })

    dataset_df = pd.DataFrame(all_data_rows)

    # 確保輸出的 data 目錄存在
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    dataset_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ CSV 自動生成完畢！共包含 {len(dataset_df)} 張圖片。")
    print(f"✅ 檔案已成功更新至: {CSV_PATH}")

if __name__ == '__main__':
    generate_manifest()
