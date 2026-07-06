import os
import glob
import pandas as pd
import cv2
import numpy as np

# 1. 設定你的基本路徑 (根據你電腦的目錄)
BASE_DIR = r"C:\Users\tony9\Desktop\5000 - BETTER\data"
CSV_PATH = os.path.join(BASE_DIR, "label_data.csv") # 你在步驟一存好的 CSV

# 2. 建立一個虛擬的對應字典 (如果你的 CSV 還沒做好，程式會先用這個示範數值跑跑看)
# 這裡對應的是你每個資料夾的真實 Ra 量測值
ra_lookup = {
    "5000-0": 1.354,
    "5000-1": 1.582,
    "5000-2": 2.624,
    "5000-3": 2.345,
    "7000-0": 3.037,
    "7000-1": 2.650
}

print("開始掃描資料夾並進行資料配對...")

all_data_rows = []

# 3. 定義你要掃描的加工資料夾清單
folder_names = ["5000-0", "5000-1", "5000-2", "5000-3", "7000-0", "7000-1"]

for folder in folder_names:
    folder_path = os.path.join(BASE_DIR, folder)
    
    # 因為你的資料夾裡，有的是放在 'pc'，有的是放在 '照片'，這裡做一個自動相容判斷
    img_dir = os.path.join(folder_path, "pc")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(folder_path, "照片")
        
    # 如果找到了存放圖片的子資料夾，就開始抓裡面的所有 .jpg 檔
    if os.path.exists(img_dir):
        search_path = os.path.join(img_dir, "*.jpg")
        image_files = glob.glob(search_path)
        print(f"資料夾 [{folder}] 找到 {len(image_files)} 張切削表面圖片")
        
        # 解析轉速與進給編號 (從資料夾名稱切開，例如 '5000-0' 切成 '5000' 和 '0')
        speed, cond_id = folder.split('-')
        
        # 抓取對應的 Ra 標籤值
        ra_value = ra_lookup.get(folder, 0.0)
        
        # 把每一張圖片的路徑、加工參數、Ra值全部綁在一起
        for img_path in image_files:
            all_data_rows.append({
                "image_path": img_path,
                "speed": float(speed),
                "condition_id": float(cond_id),
                "ra_target": float(ra_value)
            })
    else:
        print(f"警告：找不到資料夾 {folder} 的圖片目錄(pc或照片)")

# 4. 將所有人腦看得到的配對結果，轉換成機器學習專用的清單表格 (DataFrame)
dataset_df = pd.DataFrame(all_data_rows)

print("\n--- 資料集配對完成 ---")
print(f"總共成功配對了 {len(dataset_df)} 筆資料（圖片+參數+Ra值）")
print("前 5 筆資料預覽：")
print(dataset_df.head())

# 5. 將這個配對好的清單儲存起來，後面訓練模型直接讀這張表就可以了
dataset_df.to_csv(os.path.join(BASE_DIR, "final_training_manifest.csv"), index=False, encoding="utf-8-sig")
print("\n配對清單已儲存至：final_training_manifest.csv")


# ==========================================
# 補充教學：如何用這張表把圖片真正讀進 Python 變矩陣？
# ==========================================
def load_and_preprocess_image(img_path):
    # 讀取圖片 (OpenCV 預設是 BGR)
    img = cv2.imread(img_path)
    # 將圖片縮放到統一尺寸 (224x224)，這是神經網路最喜歡的大小
    img_resized = cv2.resize(img, (224, 224))
    # 將像素值從 0~255 正規化到 0~1 之間
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

print("\n測試讀取第一張圖片進行前處理...")
if len(all_data_rows) > 0:
    test_img = load_and_preprocess_image(all_data_rows[0]["image_path"])
    print(f"測試成功！圖片已被轉換為形狀為 {test_img.shape} 的數值矩陣，可以準備餵給 AI 了。")