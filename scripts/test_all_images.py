import torch
import pandas as pd
from predict_roughness import predict_single_image # 引用我們之前寫好的預測函式

# 1. 讀取你的訓練清單
csv_file = r"C:\Users\tony9\Desktop\5000 - BETTER\data\final_training_manifest.csv"
df = pd.DataFrame(pd.read_csv(csv_file))

# 2. 建立儲存預測結果的清單
results = []

print(f"🔍 正在開始對 {len(df)} 張圖片進行批量預測...")

# 3. 用迴圈跑完所有資料
for i, row in df.iterrows():
    img_path = row['image_path']
    speed = row['speed']
    cond = row['condition_id']
    true_ra = row['ra_target']
    
    # 呼叫我們之前寫好的預測函式
    pred_ra = predict_single_image(img_path, speed, cond)
    
    if pred_ra is not None:
        results.append({
            "True_Ra": true_ra,
            "Predicted_Ra": pred_ra
        })
    
    if (i+1) % 20 == 0:
        print(f"已完成 {i+1} / {len(df)} 張...")

# 4. 存成 CSV 讓你可以直接畫圖
results_df = pd.DataFrame(results)
results_df.to_csv(r"C:\Users\tony9\Desktop\5000 - BETTER\results\results.csv", index=False)

print("\n✅ 批量測試完成！結果已儲存至 results.csv")
print("你可以直接用 Excel 開啟這個檔案，選中兩欄資料畫散佈圖！")