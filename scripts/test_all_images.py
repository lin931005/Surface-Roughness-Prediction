import os
import torch
import pandas as pd
from predict_roughness import load_model, predict_images

BASE_DIR = r"C:\Users\tony9\Desktop\5000 - BETTER"
CSV_FILE = os.path.join(BASE_DIR, "data", "final_training_manifest.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "results.csv")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 目前使用裝置：{device}")

    df = pd.read_csv(CSV_FILE)
    img_paths = df['image_path'].tolist()
    speed_list = df['speed'].tolist()
    cond_list = df['condition_id'].tolist()
    true_ra = df['ra_target'].tolist()

    print(f"🔍 正在開始對 {len(df)} 張圖片進行批量預測...")

    model = load_model(device=device)
    predictions = predict_images(
        img_paths,
        speed_list,
        cond_list,
        model=model,
        device=device,
        batch_size=16,
    )

    results_df = pd.DataFrame({
        "True_Ra": true_ra,
        "Predicted_Ra": predictions,
    })
    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"\n✅ 批量測試完成！結果已儲存至 {RESULTS_PATH}")
print("你可以直接用 Excel 開啟這個檔案，選中兩欄資料畫散佈圖！")