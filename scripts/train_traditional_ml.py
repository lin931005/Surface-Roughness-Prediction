import os
import sys
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path
from model_versions import archive_current, current_path, write_meta

def extract_cv_features(img_path):
    img_array = np.fromfile(img_path, dtype=np.uint8)
    img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return None

    # 萃取 3 大傳統特徵
    brightness_var = np.var(img_gray) # 亮度變異數
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size # 邊緣密度
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var() # 模糊度/銳利度

    return [brightness_var, edge_density, laplacian_var]

def load_features(df):
    """萃取每張影像的特徵，回傳 (X, y, 成功讀取的資料列)"""
    X, y, kept = [], [], []
    for idx, row in df.iterrows():
        features = extract_cv_features(row['image_path'])
        if features:
            X.append(features)
            y.append(row['ra_target'])
            kept.append(idx)
    return np.array(X), np.array(y), df.loc[kept]

def main():
    print("🚀 開始萃取傳統特徵並訓練 Random Forest 模型...")
    df = pd.read_csv(str_path('data', 'final_training_manifest.csv'))
    if 'split' not in df.columns:
        raise ValueError("❌ CSV 缺少 split 欄位，請先執行 scripts/dataset_prepare.py 重新產生清單")
    df = df[df['machining_type'] != 'Other'].dropna()

    # 依工件切分 (由 dataset_prepare.py 決定)，和深度學習模型保留相同的驗證工件，比較才公平
    X_train, y_train, _ = load_features(df[df['split'] == 'train'])
    X_val, y_val, val_df = load_features(df[df['split'] == 'val'])
    print(f"🧪 訓練 {len(X_train)} 張 / 驗證 {len(X_val)} 張（驗證集的工件訓練時不會看到）")

    # 訓練隨機森林回歸模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 用訓練時沒看過的工件評估
    abs_err = np.abs(rf_model.predict(X_val) - y_val)
    val_mae = float(abs_err.mean())
    val_mape = float((abs_err / y_val).mean() * 100)
    print(f"📏 驗證集 MAE: {val_mae:.4f} μm | MAPE: {val_mape:.2f} %")
    for machining_type in sorted(val_df['machining_type'].unique()):
        mask = (val_df['machining_type'] == machining_type).to_numpy()
        print(f"   {machining_type}: MAE {abs_err[mask].mean():.4f} μm（{mask.sum()} 張）")

    # 儲存模型 (覆蓋前先備份舊版本)
    archived = archive_current('Traditional')
    if archived:
        print(f"🗄️ 舊版模型已備份為 results/archive/{archived}")
    save_path = current_path('Traditional')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, save_path)
    write_meta('Traditional', {
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'val_mae': val_mae,
        'val_mape': val_mape,
        'train_images': len(X_train),
        'val_images': len(X_val),
        'val_conditions': sorted({f"{t}/{c}" for t, c in zip(val_df['machining_type'], val_df['condition_id'])}),
    })
    print(f"✅ 傳統機器學習模型訓練完成！已儲存至：{save_path}")

if __name__ == "__main__":
    main()
