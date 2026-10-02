import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path

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

def main():
    print("🚀 開始萃取傳統特徵並訓練 Random Forest 模型...")
    df = pd.read_csv(str_path('data', 'final_training_manifest.csv'))
    df = df[df['machining_type'] != 'Other'].dropna()

    X, y = [], []
    for idx, row in df.iterrows():
        features = extract_cv_features(row['image_path'])
        if features:
            X.append(features)
            y.append(row['ra_target'])

    # 訓練隨機森林回歸模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # 儲存模型
    save_path = str_path('results', 'traditional_rf_model.joblib')
    joblib.dump(rf_model, save_path)
    print(f"✅ 傳統機器學習模型訓練完成！已儲存至：{save_path}")

if __name__ == "__main__":
    main()
