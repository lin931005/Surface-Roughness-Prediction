import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 確保能讀取到 project_root[cite: 10]
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path

# 設定支援中文顯示的字體 (避免 Matplotlib 圖表中文變方塊)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_features(image_path):
    """使用 OpenCV 提取傳統影像特徵"""
    # 支援中文路徑的讀取方式
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        return None, None

    # 1. 亮度變異數 (Brightness Variance)
    # 物理意義：粗糙度越高，表面高低起伏越大，反光與陰影的對比越強烈，變異數通常越大。
    brightness_var = np.var(img_gray)

    # 2. 邊緣密度 (Edge Density)
    # 物理意義：使用 Canny 邊緣偵測抓取刀痕。刀痕越深、越密集，偵測到的邊緣像素比例就越高。
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    edge_density = np.sum(edges > 0) / edges.size

    return brightness_var, edge_density

def main():
    csv_path = str_path('data', 'final_training_manifest.csv')

    print("🚀 開始讀取資料庫並分析立銑影像的傳統特徵...")
    df = pd.read_csv(csv_path)

    # 過濾出「立銑」且排除 Other 的正常影像
    df_end_milling = df[df['machining_type'] == 'End_Milling'].copy()

    ra_list = []
    var_list = []
    edge_list = []

    # 萃取特徵
    for idx, row in df_end_milling.iterrows():
        b_var, e_density = extract_features(row['image_path'])
        if b_var is not None:
            ra_list.append(row['ra_target'])
            var_list.append(b_var)
            edge_list.append(e_density)

    # 將結果轉為 DataFrame 方便計算
    results_df = pd.DataFrame({
        'Ra': ra_list,
        'Brightness_Variance': var_list,
        'Edge_Density': edge_list
    })

    # 計算相關係數 (Correlation)
    corr_var, _ = pearsonr(results_df['Ra'], results_df['Brightness_Variance'])
    corr_edge, _ = pearsonr(results_df['Ra'], results_df['Edge_Density'])

    print("-" * 40)
    print("📊 【特徵與 Ra 真實值相關性分析報告】")
    print(f"🔹 亮度變異數 相關係數: {corr_var:.4f}")
    print(f"🔹 邊緣密度   相關係數: {corr_edge:.4f}")
    print("💡 註：絕對值越接近 1 代表相關性越強，0 代表無相關。")
    print("-" * 40)

    # 視覺化散佈圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('傳統機器視覺特徵 vs CNC 表面粗糙度 (Ra)', fontsize=16)

    # 圖表 1：亮度變異數
    ax1.scatter(results_df['Ra'], results_df['Brightness_Variance'], alpha=0.7, c='#3b82f6')
    ax1.set_title(f'亮度變異數 (r = {corr_var:.2f})')
    ax1.set_xlabel('真實粗糙度 Ra (μm)')
    ax1.set_ylabel('亮度變異數')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 圖表 2：邊緣密度
    ax2.scatter(results_df['Ra'], results_df['Edge_Density'], alpha=0.7, c='#10b981')
    ax2.set_title(f'邊緣密度 (r = {corr_edge:.2f})')
    ax2.set_xlabel('真實粗糙度 Ra (μm)')
    ax2.set_ylabel('邊緣密度 (比例)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
