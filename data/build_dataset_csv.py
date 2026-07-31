import os
import csv

def generate_ra_manifest_manual():
    dataset_dir = "."
    output_csv = "final_labels.csv"

    print("🔍 步驟 1：載入自訂的 Ra 數值雙層字典...")
    # ==========================================
    # 💡 雙層字典：先分加工法，再填實驗編號與 Ra 值
    # ==========================================
    ra_dict = {
        # 🟢 立銑  彎 (End Milling) 的 Ra 數據
        "End_Milling": {
            "5000-1": 1.4786,
            "5000-2": 1.515,
            "5000-3": 1.5086,
            "5000-4": 1.19,
            "5000-5": 1.5132,
            "5000-6": 1.5084,
            "5000-7": 1.4602,
            "5000-7.5": 1.4602,
            "5000-8": 2.1068,
            "5000-9": 2.0038,
            "5000-10": 2.0156,

            "7000-6": 1.8972,
            "7000-7": 1.6492,
            "7000-8": 1.462,
            "7000-9": 1.752,
            "7000-10": 1.662,
        },
        
        # 🔵 直銑  直 (Peripheral Milling) 的 Ra 數據
        "Peripheral_Milling": {
            "5000-0": 2.0276,
            "5000-1": 1.6932,
            "5000-2": 1.7772,
            "5000-3": 2.0786,
              
            "7000-0": 3.1566, 
            "7000-0.5": 2.3746, 
            "7000-1": 1.0238,  
            "7000-2": 1.321,  
            "7000-3": 1.565,  
            "7000-4": 1.0698,  
            "7000-5": 1.0558,  

            "9000-1": 0.9086,
            "9000-2": 1.2042,
            "9000-3": 0.7202,
            "9000-4": 1.392,
            "9000-5": 1.473,
            "9000-6": 1.29,
            "9000-7": 1.021,
            "9000-8": 1.2878,
            "9000-9": 1.1274,
        }
    }

    # 計算總共輸入了幾筆
    total_records = sum(len(records) for records in ra_dict.values())
    print(f"✅ 成功載入 {total_records} 組 Ra 數值對應！")
    
    print("\n🔍 步驟 2：掃描圖片並配對 Ra 值...")
    csv_data = []
    valid_exts = ('.png', '.jpg', '.jpeg')
    missing_ra_folders = set()

    if not os.path.exists(dataset_dir):
        print(f"❌ 找不到資料夾 '{dataset_dir}'，請確認它和這支程式在同一個目錄。")
        return

    # 遍歷 Dataset_Cleaned 目錄
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(valid_exts):
                rel_path = os.path.relpath(os.path.join(root, file), start=".")
                path_parts = rel_path.split(os.sep)
                
                # 確保路徑深度符合預期
                if len(path_parts) >= 4:
                    machining_type = path_parts[0] # "End_Milling" 或 "Peripheral_Milling"
                    speed = path_parts[1]          # "5000", "7000"...
                    condition_id = path_parts[2]   # "5000-0", "5000-1"...

                    # 💡 全新配對邏輯：先找加工法，再找實驗編號
                    if machining_type in ra_dict and condition_id in ra_dict[machining_type]:
                        ra_val = ra_dict[machining_type][condition_id]
                        csv_data.append([rel_path, machining_type, speed, condition_id, ra_val])
                    else:
                        # 記錄下找不到數值的「加工法 + 編號」，方便你除錯
                        missing_ra_folders.add(f"{machining_type} / {condition_id}")

    print("\n📝 步驟 3：寫入 CSV 檔案...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'machining_type', 'speed', 'condition_id', 'ra_target'])
        writer.writerows(csv_data)

    print(f"🎉 完成！共成功配對 {len(csv_data)} 張圖片，已匯出至 {output_csv}")

    if missing_ra_folders:
        print("\n⚠️ 警告：以下資料夾在你的 ra_dict 字典中找不到對應的 Ra 值，已跳過：")
        for f in missing_ra_folders:
            print(f"  - {f}")

if __name__ == "__main__":
    # 修正了這裡的呼叫名稱
    generate_ra_manifest_manual()