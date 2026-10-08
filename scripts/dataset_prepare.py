import hashlib
import os
import random
import pandas as pd
import sys
from collections import Counter

# 確保能讀取到上一層的 project_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path

# 驗證集比例與抽樣種子：同一份資料每次都會切出相同的驗證集
VAL_RATIO = 0.2
SPLIT_SEED = 42


def condition_sort_key(condition_id):
    # "7000-0.5" -> (7000.0, 0.5)，讓 5000-2 排在 5000-10 前面
    return tuple(float(x) for x in condition_id.split('-'))


def find_duplicate_photos(paths):
    """找出內容完全相同的照片，回傳 {重複的照片: 第一次出現的照片}。

    先比對檔案大小，大小相同才計算雜湊，不必把整個資料集讀過一遍。
    """
    by_size = {}
    for p in paths:
        by_size.setdefault(os.path.getsize(p), []).append(p)

    first_copy = {}
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        by_digest = {}
        for p in same_size:
            with open(p, 'rb') as f:
                by_digest.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(p)
        for copies in by_digest.values():
            for p in copies[1:]:
                first_copy[p] = copies[0]
    return first_copy


def assign_split(df):
    """依刀切分訓練集 (train) 與驗證集 (val)，驗證集的刀在訓練時完全看不到。

    下列照片會分在同一組，整組一起進訓練集或驗證集：
    - 同一個條件資料夾的照片（同一刀）
    - Ra 實測值相同的條件（例如 5000-7 與 5000-7.5）
    - 內容完全相同的照片（同一張照片被放進不同資料夾）
    分組後依「銑法 + 轉速」分層，每層抽 VAL_RATIO 的組別當驗證集；Other 沒有刀之分，逐張抽樣。
    """
    df = df.reset_index(drop=True)
    labels = [
        f"{row.machining_type}/{row.condition_id}" if row.machining_type != 'Other'
        else f"Other/{os.path.basename(row.image_path)}"
        for row in df.itertuples()
    ]

    # 用 union-find 把有關聯的照片併成同一組
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen = {}

    def link(key, i):
        if key in first_seen:
            parent[find(i)] = find(first_seen[key])
        else:
            first_seen[key] = i

    duplicates = find_duplicate_photos(df['image_path'].tolist())
    for i, row in enumerate(df.itertuples()):
        link(('photo', duplicates.get(row.image_path, row.image_path)), i)
        if row.machining_type != 'Other':
            link(('condition', labels[i]), i)
            link(('ra', row.machining_type, row.ra_target), i)

    # 同一張照片出現在 Ra 不同的條件裡，代表照片可能放錯資料夾，提醒使用者檢查
    row_of = {p: i for i, p in enumerate(df['image_path'])}
    conflicts = Counter()
    for dup, original in duplicates.items():
        a, b = row_of[dup], row_of[original]
        if df.at[a, 'ra_target'] != df.at[b, 'ra_target']:
            conflicts[tuple(sorted((labels[a], labels[b])))] += 1
    for (a, b), n in sorted(conflicts.items()):
        print(f"⚠ 有 {n} 張照片同時出現在 {a} 和 {b}，但兩邊的 Ra 標籤不同，請確認照片是否放錯資料夾。")

    groups = {}
    for i in range(len(df)):
        groups.setdefault(find(i), []).append(i)

    strata = {}
    for members in groups.values():
        first = min(members, key=lambda i: labels[i])
        merged = sorted({labels[i] for i in members if df.at[i, 'machining_type'] != 'Other'})
        if len(merged) > 1:
            print(f"🔗 這些條件被分在同一組（照片重複或 Ra 相同）：{'、'.join(merged)}")
        stratum = (df.at[first, 'machining_type'], df.at[first, 'speed'])
        strata.setdefault(stratum, []).append((labels[first], members))

    df['split'] = 'train'
    for stratum, stratum_groups in sorted(strata.items()):
        stratum_groups.sort(key=lambda g: g[0])
        random.Random(f"{SPLIT_SEED}-{stratum[0]}-{stratum[1]}").shuffle(stratum_groups)
        # 只有一組時全部留給訓練，否則該轉速的資料模型完全學不到
        n_val = max(1, round(len(stratum_groups) * VAL_RATIO)) if len(stratum_groups) >= 2 else 0
        for _, members in stratum_groups[:n_val]:
            df.loc[members, 'split'] = 'val'

    val_df = df[df['split'] == 'val']
    print(f"🧪 驗證集共 {len(val_df)} 張（{len(val_df) / len(df):.0%}），這些刀在訓練時不會出現：")
    for machining_type, sub in val_df.groupby('machining_type'):
        if machining_type == 'Other':
            print(f"   Other：{len(sub)} 張")
        else:
            conditions = sorted(sub['condition_id'].unique(), key=condition_sort_key)
            print(f"   {machining_type}：{'、'.join(conditions)}（{len(sub)} 張）")
    return df


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

            "7000-1": 1.8972,
            "7000-2": 1.6492,
            "7000-3": 1.462,
            "7000-4": 1.752,
            "7000-5": 1.662,
            "7000-6": 2.0548,
            "7000-7": 1.4974,
            "7000-8": 1.7248,
            "7000-9": 1.5922,
            "7000-10": 1.7792,
            "7000-11": 1.579,

            "9000-1": 0.9436,
            "9000-2": 1.1242,
            "9000-3": 1.3222,
            "9000-4": 0.8856,
            "9000-5": 1.0288,
            "9000-6": 1.3216,
            "9000-7": 1.1508,
            "9000-8": 1.363,
            "9000-9": 0.9638,
            "9000-10": 1.0256,
            "9000-11": 0.3328,
            "9000-12": 0.43,
        },
        "Peripheral_Milling": {
            "5000-0": 2.0276,
            "5000-1": 1.6932,
            "5000-2": 1.7772,
            "5000-3": 2.0786,
            # 9/16–9/17 拍攝的新直銑 5000 第 1~8 刀（ra資料 自動儲存版 new5000 第二張表），編號加 10 與舊的 5000-0~3 區分；
            # 第 6、7 刀依拍攝順序對應：原始資料夾 5000-7（14:03 先拍）是第 6 刀，5000-6（14:20 後拍）是第 7 刀
            "5000-11": 1.1306,
            "5000-12": 1.818,
            "5000-13": 1.5792,
            "5000-14": 1.7494,
            "5000-15": 1.9058,
            "5000-16": 2.1616,
            "5000-17": 1.978,
            "5000-18": 2.1628,

            "7000-0": 3.1566,
            "7000-0.5": 2.3746,
            "7000-1": 1.0238,
            "7000-2": 1.321,
            "7000-3": 1.565,
            "7000-4": 1.0698,
            "7000-5": 1.0558,
            "7000-6": 1.6996,
            "7000-7": 1.7408,
            "7000-8": 1.443,
            "7000-9": 1.4448,
            "7000-10": 1.2726,

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

    dataset_df = assign_split(pd.DataFrame(all_data_rows))

    # 確保輸出的 data 目錄存在
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    dataset_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ CSV 自動生成完畢！共包含 {len(dataset_df)} 張圖片。")
    print(f"✅ 檔案已成功更新至: {CSV_PATH}")

if __name__ == '__main__':
    generate_manifest()
