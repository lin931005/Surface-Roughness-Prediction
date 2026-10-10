import hashlib
import os
import random
import re
import pandas as pd
import sys
from collections import Counter

# 確保能讀取到上一層的 project_root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path
from photo_overlap import overlap_ratio, thumbnail

# 驗證集比例與抽樣種子：同一份資料每次都會切出相同的驗證集
VAL_RATIO = 0.2
SPLIT_SEED = 42

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')

# 測試照片（每刀一張）只用來檢測模型，永遠不放進訓練資料
TEST_PHOTO_DIR = str_path('data', 'example')
# 每張測試照片的原始檔名，用來認出同一次拍攝、但裁切過或另存的照片
TEST_PHOTO_SOURCES = str_path('data', 'example_來源.csv')
# 相機用拍攝時間命名照片，例如 20260630164127828.jpg
SHOT_STAMP = re.compile(r'(?<!\d)\d{17}(?!\d)')
# 測試照片的檔名格式，例如「直銑_7000-0.5_2.3746.jpg」，用來找出同一刀的照片
TEST_PHOTO_NAME = re.compile(r'(立銑|直銑)_(\d{4,5}-\d+(?:\.\d+)?)_')
TEST_PHOTO_TYPES = {'立銑': 'End_Milling', '直銑': 'Peripheral_Milling'}


def md5_of(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def list_test_photos():
    if not os.path.isdir(TEST_PHOTO_DIR):
        return []
    return [f for f in os.listdir(TEST_PHOTO_DIR) if f.lower().endswith(IMAGE_EXTS)]


def exclude_test_photos(df):
    """從訓練資料排除 data/example 的測試照片，以及和測試照片拍到同一個位置的照片：

    - 內容和測試照片完全相同
    - 檔名的拍攝時間和測試照片的原始檔相同（同一次拍攝、裁切過的版本）
    - 同一刀裡和測試照片畫面重疊的照片（連續拍攝時拍到同一塊表面）
    """
    test_names = list_test_photos()
    if not test_names:
        return df

    test_digests = {}
    for name in test_names:
        path = os.path.join(TEST_PHOTO_DIR, name)
        test_digests.setdefault(os.path.getsize(path), set()).add(md5_of(path))

    test_stamps = set()
    if os.path.exists(TEST_PHOTO_SOURCES):
        sources = pd.read_csv(TEST_PHOTO_SOURCES, encoding='utf-8-sig')
        for name, source in zip(sources['測試照片'], sources['原始檔名']):
            stamp = SHOT_STAMP.search(str(source))
            if name in test_names and stamp:
                test_stamps.add(stamp.group())

    def is_test_photo(path):
        stamp = SHOT_STAMP.search(os.path.basename(path))
        if stamp and stamp.group() in test_stamps:
            return True
        same_size = test_digests.get(os.path.getsize(path))
        return bool(same_size) and md5_of(path) in same_size

    same_photo = df['image_path'].map(is_test_photo)

    # 同一刀的照片才可能拍到同一塊表面，只和同一刀的照片比對
    same_spot = pd.Series(False, index=df.index)
    for name in test_names:
        m = TEST_PHOTO_NAME.match(name)
        if not m:
            print(f"⚠ 無法從檔名判斷測試照片 {name} 屬於哪一刀，沒辦法檢查同一刀裡拍到同一位置的照片，請改用「直銑_7000-3_1.565.jpg」的格式命名。")
            continue
        same_cut = df[~same_photo & (df['machining_type'] == TEST_PHOTO_TYPES[m.group(1)]) & (df['condition_id'] == m.group(2))]
        test_thumb = thumbnail(os.path.join(TEST_PHOTO_DIR, name))
        for idx, path in same_cut['image_path'].items():
            if not same_spot[idx] and overlap_ratio(test_thumb, thumbnail(path)) > 0:
                same_spot[idx] = True

    print(f"🔒 測試照片（data/example）共 {len(test_names)} 張，不會放進訓練資料。")
    if same_photo.any():
        print(f"⚠ 有 {same_photo.sum()} 張照片和測試照片相同或是同一次拍攝，已從訓練資料排除：")
        for path in df.loc[same_photo, 'image_path']:
            print(f"   {os.path.relpath(path, str_path('data'))}")
    if same_spot.any():
        print(f"🔍 另有 {same_spot.sum()} 張照片和測試照片拍到同一個位置（畫面重疊），也從訓練資料排除，測試照片拍到的位置訓練時都看不到。")
    return df[~(same_photo | same_spot)]


def assign_split(df):
    """依照片切分訓練集 (train) 與驗證集 (val)：每一刀各抽約 VAL_RATIO 的照片當驗證集，用來挑選最好的訓練回合。

    每一刀的其他照片都拿去訓練。內容完全相同的照片（同一張照片被放進不同資料夾）一定分在同一邊；
    Other 沒有刀之分，整個類別一起抽樣。測試照片在這之前就已經排除，不在任何一邊。
    """
    df = df.reset_index(drop=True)

    # 同一張照片出現在 Ra 不同的條件裡，代表照片可能放錯資料夾，提醒使用者檢查
    conflicts = Counter()
    groups = []
    for _, copies in df.groupby('md5'):
        members = sorted(copies.index, key=lambda i: df.at[i, 'image_path'])
        groups.append(members)
        if copies['ra_target'].nunique() > 1:
            conflicts[tuple(sorted({f"{t}/{c}" for t, c in zip(copies['machining_type'], copies['condition_id'])}))] += 1
    for labels, n in sorted(conflicts.items()):
        print(f"⚠ 有 {n} 張照片同時出現在 {'、'.join(labels)}，但 Ra 標籤不同，請確認照片是否放錯資料夾。")

    strata = {}
    for members in groups:
        first = members[0]
        stratum = (df.at[first, 'machining_type'], df.at[first, 'condition_id'])
        strata.setdefault(stratum, []).append((df.at[first, 'image_path'], members))

    df['split'] = 'train'
    for stratum, stratum_groups in sorted(strata.items()):
        stratum_groups.sort(key=lambda g: g[0])
        random.Random(f"{SPLIT_SEED}-{stratum[0]}-{stratum[1]}").shuffle(stratum_groups)
        # 只有一張照片時留給訓練
        n_val = max(1, round(len(stratum_groups) * VAL_RATIO)) if len(stratum_groups) >= 2 else 0
        for _, members in stratum_groups[:n_val]:
            df.loc[members, 'split'] = 'val'

    val_df = df[df['split'] == 'val']
    print(f"🧪 驗證集共 {len(val_df)} 張（{len(val_df) / len(df):.0%}），每一刀各抽約 {VAL_RATIO:.0%} 的照片，用來挑選最好的訓練回合：")
    for machining_type, sub in val_df.groupby('machining_type'):
        cuts = "" if machining_type == 'Other' else f"，來自 {sub['condition_id'].nunique()} 刀"
        print(f"   {machining_type}：{len(sub)} 張{cuts}")
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

    if not os.path.exists(DATASET_DIR):
        print(f"❌ 找不到資料夾：{DATASET_DIR}，請確認 Dataset_Cleaned 是否在正確位置。")
        return

    # 自動掃描 Dataset_Cleaned 底下的所有東西
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(IMAGE_EXTS):
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

    dataset_df = exclude_test_photos(pd.DataFrame(all_data_rows))
    # 記錄每張照片的內容雜湊：訓練時會存進模型紀錄，批量驗證時用來判斷上傳的照片是否訓練過
    dataset_df = assign_split(dataset_df.assign(md5=dataset_df['image_path'].map(md5_of)))

    # 確保輸出的 data 目錄存在
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    dataset_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ CSV 自動生成完畢！共包含 {len(dataset_df)} 張圖片。")
    print(f"✅ 檔案已成功更新至: {CSV_PATH}")

if __name__ == '__main__':
    generate_manifest()
