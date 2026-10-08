import os
import sys
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import functools
from datetime import datetime

print = functools.partial(print, flush=True)

# ------------------------------------------
# 參數設定
# ------------------------------------------
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)
PATIENCE = 30
LR = 1e-4
EPOCHS = 100

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path
from model_versions import archive_current, current_path, write_meta
from preprocessing import (PATCH_SIZE, EVAL_GRID, PREPROCESS_VERSION, to_input, load_gray, grid_boxes,
                           random_train_patch)

CSV_PATH = str_path('data', 'final_training_manifest.csv')
RESULTS_DIR = str_path('results')
BEST_MODEL_PATH = str(current_path('Classifier'))
LOSS_CSV_PATH = os.path.join(RESULTS_DIR, 'loss_record_Classifier.csv')

# ==========================================
# 1. 視覺分類專屬資料讀取器 (0: 立銑, 1: 直銑)
# ==========================================
class ClassifierDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        # 位置、大小與旋轉的隨機變化在 random_train_patch 裡處理，這裡只做翻轉與亮度對比
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])
        # 將字串轉換為數字標籤
        self.label_map = {"End_Milling": 0, "Peripheral_Milling": 1, "Other": 2}

    def __len__(self):
        return len(self.df) * 2 if self.is_train else len(self.df)

    def __getitem__(self, idx):
        real_idx = idx % len(self.df)
        row = self.df.iloc[real_idx]
        img_path = row['image_path']
        label = self.label_map[row['machining_type']]

        try:
            img_pil = load_gray(img_path)
        except Exception as e:
            raise IOError(f"讀取圖片發生錯誤 {img_path}: {str(e)}")

        # 從原始解析度切固定大小的方塊；驗證時均勻取 3x3 個方塊，評估時取平均
        if self.is_train:
            img_tensor = to_input(self.augment(random_train_patch(img_pil)))
        else:
            img_tensor = torch.stack([to_input(img_pil.crop(box)) for box in grid_boxes(img_pil)])
        return img_tensor, torch.tensor(label, dtype=torch.long)

# ==========================================
# 2. 輕量級視覺大腦 (ResNet-18)
# ==========================================
class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        # 輸出 2 個神經元 (立銑與直銑的機率)
        self.resnet.fc = nn.Linear(num_ftrs, 3)

    def forward(self, img):
        return self.resnet(img)

# ==========================================
# 3. 核心訓練邏輯
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            n_imgs, n_patches = imgs.shape[:2]
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                # 和線上推論相同：同一張照片各方塊的輸出取平均後再判斷類別
                preds = model(imgs.flatten(0, 1)).view(n_imgs, n_patches, -1).mean(dim=1)
                loss = criterion(preds, labels)

            total_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total if total > 0 else 0
    return total_loss / len(dataloader), acc

def main():
    print("🚀 準備啟動【視覺分類器大腦】專屬訓練管線...")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 檔案：{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if df.empty or 'machining_type' not in df.columns:
        raise ValueError("❌ CSV 格式錯誤或無資料")
    if 'split' not in df.columns:
        raise ValueError("❌ CSV 缺少 split 欄位，請先執行 scripts/dataset_prepare.py 重新產生清單")

    # 依刀切分 (由 dataset_prepare.py 決定)：驗證集的刀訓練時完全看不到
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    val_milling = val_df[val_df['machining_type'] != 'Other']
    val_conditions = sorted({f"{t}/{c}" for t, c in zip(val_milling['machining_type'], val_milling['condition_id'])})

    print(f"📂 成功載入 {len(df)} 筆影像資料！(包含立銑、直銑與 Other 負面教材)")
    print(f"🧪 訓練 {len(train_df)} 張 / 驗證 {len(val_df)} 張（驗證集的刀訓練時不會看到）")

    archived = archive_current('Classifier')
    if archived:
        print(f"🗄️ 舊版模型已備份為 results/archive/{archived}")

    train_loader = DataLoader(ClassifierDataset(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # 驗證時每張照片有 EVAL_GRID² 個方塊，批次張數相應縮小
    val_loader = DataLoader(ClassifierDataset(val_df, False), batch_size=max(1, BATCH_SIZE // EVAL_GRID ** 2), shuffle=False, num_workers=NUM_WORKERS)
    print(f"🔍 前處理：從原始照片切 {PATCH_SIZE}x{PATCH_SIZE} 方塊後縮成 224x224，驗證時每張照片取 {EVAL_GRID}x{EVAL_GRID} 個方塊平均")

    model = ClassifierModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    epochs_without_improve = 0
    stats = []

    print(f"🔥 開始進行 {EPOCHS} 回合的【三元分類】防禦大腦訓練！\n" + "-" * 50)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(imgs)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(avg_val_loss)

        stats.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_acc': val_acc})
        print(f"第 {epoch:02d}/{EPOCHS} 回合 | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | val_acc: {val_acc*100:.2f}%")

        import json
        print(json.dumps({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            write_meta('Classifier', {
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'train_images': len(train_df),
                'val_images': len(val_df),
                'val_conditions': val_conditions,
                'preprocess': PREPROCESS_VERSION,
            })
            print("  👉 已儲存最佳分類器模型！")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= PATIENCE:
            print(f"已連續 {PATIENCE} 個 epoch 未進步，提前停止訓練。")
            break

    pd.DataFrame(stats).to_csv(LOSS_CSV_PATH, index=False)
    print(f"🎉 分類器訓練完成！模型已儲存至：{BEST_MODEL_PATH}")

if __name__ == '__main__':
    main()
