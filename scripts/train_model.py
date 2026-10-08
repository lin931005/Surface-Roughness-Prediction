import os
import sys
import random
import inspect
import pandas as pd
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import time
import functools
from datetime import datetime

print = functools.partial(print, flush=True)

# ------------------------------------------
# 參數設定
# ------------------------------------------
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)
PATIENCE = 40
LR = 1e-4
EPOCHS = 200

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path
from model_versions import archive_current, current_path, write_meta
from preprocessing import (PATCH_SIZE, EVAL_GRID, PREPROCESS_VERSION, to_input, load_gray, grid_boxes,
                           random_train_patch)

BASE_DIR = str_path()
CSV_PATH = str_path('data', 'final_training_manifest.csv')
RESULTS_DIR = str_path('results')

# ==========================================
# 1. 定義資料讀取器
# ==========================================
class SurfaceDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, is_train: bool = True):
        self.data_info = data_frame.reset_index(drop=True)
        self.is_train = is_train

        # 💡 位置、大小與旋轉的隨機變化在 random_train_patch 裡處理，這裡只做翻轉與亮度對比
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])

    def __len__(self):
        return len(self.data_info) * 2 if self.is_train else len(self.data_info)

    def __getitem__(self, idx):
        real_idx = idx % len(self.data_info)
        row = self.data_info.iloc[real_idx]
        img_path = row['image_path']

        try:
            # 💡 核心修正：讀圖後立刻轉灰階 (L) 去除色彩，再轉回三通道 (RGB)
            img_pil = load_gray(img_path)
        except Exception as e:
            raise IOError(f"讀取圖片時發生錯誤 {img_path}: {str(e)}")

        # 💡 從原始解析度切固定大小的方塊再縮放，每張照片的刀痕放大倍率都相同
        if self.is_train:
            img_tensor = to_input(self.augment(random_train_patch(img_pil)))
        else:
            # 驗證：每張照片均勻取 3x3 個方塊，評估時取平均，和線上推論的做法一致
            img_tensor = torch.stack([to_input(img_pil.crop(box)) for box in grid_boxes(img_pil)])

        speed = float(row['speed']) / 10000.0
        cond = 0.0 # 💡 修正：廢棄字串編號轉換，統一設為 0.0 配合推論端
        params = np.array([speed, cond], dtype=np.float32)
        ra_target = np.array([row['ra_target']], dtype=np.float32)

        return img_tensor, torch.tensor(params), torch.tensor(ra_target)

# ==========================================
# 2. 雙輸入 AI 模型 (ResNet-50 全局微調)
# ==========================================
class ResNetDualInputModel(nn.Module):
    def __init__(self):
        super(ResNetDualInputModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU(),
        )

        self.dnn = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, img, params):
        img_features = self.resnet(img)
        param_features = self.dnn(params)
        combined = torch.cat((img_features, param_features), dim=1)
        return self.fc(combined)

# ==========================================
# 3. 工具函式與主程式
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, criterion, device):
    """回傳 (平均 loss, 平均絕對誤差 MAE μm)；每張照片的多個方塊先平均成一個預測值"""
    model.eval()
    total_loss = 0.0
    total_abs_err, total_count = 0.0, 0
    with torch.no_grad():
        for imgs, params, targets in dataloader:
            imgs = imgs.to(device)
            params = params.to(device)
            targets = targets.to(device)
            n_imgs, n_patches = imgs.shape[:2]
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(imgs.flatten(0, 1), params.repeat_interleave(n_patches, dim=0))
                preds = preds.view(n_imgs, n_patches).mean(dim=1, keepdim=True)
                loss = criterion(preds, targets)
            total_loss += loss.item()
            total_abs_err += (preds.float() - targets).abs().sum().item()
            total_count += targets.size(0)
    return total_loss / len(dataloader), total_abs_err / total_count

def main():
    parser = argparse.ArgumentParser()
    # 💡 核心新增：必須指定訓練哪一種加工法！
    parser.add_argument('--milling_type', type=str, required=True, choices=['End_Milling', 'Peripheral_Milling'], help='指定要訓練的銑削類型')
    parser.add_argument('--output', type=str, default=None, help='output model path')
    parser.add_argument('--log', type=str, default=None, help='progress log path')
    args = parser.parse_args()

    print(f"🚀 準備啟動【{args.milling_type} 專家大腦】專屬訓練管線...")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # 💡 動態命名輸出檔案
    BEST_MODEL_PATH = str(current_path(args.milling_type))
    LOSS_CSV_PATH = os.path.join(RESULTS_DIR, f'loss_record_{args.milling_type}.csv')

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 檔案：{CSV_PATH}")

    data_df = pd.read_csv(CSV_PATH)
    if 'split' not in data_df.columns:
        raise ValueError("❌ CSV 缺少 split 欄位，請先執行 scripts/dataset_prepare.py 重新產生清單")

    # 💡 核心過濾：只挑選符合當前 milling_type 的資料來訓練！
    data_df = data_df[data_df['machining_type'] == args.milling_type].copy()

    if data_df.empty:
        raise ValueError(f"❌ 在 CSV 中找不到任何 {args.milling_type} 的資料，請檢查清單！")

    print(f"📂 成功載入 {len(data_df)} 筆 {args.milling_type} 影像資料！")

    # 💡 依刀切分 (由 dataset_prepare.py 決定)：驗證集的刀訓練時完全看不到，驗證成績才不會偏樂觀
    train_df = data_df[data_df['split'] == 'train'].reset_index(drop=True)
    val_df = data_df[data_df['split'] == 'val'].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("❌ 訓練集或驗證集是空的，請檢查 dataset_prepare.py 的切分結果")
    val_conditions = sorted(val_df['condition_id'].unique())
    print(f"🧪 訓練 {len(train_df)} 張 / 驗證 {len(val_df)} 張，驗證用的刀：{', '.join(val_conditions)}")

    archived = archive_current(args.milling_type)
    if archived:
        print(f"🗄️ 舊版模型已備份為 results/archive/{archived}")

    train_dataset = SurfaceDataset(train_df, is_train=True)
    val_dataset = SurfaceDataset(val_df, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    # 驗證時每張照片有 EVAL_GRID² 個方塊，批次張數相應縮小，避免顯示卡記憶體不足
    val_loader = DataLoader(val_dataset, batch_size=max(1, BATCH_SIZE // EVAL_GRID ** 2), shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    print(f"🔍 前處理：從原始照片切 {PATCH_SIZE}x{PATCH_SIZE} 方塊後縮成 224x224，驗證時每張照片取 {EVAL_GRID}x{EVAL_GRID} 個方塊平均")

    model = ResNetDualInputModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    epochs_without_improve = 0
    stats = []
    torch.backends.cudnn.benchmark = True

    print(f"🔥 開始進行 {EPOCHS} 回合的全局微調訓練！\n" + "-" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch_imgs, batch_params, batch_targets in train_loader:
            batch_imgs, batch_params, batch_targets = batch_imgs.to(device), batch_params.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                predictions = model(batch_imgs, batch_params)
                loss = criterion(predictions, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(avg_val_loss)

        stats.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_mae': val_mae})

        print(f"第 {epoch:02d}/{EPOCHS} 回合 | {args.milling_type} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | val_mae: {val_mae:.4f} μm")

        import json
        print(json.dumps({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_mae': val_mae}))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            # 記錄這個版本的驗證成績與保留的驗證刀，批量驗證頁靠它判斷哪些影像是模型沒看過的
            write_meta(args.milling_type, {
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_mae': val_mae,
                'train_images': len(train_df),
                'val_images': len(val_df),
                'val_conditions': [f"{args.milling_type}/{c}" for c in val_conditions],
                'preprocess': PREPROCESS_VERSION,
            })
            print(f"  👉 已儲存最佳 {args.milling_type} 模型！")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= PATIENCE:
            print(f"已連續 {PATIENCE} 個 epoch 未進步，提前停止訓練。")
            break

    loss_df = pd.DataFrame(stats)
    loss_df.to_csv(LOSS_CSV_PATH, index=False)
    print(f"🎉 {args.milling_type} 訓練完成！模型已儲存至：{BEST_MODEL_PATH}")

if __name__ == '__main__':
    main()
