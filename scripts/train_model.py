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
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import functools

print = functools.partial(print, flush=True)

# ------------------------------------------
# 參數設定
# ------------------------------------------
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)
VALIDATION_SPLIT = 0.2
PATIENCE = 40
LR = 1e-4
EPOCHS = 200

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path

BASE_DIR = str_path()
CSV_PATH = str_path('data', 'final_training_manifest.csv')
RESULTS_DIR = str_path('results')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# 1. 定義資料讀取器
# ==========================================
class SurfaceDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, is_train: bool = True):
        self.data_info = data_frame.reset_index(drop=True)
        self.is_train = is_train

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.data_info) * 2 if self.is_train else len(self.data_info)

    def __getitem__(self, idx):
        real_idx = idx % len(self.data_info)
        row = self.data_info.iloc[real_idx]
        img_path = row['image_path']

        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError(f"無法解碼圖片，檔案可能損壞: {img_path}")

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            # 💡 核心修正：將陣列轉成 PIL 圖片後，立刻轉灰階 (L) 去除色彩，再轉回三通道 (RGB)
            img_pil = Image.fromarray(img_rgb).convert('L').convert('RGB')
        except Exception as e:
            raise IOError(f"讀取圖片時發生錯誤 {img_path}: {str(e)}")

        transform = self.train_transform if self.is_train else self.eval_transform
        img_tensor = transform(img_pil)

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
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, params, targets in dataloader:
            imgs = imgs.to(device)
            params = params.to(device)
            targets = targets.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(imgs, params)
                loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

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
    BEST_MODEL_PATH = os.path.join(RESULTS_DIR, f'best_model_{args.milling_type}.pth')
    LOSS_CSV_PATH = os.path.join(RESULTS_DIR, f'loss_record_{args.milling_type}.csv')

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 檔案：{CSV_PATH}")

    data_df = pd.read_csv(CSV_PATH)

    # 💡 核心過濾：只挑選符合當前 milling_type 的資料來訓練！
    data_df = data_df[data_df['machining_type'] == args.milling_type].copy()

    if data_df.empty:
        raise ValueError(f"❌ 在 CSV 中找不到任何 {args.milling_type} 的資料，請檢查清單！")

    print(f"📂 成功載入 {len(data_df)} 筆 {args.milling_type} 影像資料！")

    data_df = data_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = max(1, int(len(data_df) * VALIDATION_SPLIT))
    train_df = data_df.iloc[val_size:].reset_index(drop=True)
    val_df = data_df.iloc[:val_size].reset_index(drop=True)

    train_dataset = SurfaceDataset(train_df, is_train=True)
    val_dataset = SurfaceDataset(val_df, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))

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
        avg_val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(avg_val_loss)

        stats.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

        print(f"第 {epoch:02d}/{EPOCHS} 回合 | {args.milling_type} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        import json
        print(json.dumps({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
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
