import os
import random
import inspect
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ------------------------------------------
# 參數設定
# ------------------------------------------
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = min(4, os.cpu_count() or 1)
VALIDATION_SPLIT = 0.2
PATIENCE = 5
LR = 1e-4
EPOCHS = 40

BASE_DIR = r"C:\Users\tony9\Desktop\5000 - BETTER"
CSV_PATH = os.path.join(BASE_DIR, "data", "final_training_manifest.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_surface_model.pth")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "best_surface_checkpoint.pth")
LOSS_CSV_PATH = os.path.join(RESULTS_DIR, "loss_record.csv")

# ImageNet normalization for ResNet pretrained backbone
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
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        img_path = row['image_path']

        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError(f"無法解碼圖片，檔案可能損壞: {img_path}")

            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        except Exception as e:
            raise IOError(f"讀取圖片時發生錯誤 {img_path}: {str(e)}")

        transform = self.train_transform if self.is_train else self.eval_transform
        img_tensor = transform(img_pil)

        speed = float(row['speed']) / 10000.0
        cond = float(row['condition_id']) / 10.0
        params = np.array([speed, cond], dtype=np.float32)
        ra_target = np.array([row['ra_target']], dtype=np.float32)

        return img_tensor, torch.tensor(params), torch.tensor(ra_target)

# ==========================================
# 2. 雙輸入 AI 模型 (ResNet-18 全局微調)
# ==========================================
class ResNetDualInputModel(nn.Module):
    def __init__(self):
        super(ResNetDualInputModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

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
# 3. 工具函式
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
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                preds = model(imgs, params)
                loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    print("準備載入資料與模型...")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 目前使用的運算設備：{device}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到 CSV 檔案：{CSV_PATH}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data_df = pd.read_csv(CSV_PATH)
    if data_df.empty:
        raise ValueError("訓練清單為空，請確認 final_training_manifest.csv 是否包含資料。")

    data_df = data_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = max(1, int(len(data_df) * VALIDATION_SPLIT))
    train_df = data_df.iloc[val_size:].reset_index(drop=True)
    val_df = data_df.iloc[:val_size].reset_index(drop=True)

    train_dataset = SurfaceDataset(train_df, is_train=True)
    val_dataset = SurfaceDataset(val_df, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = ResNetDualInputModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler_kwargs = {
        'mode': 'min',
        'patience': 3,
        'factor': 0.5,
    }
    if 'verbose' in inspect.signature(optim.lr_scheduler.ReduceLROnPlateau).parameters:
        scheduler_kwargs['verbose'] = True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    epochs_without_improve = 0
    stats = []

    torch.backends.cudnn.benchmark = True

    print(f"開始進行 {EPOCHS} 回合的全局微調訓練！(Data Augmentation + AMP Enabled)\n" + "-" * 50)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch_imgs, batch_params, batch_targets in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_params = batch_params.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                predictions = model(batch_imgs, batch_params)
                loss = criterion(predictions, batch_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(avg_val_loss)

        stats.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        })

        print(
            f"第 {epoch:02d}/{EPOCHS} 回合完成 | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                },
                CHECKPOINT_PATH,
            )
            print(f"💾 已儲存最佳模型與檢查點：{BEST_MODEL_PATH}")
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= PATIENCE:
            print(
                f"⏱️ 已連續 {PATIENCE} 個 epoch 未進步，提前停止訓練。"
            )
            break

    loss_df = pd.DataFrame(stats)
    loss_df.to_csv(LOSS_CSV_PATH, index=False)
    print(f"📊 Loss 歷史紀錄已儲存至：{LOSS_CSV_PATH}")
    print("訓練完成！")


if __name__ == '__main__':
    main()