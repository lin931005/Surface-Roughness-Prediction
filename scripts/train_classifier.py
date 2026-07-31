import os
import sys
import random
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
import functools

print = functools.partial(print, flush=True)

# ------------------------------------------
# 參數設定
# ------------------------------------------
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 4)
VALIDATION_SPLIT = 0.2
PATIENCE = 30
LR = 1e-4
EPOCHS = 100

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_root import str_path

CSV_PATH = str_path('data', 'final_training_manifest.csv')
RESULTS_DIR = str_path('results')
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, 'best_classifier.pth')
LOSS_CSV_PATH = os.path.join(RESULTS_DIR, 'loss_record_Classifier.csv')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==========================================
# 1. 視覺分類專屬資料讀取器 (0: 立銑, 1: 直銑)
# ==========================================
class ClassifierDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        # 將字串轉換為數字標籤
        self.label_map = {"End_Milling": 0, "Peripheral_Milling": 1}

    def __len__(self):
        return len(self.df) * 2 if self.is_train else len(self.df)

    def __getitem__(self, idx):
        real_idx = idx % len(self.df)
        row = self.df.iloc[real_idx]
        img_path = row['image_path']
        label = self.label_map[row['machining_type']]

        try:
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).convert('L').convert('RGB')
        except Exception as e:
            raise IOError(f"讀取圖片發生錯誤 {img_path}: {str(e)}")

        transform = self.train_transform if self.is_train else self.eval_transform
        return transform(img_pil), torch.tensor(label, dtype=torch.long)

# ==========================================
# 2. 輕量級視覺大腦 (ResNet-18)
# ==========================================
class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        # 輸出 2 個神經元 (立銑與直銑的機率)
        self.resnet.fc = nn.Linear(num_ftrs, 2)

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
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(imgs)
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

    # 洗牌並切割訓練與驗證集
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = max(1, int(len(df) * VALIDATION_SPLIT))
    train_df = df.iloc[val_size:].reset_index(drop=True)
    val_df = df.iloc[:val_size].reset_index(drop=True)

    print(f"📂 成功載入 {len(df)} 筆影像資料！(包含立銑與直銑)")

    train_loader = DataLoader(ClassifierDataset(train_df, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ClassifierDataset(val_df, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ClassifierModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    epochs_without_improve = 0
    stats = []

    print(f"🔥 開始進行 {EPOCHS} 回合的二元分類訓練！\n" + "-" * 50)
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
