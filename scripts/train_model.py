import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision.models as models
# 💡 【新增】載入 PyTorch 強大的影像擴增工具與 PIL 圖片處理套件
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# 1. 定義資料讀取器 (加入影像擴增 Data Augmentation)
# ==========================================
class SurfaceDataset(Dataset):
    def __init__(self, csv_path):
        self.data_info = pd.read_csv(csv_path)
        
        # 💡 【優化核心】：設定影像擴增的隨機變化規則
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5), # 50% 機率左右翻轉
            transforms.RandomVerticalFlip(p=0.5),   # 50% 機率上下翻轉
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # 隨機微調亮度和對比度，模擬反光
            transforms.ToTensor() # 自動將數值縮放至 0~1 並轉成 PyTorch 張量格式
        ])
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        img_path = row['image_path']
        
        try:
            # 相容 Windows 中文路徑的讀取法
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError(f"無法解碼圖片，檔案可能損壞: {img_path}")
                
            # 將 OpenCV 的 BGR 色彩格式轉成標準的 RGB，並轉成 PIL 圖片格式交給擴增工具
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        except Exception as e:
            raise IOError(f"讀取圖片時發生錯誤 {img_path}: {str(e)}")
            
        # 套用影像擴增 (每次讀取同一張圖，都會長得稍微不一樣！)
        img_tensor = self.transform(img_pil)
        
        # 切削參數縮放
        speed = float(row['speed']) / 10000.0  
        cond = float(row['condition_id']) / 10.0
        params = np.array([speed, cond], dtype=np.float32)
        
        # 標準答案 Ra 值
        ra_target = np.array([row['ra_target']], dtype=np.float32)
        
        # 注意：img_tensor 已經是 torch.tensor 了，所以不用再包一層
        return img_tensor, torch.tensor(params), torch.tensor(ra_target)

# ==========================================
# 2. 雙輸入 AI 模型 (ResNet-18 全局微調)
# ==========================================
class ResNetDualInputModel(nn.Module):
    def __init__(self):
        super(ResNetDualInputModel, self).__init__()
        
        # 載入預訓練的 ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 💡 【優化核心】：解除了原本凍結權重的程式碼
        # 現在 ResNet 的每一層都會跟著你的切削刀痕一起進化 (Fine-tuning)！
            
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.ReLU()
        )
        
        self.dnn = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img, params):
        img_features = self.resnet(img) 
        param_features = self.dnn(params)
        combined = torch.cat((img_features, param_features), dim=1)
        output = self.fc(combined)
        return output

# ==========================================
# 3. 主程式：開始訓練
# ==========================================
if __name__ == '__main__':
    print("準備載入資料與模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 目前使用的運算設備：{device}")
    
    csv_file = r"C:\Users\tony9\Desktop\5000 - BETTER\data\final_training_manifest.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ 錯誤：找不到 CSV 檔案！")
        exit()
        
    dataset = SurfaceDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    model = ResNetDualInputModel().to(device)
    criterion = nn.MSELoss()
    
    # 💡 【優化核心】：因為解凍了神經網路，學習率必須調小到 1e-4，避免破壞預訓練的視覺大腦
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 💡 【優化核心】：將訓練回合數增加到 40，讓模型有足夠時間看過擴增後的各種翻轉圖片
    epochs = 40 
    loss_history = []
    
    print(f"開始進行 {epochs} 回合的全局微調訓練！(Data Augmentation Enabled)\n" + "-"*50)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_imgs, batch_params, batch_targets in dataloader:
            batch_imgs = batch_imgs.to(device)
            batch_params = batch_params.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()                     
            predictions = model(batch_imgs, batch_params) 
            loss = criterion(predictions, batch_targets)  
            
            loss.backward()                           
            optimizer.step()                          
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"第 {epoch+1:02d}/{epochs} 回合完成 | 平均誤差 (Loss): {avg_loss:.4f}")
        
    print("-" * 50)
    print("訓練完成！")
    
    model_save_path = r"C:\Users\tony9\Desktop\5000 - BETTER\results\best_surface_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 進化版模型大腦已成功儲存至：{model_save_path}")
    
    loss_df = pd.DataFrame({"Epoch": range(1, epochs+1), "Loss": loss_history})
    loss_csv_path = r"C:\Users\tony9\Desktop\5000 - BETTER\results\loss_record.csv"
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"📊 Loss 歷史紀錄已儲存至：{loss_csv_path}")