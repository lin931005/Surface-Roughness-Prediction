import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("\n[進度 1/4] 🚀 啟動 Python，載入核心 AI 套件...")

import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.models as models

print("[進度 2/4] 🎉 套件載入完成！正在設定 ResNet-18 模型架構...")

# ==========================================
# 必須與訓練時完全一模一樣的 ResNet 架構
# ==========================================
class ResNetDualInputModel(nn.Module):
    def __init__(self):
        super(ResNetDualInputModel, self).__init__()
        
        # 這裡不需要下載權重了，因為等一下會直接讀取你訓練好的 .pth
        self.resnet = models.resnet18()
        
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

def predict_single_image(img_path, speed_rpm, condition_id):
    if not os.path.exists(img_path):
        print(f"❌ 錯誤：找不到圖片檔案 {img_path}")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 這裡改成新的模型名稱
    model = ResNetDualInputModel().to(device)
    model_path = r"C:\Users\tony9\Desktop\5000\best_surface_model.pth"
    
    # 載入你剛剛訓練好的 ResNet 權重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 影像前處理
    img_array = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 錯誤：圖片無法讀取 {img_path}")
        return None
        
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) 
    img_tensor = torch.tensor(img).unsqueeze(0).to(device) 
    
    # 參數前處理
    speed_scaled = float(speed_rpm) / 10000.0
    cond_scaled = float(condition_id) / 10.0
    param_tensor = torch.tensor([[speed_scaled, cond_scaled]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predicted_ra = model(img_tensor, param_tensor)
        
    return predicted_ra.item()

if __name__ == '__main__':
    print("[進度 3/4] 🔍 進入主執行區，準備讀取測試影像...")
    print("="*50)
    print("🚀 加工表面粗糙度 AI 預測系統 (ResNet-18 升級版) 🚀")
    print("="*50)
    
    test_image = r"C:\Users\tony9\Desktop\5000\5000-0\pc\20260630164028698.jpg"
    test_speed = 5000
    test_cond = 0
    
    print(f"▶ 測試圖片：{os.path.basename(test_image)}")
    print(f"▶ 加工條件：轉速 {test_speed} RPM, 條件編號 {test_cond}\n")
    
    prediction = predict_single_image(test_image, test_speed, test_cond)
    
    if prediction is not None:
        print(f"✨ [進度 4/4] 【AI 預測成功】")
        print(f"🔮 預測表面粗糙度 (Ra) 為: {prediction:.4f}")
        print(f"📌 （真實量測值為: 1.354，來看看 ResNet 猜得有沒有比之前準！）")
    print("="*50)