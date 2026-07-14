import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms

from project_root import path, str_path

BASE_DIR = path()
MODEL_PATH = str_path('results', 'best_surface_model.pth')
IMG_SIZE = 224
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

class ResNetDualInputModel(nn.Module):
    def __init__(self):
        super(ResNetDualInputModel, self).__init__()
        self.resnet = models.resnet18()

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


def preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到圖片檔案 {img_path}")

    img_array = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"圖片無法讀取或已損壞 {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.tensor(img)
    return NORMALIZE(img_tensor)


def load_model(model_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetDualInputModel().to(device)
    if model_path is None:
        model_path = MODEL_PATH

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_images(img_paths, speed_rpms, condition_ids, model=None, device=None, batch_size=16):
    if len(img_paths) != len(speed_rpms) or len(img_paths) != len(condition_ids):
        raise ValueError("img_paths, speed_rpms 和 condition_ids 長度必須相同。")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model or load_model(device=device)

    predictions = []
    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]
        batch_speeds = speed_rpms[start:start + batch_size]
        batch_conds = condition_ids[start:start + batch_size]

        img_tensors = [preprocess_image(path) for path in batch_paths]
        img_batch = torch.stack(img_tensors).to(device)

        params = [
            [float(speed) / 10000.0, float(cond) / 10.0]
            for speed, cond in zip(batch_speeds, batch_conds)
        ]
        param_tensor = torch.tensor(params, dtype=torch.float32).to(device)

        with torch.no_grad():
            batch_preds = model(img_batch, param_tensor).squeeze(1).cpu().tolist()
            predictions.extend(batch_preds)

    return predictions


def predict_single_image(img_path, speed_rpm, condition_id, model=None, device=None):
    return predict_images(
        [img_path], [speed_rpm], [condition_id], model=model, device=device, batch_size=1
    )[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict CNC surface roughness from one image')
    parser.add_argument('--image', type=str, default=str_path('data', '5000-0', 'pc', '20260630164028698.jpg'),
                        help='image file path to predict')
    parser.add_argument('--speed', type=float, default=5000.0, help='spindle speed in RPM')
    parser.add_argument('--cond', type=float, default=0.0, help='condition id')
    args = parser.parse_args()

    print("[進度 3/4] 🔍 進入主執行區，準備讀取測試影像...")
    print("=" * 50)
    print("🚀 加工表面粗糙度 AI 預測系統 (ResNet-18 升級版) 🚀")
    print("=" * 50)

    test_image = args.image
    test_speed = args.speed
    test_cond = args.cond

    model = load_model(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    prediction = predict_single_image(test_image, test_speed, test_cond, model=model)

    if prediction is not None:
        print(f"✨ [進度 4/4] 【AI 預測成功】")
        print(f"🔮 預測表面粗糙度 (Ra) 為: {prediction:.4f}")
        print("📌 （真實量測值為: 1.354，來看看 ResNet 猜得有沒有比之前準！）")
    print("=" * 50)