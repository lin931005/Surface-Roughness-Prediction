from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import sys
import io
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import os
import subprocess
import time
import cv2
import pandas as pd
import psutil
import base64
import numpy as np
import matplotlib
import random
import asyncio
import gc
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .auth import admin_auth, verify_credentials, create_token

app = FastAPI()

# ==========================================
# 🔍 1. 模型路徑與載入設定 (雙專家 + 分類器)
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_END_PATH = os.path.join(MODELS_DIR, 'best_model_End_Milling.pth')
MODEL_PERI_PATH = os.path.join(MODELS_DIR, 'best_model_Peripheral_Milling.pth')
CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'best_classifier.pth')

# ==========================================
# 🧠 2. 智慧動態記憶體管理 (動態加載/卸載)
# ==========================================
expert_models = {"End_Milling": None, "Peripheral_Milling": None}
classifier_model = None

# 全域狀態與計時器
models_are_loaded = False
last_active_time = time.time()
IDLE_TIMEOUT_SECONDS = 600  # 閒置 10 分鐘 (600 秒) 後自動卸載

class ResNetDualInputModel(nn.Module):
    def __init__(self):
        super(ResNetDualInputModel, self).__init__()
        self.resnet = models.resnet50(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs, 64), nn.ReLU())
        self.dnn = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64 + 16, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, img, params):
        img_features = self.resnet(img)
        param_features = self.dnn(params)
        combined = torch.cat((img_features, param_features), dim=1)
        return self.fc(combined)

class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        # 💡 這裡也要改成 3！
        self.resnet.fc = nn.Linear(num_ftrs, 3)

    def forward(self, img):
        return self.resnet(img)

def load_models_on_demand():
    """需要預測時才掛載模型"""
    global expert_models, classifier_model, models_are_loaded
    if models_are_loaded:
        return

    print("⏳ 偵測到模型尚未載入，正在將大腦掛載至 GPU 記憶體...")
    if os.path.exists(MODEL_END_PATH):
        model = ResNetDualInputModel()
        model.load_state_dict(torch.load(MODEL_END_PATH, map_location='cpu'))
        model.eval()
        expert_models["End_Milling"] = model

    if os.path.exists(MODEL_PERI_PATH):
        model = ResNetDualInputModel()
        model.load_state_dict(torch.load(MODEL_PERI_PATH, map_location='cpu'))
        model.eval()
        expert_models["Peripheral_Milling"] = model

    if os.path.exists(CLASSIFIER_PATH):
        model = ClassifierModel()
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location='cpu'))
        model.eval()
        classifier_model = model

    models_are_loaded = True
    print("✅ 模型掛載完成，系統已進入戰鬥狀態！")

def unload_models_to_free_vram():
    """徹底卸載模型並清空 GPU 記憶體"""
    global expert_models, classifier_model, models_are_loaded
    if not models_are_loaded:
        return

    print("💤 系統閒置或即將進行訓練，正在卸載模型並釋放 GPU 資源...")
    expert_models["End_Milling"] = None
    expert_models["Peripheral_Milling"] = None
    classifier_model = None
    models_are_loaded = False

    # 強制執行垃圾回收與 CUDA 記憶體清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 GPU 記憶體已清空！")

# 背景計時器：每 60 秒檢查一次是否超時 10 分鐘
async def idle_timeout_checker():
    global last_active_time
    while True:
        await asyncio.sleep(60)
        if models_are_loaded and (time.time() - last_active_time > IDLE_TIMEOUT_SECONDS):
            unload_models_to_free_vram()

@app.on_event("startup")
async def startup_event():
    # 伺服器啟動時，開始跑背景的閒置計時器 (此時不載入模型)
    asyncio.create_task(idle_timeout_checker())
    print("🚀 API 伺服器已啟動，閒置卸載監控已開啟 (10分鐘超時)。")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==========================================
# 🚀 3. 預測核心 API
# ==========================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    gradcam: bool = Query(False),
    speed: float = Query(None),
    milling_type: str = Query("Auto")
):
    global last_active_time
    last_active_time = time.time()  # 💡 有人呼叫預測，重置 10 分鐘計時器！

    # 💡 確保模型有掛載
    load_models_on_demand()

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 💡 1. 決定最終要用的 milling_type (加入 AI 自信度偵測)
    img_bw = img.convert('L').convert('RGB')

    ai_confidence = 1.0      # 初始化自信度
    ai_is_confused = False   # 初始化困惑狀態

    final_milling_type = milling_type
    if milling_type == "Auto":
        if classifier_model is not None:
            classifier_model.to(device)
            with torch.no_grad():
                img_tensor = transform(img_bw).unsqueeze(0).to(device)
                out = classifier_model(img_tensor)

                probs = torch.nn.functional.softmax(out, dim=1)
                max_prob = torch.max(probs).item()
                pred_class = torch.argmax(out, dim=1).item()

                # 🌟 攔截邏輯升級：
                if pred_class == 2:
                    # 如果 AI 判定這是一張「Other (垃圾桶)」的照片
                    final_milling_type = "End_Milling" # 給個預設值防呆
                    ai_confidence = 0.0                # 直接把自信度無情歸零！
                    ai_is_confused = True              # 觸發異常警告
                else:
                    # 如果是正常的立銑 (0) 或 直銑 (1)
                    final_milling_type = "Peripheral_Milling" if pred_class == 1 else "End_Milling"
                    ai_confidence = max_prob

                    # 依然保留 85% 的防線，防止 AI 遇到模糊金屬時亂猜
                    if max_prob < 0.85:
                        ai_is_confused = True
        else:
            final_milling_type = "End_Milling"

    target_model = expert_models.get(final_milling_type)
    if target_model is None:
        return JSONResponse({"error": f"尚未載入 {final_milling_type} 的模型，請先訓練！"}, status_code=500)

    target_model.to(device)

    used_default = False
    if speed is None:
        speed = 5000.0
        used_default = True
    dummy_condition = 0.0

    img_color_np = np.array(img)
    img_cv = cv2.cvtColor(img_color_np, cv2.COLOR_RGB2BGR)
    img_bw = img.convert('L').convert('RGB')
    img_np = np.array(img_bw)
    h, w, _ = img_np.shape

    num_patches = 32
    crop_h, crop_w = int(h * 0.8), int(w * 0.8)
    patches, patch_coords = [], []

    for _ in range(num_patches):
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        patch_coords.append({"top": top, "left": left, "bottom": top + crop_h, "right": left + crop_w})
        patch_arr = img_np[top:top+crop_h, left:left+crop_w]
        patches.append(Image.fromarray(patch_arr))

    batch_tensors = torch.stack([transform(p) for p in patches]).to(device)
    params_tensor = torch.tensor([[speed / 10000.0, dummy_condition / 10.0]] * num_patches, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = target_model(batch_tensors, params_tensor).cpu().numpy().flatten()

    # 🌟 終極防禦：全面信任三元分類大腦的「Other 攔截」與「85% 自信度門檻」
    is_anomaly = bool(ai_is_confused)

    # 保留這兩行只是為了餵給前端，避免報錯
    color_std_score = 0.0
    edge_score = 0.0

    preds_with_coords = list(zip(preds, patch_coords))
    preds_sorted_with_coords = sorted(preds_with_coords, key=lambda x: x[0])
    preds_sorted = [x[0] for x in preds_sorted_with_coords]

    trim_count = int(num_patches * 0.15)
    valid_preds = preds_sorted[trim_count:-trim_count] if trim_count > 0 else preds_sorted
    final_ra = float(np.mean(valid_preds))

    detailed_patches = []
    for i, (val, coords) in enumerate(preds_sorted_with_coords):
        if i < trim_count:
            status = "剔除 (異常低值)"
        elif i >= len(preds_sorted_with_coords) - trim_count:
            status = "剔除 (異常高值/可能含灰塵)"
        else:
            status = "保留 (有效計算區間)"
        detailed_patches.append({"id": i + 1, "ra": float(val), "status": status, "coords": coords})

    result = {
        "ra": final_ra,
        "used_default_params": used_default,
        "is_anomaly": is_anomaly,
        "preds_std": color_std_score,
        "preds_edge": edge_score,
        "ai_confidence": ai_confidence,  # 💡 修改點 2：把 AI 的自信度數據打包傳給網頁
        "detected_milling": final_milling_type,
        "xai_details": {
            "num_patches": num_patches,
            "trim_count": trim_count,
            "patches_info": detailed_patches
        }
    }

    try:
        fn = getattr(file, 'filename', f'upload_{int(time.time())}')
        log_prediction(fn, final_ra)
    except Exception:
        pass

    return result

# ==========================================
# 🛠️ 4. 訓練 API 升級 (附帶強制卸載防護)
# ==========================================
def run_training_script(milling_type: str):
    """這支函式會在背景獨立執行"""
    unload_models_to_free_vram()

    try:
        from datetime import datetime
        python_exe = sys.executable

        script_dataset = os.path.join(BASE_DIR, "scripts", "dataset_prepare.py")

        # 💡 核心修改：判斷要呼叫哪一支訓練腳本
        if milling_type == "Classifier":
            script_train = os.path.join(BASE_DIR, "scripts", "train_classifier.py")
        else:
            script_train = os.path.join(BASE_DIR, "scripts", "train_model.py")

        log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
        os.makedirs(log_dir, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(log_dir, f'train_{milling_type}_{time_str}.log')

        custom_env = os.environ.copy()
        custom_env["PYTHONIOENCODING"] = "utf-8"

        with open(logfile, 'wb') as f:
            f.write(f"🚀 開始為【{milling_type}】準備資料庫...\n".encode('utf-8'))
            subprocess.run([python_exe, script_dataset], cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT, env=custom_env)

            f.write(f"\n🚀 啟動【{milling_type}】神經網路訓練...\n".encode('utf-8'))

            # 💡 核心修改：分類器不需要後面的參數，專家大腦才需要
            if milling_type == "Classifier":
                subprocess.Popen([python_exe, script_train], cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT, env=custom_env)
            else:
                subprocess.Popen([python_exe, script_train, "--milling_type", milling_type], cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT, env=custom_env)

    except Exception as e:
        print(f"❌ [背景任務] 訓練發生錯誤: {str(e)}")

@app.post("/train")
async def start_training(milling_type: str = Query(...), background_tasks: BackgroundTasks = BackgroundTasks(), user: str = Depends(admin_auth)):
    if milling_type not in ["End_Milling", "Peripheral_Milling", "Classifier"]:
        return JSONResponse({"error": "未知的訓練類型"}, status_code=400)

    background_tasks.add_task(run_training_script, milling_type)
    return {"message": f"✅ 【{milling_type}】訓練排程已在背景啟動！已自動釋放記憶體，請至 Train Logs 查看進度。"}

def log_prediction(filename: str, ra: float):
    try:
        log_file = os.path.join(BASE_DIR, 'results', 'predictions.csv')
        header = not os.path.exists(log_file)
        import csv
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if header: writer.writerow(['timestamp', 'file', 'ra'])
            writer.writerow([int(time.time()), filename, ra])
    except Exception:
        pass

@app.post('/login')
async def login(username: str = Query(...), password: str = Query(...)):
    if verify_credentials(username, password): return {"access_token": create_token(username)}
    return JSONResponse({"error": "invalid credentials"}, status_code=401)

@app.get('/train_logs')
async def list_train_logs(user: str = Depends(admin_auth)):
    log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
    return {"logs": sorted(os.listdir(log_dir), reverse=True)} if os.path.exists(log_dir) else {"logs": []}

@app.get('/train_logs/{name}')
async def get_train_log(name: str, user: str = Depends(admin_auth)):
    path = os.path.join(BASE_DIR, 'results', 'train_logs', name)
    if not os.path.exists(path): return JSONResponse({"error": "not found"}, status_code=404)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f: return {"log": f.read()}

# 💡 補回來的折線圖讀取 API
@app.get('/train_progress/{name}')
async def train_progress(name: str, user: str = Depends(admin_auth)):
    log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
    path = os.path.join(log_dir, name)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    entries = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: entries.append(pd.read_json(pd.io.common.StringIO(line), typ='series').to_dict())
                except Exception:
                    try:
                        import json
                        entries.append(json.loads(line))
                    except Exception: continue
    except Exception:
        return JSONResponse({"error": "read error"}, status_code=500)
    return {"progress": entries}

@app.get('/admin/stats')
async def admin_stats(user: str = Depends(admin_auth)):
    return {"cpu": psutil.cpu_percent(interval=0.5), "mem": psutil.virtual_memory()._asdict(), "gpu": {'available': torch.cuda.is_available()}}

if __name__ == '__main__':
    uvicorn.run('webapp.app.main:app', host='0.0.0.0', port=2578, reload=False)
