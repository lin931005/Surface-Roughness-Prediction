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
import zipfile
import time
import pandas as pd
import psutil
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .auth import admin_auth, verify_credentials, create_token

app = FastAPI()

# ==========================================
# 🔍 1. 模型路徑與載入設定
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

ACTIVE_MODEL_LINK = os.path.join(BASE_DIR, 'results', 'current_model.pth')
MODEL_PATH = ACTIVE_MODEL_LINK if os.path.exists(ACTIVE_MODEL_LINK) else os.path.join(BASE_DIR, 'results', 'best_surface_model.pth')

# ==========================================
# 🧠 2. 定義雙輸入神經網路骨架 (ResNet18 + DNN)
# ==========================================
class ResNetDualInputModel(nn.Module):
    def __init__(self):
        super(ResNetDualInputModel, self).__init__()
        self.resnet = models.resnet18(weights=None)

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

model = None

def load_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                print("▶️ 正在建立雙輸入 ResNet-18 骨架...")
                model = ResNetDualInputModel()

                print("▶️ 正在載入模型權重...")
                weights = torch.load(MODEL_PATH, map_location='cpu')

                if isinstance(weights, dict) and 'state_dict' in weights:
                    model.load_state_dict(weights['state_dict'])
                else:
                    model.load_state_dict(weights)

                model.eval()
                print("✅ AI 雙輸入模型載入成功，大腦已上線！\n")
            except Exception as e:
                print(f"❌ 模型載入發生異常：{str(e)}\n")
                model = None
        else:
            print("❌ 找不到模型檔案！")
            model = None

load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==========================================
# 🚀 3. 預測核心 API (支援選填參數)
# ==========================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    gradcam: bool = Query(False),
    speed: float = Query(None)  # 移除 condition，只接收 speed
):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    if model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=500)

    x = transform(img).unsqueeze(0)

    # 處理「選填」的機台參數
    used_default = False
    if speed is None:
        speed = 5000.0  # 如果沒填，給予一個預設平均轉速
        used_default = True

    # 💡 核心工程技巧：
    # 雖然前端不傳切削條件了，但 PyTorch 雙輸入模型當初訓練時是吃 2 個維度。
    # 所以我們在這裡手動補上一個「0.0」作為 dummy 參數，避免維度錯誤崩潰。
    dummy_condition = 0.0
    params_tensor = torch.tensor([[speed / 10000.0, dummy_condition / 10.0]], dtype=torch.float32)

    x = x.requires_grad_(True)
    out = model(x, params_tensor)

    if isinstance(out, torch.Tensor):
        try:
            val = float(out.item())
        except Exception:
            val = float(out.detach().cpu().numpy().tolist())
    else:
        val = float(out)

    result = {
        "ra": val,
        "used_default_params": used_default
    }

    if gradcam:
        try:
            heatmap_b64 = compute_gradcam(model, x, params_tensor, img)
            result['heatmap'] = heatmap_b64
        except Exception as e:
            result['heatmap_error'] = str(e)

    try:
        fn = getattr(file, 'filename', f'upload_{int(time.time())}')
        log_prediction(fn, float(val))
    except Exception:
        pass

    return result

# ==========================================
# 🎨 4. Grad-CAM 運算與其他工具
# ==========================================
def find_last_conv(module):
    last = None
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def compute_gradcam(model, input_tensor, params_tensor, orig_img, target_layer=None):
    model.zero_grad()
    if target_layer is None:
        target_layer = find_last_conv(model)
    if target_layer is None:
        raise RuntimeError('No conv layer found for Grad-CAM')

    activations = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    # 必須同時傳入影像與參數
    output = model(input_tensor, params_tensor)
    if isinstance(output, torch.Tensor):
        score = output.squeeze()
        if score.dim() > 0:
            score = score.mean()
    else:
        score = torch.tensor(float(output))

    score.backward(retain_graph=True)
    fh.remove()
    bh.remove()

    if activations is None or gradients is None:
        raise RuntimeError('Failed to get activations or gradients')

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = torch.nn.functional.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    cam_img = np.uint8(255 * cam)
    cam_img = Image.fromarray(cam_img).resize(orig_img.size, resample=Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0

    cmap = plt.get_cmap('jet')
    colored = cmap(cam_arr)[:, :, :3]
    colored = np.uint8(255 * colored)
    orig_arr = np.array(orig_img).astype(np.uint8)
    overlay = (0.5 * orig_arr + 0.5 * colored).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"

def log_prediction(filename: str, ra: float):
    try:
        log_file = os.path.join(BASE_DIR, 'results', 'predictions.csv')
        header = not os.path.exists(log_file)
        import csv
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(['timestamp', 'file', 'ra'])
            writer.writerow([int(time.time()), filename, ra])
    except Exception:
        pass

# ==========================================
# 其餘 MLOps API (未更動，為節省版面省略實作內部，但請保留你原本後面的那些 router)
# ==========================================
@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, token: str = Query(None), user: str = Depends(admin_auth)):
    def run_train():
        try:
            import sys
            from datetime import datetime # 引入日期套件

            train_script = os.path.join(BASE_DIR, 'scripts', 'train_model.py')
            csv_script = os.path.join(BASE_DIR, 'scripts', 'dataset_prepare.py')

            log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
            os.makedirs(log_dir, exist_ok=True)

            # 💡 核心修改：改用 YYYYMMDD_HHMMSS 作為檔名，告別醜陋的數字串！
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            logfile = os.path.join(log_dir, f'train_{time_str}.log')
            out_model = os.path.join(MODELS_DIR, f'model_{time_str}.pth')

            custom_env = os.environ.copy()
            custom_env["PYTHONIOENCODING"] = "utf-8"

            if os.path.exists(train_script):
                print(f"🚀 開始執行背景任務，日誌將輸出至: {logfile}")
                with open(logfile, 'wb') as f:
                    if os.path.exists(csv_script):
                        f.write("🔄 正在執行自動化資料管線 (重新生成 CSV)...\n".encode('utf-8'))
                        subprocess.run([sys.executable, csv_script], cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT, env=custom_env)
                        f.write("\n".encode('utf-8'))

                    f.write("🚀 開始執行神經網路模型訓練...\n".encode('utf-8'))
                    subprocess.Popen(
                        [sys.executable, train_script, "--output", out_model, "--log", logfile],
                        cwd=BASE_DIR,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        env=custom_env
                    )
            else:
                print(f"❌ 找不到訓練腳本：{train_script}")
        except Exception as e:
            print(f"❌ 背景任務啟動失敗：{str(e)}")

    background_tasks.add_task(run_train)
    return {"status": "retraining started, CSV will be auto-generated"}

@app.post('/login')
async def login(username: str = Query(...), password: str = Query(...)):
    if verify_credentials(username, password):
        tok = create_token(username)
        return {"access_token": tok}
    return JSONResponse({"error": "invalid credentials"}, status_code=401)

@app.get('/train_logs')
async def list_train_logs(user: str = Depends(admin_auth)):
    log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
    if not os.path.exists(log_dir):
        return {"logs": []}
    files = sorted(os.listdir(log_dir), reverse=True)
    return {"logs": files}

@app.get('/train_logs/{name}')
async def get_train_log(name: str, user: str = Depends(admin_auth)):
    log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
    path = os.path.join(log_dir, name)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return {"log": f.read()}

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
                if not line:
                    continue
                try:
                    entries.append(pd.read_json(pd.io.common.StringIO(line), typ='series').to_dict())
                except Exception:
                    try:
                        import json
                        entries.append(json.loads(line))
                    except Exception:
                        continue
    except Exception:
        return JSONResponse({"error": "read error"}, status_code=500)
    return {"progress": entries}

@app.get('/models')
async def list_models():
    files = sorted(os.listdir(MODELS_DIR), reverse=True)
    models = []
    for fn in files:
        if not fn.lower().endswith('.pth'):
            continue
        p = os.path.join(MODELS_DIR, fn)
        stat = os.path.getmtime(p)
        models.append({"file": fn, "mtime": stat})
    return {"models": models}

@app.get('/predictions/stats')
async def prediction_stats(user: str = Depends(admin_auth)):
    log_file = os.path.join(BASE_DIR, 'results', 'predictions.csv')
    if not os.path.exists(log_file):
        return {"total": 0, "last": []}
    import csv
    rows = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception:
        return {"total": 0, "last": []}
    return {"total": len(rows), "last": rows[-20:]}

@app.post('/predict_batch')
async def predict_batch(file: UploadFile = File(...), user: str = Depends(admin_auth)):
    return JSONResponse({"error": "batch predict needs to be updated for dual inputs"}, status_code=501)

@app.post('/report_true')
async def report_true(filename: str = Query(...), ra: float = Query(...), user: str = Depends(admin_auth)):
    label_file = os.path.join(BASE_DIR, 'data', 'label_data.csv')
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    header = not os.path.exists(label_file)
    df = pd.DataFrame([{'file': filename, 'ra': ra}])
    df.to_csv(label_file, mode='a', header=header, index=False)
    return {"status": "ok"}

@app.get('/admin/stats')
async def admin_stats(user: str = Depends(admin_auth)):
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()._asdict()
    gpu = {'available': torch.cuda.is_available()}
    return {"cpu": cpu, "mem": mem, "gpu": gpu}

@app.post('/admin/set_active_model')
async def set_active_model(model_file: str = Query(...), user: str = Depends(admin_auth)):
    try:
        target_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(target_path):
            return JSONResponse({"error": "找不到該模型檔案"}, status_code=404)

        # 複製選定的歷史模型，覆蓋掉 current_model.pth
        shutil.copy(target_path, ACTIVE_MODEL_LINK)

        # 強制後端清空記憶，重新載入新模型
        global model
        model = None
        load_model()

        return {"status": "ok", "msg": f"✅ 系統已成功切換至模型：{model_file}"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == '__main__':
    uvicorn.run('webapp.app.main:app', host='0.0.0.0', port=2578, reload=False)
