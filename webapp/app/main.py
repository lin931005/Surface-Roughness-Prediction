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
import os
import subprocess
import time
import joblib
import cv2
import pandas as pd
import psutil
import base64
import numpy as np
import random
import asyncio
import gc
import secrets
import zlib
from dotenv import load_dotenv
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# ==========================================
# 🔍 1. 先定義好所有的路徑變數！
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)

# 讓後端能 import 專案根目錄的共用模組
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from model_versions import (MODEL_FILES, ARCHIVE_DIR, current_path, read_meta, version_of,
                            archived_name, list_archived, deploy_archived)
from preprocessing import (PREPROCESS_VERSION, to_input, load_gray, patch_side, grid_boxes, tile_boxes,
                           random_box)

# ==========================================
# 🔒 API 金鑰：從專案根目錄的 .env 讀取，不寫死在程式碼裡
# ==========================================
load_dotenv(os.path.join(BASE_DIR, '.env'))
API_SECRET_KEY = os.environ.get("CNC_API_KEY", "")
if not API_SECRET_KEY:
    raise RuntimeError("找不到 CNC_API_KEY：請參考 .env.example 在專案根目錄建立 .env 並設定 API 金鑰")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or not secrets.compare_digest(api_key.encode(), API_SECRET_KEY.encode()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="拒絕存取：無效的 API 金鑰"
        )
    return api_key

# 💡 所有 API 都必須帶金鑰，避免有人繞過前端登入直接呼叫
app = FastAPI(dependencies=[Depends(verify_api_key)])

# ==========================================
# 🤖 2. 載入傳統機器學習引擎
# ==========================================
RF_MODEL = None
RF_MODEL_MTIME = None

def get_rf_model():
    """取得傳統 RF 模型；模型檔重新訓練或切換版本後會自動重新載入"""
    global RF_MODEL, RF_MODEL_MTIME
    rf_path = current_path("Traditional")
    if not rf_path.exists():
        return None
    mtime = rf_path.stat().st_mtime
    if RF_MODEL is None or mtime != RF_MODEL_MTIME:
        RF_MODEL = joblib.load(rf_path)
        RF_MODEL_MTIME = mtime
    return RF_MODEL

try:
    if get_rf_model() is not None:
        print("✅ 傳統 Random Forest 引擎載入成功")
    else:
        print("⚠ 找不到傳統模型，請先訓練")
except Exception as e:
    print(f"⚠ 找不到傳統模型或發生錯誤: {e}")

# ==========================================
# 🧠 2. 智慧動態記憶體管理 (動態加載/卸載)
# ==========================================
expert_models = {"End_Milling": None, "Peripheral_Milling": None}
classifier_model = None
model_preprocess = {}  # 各模型訓練時使用的前處理版本（來自 meta.json），用來提醒舊模型需要重新訓練

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

def load_torch_model(role, model_path):
    """依模型類型建立網路架構並載入權重"""
    model = ClassifierModel() if role == "Classifier" else ResNetDualInputModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def load_models_on_demand():
    """需要預測時才掛載模型"""
    global expert_models, classifier_model, models_are_loaded
    if models_are_loaded:
        return

    print("⏳ 偵測到模型尚未載入，正在將大腦掛載至 GPU 記憶體...")
    for role in expert_models:
        if current_path(role).exists():
            expert_models[role] = load_torch_model(role, current_path(role))

    if current_path("Classifier").exists():
        classifier_model = load_torch_model("Classifier", current_path("Classifier"))

    for role in ("End_Milling", "Peripheral_Milling", "Classifier"):
        model_preprocess[role] = (read_meta(current_path(role)) or {}).get("preprocess")

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

# ==========================================
# 🎨 Grad-CAM 熱力圖
# ==========================================
def compute_gradcam(model, img_bw, params_row):
    """回歸模型的 Grad-CAM：把影像切成蓋滿整張的方塊（與預測相同的放大倍率），
    各方塊以預測的 Ra 對 ResNet 最後一層特徵圖取梯度，再拼回整張圖，標出影響預測最大的區域"""
    device = next(model.parameters()).device
    boxes = tile_boxes(img_bw)
    batch = torch.stack([to_input(img_bw.crop(box)) for box in boxes]).to(device)
    feature_maps = {}
    handle = model.resnet.layer4.register_forward_hook(lambda module, inp, out: feature_maps.update(value=out))
    try:
        with torch.enable_grad():
            output = model(batch, params_row.expand(len(boxes), -1))
            grads = torch.autograd.grad(output.sum(), feature_maps['value'])[0]
    finally:
        handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * feature_maps['value']).sum(dim=1)).detach().cpu().numpy()

    # 拼回整張圖（縮小到最長邊 800px 以減少傳輸量），重疊處取平均，全圖統一正規化
    scale = min(1.0, 800 / max(img_bw.size))
    W, H = max(1, round(img_bw.width * scale)), max(1, round(img_bw.height * scale))
    total, count = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for cam, (left, top, right, bottom) in zip(cams, boxes):
        x0, y0, x1, y1 = (round(v * scale) for v in (left, top, right, bottom))
        if x1 > x0 and y1 > y0:
            total[y0:y1, x0:x1] += cv2.resize(cam, (x1 - x0, y1 - y0))
            count[y0:y1, x0:x1] += 1
    cam = total / np.maximum(count, 1)
    if cam.max() > 0:
        cam = cam / cam.max()

    # 疊在灰階原圖上
    base = cv2.resize(np.array(img_bw), (W, H), interpolation=cv2.INTER_AREA)
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.5 * base + 0.5 * colored).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='JPEG', quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('ascii')

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
        # 💡 讀圖方式與訓練相同：依 EXIF 轉正後轉灰階
        img_bw = load_gray(contents)
    except Exception:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 💡 1. 決定最終要用的 milling_type (加入 AI 自信度偵測)
    ai_confidence = 1.0      # 初始化自信度
    ai_is_confused = False   # 初始化困惑狀態

    final_milling_type = milling_type
    if milling_type == "Auto":
        if classifier_model is not None:
            classifier_model.to(device)
            with torch.no_grad():
                # 💡 均勻取 3x3 個方塊，輸出取平均後再判斷（與訓練時的驗證方式相同）
                patches = torch.stack([to_input(img_bw.crop(box)) for box in grid_boxes(img_bw)]).to(device)
                out = classifier_model(patches).mean(dim=0, keepdim=True)

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

    num_patches = 32
    # 💡 從原始解析度隨機切 32 個固定大小的方塊（放大倍率與訓練時相同）；
    #    取樣位置由影像內容決定，同一張照片每次預測的結果都相同
    rng = random.Random(zlib.crc32(contents))
    boxes = [random_box(img_bw, rng) for _ in range(num_patches)]
    patch_coords = [{"top": top, "left": left, "bottom": bottom, "right": right} for left, top, right, bottom in boxes]

    batch_tensors = torch.stack([to_input(img_bw.crop(box)) for box in boxes]).to(device)
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

    # 💡 這次用到的模型若是用舊版前處理訓練的，提醒使用者重新訓練
    used_roles = [final_milling_type] + (["Classifier"] if milling_type == "Auto" and classifier_model is not None else [])
    preprocess_mismatch = any(model_preprocess.get(role) != PREPROCESS_VERSION for role in used_roles)

    result = {
        "ra": final_ra,
        "used_default_params": used_default,
        "is_anomaly": is_anomaly,
        "preds_std": color_std_score,
        "preds_edge": edge_score,
        "ai_confidence": ai_confidence,  # 💡 修改點 2：把 AI 的自信度數據打包傳給網頁
        "detected_milling": final_milling_type,
        "preprocess_mismatch": preprocess_mismatch,
        "xai_details": {
            "num_patches": num_patches,
            "trim_count": trim_count,
            "patch_size": patch_side(img_bw),
            "patches_info": detailed_patches
        }
    }

    # 🎨 Grad-CAM：逐塊分析後拼回整張圖，標出模型主要看哪些區域
    if gradcam:
        try:
            result['heatmap'] = compute_gradcam(target_model, img_bw, params_tensor[:1])
        except Exception as e:
            result['heatmap_error'] = str(e)

    try:
        fn = getattr(file, 'filename', f'upload_{int(time.time())}')
        log_prediction(fn, final_ra)
    except Exception:
        pass

    return result

@app.post("/predict/traditional")
async def predict_traditional(file: UploadFile = File(...)):
    try:
        rf_model = get_rf_model()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"傳統 ML 引擎載入失敗：{e}"})
    if rf_model is None:
        return JSONResponse(status_code=500, content={"error": "傳統 ML 引擎未啟動"})

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        # 萃取與訓練時完全相同的 3 個特徵
        brightness_var = np.var(img_gray)
        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        features = np.array([[brightness_var, edge_density, laplacian_var]])

        # 進行預測
        pred_ra = rf_model.predict(features)[0]

        return {
            "engine": "Random_Forest",
            "ra": float(pred_ra),
            "features_extracted": {
                "brightness_variance": float(brightness_var),
                "edge_density": float(edge_density),
                "laplacian_variance": float(laplacian_var)
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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
        elif milling_type == "Traditional":
            script_train = os.path.join(BASE_DIR, "scripts", "train_traditional_ml.py")
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

            # 💡 核心修改：分類器與傳統模型不需要後面的參數，專家大腦才需要
            if milling_type in ("Classifier", "Traditional"):
                subprocess.Popen([python_exe, script_train], cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT, env=custom_env)
            else:
                subprocess.Popen([python_exe, script_train, "--milling_type", milling_type], cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT, env=custom_env)

    except Exception as e:
        print(f"❌ [背景任務] 訓練發生錯誤: {str(e)}")

@app.post("/train")
async def start_training(milling_type: str = Query(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if milling_type not in ["End_Milling", "Peripheral_Milling", "Classifier", "Traditional"]:
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

def train_log_path(name: str):
    """只允許讀取 train_logs 資料夾裡的 .log 檔，避免用 ..\\ 之類的路徑讀到其他檔案"""
    if os.path.basename(name) != name or not name.endswith('.log'):
        return None
    return os.path.join(BASE_DIR, 'results', 'train_logs', name)

@app.get('/train_logs')
async def list_train_logs():
    log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
    return {"logs": sorted(os.listdir(log_dir), reverse=True)} if os.path.exists(log_dir) else {"logs": []}

@app.get('/train_logs/{name}')
async def get_train_log(name: str):
    path = train_log_path(name)
    if not path or not os.path.exists(path): return JSONResponse({"error": "not found"}, status_code=404)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f: return {"log": f.read()}

# 💡 補回來的折線圖讀取 API
@app.get('/train_progress/{name}')
async def train_progress(name: str):
    path = train_log_path(name)
    if not path or not os.path.exists(path):
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
async def admin_stats():
    return {"cpu": psutil.cpu_percent(interval=0.5), "mem": psutil.virtual_memory()._asdict(), "gpu": {'available': torch.cuda.is_available()}}

# ==========================================
# 🔄 5. 模型版本管理 (熱切換 / Rollback)
# ==========================================
def preprocess_ok(role, meta):
    """傳統 RF 不受影像前處理影響；深度學習模型必須是用目前版本的前處理訓練的"""
    return role == "Traditional" or (meta or {}).get("preprocess") == PREPROCESS_VERSION

def describe_role(role):
    current = current_path(role)
    version = version_of(current) if current.exists() else None
    meta = read_meta(current) if version else None
    archived = []
    for name in list_archived(role):
        archived_meta = read_meta(ARCHIVE_DIR / name)
        archived.append({
            "file": name,
            "is_current": version is not None and name == archived_name(role, version),
            "meta": archived_meta,
            "preprocess_ok": preprocess_ok(role, archived_meta),
        })
    return {
        "file": MODEL_FILES[role],
        "exists": version is not None,
        "version": version,
        "meta": meta,
        "preprocess_ok": preprocess_ok(role, meta),
        "archived": archived,
    }

@app.get('/models')
async def list_models():
    """列出每種模型的線上版本，以及 results/archive/ 裡可以切換的歷史版本"""
    return {"roles": {role: describe_role(role) for role in MODEL_FILES}}

@app.post('/admin/set_active_model')
async def set_active_model(role: str = Query(...), model_file: str = Query(...)):
    """把 results/archive/ 裡的某個版本換成線上版本"""
    if role not in MODEL_FILES:
        return JSONResponse(status_code=400, content={"error": "未知的模型類型"})
    if model_file not in list_archived(role):
        return JSONResponse(status_code=404, content={"error": "找不到該模型檔案"})

    # 先確認檔案能正常載入，避免把壞掉的檔案換上線
    try:
        if role == "Traditional":
            if not hasattr(joblib.load(ARCHIVE_DIR / model_file), "predict"):
                raise ValueError("不是可用的迴歸模型")
        else:
            load_torch_model(role, ARCHIVE_DIR / model_file)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"模型檔案無法載入：{e}"})

    deploy_archived(role, model_file)
    # 深度學習模型先卸載，下一次預測就會載入新版本；RF 模型會在下一次預測時自動重新載入
    if role != "Traditional":
        unload_models_to_free_vram()
    return {"msg": f"✅ 模型 {model_file} 已成功設為上線版本！"}

if __name__ == '__main__':
    uvicorn.run('webapp.app.main:app', host='0.0.0.0', port=2578, reload=False)
