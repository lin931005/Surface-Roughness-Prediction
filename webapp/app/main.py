from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import torch
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
ACTIVE_MODEL_LINK = os.path.join(BASE_DIR, 'results', 'current_model.pth')
MODEL_PATH = ACTIVE_MODEL_LINK if os.path.exists(ACTIVE_MODEL_LINK) else os.path.join(BASE_DIR, 'results', 'best_surface_model.pth')

model = None

def load_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = torch.load(MODEL_PATH, map_location='cpu')
                model.eval()
            except Exception as e:
                model = None
        else:
            model = None

load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...), gradcam: bool = Query(False)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    x = transform(img).unsqueeze(0)
    if model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=500)
    # forward
    x = x.requires_grad_(True)
    out = model(x)
    if isinstance(out, torch.Tensor):
        try:
            val = float(out.item())
        except Exception:
            val = float(out.detach().cpu().numpy().tolist())
    else:
        val = float(out)

    result = {"ra": val}

    if gradcam:
        try:
            heatmap_b64 = compute_gradcam(model, x, img)
            result['heatmap'] = heatmap_b64
        except Exception as e:
            result['heatmap_error'] = str(e)

    # log prediction (non-blocking)
    try:
        fn = getattr(file, 'filename', f'upload_{int(time.time())}')
        log_prediction(fn, float(val))
    except Exception:
        pass

    return result



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


def find_last_conv(module):
    last = None
    for name, m in module.named_modules():
        # heuristic: conv layers produce 4D outputs
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def compute_gradcam(model, input_tensor, orig_img, target_layer=None):
    # simple Grad-CAM implementation
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

    output = model(input_tensor)
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

    # normalize
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    # resize to original image size
    cam_img = np.uint8(255 * cam)
    cam_img = Image.fromarray(cam_img).resize(orig_img.size, resample=Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0

    # apply colormap
    cmap = plt.get_cmap('jet')
    colored = cmap(cam_arr)[:, :, :3]
    colored = np.uint8(255 * colored)

    orig_arr = np.array(orig_img).astype(np.uint8)
    overlay = (0.5 * orig_arr + 0.5 * colored).astype(np.uint8)

    # encode to base64 PNG
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, token: str = Query(None), user: str = Depends(admin_auth)):

    def run_train():
        try:
            train_script = os.path.join(BASE_DIR, 'scripts', 'train_model.py')
            log_dir = os.path.join(BASE_DIR, 'results', 'train_logs')
            os.makedirs(log_dir, exist_ok=True)
            ts = int(time.time())
            logfile = os.path.join(log_dir, f'train_{ts}.log')
            out_model = os.path.join(MODELS_DIR, f'model_{ts}.pth')
            if os.path.exists(train_script):
                # call training script and redirect stdout to logfile; pass output path as arg if supported
                with open(logfile, 'wb') as f:
                    subprocess.Popen(["python", train_script, "--output", out_model, "--log", logfile], cwd=BASE_DIR, stdout=f, stderr=subprocess.STDOUT)
        except Exception:
            pass

    background_tasks.add_task(run_train)
    return {"status": "retraining started"}


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
    # return parsed JSON lines from the log file as a list
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
        meta = None
        meta_path = p + '.meta.json'
        if os.path.exists(meta_path):
            try:
                import json
                with open(meta_path, 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
            except Exception:
                meta = None
        models.append({"file": fn, "mtime": stat, "meta": meta})
    return {"models": models}


@app.get('/predictions/stats')
async def prediction_stats(user: str = Depends(admin_auth)):
    # return total count and last 20 predictions
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
    total = len(rows)
    last = rows[-20:]
    return {"total": total, "last": last}


@app.post('/models/set')
async def set_active_model(name: str = Query(...), user: str = Depends(admin_auth)):
    src = os.path.join(MODELS_DIR, name)
    if not os.path.exists(src):
        return JSONResponse({"error": "not found"}, status_code=404)
    # copy to active link
    try:
        import shutil
        shutil.copy2(src, ACTIVE_MODEL_LINK)
        # reload model
        global model
        model = None
        load_model()
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post('/predict_batch')
async def predict_batch(file: UploadFile = File(...), user: str = Depends(admin_auth)):
    # accept zip of images, return csv and excel with predictions
    contents = await file.read()
    try:
        z = zipfile.ZipFile(io.BytesIO(contents))
    except Exception:
        return JSONResponse({"error": "invalid zip"}, status_code=400)
    tmpdir = os.path.join(BASE_DIR, 'results', 'tmp', str(int(time.time())))
    os.makedirs(tmpdir, exist_ok=True)
    z.extractall(tmpdir)
    rows = []
    for root, _, files in os.walk(tmpdir):
        for fn in files:
            if fn.lower().endswith(('.png','.jpg','.jpeg')):
                path = os.path.join(root, fn)
                try:
                    img = Image.open(path).convert('RGB')
                    x = transform(img).unsqueeze(0)
                    with torch.no_grad():
                        out = model(x) if model is not None else None
                        val = float(out.item()) if out is not None else None
                    rows.append({'file': fn, 'ra': val})
                except Exception:
                    rows.append({'file': fn, 'ra': None})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(BASE_DIR, 'results', f'batch_{int(time.time())}.csv')
    excel_path = csv_path.replace('.csv', '.xlsx')
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)
    return {"csv": os.path.relpath(csv_path, BASE_DIR), "excel": os.path.relpath(excel_path, BASE_DIR)}


@app.post('/report_true')
async def report_true(filename: str = Query(...), ra: float = Query(...), user: str = Depends(admin_auth)):
    # append to data/label_data.csv
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
    if torch.cuda.is_available():
        gpu['count'] = torch.cuda.device_count()
        gpu['name'] = torch.cuda.get_device_name(0)
        gpu['memory_allocated'] = torch.cuda.memory_allocated(0)
        gpu['memory_reserved'] = torch.cuda.memory_reserved(0)
    return {"cpu": cpu, "mem": mem, "gpu": gpu}


if __name__ == '__main__':
    uvicorn.run('webapp.app.main:app', host='0.0.0.0', port=8000, reload=False)
