import time
import pandas as pd
from predict_roughness import load_model, predict_images, predict_single_image
import torch

CSV = r"data/final_training_manifest.csv"
df = pd.read_csv(CSV)
# limit to 100 images (or fewer if dataset smaller)
N = min(100, len(df))
img_paths = df['image_path'].tolist()[:N]
speeds = df['speed'].tolist()[:N]
conds = df['condition_id'].tolist()[:N]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Approach A: load once, batch predict
start = time.time()
model = load_model(device=device)
_preds = predict_images(img_paths, speeds, conds, model=model, device=device, batch_size=16)
durA = time.time() - start

# Approach B: reload per image
start = time.time()
_preds2 = []
for p,s,c in zip(img_paths, speeds, conds):
    m = load_model(device=device)
    _preds2.append(predict_single_image(p,s,c,model=m,device=device))
durB = time.time() - start

print(f"Approach A (load once, batch): {durA:.3f} s for {len(img_paths)} images")
print(f"Approach B (reload per image): {durB:.3f} s for {len(img_paths)} images")
print('speedup', round(durB/durA, 2) if durA>0 else float('inf'))
