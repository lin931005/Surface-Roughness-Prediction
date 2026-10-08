"""影像前處理：訓練、驗證與線上推論共用，確保模型在三個階段看到的影像條件一致。

所有照片都在同一個倍率（4.0X）下拍攝，原始解析度下固定的像素數就代表固定的實際尺寸。
因此一律從原始照片切出 PATCH_SIZE × PATCH_SIZE 的正方形方塊，再縮成 INPUT_SIZE × INPUT_SIZE：
不論照片原本多大、有沒有裁切過，刀痕的放大倍率都相同，也不會因為整張縮放而變形。
"""
import io
import math
import random

import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageOps

PATCH_SIZE = 672  # 原始解析度下的方塊邊長（像素），是 INPUT_SIZE 的 3 倍，最小的訓練照片短邊為 746
INPUT_SIZE = 224
EVAL_GRID = 3  # 驗證與銑法分類時，在影像上均勻取 3×3 個方塊
PREPROCESS_VERSION = "patch672-v1"  # 寫進模型 meta，用來辨識模型是用哪一版前處理訓練的

to_input = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_gray(src):
    """讀取影像（路徑或 bytes），依 EXIF 轉正後去除色彩，再複製成三通道"""
    img = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    return ImageOps.exif_transpose(img).convert("L").convert("RGB")


def patch_side(img):
    """方塊邊長；影像比標準方塊小時退而求其次用最短邊（例如 Other 類別的小圖）"""
    return min(PATCH_SIZE, img.width, img.height)


def _spread(length, side, n):
    return [int(v) for v in np.linspace(0, length - side, n).round()]


def grid_boxes(img, n=EVAL_GRID):
    """在影像上均勻排列 n×n 個方塊（影像較小時方塊會重疊），回傳 (left, top, right, bottom)"""
    s = patch_side(img)
    return [(x, y, x + s, y + s) for y in _spread(img.height, s, n) for x in _spread(img.width, s, n)]


def tile_boxes(img):
    """用最少的方塊蓋滿整張影像（Grad-CAM 熱力圖用）"""
    s = patch_side(img)
    nx, ny = math.ceil(img.width / s), math.ceil(img.height / s)
    return [(x, y, x + s, y + s) for y in _spread(img.height, s, ny) for x in _spread(img.width, s, nx)]


def random_box(img, rng=random):
    """隨機位置的標準方塊（線上推論的蒙地卡羅取樣用）"""
    s = patch_side(img)
    x, y = rng.randint(0, img.width - s), rng.randint(0, img.height - s)
    return (x, y, x + s, y + s)


def random_train_patch(img, rng=random, scale_jitter=0.1, max_rotation=15.0):
    """訓練用的隨機方塊：位置隨機、邊長 ±10%、旋轉 ±15°。
    先取能容納旋轉的較大範圍，旋轉後再從中央切出方塊，避免角落出現黑色三角形。"""
    s = min(int(PATCH_SIZE * rng.uniform(1 - scale_jitter, 1 + scale_jitter)), img.width, img.height)
    angle = rng.uniform(-max_rotation, max_rotation)
    need = math.ceil(s * (abs(math.cos(math.radians(angle))) + abs(math.sin(math.radians(angle)))))
    if need > min(img.width, img.height):
        angle, need = 0.0, s
    x, y = rng.randint(0, img.width - need), rng.randint(0, img.height - need)
    patch = img.crop((x, y, x + need, y + need))
    if angle:
        off = (need - s) // 2
        patch = patch.rotate(angle, resample=Image.BILINEAR).crop((off, off, off + s, off + s))
    return patch
