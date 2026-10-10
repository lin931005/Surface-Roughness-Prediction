"""判斷兩張照片是否拍到同一塊表面，以及重疊多少。

同一刀的照片常是一邊移動載物台一邊連續拍的，相鄰的照片可能拍到大半相同的區域。
做法：照片縮成 1/8 並去掉照明漸層後，把 A 切成 3x3 個小區域，各自在 B 裡找最吻合的位置；
多個區域指向同一個位移，代表兩張照片拍到同一塊表面，再由位移算出重疊比例。
刀痕有週期性，單一區域可能對到錯的週期，所以要求多個區域的位移一致，
而且位移後應該落在 B 裡的區域，大多都要真的吻合。

門檻用已知的照片對校正過：同一次拍攝、裁切過的 44 對全部偵測到；
不同刀（同銑法、同轉速）的 1,634 對沒有任何一對被誤判為重疊。
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

SCALE = 8          # 縮小倍數
GRID = 3           # 每張照片切成 GRID x GRID 個比對區域
MIN_SCORE = 0.5    # 單一區域算「吻合」的最低相關係數
MIN_AGREE = 3      # 至少要有幾個區域的位移一致
MIN_SHARE = 0.6    # 位移後應該落在另一張照片裡的區域，至少要有這個比例真的吻合
TOL = 2            # 位移一致的容許誤差（縮圖像素，約原圖 16 px）


def thumbnail(path):
    """縮成 1/8 的灰階圖，並減掉模糊版本去除照明漸層，只留下刀痕與表面細節"""
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # EXIF 記錄要轉 90° 的照片，轉正後寬高互換
            w, h = h, w
        im.draft('L', (im.width // SCALE, im.height // SCALE))  # JPEG 直接以 1/8 解碼，快很多
        im = ImageOps.exif_transpose(im).convert('L')
        size = (max(1, round(w / SCALE)), max(1, round(h / SCALE)))
        if im.size != size:
            im = im.resize(size, Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
    return a - cv2.GaussianBlur(a, (0, 0), 8)


def _regions(a):
    h, w = a.shape
    rh, rw = h // 4, w // 4
    for i in range(GRID):
        for j in range(GRID):
            y = int(h * (i + 0.5) / GRID - rh / 2)
            x = int(w * (j + 0.5) / GRID - rw / 2)
            yield x, y, a[y:y + rh, x:x + rw]


def _one_way(a, b):
    """A 的各區域在 B 中的位移；回傳 (一致的區域數, 位移後應落在 B 裡的區域數, 位移)"""
    boxes, shifts = [], []
    for x, y, region in _regions(a):
        boxes.append((x, y, region.shape[1], region.shape[0]))
        if region.shape[0] > b.shape[0] or region.shape[1] > b.shape[1] or region.std() < 1e-3:
            continue
        _, score, _, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(b, region, cv2.TM_CCOEFF_NORMED))
        if score >= MIN_SCORE:
            shifts.append((bx - x, by - y))

    best, best_shift = 0, None
    for sx, sy in shifts:
        agree = [(dx, dy) for dx, dy in shifts if abs(dx - sx) <= TOL and abs(dy - sy) <= TOL]
        if len(agree) > best:
            best, best_shift = len(agree), np.median(np.array(agree), axis=0)
    if best_shift is None:
        return 0, 0, None
    dx, dy = best_shift
    expected = sum(0 <= x + dx and x + dx + w <= b.shape[1] and 0 <= y + dy and y + dy + h <= b.shape[0]
                   for x, y, w, h in boxes)
    return best, expected, best_shift


def overlap_ratio(a, b):
    """兩張縮圖（thumbnail 的結果）拍到同一塊表面的面積比例；沒有可信的重疊時回傳 0"""
    ratios = [0.0]
    for src, dst in ((a, b), (b, a)):
        agree, expected, shift = _one_way(src, dst)
        if agree >= MIN_AGREE and agree >= MIN_SHARE * expected:
            h, w = src.shape
            dx, dy = shift
            ratios.append(max(0.0, (w - abs(dx)) * (h - abs(dy)) / (w * h)))
    return max(ratios)
