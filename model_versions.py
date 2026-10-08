"""模型版本管理工具。

results/ 底下固定檔名的模型就是線上版本，預測時一律載入這些檔案。
重新訓練或切換版本前，會先把當時的線上版本複製到 results/archive/，之後可以從管理介面換回來。
每個模型旁邊可能有一個「<模型檔名>.meta.json」，記錄訓練時間、驗證成績與訓練時保留的驗證刀。
"""
import json
import shutil
from datetime import datetime

from project_root import path

# 每種模型的線上版本檔名
MODEL_FILES = {
    "End_Milling": "best_model_End_Milling.pth",
    "Peripheral_Milling": "best_model_Peripheral_Milling.pth",
    "Classifier": "best_classifier.pth",
    "Traditional": "traditional_rf_model.joblib",
}

RESULTS_DIR = path("results")
ARCHIVE_DIR = path("results", "archive")
META_SUFFIX = ".meta.json"


def current_path(role):
    """線上版本的模型路徑"""
    return RESULTS_DIR / MODEL_FILES[role]


def meta_path(model_path):
    return model_path.with_name(model_path.name + META_SUFFIX)


def read_meta(model_path):
    """讀取模型旁的 meta.json；舊版模型沒有這個檔案時回傳 None"""
    p = meta_path(model_path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_meta(role, meta):
    with open(meta_path(current_path(role)), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def version_of(model_path):
    """用檔案修改時間當版本號，例如 20260916_205100"""
    return datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")


def archived_name(role, version):
    stem, ext = MODEL_FILES[role].rsplit(".", 1)
    return f"{stem}_{version}.{ext}"


def _copy_with_meta(src, dst):
    # copy2 會保留修改時間，版本號才不會因為複製而改變
    shutil.copy2(src, dst)
    if meta_path(src).exists():
        shutil.copy2(meta_path(src), meta_path(dst))
    elif meta_path(dst).exists():
        meta_path(dst).unlink()


def archive_current(role):
    """把目前的線上版本備份到 archive/，同一個版本只會備份一次"""
    current = current_path(role)
    if not current.exists():
        return None
    target = ARCHIVE_DIR / archived_name(role, version_of(current))
    if not target.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        _copy_with_meta(current, target)
    return target.name


def list_archived(role):
    """列出 archive/ 裡這種模型的所有版本，新的在前"""
    if not ARCHIVE_DIR.exists():
        return []
    stem, ext = MODEL_FILES[role].rsplit(".", 1)
    return sorted(
        (p.name for p in ARCHIVE_DIR.iterdir()
         if p.name.startswith(stem + "_") and p.name.endswith("." + ext)),
        reverse=True,
    )


def deploy_archived(role, file_name):
    """把 archive/ 裡的某個版本換成線上版本，換之前會先備份目前的版本"""
    if file_name not in list_archived(role):
        raise FileNotFoundError(file_name)
    archive_current(role)
    _copy_with_meta(ARCHIVE_DIR / file_name, current_path(role))
