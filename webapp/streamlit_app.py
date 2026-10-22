import streamlit as st
import requests
import pandas as pd
import numpy as np
import hashlib
import html
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import os
import random
import re
import secrets
import altair as alt
from dotenv import load_dotenv

# ==========================================
# 🌐 全域變數設定
# ==========================================
API_URL = "http://127.0.0.1:2578"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STYLE_PATH = Path(__file__).with_name('style.css')

# 🔒 密碼與 API 金鑰從專案根目錄的 .env 讀取，不寫死在程式碼裡
load_dotenv(os.path.join(BASE_DIR, '.env'))
API_SECRET_KEY = os.environ.get("CNC_API_KEY", "")
ADMIN_PASSWORD = os.environ.get("CNC_ADMIN_PASSWORD", "")
USER_PASSWORD = os.environ.get("CNC_USER_PASSWORD", "")
HEADERS = {'X-API-Key': API_SECRET_KEY}

# 各模型在介面上顯示的名稱
ROLE_LABELS = {
    "End_Milling": "立銑回歸模型",
    "Peripheral_Milling": "直銑回歸模型",
    "Classifier": "銑法分類器",
    "Traditional": "傳統視覺 (Random Forest)",
}

# 單筆檢測的選項：key 是程式裡用的值，value 是畫面上的文字
ENGINES = {"deep": ":material/neurology: 深度學習 (ResNet)", "traditional": ":material/forest: 傳統視覺 (RF)"}
MILLING_OPTIONS = {"Auto": "自動辨識", "End_Milling": "立銑", "Peripheral_Milling": "直銑 (躺銑)"}

# 取樣方塊的狀態（文字由後端回傳）與對應的顏色
PATCH_STATUS_COLORS = {
    '保留 (有效計算區間)': '#10B981',
    '剔除 (異常低值)': '#F59E0B',
    '剔除 (異常高值/可能含灰塵)': '#EF4444',
}
ENGINE_COLORS = {'深度學習 (ResNet)': '#6366F1', '傳統視覺 (RF)': '#F59E0B'}

# 管理頁的單一模型訓練按鈕
TRAIN_BUTTONS = [
    ("End_Milling", "訓練立銑回歸模型", ":material/show_chart:"),
    ("Peripheral_Milling", "訓練直銑回歸模型", ":material/show_chart:"),
    ("Classifier", "訓練銑法分類器", ":material/category:"),
    ("Traditional", "訓練傳統視覺 RF", ":material/forest:"),
]

# ==========================================
# 🎨 網頁基礎設定與共用元件
# ==========================================
st.set_page_config(page_title='CNC 表面粗糙度自動化檢測系統', page_icon=":material/precision_manufacturing:", layout='wide')
st.html(STYLE_PATH)

if not (API_SECRET_KEY and ADMIN_PASSWORD and USER_PASSWORD):
    st.error("系統尚未設定密碼與 API 金鑰：請參考專案根目錄的 .env.example 建立 .env，再重新啟動系統。", icon=":material/key_off:")
    st.stop()


def esc(value) -> str:
    """放進 st.html 的文字先跳脫，避免檔名之類的內容被當成 HTML"""
    return html.escape(str(value))


def page_header(icon, eyebrow, title, description):
    st.html(f'<header class="page-head"><span class="eyebrow"><span class="ms">{icon}</span>{esc(eyebrow)}</span>'
            f'<h1>{esc(title)}</h1><p>{esc(description)}</p></header>')


def card_header(icon, title, subtitle=None):
    sub = f'<div class="s">{esc(subtitle)}</div>' if subtitle else ''
    st.html(f'<div class="card-head"><span class="icon"><span class="ms">{icon}</span></span>'
            f'<div><div class="t">{esc(title)}</div>{sub}</div></div>')


def ra_hero(ra, note, facts, variant=""):
    """Ra 估算值的大字卡片，右側列出模型與取樣等資訊"""
    rows = ''.join(f'<div class="fact"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in facts)
    st.html(f'<section class="ra-hero {variant}"><div>'
            f'<div class="ra-label"><span class="ms">straighten</span>表面粗糙度估算值 (Ra)</div>'
            f'<div class="ra-value">{ra:.4f}<span class="ra-unit">μm</span></div>'
            f'<div class="ra-note">{esc(note)}</div></div><div class="ra-facts">{rows}</div></section>')


# ==========================================
# 🔒 系統安全登入驗證區塊 (純密碼分級制)
# ==========================================
def password_matches(entered: str, expected: str) -> bool:
    return secrets.compare_digest(entered.encode('utf-8'), expected.encode('utf-8'))


def page_login():
    _, center, _ = st.columns([1, 1.1, 1])
    with center:
        st.html('<div class="login-head"><div class="brand-mark"><span class="ms">precision_manufacturing</span></div>'
                '<h1>CNC 表面粗糙度自動化檢測系統</h1><p>以刀痕影像預測銑削表面粗糙度 Ra</p></div>')
        with st.container(border=True, key="card_login"):
            with st.form("login_form", border=False):
                pwd_input = st.text_input("系統通行密碼", type="password", placeholder="請輸入系統通行碼",
                                          help="輸入不同密碼將解鎖不同權限")
                submitted = st.form_submit_button("解鎖系統", type="primary", icon=":material/lock_open:", width="stretch")
            if submitted:
                if password_matches(pwd_input, ADMIN_PASSWORD):
                    st.session_state['role'] = 'admin'
                    st.rerun()
                elif password_matches(pwd_input, USER_PASSWORD):
                    st.session_state['role'] = 'user'
                    st.rerun()
                else:
                    st.error("通行碼錯誤，請重新輸入。", icon=":material/error:")
        st.html('<p class="login-foot"><span class="ms">shield_lock</span>安全防護鎖：管理員與現場作業員使用不同的通行碼</p>')


# ==========================================
# 🧭 側邊欄
# ==========================================
@st.cache_data(ttl=15, show_spinner=False)
def backend_online() -> bool:
    """側邊欄顯示的後端連線狀態（15 秒內不重複檢查）"""
    try:
        return requests.get(f'{API_URL}/train_status', headers=HEADERS, timeout=2).status_code == 200
    except requests.RequestException:
        return False


def render_sidebar(pages):
    is_admin = st.session_state['role'] == 'admin'
    with st.sidebar:
        st.html('<div class="brand"><div class="brand-mark"><span class="ms">precision_manufacturing</span></div>'
                '<div><div class="brand-name">CNC 粗糙度檢測</div><div class="brand-sub">表面粗糙度 Ra 自動化檢測</div></div></div>')
        st.html('<div class="nav-label">操作環境</div>')
        for page in pages:
            st.page_link(page, width="stretch")
        st.space("medium")
        online = backend_online()
        st.html(f'<div class="status-pill{"" if online else " off"}"><span class="dot"></span>'
                f'後端 API {"已連線" if online else "未連線"}</div>')
        name, sub, icon = ("系統管理員", "完整權限", "shield_person") if is_admin else ("現場作業員", "檢測與驗證", "engineering")
        st.html(f'<div class="user-card"><div class="avatar"><span class="ms">{icon}</span></div>'
                f'<div><div class="name">{name}</div><div class="sub">已登入 · {sub}</div></div></div>')
        if st.button("登出系統", icon=":material/logout:", width="stretch"):
            st.session_state['role'] = None
            st.rerun()


# ==========================================
# 🔧 共用的資料處理
# ==========================================
def parse_filename_gt(filename: str):
    gt_type_code = None
    gt_type_text = "未知"
    if "立銑" in filename or "End_Milling" in filename or "End" in filename:
        gt_type_code = "End_Milling"
        gt_type_text = "立銑"
    elif "直銑" in filename or "Peripheral_Milling" in filename or "Peripheral" in filename:
        gt_type_code = "Peripheral_Milling"
        gt_type_text = "直銑 (躺銑)"

    gt_ra = None
    matches = re.findall(r'(\d+\.\d+)', filename)
    if matches:
        gt_ra = float(matches[-1])
    return gt_type_code, gt_type_text, gt_ra

def parse_filename_cut(filename: str):
    """從檔名讀取條件編號（哪一刀），顯示在批量驗證的明細表，例如「直銑_7000-0.5_2.3746.jpg」→ '7000-0.5'"""
    match = re.search(r'(?<![\d.])(\d{4,5})-(\d+(?:\.\d+)?)(?![\d.\-])', filename)
    if not match or not 1000 <= int(match.group(1)) <= 10000:
        return None
    return f"{match.group(1)}-{match.group(2)}"

def describe_model_meta(meta, up_to_date=True):
    """把模型的 meta.json 整理成一行說明"""
    warning = "" if up_to_date else "　⚠️ 舊版模型，需重新訓練"
    if not meta:
        return "（舊版模型，沒有驗證紀錄）" + warning
    if 'val_mae' in meta:
        score = f"驗證 MAE {meta['val_mae']:.4f} μm"
    elif 'val_acc' in meta:
        score = f"驗證正確率 {meta['val_acc'] * 100:.1f}%"
    else:
        score = ""
    return f"訓練於 {meta.get('trained_at', '?')}，{score}".rstrip("，") + warning

def next_example_photo():
    """輪流取 data/example 的測試照片，當作系統能辨識的影像範例"""
    folder = os.path.join(BASE_DIR, "data", "example")
    if not os.path.exists(folder):
        return None
    all_images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_images:
        return None
    if not st.session_state.get('meme_playlist'):
        st.session_state['meme_playlist'] = all_images.copy()
        random.shuffle(st.session_state['meme_playlist'])
    return os.path.join(folder, st.session_state['meme_playlist'].pop(0))

def patch_color(status):
    """取樣方塊的框線顏色：綠色採用、紅色剔除高值、黃色剔除低值"""
    colors = list(PATCH_STATUS_COLORS.values())
    return colors[0] if "保留" in status else colors[2] if "異常高值" in status else colors[1]

def draw_patch_boxes(img, patches_info):
    boxed = img.copy()
    draw = ImageDraw.Draw(boxed)
    line = max(4, round(max(img.size) / 500))  # 照片縮小顯示後框線仍看得清楚
    for p in patches_info:
        c = p['coords']
        draw.rectangle([c['left'], c['top'], c['right'], c['bottom']], outline=patch_color(p['status']), width=line)
    return boxed

def patch_chart(patches_info, final_ra):
    """每個取樣方塊的 Ra（依數值排序），虛線是去掉極端值後的平均，也就是最終的 Ra"""
    df = pd.DataFrame([{"排序": i + 1, "區塊編號": f"Patch {p['id']}", "預測粗糙度 (Ra)": p['ra'], "狀態": p['status']}
                       for i, p in enumerate(patches_info)])
    points = alt.Chart(df).mark_circle(size=110, opacity=0.95).encode(
        x=alt.X('排序:Q', title='隨機取樣區塊 (依數值排序)', scale=alt.Scale(domain=[0, len(df) + 1], nice=False),
                axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y('預測粗糙度 (Ra):Q', scale=alt.Scale(zero=False), title='Ra 值 (μm)'),
        color=alt.Color('狀態:N', scale=alt.Scale(domain=list(PATCH_STATUS_COLORS), range=list(PATCH_STATUS_COLORS.values())),
                        legend=None),
        tooltip=['區塊編號', alt.Tooltip('預測粗糙度 (Ra):Q', format='.4f'), '狀態']
    )
    final = alt.Chart(pd.DataFrame({'最終 Ra': [final_ra]}))
    rule = final.mark_rule(color='#6366F1', strokeDash=[6, 4], strokeWidth=1.5).encode(
        y='最終 Ra:Q', tooltip=[alt.Tooltip('最終 Ra:Q', format='.4f')])
    label = final.mark_text(align='left', baseline='bottom', dx=6, dy=-4, color='#6366F1', fontWeight=600).encode(
        y='最終 Ra:Q', x=alt.value(0), text=alt.value(f'最終 Ra {final_ra:.4f} μm'))
    return (rule + label + points).properties(height=380, background='transparent').interactive()

def sample_labels(df):
    """圖表 x 軸用的短標籤，例如「立銑 9000-11」；有重複時在後面加編號"""
    labels = [f"{gt_type[:2]} {cut}" if cut else name for gt_type, cut, name in zip(df['真實銑法'], df['條件編號'], df['圖片檔名'])]
    total, seen = Counter(labels), Counter()
    result = []
    for label in labels:
        seen[label] += 1
        result.append(f"{label} #{seen[label]}" if total[label] > 1 else label)
    return result

def error_chart(valid_df):
    """每張影像兩個引擎的絕對誤差，並排比較"""
    chart_df = valid_df.assign(樣本=sample_labels(valid_df)).melt(
        id_vars=['樣本', '圖片檔名'], value_vars=['絕對誤差 (深度學習)', '絕對誤差 (傳統視覺)'],
        var_name='引擎', value_name='絕對誤差 (μm)')
    chart_df['引擎'] = chart_df['引擎'].map({'絕對誤差 (深度學習)': '深度學習 (ResNet)', '絕對誤差 (傳統視覺)': '傳統視覺 (RF)'})
    n = len(valid_df)
    many = n > 40  # 張數多時不顯示標籤，滑鼠移到長條上看檔名
    return alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('樣本:N', sort=None, title='測試樣本 (滑鼠移到長條上可看檔名)' if many else '測試樣本',
                axis=alt.Axis(labels=not many, ticks=not many, labelAngle=0 if n <= 8 else -45, labelLimit=160)),
        xOffset=alt.XOffset('引擎:N', sort=list(ENGINE_COLORS)),
        y=alt.Y('絕對誤差 (μm):Q', title='絕對誤差 (μm，越低越好)'),
        color=alt.Color('引擎:N', scale=alt.Scale(domain=list(ENGINE_COLORS), range=list(ENGINE_COLORS.values())),
                        legend=alt.Legend(orient='top', title=None)),
        tooltip=['圖片檔名', '引擎', alt.Tooltip('絕對誤差 (μm):Q', format='.4f')]
    ).properties(height=360, background='transparent')

def loss_chart(progress_df):
    """訓練監控的 loss 曲線"""
    data = progress_df.melt(id_vars='epoch', value_vars=['train_loss', 'val_loss'], var_name='曲線', value_name='loss')
    return alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=alt.Y('loss:Q', title='Loss'),
        color=alt.Color('曲線:N', legend=alt.Legend(orient='top', title=None)),
        tooltip=['epoch', '曲線', alt.Tooltip('loss:Q', format='.4f')]
    ).properties(height=320, background='transparent')


# ==========================================
# 👨‍🔧 模式 A：單張檢測 (支援雙引擎切換)
# ==========================================
def page_single():
    page_header("center_focus_strong", "Inspection", "單筆影像檢測作業",
                "上傳工件表面影像，系統會估算表面粗糙度 Ra；可以切換不同的 AI 引擎來比較預測結果。")

    col_settings, col_preview = st.columns([4, 5], gap="large")
    with col_settings:
        with st.container(border=True, key="card_settings"):
            card_header("tune", "分析設定", "選擇 AI 運算引擎與銑削加工法")
            engine = st.segmented_control("AI 運算引擎", list(ENGINES), format_func=ENGINES.get, default="deep",
                                          required=True, width="stretch",
                                          help="深度學習：ResNet 專家模型；傳統視覺：OpenCV 影像特徵 + Random Forest")
            use_deep = engine == "deep"
            selected_type_api = st.segmented_control(
                "銑削加工法", list(MILLING_OPTIONS), format_func=MILLING_OPTIONS.get, default="Auto", required=True,
                disabled=not use_deep, width="stretch",
                help="自動辨識：先用銑法分類器判斷立銑或直銑，再交給對應的 Ra 模型。傳統視覺引擎不需要指定銑法。")
            use_gc = st.toggle('顯示 Grad-CAM 特徵啟動熱力圖', value=True, disabled=not use_deep, help='僅支援深度學習引擎')
            uploaded = st.file_uploader('上傳表面影像 (支援 png, jpg, jpeg)', type=['png', 'jpg', 'jpeg'])
            predict_btn = st.button('執行粗糙度 (Ra) 分析', type="primary", icon=":material/play_arrow:", width="stretch",
                                    disabled=uploaded is None)

    img = ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB') if uploaded is not None else None  # 與後端相同，依 EXIF 轉正
    with col_preview:
        with st.container(border=True, key="card_preview"):
            if img is None:
                card_header("image", "待測工件影像", "尚未上傳影像")
                st.html('<div class="empty"><span class="ms">add_photo_alternate</span><b>尚未上傳影像</b>'
                        '<span>在左側上傳工件表面影像後，這裡會顯示預覽</span></div>')
            else:
                card_header("image", "待測工件影像", f"{uploaded.name} · {img.width} × {img.height} px")
                st.image(img, width='stretch')

    if uploaded is None:
        return

    if 'force_override' not in st.session_state: st.session_state['force_override'] = False
    if 'do_predict' not in st.session_state: st.session_state['do_predict'] = False
    if 'last_file' not in st.session_state or st.session_state['last_file'] != uploaded.name:
        st.session_state['last_file'] = uploaded.name
        st.session_state['force_override'] = False
        st.session_state['do_predict'] = False

    def trigger_override():
        st.session_state['force_override'] = True
        st.session_state['do_predict'] = True

    def trigger_next_meme():
        st.session_state['do_predict'] = True

    if not (predict_btn or st.session_state['do_predict']):
        return
    st.session_state['do_predict'] = False

    files = {'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')}
    params = {'milling_type': selected_type_api}
    if use_gc: params['gradcam'] = 'true'
    # 💡 判斷要打哪一個後端 API 路徑
    api_endpoint = f'{API_URL}/predict' if use_deep else f'{API_URL}/predict/traditional'

    with st.spinner('影像特徵萃取與數值運算中...', show_time=True):
        try:
            r = requests.post(api_endpoint, files=files, params=params, headers=HEADERS, timeout=30)
            j = r.json() if r.status_code == 200 else None
        except Exception as e:
            st.error(f"連線失敗：{str(e)}")
            return

    if r.status_code != 200:
        st.error(f"伺服器回傳異常 (狀態碼 {r.status_code})：{r.text}")
    elif 'error' in j:
        st.error(f"分析失敗：{j['error']}")
    elif use_deep:
        show_deep_result(j, img, use_gc, trigger_next_meme, trigger_override)
    else:
        show_traditional_result(j, img)


def show_traditional_result(j, img):
    import cv2  # 只有傳統視覺的說明圖會用到

    ra_hero(j.get('ra'), "傳統機器視覺 (OpenCV) + 隨機森林 (Random Forest)",
            [("特徵萃取", "OpenCV 固定濾波器"), ("特徵向量", "亮度・邊緣・模糊度"), ("迴歸模型", "Random Forest")],
            variant="cv")

    # --- 視覺化特徵萃取過程 (XAI) ---
    with st.container(border=True, key="card_cv_pipeline"):
        card_header("filter", "影像特徵物理轉換過程 (Feature Extraction Pipeline)",
                    "傳統演算法不依賴黑盒子神經網路，而是透過固定濾波器提取具有物理意義的表面紋理，再交由機器學習進行迴歸預測。")
        # 💡 為了在網頁展示，我們用 OpenCV 重現後端萃取特徵時的影像變化
        img_gray = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        laplacian_disp = cv2.convertScaleAbs(cv2.Laplacian(img_gray, cv2.CV_64F))  # 轉為可顯示的圖片格式

        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.image(img_gray, caption='1. 灰階轉換 (評估整體明暗對比)', width='stretch')
        col_v2.image(edges, caption='2. Canny 邊緣偵測 (擷取刀痕特徵)', width='stretch')
        col_v3.image(laplacian_disp, caption='3. Laplacian 濾波 (捕捉表面銳利度)', width='stretch')

    # --- 顯示最終萃取數值 ---
    with st.container(border=True, key="card_cv_features"):
        card_header("data_array", "轉換為機器學習特徵向量 (Feature Vector)")
        feats = j.get('features_extracted', {})
        c1, c2, c3 = st.columns(3)
        c1.metric("亮度變異數", f"{feats.get('brightness_variance', 0):.1f}", "對應圖 1", delta_color="off",
                  delta_arrow="off", icon=":material/brightness_6:", border=True)
        c2.metric("邊緣密度", f"{feats.get('edge_density', 0):.4f}", "對應圖 2", delta_color="off",
                  delta_arrow="off", icon=":material/texture:", border=True)
        c3.metric("模糊變異數", f"{feats.get('laplacian_variance', 0):.1f}", "對應圖 3", delta_color="off",
                  delta_arrow="off", icon=":material/blur_on:", border=True)
        st.info("**運算邏輯總結**：系統將上述 3 個特徵值打包成一維數學向量 `[亮度, 邊緣, 模糊度]`，輸入至已訓練完畢的 "
                "**Random Forest 決策樹群集** 中，透過 100 棵決策樹的投票機制，得出最終的 Ra 預測數值。",
                icon=":material/lightbulb:")


def show_deep_result(j, img, use_gc, trigger_next_meme, trigger_override):
    ai_conf = j.get('ai_confidence', 1.0) * 100

    if j.get('is_anomaly'):
        if not st.session_state['force_override']:
            st.error("**資料驗證失敗：已中止分析流程**", icon=":material/gpp_bad:")
            if ai_conf == 0.0:
                st.warning("**影像特徵不符警告：** 系統判定此影像缺乏有效之金屬切削紋理，已中斷流程。",
                           icon=":material/image_not_supported:")
            elif ai_conf < 85.0:
                st.warning(f"**影像品質警告：** 系統對此影像特徵辨識度偏低 (置信度 {ai_conf:.1f}%)。", icon=":material/blur_on:")

            example = next_example_photo()
            with st.container(border=True, key="card_example"):
                col_img, col_act = st.columns([3, 2], gap="large", vertical_alignment="center")
                with col_act:
                    card_header("photo_library", "參考範例", "系統能辨識的刀痕影像長這樣。確認影像沒有問題的話，也可以忽略警告繼續分析。")
                    if example:
                        st.button("載入其他參考範例", on_click=trigger_next_meme, icon=":material/refresh:", width="stretch")
                    st.button("強制忽略警告並執行分析", on_click=trigger_override, icon=":material/warning:", width="stretch")
                if example:
                    col_img.image(example, width='stretch')
            return
        st.warning("提示：已手動覆寫安全攔截設定，強制執行特徵數值分析。", icon=":material/warning:")

    detected_type = j.get('detected_milling')
    display_type = "立銑 (End Milling)" if detected_type == "End_Milling" else "直銑 (躺銑) (Peripheral Milling)"
    details = j.get('xai_details') or {}
    patches_info = details.get('patches_info', [])
    counts = Counter(patch_color(p['status']) for p in patches_info)
    keep_color, low_color, high_color = PATCH_STATUS_COLORS.values()

    note = (f"隨機取樣 {details.get('num_patches', len(patches_info))} 塊 {details.get('patch_size', '?')} px 方塊，"
            f"去除最高與最低各 {details.get('trim_count', '?')} 塊後取平均") if patches_info else ""
    facts = [("特徵萃取模型", display_type), ("特徵置信度", f"{ai_conf:.1f}%")]
    if patches_info:
        facts.append(("有效取樣", f"{counts[keep_color]} / {len(patches_info)} 塊"))
    ra_hero(j.get('ra'), note, facts)

    if j.get('outdated_model'):
        st.warning("目前上線的模型是舊版本，請到「系統管理與模型控制台」重新訓練，否則估算值不準確。", icon=":material/update:")

    tab_labels = [":material/scatter_plot: 局部特徵取樣與離群值"]
    if use_gc:
        tab_labels.append(":material/local_fire_department: Grad-CAM 熱力圖")
    tab_labels.append(":material/layers: 影像預處理")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if patches_info:
            chips = [(keep_color, "採用 (有效計算區間)", counts[keep_color]),
                     (low_color, "剔除異常低值", counts[low_color]),
                     (high_color, "剔除異常高值 / 可能含灰塵", counts[high_color])]
            st.html('<div class="legend">' + ''.join(
                f'<span class="chip"><i style="background:{color}"></i>{label}<b>{n}</b></span>' for color, label, n in chips) + '</div>')
            col_img, col_chart = st.columns([7, 5], gap="medium")
            col_img.image(draw_patch_boxes(img, patches_info), width='stretch',
                          caption=f"隨機取樣區域可視化 (每個方塊 {details.get('patch_size', '?')} px；綠框: 採用, 紅/黃框: 剔除極端值)")
            col_chart.altair_chart(patch_chart(patches_info, j['ra']), width='stretch')
        else:
            st.info("後端沒有回傳取樣明細。")

    if use_gc:
        with tabs[1]:
            if j.get('heatmap'):
                col_gray, col_heat = st.columns(2, gap="medium")
                col_gray.image(img.convert('L'), caption='灰階影像 (對照用)', width='stretch')
                col_heat.image(j['heatmap'], caption='Grad-CAM 表面紋理熱力圖 (整塊越紅，模型估出的 Ra 越高；塊內紅點位置僅供參考)', width='stretch')
                st.caption(f"熱力圖把照片切成多個平均分布的 {details.get('patch_size', '?')} px 方塊，各自計算後再拼回整張；"
                           "方塊交界的格線是計算方式造成的，不是工件上的紋路。熱力圖只用來輔助解釋，不影響 Ra 的計算。")
            elif j.get('heatmap_error'):
                st.warning(f"熱力圖產生失敗：{j['heatmap_error']}")
            else:
                st.info("後端沒有回傳熱力圖。")

    with tabs[-1]:
        col_orig, col_bw = st.columns(2, gap="medium")
        col_orig.image(img, caption='1. 原始彩色輸入', width='stretch')
        col_bw.image(img.convert('L'), caption='2. 灰階紋理強化 (演算法分析特徵)', width='stretch')


# ==========================================
# 🧪 模式 B：批量驗證與精度分析 (雙引擎大對決)
# ==========================================
def page_batch():
    page_header("fact_check", "Batch Evaluation", "批量驗證與精度分析",
                "上傳多張影像，系統將同時啟動「深度學習」與「傳統視覺」引擎，並對比兩者之精準度。")

    with st.container(border=True, key="card_batch_upload"):
        card_header("upload_file", "批量上傳測試影像", "可按 Ctrl+A 全選上傳")
        batch_files = st.file_uploader('批量上傳測試影像', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True,
                                       label_visibility="collapsed")
        st.caption("兩個引擎都只看刀痕影像預測 Ra；檔名建議使用「直銑_7000-3_1.565.jpg」的格式，"
                   "檔名中的銑法、條件編號與真實 Ra 只用來計算誤差，不會用於預測。")
        run_batch = False
        if batch_files:
            col_count, col_run = st.columns([3, 2], vertical_alignment="center")
            col_count.success(f"已成功載入 **{len(batch_files)}** 張待測影像！", icon=":material/folder_open:")
            with col_run:
                run_batch = st.button('執行雙引擎批量分析', type="primary", icon=":material/rocket_launch:", width="stretch")

    if not run_batch:
        return

    progress_bar = st.progress(0, text="準備開始...")
    results = []
    outdated_model = False

    # 💡 讀取各模型訓練時用過的照片（內容的 MD5），用來判斷上傳的照片是否訓練過
    try:
        r_photos = requests.get(f'{API_URL}/models/train_photos', headers=HEADERS, timeout=10)
        used_photos = {role: set(md5s) for role, md5s in r_photos.json().items()} if r_photos.status_code == 200 else {}
    except Exception:
        used_photos = {}

    for idx, file in enumerate(batch_files):
        progress_bar.progress(idx / len(batch_files), text=f"雙引擎運算中：第 ({idx+1}/{len(batch_files)}) 筆影像...")
        # 檔名的真實值只用來計算誤差，不會送進模型
        gt_type_code, gt_type_text, gt_ra = parse_filename_gt(file.name)
        condition_id = parse_filename_cut(file.name)

        # 用照片內容比對：四個模型訓練時都沒用過這張照片，測試才公平
        digest = hashlib.md5(file.getvalue()).hexdigest()
        if any(digest in md5s for md5s in used_photos.values()):
            seen_text = "是"
        elif len(used_photos) == len(ROLE_LABELS):
            seen_text = "否"
        else:
            seen_text = "❓ 未知"  # 有模型是舊版本，沒有記錄訓練用的照片

        # --- 1. 深度學習引擎（只送影像，由分類器自動判斷銑法）---
        files_payload_dl = {'file': (file.name, file.getvalue(), 'image/jpeg')}
        try:
            r_dl = requests.post(f'{API_URL}/predict', files=files_payload_dl, params={'milling_type': 'Auto'}, headers=HEADERS, timeout=15)
            if r_dl.status_code == 200:
                j_dl = r_dl.json()
                outdated_model |= bool(j_dl.get('outdated_model'))
                pred_ra_dl = j_dl.get('ra')
                abs_err_dl = abs(pred_ra_dl - gt_ra) if gt_ra else None
                pred_type_code = j_dl.get('detected_milling')
                pred_type_text = "立銑" if pred_type_code == "End_Milling" else "直銑 (躺銑)" if pred_type_code == "Peripheral_Milling" else "未知"
                type_correct = (gt_type_code == pred_type_code) if gt_type_code else None
                ai_conf = j_dl.get('ai_confidence', 1.0) * 100
            else:
                pred_ra_dl, abs_err_dl, pred_type_text, type_correct, ai_conf = np.nan, np.nan, "API 錯誤", None, 0.0
        except:
            pred_ra_dl, abs_err_dl, pred_type_text, type_correct, ai_conf = np.nan, np.nan, "連線錯誤", None, 0.0

        # --- 2. 傳統視覺引擎 ---
        files_payload_ml = {'file': (file.name, file.getvalue(), 'image/jpeg')}
        try:
            r_ml = requests.post(f'{API_URL}/predict/traditional', files=files_payload_ml, headers=HEADERS, timeout=15)
            if r_ml.status_code == 200:
                pred_ra_ml = r_ml.json().get('ra')
                abs_err_ml = abs(pred_ra_ml - gt_ra) if gt_ra else None
            else:
                pred_ra_ml, abs_err_ml = np.nan, np.nan
        except:
            pred_ra_ml, abs_err_ml = np.nan, np.nan

        results.append({
            "圖片檔名": file.name,
            "條件編號": condition_id or "",
            "訓練時用過這張照片": seen_text,
            "真實銑法": gt_type_text,
            "系統判定銑法": pred_type_text,
            "特徵置信度 (%)": round(ai_conf, 1),
            "銑法辨識": "✅ 正確" if type_correct else "❌ 誤判" if type_correct is False else "❓ 未知",
            "真實 Ra (μm)": round(gt_ra, 4) if gt_ra else np.nan,
            "預測 Ra (深度學習)": round(pred_ra_dl, 4) if not np.isnan(pred_ra_dl) else np.nan,
            "絕對誤差 (深度學習)": round(abs_err_dl, 4) if abs_err_dl else np.nan,
            "預測 Ra (傳統視覺)": round(pred_ra_ml, 4) if not np.isnan(pred_ra_ml) else np.nan,
            "絕對誤差 (傳統視覺)": round(abs_err_ml, 4) if abs_err_ml else np.nan
        })

    progress_bar.progress(1.0, text="雙引擎批量分析完畢！")
    if outdated_model:
        st.warning("目前上線的深度學習模型是舊版本，請先重新訓練，以下深度學習的數字不準確。", icon=":material/update:")
    df_res = pd.DataFrame(results)

    # 💡 計算 MAPE (平均相對偏差率)
    df_res['偏差率 (深度學習) (%)'] = (df_res['絕對誤差 (深度學習)'] / df_res['真實 Ra (μm)']) * 100
    df_res['偏差率 (傳統視覺) (%)'] = (df_res['絕對誤差 (傳統視覺)'] / df_res['真實 Ra (μm)']) * 100

    # 💡 只用訓練時沒用過的照片計算 KPI，數字才不會偏樂觀
    unseen_df = df_res[df_res['訓練時用過這張照片'] == "否"]
    scope_df = unseen_df if not unseen_df.empty else df_res
    valid_df = scope_df.dropna(subset=['真實 Ra (μm)', '預測 Ra (深度學習)', '預測 Ra (傳統視覺)'])

    # ==========================================
    # 🎯 雙引擎對決 KPI 統計儀表板
    # ==========================================
    st.space("small")
    card_header("insights", "雙引擎綜合效能 KPI 統計與銑法對照")
    if len(unseen_df) == len(df_res):
        st.success(f"這 **{len(df_res)}** 張照片訓練時都沒有用過，以下是它們的測試結果。", icon=":material/verified:")
    elif not unseen_df.empty:
        st.success(f"以下統計只計算訓練時沒用過的 **{len(unseen_df)}** 張照片，全部 {len(df_res)} 張的結果請看最下方的明細表。",
                   icon=":material/verified:")
    else:
        st.warning("這批照片訓練時都用過（或模型是舊版本，沒有記錄訓練用的照片），以下數字會偏樂觀。"
                   "請改用 data/example 的測試照片，或重新訓練全部模型後再測。", icon=":material/warning:")

    if not valid_df.empty:
        # 總結數據
        type_checked = scope_df[scope_df['銑法辨識'] != "❓ 未知"]
        acc = (type_checked['銑法辨識'] == "✅ 正確").mean() * 100 if not type_checked.empty else 0.0

        def engine_scores(engine):
            """一個引擎的綜合、立銑、直銑 MAE 與 MAPE"""
            groups = [valid_df, valid_df[valid_df['真實銑法'] == '立銑'], valid_df[valid_df['真實銑法'] == '直銑 (躺銑)']]
            mae = [g[f'絕對誤差 ({engine})'].mean() for g in groups]
            mape = [g[f'偏差率 ({engine}) (%)'].mean() for g in groups]
            return mae, mape

        dl_mae_all, dl_mape_all = engine_scores('深度學習')
        ml_mae_all, ml_mape_all = engine_scores('傳統視覺')
        dl_mae, ml_mae = dl_mae_all[0], ml_mae_all[0]

        # --- 區塊 1：總覽 ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("測試樣本總數", f"{len(scope_df)} 張", icon=":material/photo_library:", border=True)
        c2.metric("深度學習銑法辨識正確率", f"{acc:.1f} %", icon=":material/visibility:", border=True)
        c3.metric("深度學習 綜合 MAE", f"{dl_mae:.4f} μm", icon=":material/neurology:", border=True)
        c4.metric("傳統視覺 綜合 MAE", f"{ml_mae:.4f} μm", icon=":material/forest:", border=True)

        # 💡 結論依本批實際數字產生，不預設哪個引擎比較好
        if pd.notna(dl_mae) and pd.notna(ml_mae):
            if dl_mae < ml_mae:
                verdict = f"深度學習的平均誤差較低，比傳統視覺少 {ml_mae - dl_mae:.4f} μm"
            elif ml_mae < dl_mae:
                verdict = f"傳統視覺的平均誤差較低，比深度學習少 {dl_mae - ml_mae:.4f} μm"
            else:
                verdict = "兩個引擎的平均誤差相同"
            caution = "；樣本少於 10 張，差距不一定有代表性" if len(valid_df) < 10 else ""
            st.info(f"**本批結果**：深度學習 MAE {dl_mae:.4f} μm、傳統視覺 MAE {ml_mae:.4f} μm，{verdict}{caution}。",
                    icon=":material/lightbulb:")

        # --- 區塊 2、3：兩個引擎的細項 ---
        def fmt(value, unit):
            return f"{value:.4f} μm" if unit == "μm" and pd.notna(value) else f"{value:.2f} %" if pd.notna(value) else "N/A"

        engines = [("card_kpi_dl", "neurology", "深度學習引擎 (ResNet-50 雙專家)", "影像 → 銑法分類器 → 立銑／直銑 Ra 模型",
                    dl_mae_all, dl_mape_all),
                   ("card_kpi_ml", "forest", "傳統機器視覺引擎 (OpenCV + RF)", "影像 → 亮度・邊緣・模糊度特徵 → Random Forest",
                    ml_mae_all, ml_mape_all)]
        for col, (key, icon, title, subtitle, mae, mape) in zip(st.columns(2, gap="medium"), engines):
            with col, st.container(border=True, key=key):
                card_header(icon, title, subtitle)
                for scope, mae_value, mape_value in zip(("綜合", "立銑", "直銑"), mae, mape):
                    c_mae, c_mape = st.columns(2)
                    c_mae.metric(f"{scope}平均誤差 (MAE)", fmt(mae_value, "μm"), border=True,
                                 help="平均絕對誤差：預測 Ra 與真實 Ra 平均相差多少 μm")
                    c_mape.metric(f"{scope}偏差率 (MAPE)", fmt(mape_value, "%"), border=True,
                                  help="平均偏差率：誤差占真實 Ra 的百分比")

        # ==========================================
        # 📈 視覺化圖表分析
        # ==========================================
        with st.container(border=True, key="card_batch_chart"):
            card_header("bar_chart", "雙引擎精準度對決 (深度學習 vs 傳統視覺)", "每張影像兩個引擎的絕對誤差，越低越好")
            st.altair_chart(error_chart(valid_df), width='stretch')

    # ==========================================
    # 📋 明細與匯出
    # ==========================================
    # 將數值四捨五入方便閱讀
    if '偏差率 (深度學習) (%)' in df_res:
        df_res['偏差率 (深度學習) (%)'] = df_res['偏差率 (深度學習) (%)'].round(2)
        df_res['偏差率 (傳統視覺) (%)'] = df_res['偏差率 (傳統視覺) (%)'].round(2)

    with st.container(border=True, key="card_batch_table"):
        card_header("table_view", "完整比對數據明細表", f"共 {len(df_res)} 張影像")
        ra_format = st.column_config.NumberColumn(format="%.4f")
        pct_format = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(df_res, hide_index=True, column_config={
            "特徵置信度 (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "真實 Ra (μm)": ra_format, "預測 Ra (深度學習)": ra_format, "絕對誤差 (深度學習)": ra_format,
            "預測 Ra (傳統視覺)": ra_format, "絕對誤差 (傳統視覺)": ra_format,
            "偏差率 (深度學習) (%)": pct_format, "偏差率 (傳統視覺) (%)": pct_format,
        })
        csv_out = df_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="匯出完整測試驗證報告 (CSV)", data=csv_out, file_name="Dual_Engine_Evaluation.csv",
                           mime="text/csv", icon=":material/download:")


# ==========================================
# 👑 模式 C：系統管理員 (MLOps 中控台)
# ==========================================
def start_training(milling_type):
    try:
        res = requests.post(f"{API_URL}/train", params={'milling_type': milling_type}, headers=HEADERS)
        if res.status_code == 200: st.success(res.json().get("message", "指令發送成功"))
        else:
            try: detail = res.json().get('error') or res.text
            except ValueError: detail = res.text
            st.error(f"啟動失敗：{detail}")
    except Exception as e: st.error(str(e))


def page_admin():
    page_header("admin_panel_settings", "Admin Console", "系統管理與模型控制台",
                "訓練模型、切換上線的模型版本，並查看訓練資料、預測紀錄與伺服器狀態。")

    tab_train, tab_model, tab_data, tab_history, tab_stats = st.tabs([
        ":material/model_training: 訓練與終端機", ":material/published_with_changes: 模型熱切換",
        ":material/dataset: 資料集分析", ":material/history: 預測紀錄與稽核", ":material/memory: 硬體監控"
    ])

    with tab_train:
        # 💡 同一時間只能跑一組訓練：進行中時顯示進度並停用按鈕
        try:
            status_res = requests.get(f"{API_URL}/train_status", headers=HEADERS, timeout=3)
            train_status = status_res.json() if status_res.status_code == 200 else {}
        except Exception:
            train_status = {}
        training_busy = bool(train_status.get('running'))

        with st.container(border=True, key="card_train"):
            card_header("model_training", "啟動模型訓練管線 (Training Pipeline)", "同一時間只能跑一組訓練，結束後才能再啟動")
            if training_busy:
                jobs, current = train_status.get('jobs', []), train_status.get('current')
                step = jobs.index(current) + 1 if current in jobs else len(jobs)
                st.info(f"訓練進行中：**{ROLE_LABELS.get(current, current)}**（第 {step}/{len(jobs)} 個）。同一時間只能跑一組訓練，結束後才能再啟動。",
                        icon=":material/sync:")
            elif train_status.get('failed'):
                st.warning("上一次訓練有失敗的模型：" + "、".join(ROLE_LABELS.get(m, m) for m in train_status['failed']) + "，請查看下方的訓練日誌。",
                           icon=":material/warning:")

            requested = None
            if st.button("依序訓練全部模型（立銑 → 直銑 → 銑法分類器 → 傳統視覺 RF）", type="primary",
                         icon=":material/playlist_play:", width="stretch", disabled=training_busy):
                requested = "All"
            for col, (role, label, icon) in zip(st.columns(4), TRAIN_BUTTONS):
                if col.button(label, icon=icon, width="stretch", disabled=training_busy):
                    requested = role
            if requested:
                start_training(requested)

        with st.container(border=True, key="card_train_log"):
            card_header("terminal", "訓練日誌即時監控", "選擇日誌後啟動即時監聽，每 2 秒更新一次")
            try:
                r = requests.get(f'{API_URL}/train_logs', headers=HEADERS)
                logs = r.json().get('logs', [])
            except Exception: logs = []

            sel = st.selectbox('選擇要監控的訓練日誌 (Log)', [''] + logs)
            if sel:
                col_ctrl1, col_ctrl2 = st.columns(2)
                with col_ctrl1:
                    if st.session_state.get('monitor') != sel:
                        if st.button('啟動即時監聽', icon=":material/play_circle:", width="stretch"):
                            st.session_state['monitor'] = sel
                            st.rerun()
                with col_ctrl2:
                    if st.session_state.get('monitor') == sel:
                        if st.button('停止監聽', icon=":material/stop_circle:", width="stretch"):
                            st.session_state['monitor'] = ''
                            st.rerun()

                if st.session_state.get('monitor') == sel:
                    col_chart, col_term = st.columns([1, 1])
                    chart_placeholder = col_chart.empty()
                    term_placeholder = col_term.empty()
                    while st.session_state.get('monitor') == sel:
                        try:
                            r_prog = requests.get(f'{API_URL}/train_progress/{sel}', headers=HEADERS, timeout=3)
                            progress = r_prog.json().get('progress', []) if r_prog.status_code == 200 else []
                            if progress:
                                chart_placeholder.altair_chart(loss_chart(pd.DataFrame(progress)), width='stretch')
                        except: pass
                        try:
                            r_text = requests.get(f'{API_URL}/train_logs/{sel}', headers=HEADERS, timeout=3)
                            if r_text.status_code == 200:
                                log_text = r_text.json().get('log', '')
                                lines = log_text.split('\n')
                                tail_text = '\n'.join(lines[-25:])
                                term_placeholder.code(tail_text, language='bash')
                        except: pass
                        time.sleep(2)

    with tab_model:
        with st.container(border=True, key="card_models"):
            card_header("published_with_changes", "模型版本控制 (Rollback)",
                        "每次重新訓練或切換版本前，系統會自動把當時的線上版本備份到 results/archive/，之後可以在這裡換回來。")
            try:
                r_models = requests.get(f'{API_URL}/models', headers=HEADERS)
                if r_models.status_code == 200:
                    roles_info = r_models.json().get('roles', {})
                    role = st.selectbox("選擇模型", list(ROLE_LABELS), format_func=ROLE_LABELS.get)
                    info = roles_info.get(role, {})
                    if info.get('exists'):
                        st.info(f"目前上線版本：**{info['version']}**　{describe_model_meta(info.get('meta'), info.get('up_to_date', True))}",
                                icon=":material/deployed_code:")
                    else:
                        st.warning("這個模型還沒有訓練過。", icon=":material/warning:")

                    archived = {a['file']: a for a in info.get('archived', [])}
                    if archived:
                        selected_model = st.selectbox(
                            "選擇要切換的版本", list(archived),
                            format_func=lambda f: f"{f}{'（目前上線）' if archived[f]['is_current'] else ''}　{describe_model_meta(archived[f].get('meta'), archived[f].get('up_to_date', True))}")
                        if st.button("設為上線模型 (Deploy)", type="primary", icon=":material/rocket_launch:"):
                            res = requests.post(f'{API_URL}/admin/set_active_model', params={'role': role, 'model_file': selected_model}, headers=HEADERS)
                            if res.status_code == 200: st.success(res.json().get('msg'))
                            else: st.error(res.json().get('error') or res.text)
                    else:
                        st.caption("目前還沒有備份的舊版本，重新訓練一次後就會出現。")
                else: st.error(f"獲取模型清單失敗 (狀態碼 {r_models.status_code})：{r_models.text}")
            except Exception as e: st.error(f"獲取模型清單失敗: {e}")

    with tab_data:
        csv_path = os.path.join(BASE_DIR, 'data', 'final_training_manifest.csv')
        if os.path.exists(csv_path):
            df_data = pd.read_csv(csv_path)
            type_counts = df_data['machining_type'].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("有效訓練影像", f"{len(df_data)} 張", icon=":material/photo_library:", border=True)
            c2.metric("立銑 (End_Milling)", f"{type_counts.get('End_Milling', 0)} 張", border=True)
            c3.metric("直銑 (Peripheral_Milling)", f"{type_counts.get('Peripheral_Milling', 0)} 張", border=True)
            c4.metric("其他 (Other)", f"{type_counts.get('Other', 0)} 張", border=True)
            col1, col2 = st.columns(2, gap="medium")
            with col1, st.container(border=True, key="card_data_speed"):
                card_header("speed", "主軸轉速 (RPM) 數據分佈")
                speed_df = df_data['speed'].value_counts().sort_index().reset_index()
                speed_df.columns = ['主軸轉速 (RPM)', '影像數']
                speed_chart = alt.Chart(speed_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color='#6366F1').encode(
                    x=alt.X('主軸轉速 (RPM):N', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('影像數:Q'),
                    tooltip=['主軸轉速 (RPM)', '影像數']
                ).properties(height=350, background='transparent')
                st.altair_chart(speed_chart, width='stretch')
            with col2, st.container(border=True, key="card_data_type"):
                card_header("donut_large", "銑削加工法 (特徵類別) 比例")
                summary_df = type_counts.reset_index()
                summary_df.columns = ['加工類型', '影像總數']
                pie_chart = alt.Chart(summary_df).mark_arc(innerRadius=70, cornerRadius=4, padAngle=0.01).encode(
                    theta=alt.Theta(field="影像總數", type="quantitative"),
                    color=alt.Color(field="加工類型", type="nominal",
                                    scale=alt.Scale(domain=['End_Milling', 'Peripheral_Milling', 'Other'],
                                                    range=['#6366F1', '#10B981', '#EF4444']),
                                    legend=alt.Legend(orient='bottom', title=None)),
                    tooltip=['加工類型', '影像總數']
                ).properties(height=350, background='transparent')
                st.altair_chart(pie_chart, width='stretch')

    with tab_history:
        pred_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
        if os.path.exists(pred_path):
            with st.container(border=True, key="card_history"):
                df_pred = pd.read_csv(pred_path)
                # 紀錄存的是 UTC 秒數，換成這台電腦的時區（台灣是 UTC+8）再顯示
                local_tz = datetime.now().astimezone().tzinfo
                df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'], unit='s', utc=True).dt.tz_convert(local_tz).dt.tz_localize(None)
                df_pred = df_pred.sort_values('timestamp', ascending=False).reset_index(drop=True)
                card_header("history", "歷史預測稽核日誌", f"共 {len(df_pred)} 筆預測紀錄")
                st.dataframe(df_pred, hide_index=True, column_config={"ra": st.column_config.NumberColumn(format="%.4f")})
                csv = df_pred.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="匯出預測紀錄 (CSV)", data=csv, file_name='system_predictions_log.csv', mime='text/csv',
                                   icon=":material/download:")

    with tab_stats:
        with st.container(border=True, key="card_stats"):
            card_header("memory", "伺服器即時狀態", "按下按鈕讀取後端伺服器目前的 CPU、記憶體與 GPU 狀態")
            if st.button('重新整理狀態', type="primary", icon=":material/refresh:"):
                try:
                    s = requests.get(f'{API_URL}/admin/stats', headers=HEADERS)
                    if s.status_code == 200:
                        stats = s.json()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("CPU 使用率", f"{stats['cpu']} %", icon=":material/speed:", border=True)
                        col2.metric("記憶體使用率", f"{stats['mem']['percent']} %", icon=":material/memory_alt:", border=True)
                        col3.metric("GPU 狀態", "啟動" if stats['gpu']['available'] else "未偵測到",
                                    icon=":material/developer_board:", border=True)
                except Exception as e: st.error(str(e))


# ==========================================
# 🚦 頁面路由：先登入，再依權限顯示頁面
# ==========================================
if 'role' not in st.session_state:
    st.session_state['role'] = None

pages = [
    st.Page(page_single, title="單筆影像檢測作業", icon=":material/center_focus_strong:", url_path="inspect", default=True),
    st.Page(page_batch, title="批量驗證與精度分析", icon=":material/fact_check:", url_path="batch"),
    st.Page(page_admin, title="系統管理與模型控制台", icon=":material/admin_panel_settings:", url_path="admin"),
]
current_page = st.navigation(pages, position="hidden")

if st.session_state['role'] is None:
    page_login()
    st.stop()

allowed_pages = pages if st.session_state['role'] == 'admin' else pages[:2]
render_sidebar(allowed_pages)
if current_page not in allowed_pages:  # 作業員直接輸入網址也進不了管理頁
    st.error("這個頁面需要系統管理員權限。", icon=":material/block:")
    st.stop()
current_page.run()
