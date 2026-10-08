import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from PIL import Image, ImageDraw, ImageOps
import io
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

# ==========================================
# 🎨 網頁基礎與標題設定
# ==========================================
st.set_page_config(page_title='CNC 表面粗糙度自動化檢測系統', page_icon="⚙️", layout='wide')

if not (API_SECRET_KEY and ADMIN_PASSWORD and USER_PASSWORD):
    st.error("⚠️ 系統尚未設定密碼與 API 金鑰：請參考專案根目錄的 .env.example 建立 .env，再重新啟動系統。")
    st.stop()

# ==========================================
# 🔒 系統安全登入驗證區塊 (純密碼分級制)
# ==========================================
def password_matches(entered: str, expected: str) -> bool:
    return secrets.compare_digest(entered.encode('utf-8'), expected.encode('utf-8'))

if 'role' not in st.session_state:
    st.session_state['role'] = None

if st.session_state['role'] is None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("🔒 **安全防護鎖**：請輸入系統通行碼以解鎖功能。")
        with st.form("login_form"):
            pwd_input = st.text_input("系統通行密碼 (Password)", type="password", help="輸入不同密碼將解鎖不同權限")
            submitted = st.form_submit_button("解鎖系統", use_container_width=True, type="primary")
            if submitted:
                if password_matches(pwd_input, ADMIN_PASSWORD):
                    st.session_state['role'] = 'admin'
                    st.rerun()
                elif password_matches(pwd_input, USER_PASSWORD):
                    st.session_state['role'] = 'user'
                    st.rerun()
                else:
                    st.error("❌ 通行碼錯誤，請重新輸入。")
    st.stop()

# ==========================================
# 🔓 登入成功後的主程式區塊
# ==========================================
if st.sidebar.button("🚪 登出系統", use_container_width=True):
    st.session_state['role'] = None
    st.rerun()

st.sidebar.markdown("---")

if st.session_state['role'] == 'admin':
    st.sidebar.markdown("👤 歡迎登入, **👑 系統管理員**")
    available_tabs = [
        '👨‍🔧 單筆影像檢測作業',
        '🧪 批量驗證與精度分析 (Batch Evaluation)',
        '👑 系統管理與模型控制台'
    ]
else:
    st.sidebar.markdown("👤 歡迎登入, **👨‍🔧 現場作業員**")
    available_tabs = [
        '👨‍🔧 單筆影像檢測作業',
        '🧪 批量驗證與精度分析 (Batch Evaluation)'
    ]

tab = st.sidebar.radio('切換操作環境', available_tabs)

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

def parse_filename_condition(filename: str):
    """從檔名讀取主軸轉速與條件編號，例如「直銑_7000-0.5_2.3746.jpg」→ (7000.0, '7000-0.5')"""
    match = re.search(r'(?<![\d.])(\d{4,5})-(\d+(?:\.\d+)?)(?![\d.\-])', filename)
    if not match or not 1000 <= int(match.group(1)) <= 10000:
        return None, None
    return float(match.group(1)), f"{match.group(1)}-{match.group(2)}"

def describe_model_meta(meta, preprocess_ok=True):
    """把模型的 meta.json 整理成一行說明"""
    warning = "" if preprocess_ok else "　⚠️ 舊版前處理，需重新訓練"
    if not meta:
        return "（舊版模型，沒有驗證紀錄）" + warning
    if 'val_mae' in meta:
        score = f"驗證 MAE {meta['val_mae']:.4f} μm"
    elif 'val_acc' in meta:
        score = f"驗證正確率 {meta['val_acc'] * 100:.1f}%"
    else:
        score = ""
    return f"訓練於 {meta.get('trained_at', '?')}，{score}".rstrip("，") + warning

# ==========================================
# 👨‍🔧 模式 A：單張檢測 (支援雙引擎切換)
# ==========================================
if tab == '👨‍🔧 單筆影像檢測作業':
    st.info("💡 **操作說明**：請上傳工件表面影像。您可以切換不同的 AI 引擎來比較預測結果。")

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        engine_selection = st.selectbox(
            "🧠 選擇 AI 運算引擎",
            ("🤖 深度學習 (ResNet 專家模型)", "⚙️ 傳統機器視覺 (OpenCV + RF)")
        )
    with col_opt2:
        milling_type_selection = st.selectbox(
            "設定銑削加工法 (預設: 自動辨識)",
            ("自動辨識 (Auto)", "立銑 (End Milling)", "直銑(躺銑) (Peripheral Milling)")
        )
    with col_opt3:
        has_params = st.checkbox("⚙️ 附加主軸轉速 (僅適用深度學習)")
        speed_rpm = 5000
        if has_params:
            speed_rpm = st.number_input("主軸轉速 (RPM)", min_value=1000, max_value=10000, value=5000, step=100)

    type_map = {
        "自動辨識 (Auto)": "Auto",
        "立銑 (End Milling)": "End_Milling",
        "直銑(躺銑) (Peripheral Milling)": "Peripheral_Milling"
    }
    selected_type_api = type_map[milling_type_selection]

    st.markdown("---")
    uploaded = st.file_uploader('📸 上傳表面影像 (支援 png, jpg, jpeg)', type=['png','jpg','jpeg'])

    if uploaded is not None:
        img = ImageOps.exif_transpose(Image.open(uploaded)).convert('RGB')  # 與後端相同，依 EXIF 轉正
        st.image(img, caption='待測工件影像', width=400)

        use_gc = st.checkbox('顯示 Grad-CAM 特徵啟動熱力圖 (僅支援深度學習)', value=True)

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

        predict_btn = st.button('⚙️ 執行粗糙度 (Ra) 分析', use_container_width=True)

        if predict_btn or st.session_state['do_predict']:
            st.session_state['do_predict'] = False

            files = {'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')}
            params = {'milling_type': selected_type_api}
            if use_gc: params['gradcam'] = 'true'
            if has_params: params['speed'] = speed_rpm

            with st.spinner('影像特徵萃取與數值運算中...'):
                try:
                    # 💡 判斷要打哪一個後端 API 路徑
                    if "深度學習" in engine_selection:
                        api_endpoint = f'{API_URL}/predict'
                    else:
                        api_endpoint = f'{API_URL}/predict/traditional'

                    r = requests.post(api_endpoint, files=files, params=params, headers=HEADERS, timeout=30)

                    if r.status_code == 200:
                        j = r.json()
                        if 'error' in j:
                            st.error(f"分析失敗：{j['error']}")
                        else:
                            # ============== 傳統引擎解析邏輯 ==============
                            # ============== 傳統引擎解析邏輯 ==============
                            if "深度學習" not in engine_selection:
                                st.success(f"### ✨ 表面粗糙度估算值 (Ra): **{j.get('ra'):.4f} μm**")
                                st.info("⚙️ 系統當前調用之特徵萃取模型：**傳統機器視覺 (OpenCV) + 隨機森林 (Random Forest)**")

                                # --- 新增：視覺化特徵萃取過程 (XAI) ---
                                st.markdown("#### 👁️ 影像特徵物理轉換過程 (Feature Extraction Pipeline)")
                                st.write("傳統演算法不依賴黑盒子神經網路，而是透過固定濾波器提取具有物理意義的表面紋理，再交由機器學習進行迴歸預測。")

                                # 💡 為了在網頁展示，我們用 OpenCV 重現後端萃取特徵時的影像變化
                                import cv2
                                import numpy as np
                                img_cv = np.array(img.convert('RGB'))
                                img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

                                # 產生特徵圖
                                edges = cv2.Canny(img_gray, 50, 150)
                                laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
                                laplacian_disp = cv2.convertScaleAbs(laplacian) # 轉為可顯示的圖片格式

                                col_v1, col_v2, col_v3 = st.columns(3)
                                with col_v1:
                                    st.image(img_gray, caption='1. 灰階轉換 (評估整體明暗對比)', width='stretch')
                                with col_v2:
                                    st.image(edges, caption='2. Canny 邊緣偵測 (擷取刀痕特徵)', width='stretch')
                                with col_v3:
                                    st.image(laplacian_disp, caption='3. Laplacian 濾波 (捕捉表面銳利度)', width='stretch')

                                # --- 顯示最終萃取數值 ---
                                st.markdown("#### 🔢 轉換為機器學習特徵向量 (Feature Vector)")
                                feats = j.get('features_extracted', {})
                                c1, c2, c3 = st.columns(3)
                                c1.metric("💡 亮度變異數", f"{feats.get('brightness_variance', 0):.1f}", "對應圖 1")
                                c2.metric("🔪 邊緣密度", f"{feats.get('edge_density', 0):.4f}", "對應圖 2")
                                c3.metric("🌫️ 模糊變異數", f"{feats.get('laplacian_variance', 0):.1f}", "對應圖 3")

                                st.markdown("---")
                                st.markdown("👉 **運算邏輯總結**：系統將上述 3 個特徵值打包成一維數學向量 `[亮度, 邊緣, 模糊度]`，輸入至已訓練完畢的 **Random Forest 決策樹群集** 中，透過 100 棵決策樹的投票機制，得出最終的 Ra 預測數值。")

                                st.stop()

                            # ============== 以下為深度學習解析邏輯 ==============
                            ai_conf = j.get('ai_confidence', 1.0) * 100

                            if j.get('is_anomaly'):
                                if not st.session_state['force_override']:
                                    st.error(f"🚨 **資料驗證失敗：已中止分析流程**")
                                    if ai_conf == 0.0:
                                        st.warning("🚨 **影像特徵不符警告：** 系統判定此影像缺乏有效之金屬切削紋理，已中斷流程。")
                                    elif ai_conf < 85.0:
                                        st.warning(f"🚨 **影像品質警告：** 系統對此影像特徵辨識度偏低 (置信度 {ai_conf:.1f}%)。")

                                    meme_folder = os.path.join("data", "example")
                                    if os.path.exists(meme_folder):
                                        valid_exts = ('.png', '.jpg', '.jpeg')
                                        all_images = [f for f in os.listdir(meme_folder) if f.lower().endswith(valid_exts)]
                                        if all_images:
                                            if 'meme_playlist' not in st.session_state or not st.session_state['meme_playlist']:
                                                st.session_state['meme_playlist'] = all_images.copy()
                                                random.shuffle(st.session_state['meme_playlist'])
                                            current_meme = st.session_state['meme_playlist'].pop(0)
                                            st.image(os.path.join(meme_folder, current_meme))
                                            st.button("🔄 載入其他參考範例", on_click=trigger_next_meme)
                                    st.button("⚠️ 強制忽略警告並執行分析", on_click=trigger_override)
                                    st.stop()
                                else:
                                    st.success("⚠️ 提示：已手動覆寫安全攔截設定，強制執行特徵數值分析。")

                            detected_type = j.get('detected_milling')
                            display_type = "立銑 (End Milling)" if detected_type == "End_Milling" else "直銑 (躺銑) (Peripheral Milling)"

                            st.success(f"### ✨ 表面粗糙度估算值 (Ra): **{j.get('ra'):.4f} μm**")
                            if j.get('preprocess_mismatch'):
                                st.warning("⚠️ 目前上線的模型是用舊版影像前處理訓練的，請到「系統管理與模型控制台」重新訓練，否則估算值不準確。")
                            st.info(f"⚙️ 系統當前調用之特徵萃取模型：**{display_type}**")

                            st.markdown("#### 🔬 影像預處理與特徵萃取可視化")
                            col_orig, col_bw = st.columns(2)
                            with col_orig: st.image(img, caption='1. 原始彩色輸入', width='stretch')
                            with col_bw: st.image(img.convert('L'), caption='2. 灰階紋理強化 (演算法分析特徵)', width='stretch')

                            st.info(f"📊 **系統狀態面板**：特徵置信度 (Confidence): **{ai_conf:.1f}%**")

                            if j.get('heatmap') and use_gc:
                                st.image(j.get('heatmap'), caption='Grad-CAM 表面紋理熱力圖 (越紅的區域對 Ra 預測值影響越大)', width='stretch')
                            elif j.get('heatmap_error') and use_gc:
                                st.warning(f"熱力圖產生失敗：{j['heatmap_error']}")

                            if 'xai_details' in j:
                                details = j['xai_details']
                                patches_info = details['patches_info']
                                st.markdown("---")
                                st.markdown("### 📊 局部特徵取樣與離群值 (Outlier) 檢測報告")

                                img_with_boxes = img.copy()
                                draw = ImageDraw.Draw(img_with_boxes)
                                for p in patches_info:
                                    coords = p['coords']
                                    color = "green" if "保留" in p['status'] else "red" if "異常高值" in p['status'] else "yellow"
                                    draw.rectangle([coords['left'], coords['top'], coords['right'], coords['bottom']], outline=color, width=4)
                                st.image(img_with_boxes, caption=f"隨機取樣區域可視化 (每個方塊 {details.get('patch_size', '?')} px；綠框: 採用, 紅/黃框: 剔除極端值)", width='stretch')

                                chart_data = [{"區塊編號": f"Patch {p['id']}", "預測粗糙度 (Ra)": p['ra'], "狀態": p['status']} for p in patches_info]
                                df = pd.DataFrame(chart_data)
                                chart = alt.Chart(df).mark_circle(size=100).encode(
                                    x=alt.X('區塊編號', sort=None, title='隨機取樣區塊 (依數值排序)'),
                                    y=alt.Y('預測粗糙度 (Ra)', scale=alt.Scale(zero=False), title='Ra 值 (μm)'),
                                    color=alt.Color('狀態', scale=alt.Scale(domain=['保留 (有效計算區間)', '剔除 (異常低值)', '剔除 (異常高值/可能含灰塵)'], range=['#28a745', '#ffc107', '#dc3545'])),
                                    tooltip=['區塊編號', '預測粗糙度 (Ra)', '狀態']
                                ).properties(height=300).interactive()
                                st.altair_chart(chart, use_container_width=True)
                    else: st.error(f"伺服器回傳異常 (狀態碼 {r.status_code})：{r.text}")
                except Exception as e: st.error(f"連線失敗：{str(e)}")

# ==========================================
# 🧪 模式 B：批量驗證與精度分析 (雙引擎大對決)
# ==========================================
elif tab == '🧪 批量驗證與精度分析 (Batch Evaluation)':
    st.subheader('🧪 批量測試與雙引擎模型對決')
    st.info('💡 **使用說明**：上傳多張影像，系統將同時啟動「深度學習」與「傳統視覺」引擎，並對比兩者之精準度。檔名建議使用「直銑_7000-3_1.565.jpg」的格式，系統會從檔名讀取銑法、轉速與真實 Ra。')

    batch_files = st.file_uploader('📸 批量上傳測試影像 (可按 Ctrl+A 全選上傳)', type=['png','jpg','jpeg'], accept_multiple_files=True)
    default_speed = st.number_input("檔名沒有轉速時使用的主軸轉速 (RPM)", min_value=1000, max_value=10000, value=5000, step=100)

    if batch_files:
        st.success(f"📂 已成功載入 **{len(batch_files)}** 張待測影像！")

        if st.button('🚀 執行雙引擎批量分析', use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            preprocess_mismatch = False

            # 💡 讀取各模型訓練時保留的驗證刀，用來判斷哪些影像是模型真的沒看過的
            try:
                r_models = requests.get(f'{API_URL}/models', headers=HEADERS, timeout=10)
                roles_info = r_models.json().get('roles', {}) if r_models.status_code == 200 else {}
            except Exception:
                roles_info = {}
            held_out = {role: set(((info or {}).get('meta') or {}).get('val_conditions', [])) for role, info in roles_info.items()}

            for idx, file in enumerate(batch_files):
                status_text.text(f"⏳ 雙引擎運算中：第 ({idx+1}/{len(batch_files)}) 筆影像...")
                gt_type_code, gt_type_text, gt_ra = parse_filename_gt(file.name)
                file_speed, condition_id = parse_filename_condition(file.name)
                speed_used = file_speed if file_speed else float(default_speed)

                # 專家模型、分類器、RF 三個模型訓練時都沒看過這一刀，比較才公平
                if gt_type_code and condition_id:
                    cond_key = f"{gt_type_code}/{condition_id}"
                    unseen = all(cond_key in held_out.get(role, set()) for role in (gt_type_code, 'Classifier', 'Traditional'))
                    seen_text = "否 (驗證集)" if unseen else "是"
                else:
                    seen_text = "❓ 未知"

                # --- 1. 深度學習引擎 ---
                files_payload_dl = {'file': (file.name, file.getvalue(), 'image/jpeg')}
                try:
                    r_dl = requests.post(f'{API_URL}/predict', files=files_payload_dl, params={'milling_type': 'Auto', 'speed': speed_used}, headers=HEADERS, timeout=15)
                    if r_dl.status_code == 200:
                        j_dl = r_dl.json()
                        preprocess_mismatch |= bool(j_dl.get('preprocess_mismatch'))
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
                    "主軸轉速 (RPM)": speed_used,
                    "轉速來源": "檔名" if file_speed else "預設值",
                    "訓練時看過這一刀": seen_text,
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
                progress_bar.progress((idx + 1) / len(batch_files))

            status_text.text("✅ 雙引擎批量分析完畢！")
            if preprocess_mismatch:
                st.warning("⚠️ 目前上線的深度學習模型是用舊版影像前處理訓練的，請先重新訓練，以下深度學習的數字不準確。")
            df_res = pd.DataFrame(results)

            # 💡 計算 MAPE (平均相對偏差率)
            df_res['偏差率 (深度學習) (%)'] = (df_res['絕對誤差 (深度學習)'] / df_res['真實 Ra (μm)']) * 100
            df_res['偏差率 (傳統視覺) (%)'] = (df_res['絕對誤差 (傳統視覺)'] / df_res['真實 Ra (μm)']) * 100

            # 💡 只用訓練時沒看過的刀計算 KPI，數字才不會偏樂觀
            unseen_df = df_res[df_res['訓練時看過這一刀'] == "否 (驗證集)"]
            scope_df = unseen_df if not unseen_df.empty else df_res
            valid_df = scope_df.dropna(subset=['真實 Ra (μm)', '預測 Ra (深度學習)', '預測 Ra (傳統視覺)'])

            # ==========================================
            # 🎯 雙引擎對決 KPI 統計儀表板 (究極細分版)
            # ==========================================
            st.markdown("---")
            st.markdown("### 🎯 雙引擎綜合效能 KPI 統計與銑法對照")
            if not unseen_df.empty:
                st.success(f"✅ 以下統計只計算訓練時沒看過的 **{len(unseen_df)}** 張影像（驗證集的刀），全部 {len(df_res)} 張的結果請看最下方的明細表。")
            else:
                st.warning("⚠️ 這批影像的刀在訓練時都出現過（或模型是舊版本，沒有記錄保留的驗證刀），以下數字會偏樂觀。用目前的資料切分重新訓練全部模型後，明細表中「訓練時看過這一刀」為「否」的影像才算公平的測試。")

            if not valid_df.empty:
                # 總結數據
                type_checked = scope_df[scope_df['銑法辨識'] != "❓ 未知"]
                acc = (type_checked['銑法辨識'] == "✅ 正確").mean() * 100 if not type_checked.empty else 0.0

                # --- 深度學習 (DL) 數據 ---
                dl_mae = valid_df['絕對誤差 (深度學習)'].mean()
                dl_end_mae = valid_df[valid_df['真實銑法'] == '立銑']['絕對誤差 (深度學習)'].mean()
                dl_peri_mae = valid_df[valid_df['真實銑法'] == '直銑 (躺銑)']['絕對誤差 (深度學習)'].mean()

                dl_mape = valid_df['偏差率 (深度學習) (%)'].mean()
                dl_end_mape = valid_df[valid_df['真實銑法'] == '立銑']['偏差率 (深度學習) (%)'].mean()
                dl_peri_mape = valid_df[valid_df['真實銑法'] == '直銑 (躺銑)']['偏差率 (深度學習) (%)'].mean()

                # --- 傳統視覺 (ML) 數據 ---
                ml_mae = valid_df['絕對誤差 (傳統視覺)'].mean()
                ml_end_mae = valid_df[valid_df['真實銑法'] == '立銑']['絕對誤差 (傳統視覺)'].mean()
                ml_peri_mae = valid_df[valid_df['真實銑法'] == '直銑 (躺銑)']['絕對誤差 (傳統視覺)'].mean()

                ml_mape = valid_df['偏差率 (傳統視覺) (%)'].mean()
                ml_end_mape = valid_df[valid_df['真實銑法'] == '立銑']['偏差率 (傳統視覺) (%)'].mean()
                ml_peri_mape = valid_df[valid_df['真實銑法'] == '直銑 (躺銑)']['偏差率 (傳統視覺) (%)'].mean()

                # --- 區塊 1：總覽 ---
                st.markdown("#### 🏆 第一階段：AI 銑法辨識與測試總覽")
                c1, c2 = st.columns(2)
                c1.metric("📸 測試樣本總數", f"{len(scope_df)} 張")
                c2.metric("👁️ 深度學習銑法辨識正確率", f"{acc:.1f} %")

                # --- 區塊 2：深度學習引擎 ---
                st.markdown("#### 🤖 深度學習引擎 (ResNet-50 雙專家) 準確度")
                # 第一排：MAE (絕對誤差)
                c3, c4, c5 = st.columns(3)
                c3.metric("📏 綜合平均誤差 (MAE)", f"{dl_mae:.4f} μm")
                c4.metric("⚙️ 立銑平均誤差 (MAE)", f"{dl_end_mae:.4f} μm" if pd.notna(dl_end_mae) else "N/A")
                c5.metric("⚙️ 直銑平均誤差 (MAE)", f"{dl_peri_mae:.4f} μm" if pd.notna(dl_peri_mae) else "N/A")
                # 第二排：MAPE (偏差率)
                c6, c7, c8 = st.columns(3)
                c6.metric("📉 綜合偏差率 (MAPE)", f"{dl_mape:.2f} %")
                c7.metric("⚙️ 立銑偏差率 (MAPE)", f"{dl_end_mape:.2f} %" if pd.notna(dl_end_mape) else "N/A")
                c8.metric("⚙️ 直銑偏差率 (MAPE)", f"{dl_peri_mape:.2f} %" if pd.notna(dl_peri_mape) else "N/A")

                # --- 區塊 3：傳統機器視覺 ---
                st.markdown("#### ⚙️ 傳統機器視覺引擎 (OpenCV + Random Forest) 準確度")
                # 第一排：MAE (絕對誤差)
                c9, c10, c11 = st.columns(3)
                c9.metric("📏 綜合平均誤差 (MAE)", f"{ml_mae:.4f} μm")
                c10.metric("⚙️ 立銑平均誤差 (MAE)", f"{ml_end_mae:.4f} μm" if pd.notna(ml_end_mae) else "N/A")
                c11.metric("⚙️ 直銑平均誤差 (MAE)", f"{ml_peri_mae:.4f} μm" if pd.notna(ml_peri_mae) else "N/A")
                # 第二排：MAPE (偏差率)
                c12, c13, c14 = st.columns(3)
                c12.metric("📉 綜合偏差率 (MAPE)", f"{ml_mape:.2f} %")
                c13.metric("⚙️ 立銑偏差率 (MAPE)", f"{ml_end_mape:.2f} %" if pd.notna(ml_end_mape) else "N/A")
                c14.metric("⚙️ 直銑偏差率 (MAPE)", f"{ml_peri_mape:.2f} %" if pd.notna(ml_peri_mape) else "N/A")
                # ==========================================
                # 📈 視覺化圖表分析
                # ==========================================
                st.markdown("---")
                st.markdown("### 📈 雙引擎精準度對決 (深度學習 vs 傳統視覺)")

                chart_data = []
                for _, row in valid_df.iterrows():
                    chart_data.append({"圖片": row['圖片檔名'], "引擎": "深度學習 (ResNet)", "絕對誤差 (μm)": row['絕對誤差 (深度學習)']})
                    chart_data.append({"圖片": row['圖片檔名'], "引擎": "傳統視覺 (RF)", "絕對誤差 (μm)": row['絕對誤差 (傳統視覺)']})

                df_chart = pd.DataFrame(chart_data)

                bar_chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X('引擎:N', title=None, axis=alt.Axis(labels=False)),
                    y=alt.Y('絕對誤差 (μm):Q', title="絕對誤差 MAE (越低越好)"),
                    color=alt.Color('引擎:N', scale=alt.Scale(domain=['深度學習 (ResNet)', '傳統視覺 (RF)'], range=['#3b82f6', '#ef4444'])),
                    column=alt.Column('圖片:N', title="測試樣本 (批次)")
                ).properties(width=30, height=350)

                st.altair_chart(bar_chart)

                # 💡 結論依本批實際數字產生，不預設哪個引擎比較好
                if pd.notna(dl_mae) and pd.notna(ml_mae):
                    if dl_mae < ml_mae:
                        verdict = f"深度學習的平均誤差較低，比傳統視覺少 {ml_mae - dl_mae:.4f} μm"
                    elif ml_mae < dl_mae:
                        verdict = f"傳統視覺的平均誤差較低，比深度學習少 {dl_mae - ml_mae:.4f} μm"
                    else:
                        verdict = "兩個引擎的平均誤差相同"
                    caution = "；樣本少於 10 張，差距不一定有代表性" if len(valid_df) < 10 else ""
                    st.info(f"💡 **本批結果**：深度學習 MAE {dl_mae:.4f} μm、傳統視覺 MAE {ml_mae:.4f} μm，{verdict}{caution}。")

            # ==========================================
            # 📋 明細與匯出
            # ==========================================
            st.markdown("---")
            st.markdown("### 📋 完整比對數據明細表")

            # 將數值四捨五入方便閱讀
            if '偏差率 (深度學習) (%)' in df_res:
                df_res['偏差率 (深度學習) (%)'] = df_res['偏差率 (深度學習) (%)'].round(2)
                df_res['偏差率 (傳統視覺) (%)'] = df_res['偏差率 (傳統視覺) (%)'].round(2)

            st.dataframe(df_res, use_container_width=True)

            csv_out = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 匯出完整測試驗證報告 (CSV)", data=csv_out, file_name="Dual_Engine_Evaluation.csv", mime="text/csv")
# ==========================================
# 👑 模式 C：系統管理員 (MLOps 中控台)
# ==========================================
else:
    st.subheader('👑 系統管理與模型控制台')

    tab_train, tab_model, tab_data, tab_history, tab_stats = st.tabs([
        "🚀 訓練與終端機", "🤖 模型熱切換", "📊 資料集分析", "📜 預測紀錄與稽核", "🖥️ 硬體監控"
    ])

    def start_training(milling_type):
        try:
            res = requests.post(f"{API_URL}/train", params={'milling_type': milling_type}, headers=HEADERS)
            if res.status_code == 200: st.success(res.json().get("message", "指令發送成功"))
            else: st.error(f"啟動失敗 (狀態碼 {res.status_code})：{res.text}")
        except Exception as e: st.error(str(e))

    with tab_train:
        st.markdown("#### 🚀 啟動模型訓練管線 (Training Pipeline)")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            if st.button("⚙️ 啟動【立銑】回歸模型訓練", use_container_width=True, type="primary"):
                start_training("End_Milling")
        with col_btn2:
            if st.button("⚙️ 啟動【直銑】回歸模型訓練", use_container_width=True, type="primary"):
                start_training("Peripheral_Milling")
        with col_btn3:
            if st.button("📊 啟動【銑法分類器】模型訓練", use_container_width=True):
                start_training("Classifier")
        with col_btn4:
            if st.button("🌲 啟動【傳統視覺 RF】模型訓練", use_container_width=True):
                start_training("Traditional")

        st.markdown("---")
        try:
            r = requests.get(f'{API_URL}/train_logs', headers=HEADERS)
            logs = r.json().get('logs', [])
        except Exception: logs = []

        sel = st.selectbox('📡 選擇要監控的訓練日誌 (Log)', [''] + logs)
        if sel:
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                if st.session_state.get('monitor') != sel:
                    if st.button('▶️ 啟動即時監聽', use_container_width=True):
                        st.session_state['monitor'] = sel
                        st.rerun()
            with col_ctrl2:
                if st.session_state.get('monitor') == sel:
                    if st.button('🛑 停止監聽', use_container_width=True):
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
                            df = pd.DataFrame(progress).set_index('epoch')
                            chart_placeholder.line_chart(df[['train_loss','val_loss']])
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
        st.markdown("#### 🔄 模型版本控制 (Rollback)")
        st.caption("每次重新訓練或切換版本前，系統會自動把當時的線上版本備份到 results/archive/，之後可以在這裡換回來。")
        try:
            r_models = requests.get(f'{API_URL}/models', headers=HEADERS)
            if r_models.status_code == 200:
                roles_info = r_models.json().get('roles', {})
                role = st.selectbox("選擇模型", list(ROLE_LABELS), format_func=ROLE_LABELS.get)
                info = roles_info.get(role, {})
                if info.get('exists'):
                    st.info(f"目前上線版本：**{info['version']}**　{describe_model_meta(info.get('meta'), info.get('preprocess_ok', True))}")
                else:
                    st.warning("這個模型還沒有訓練過。")

                archived = {a['file']: a for a in info.get('archived', [])}
                if archived:
                    selected_model = st.selectbox(
                        "選擇要切換的版本", list(archived),
                        format_func=lambda f: f"{f}{'（目前上線）' if archived[f]['is_current'] else ''}　{describe_model_meta(archived[f].get('meta'), archived[f].get('preprocess_ok', True))}")
                    if st.button("🌟 設為上線模型 (Deploy)", type="primary"):
                        res = requests.post(f'{API_URL}/admin/set_active_model', params={'role': role, 'model_file': selected_model}, headers=HEADERS)
                        if res.status_code == 200: st.success(res.json().get('msg'))
                        else: st.error(res.json().get('error') or res.text)
                else:
                    st.caption("目前還沒有備份的舊版本，重新訓練一次後就會出現。")
            else: st.error(f"獲取模型清單失敗 (狀態碼 {r_models.status_code})：{r_models.text}")
        except Exception as e: st.error(f"獲取模型清單失敗: {e}")

    with tab_data:
        st.markdown("#### 📊 當前訓練資料集分佈")
        csv_path = os.path.join(BASE_DIR, 'data', 'final_training_manifest.csv')
        if os.path.exists(csv_path):
            df_data = pd.read_csv(csv_path)
            st.success(f"目前資料庫中共有 **{len(df_data)}** 張有效訓練影像。")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**主軸轉速 (RPM) 數據分佈**")
                st.bar_chart(df_data['speed'].value_counts())
            with col2:
                st.markdown("**銑削加工法 (特徵類別) 比例**")
                summary_df = df_data['machining_type'].value_counts().reset_index()
                summary_df.columns = ['加工類型', '影像總數']
                pie_chart = alt.Chart(summary_df).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="影像總數", type="quantitative"),
                    color=alt.Color(field="加工類型", type="nominal",
                                    scale=alt.Scale(domain=['End_Milling', 'Peripheral_Milling', 'Other'],
                                                    range=['#3b82f6', '#10b981', '#ef4444'])),
                    tooltip=['加工類型', '影像總數']
                ).properties(height=350)
                st.altair_chart(pie_chart, use_container_width=True)

    with tab_history:
        st.markdown("#### 📜 歷史預測稽核日誌")
        pred_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
        if os.path.exists(pred_path):
            df_pred = pd.read_csv(pred_path)
            df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'], unit='s')
            df_pred = df_pred.sort_values('timestamp', ascending=False).reset_index(drop=True)
            st.dataframe(df_pred, use_container_width=True)
            csv = df_pred.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 匯出預測紀錄 (CSV)", data=csv, file_name='system_predictions_log.csv', mime='text/csv')

    with tab_stats:
        st.markdown("#### 🖥️ 伺服器即時狀態")
        if st.button('🔄 重新整理狀態', type="primary"):
            try:
                s = requests.get(f'{API_URL}/admin/stats', headers=HEADERS)
                if s.status_code == 200:
                    stats = s.json()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("CPU 使用率", f"{stats['cpu']} %")
                    col2.metric("記憶體使用率", f"{stats['mem']['percent']} %")
                    col3.metric("GPU 狀態", "✅ 啟動" if stats['gpu']['available'] else "❌ 未偵測到")
            except Exception as e: st.error(str(e))
