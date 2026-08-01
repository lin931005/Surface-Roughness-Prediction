import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import io
import os
import random
import re
import altair as alt

# ==========================================
# 🌐 全域變數設定
# ==========================================
API_URL = "http://127.0.0.1:2578"

# ==========================================
# 🎨 網頁基礎與標題設定
# ==========================================
st.set_page_config(page_title='CNC 表面粗糙度自動化檢測系統', page_icon="⚙️", layout='wide')

st.title('🚀 CNC 加工表面粗糙度自動化檢測系統')
st.markdown("本系統採用 **ResNet-50 雙通道深度學習架構**，提供高精準度 Ra 值自動估算與批量驗證分析。")

# 💡 側邊欄切換操作模式
tab = st.sidebar.radio('切換操作環境', [
    '👨‍🔧 單筆影像檢測作業',
    '🧪 批量驗證與精度分析 (Batch Evaluation)',
    '👑 系統管理與模型控制台'
])

# ==========================================
# 🛠️ 工具函式：從檔名自動解析真實標籤 (Ground Truth)
# ==========================================
def parse_filename_gt(filename: str):
    """從檔名 (如: 立銑_7000-7_1.6492.jpg) 自動解析真實標籤"""
    gt_type_code = None
    gt_type_text = "未知"

    # 1. 辨識銑法
    if "立銑" in filename or "End_Milling" in filename or "End" in filename:
        gt_type_code = "End_Milling"
        gt_type_text = "立銑"
    elif "直銑" in filename or "Peripheral_Milling" in filename or "Peripheral" in filename:
        gt_type_code = "Peripheral_Milling"
        gt_type_text = "直銑 (躺銑)"

    # 2. 提取 Ra 數值 (抓取檔名最後一個浮點數)
    gt_ra = None
    matches = re.findall(r'(\d+\.\d+)', filename)
    if matches:
        gt_ra = float(matches[-1])

    return gt_type_code, gt_type_text, gt_ra

# ==========================================
# 👨‍🔧 模式 A：單張檢測
# ==========================================
if tab == '👨‍🔧 單筆影像檢測作業':
    st.info("💡 **操作說明**：請上傳工件表面影像。系統將自動辨識銑削方式，亦可手動覆寫參數以提升分析精度。")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        milling_type_selection = st.selectbox(
            "設定銑削加工法 (預設: 自動特徵辨識)",
            ("自動辨識 (Auto)", "立銑 (End Milling)", "直銑(躺銑) (Peripheral Milling)")
        )
    with col_opt2:
        has_params = st.checkbox("⚙️ 附加主軸轉速 (提升準確度)")
        speed_rpm = 5000
        if has_params:
            speed_rpm = st.number_input("主軸轉速 (RPM)", min_value=1000, max_value=10000, value=5000, step=100)

    type_map = {
        "自動辨識 (Auto)": "Auto",
        "立銑 (End Milling)": "End_Milling",
        "直銑 (Peripheral Milling)": "Peripheral_Milling"
    }
    selected_type_api = type_map[milling_type_selection]

    st.markdown("---")
    uploaded = st.file_uploader('📸 上傳表面影像 (支援 png, jpg, jpeg)', type=['png','jpg','jpeg'])

    if uploaded is not None:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption='待測工件影像', width=400)

        use_gc = st.checkbox('顯示 Grad-CAM 特徵啟動熱力圖', value=True)

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
                    r = requests.post(f'{API_URL}/predict', files=files, params=params, timeout=30)
                    if r.status_code == 200:
                        j = r.json()
                        if 'error' in j:
                            st.error(f"分析失敗：{j['error']}")
                        else:
                            # 💡 1. 抓取所有變數
                            ai_conf = j.get('ai_confidence', 1.0) * 100
                            preds_std = j.get('preds_std', 0.0)
                            preds_edge = j.get('preds_edge', 0.0)

                            # 🛡️ 2. 異常防禦機制
                            if j.get('is_anomaly'):
                                if not st.session_state['force_override']:
                                    st.error(f"🚨 **資料驗證失敗：已中止分析流程**")

                                    # 專業判斷攔截原因
                                    if ai_conf == 0.0:
                                        st.warning("🚨 **影像特徵不符警告：** 系統判定此影像缺乏有效之金屬切削紋理 (分類標籤: OOD/其他)，已中斷粗糙度分析流程以確保數據可靠性。")
                                    elif ai_conf < 85.0:
                                        st.warning(f"🚨 **影像品質警告：** 系統對此影像之特徵辨識度偏低 (置信度 {ai_conf:.1f}%)。可能原因為對焦模糊或非標準切削表面，分析已中斷。")

                                    # 迷因圖防呆
                                    meme_folder = os.path.join("data", "Utfg2026")
                                    if os.path.exists(meme_folder):
                                        valid_exts = ('.png', '.jpg', '.jpeg')
                                        all_images = [f for f in os.listdir(meme_folder) if f.lower().endswith(valid_exts)]
                                        if all_images:
                                            if 'meme_playlist' not in st.session_state or not st.session_state['meme_playlist']:
                                                st.session_state['meme_playlist'] = all_images.copy()
                                                random.shuffle(st.session_state['meme_playlist'])

                                            current_meme = st.session_state['meme_playlist'].pop(0)
                                            st.image(os.path.join(meme_folder, current_meme))
                                            st.markdown("<h4 style='text-align: center; color: #ff4b4b;'>⚠️ 系統已中斷非標準影像之分析<br>（以上為參考圖片）</h4>", unsafe_allow_html=True)
                                            st.button("🔄 載入其他參考範例", on_click=trigger_next_meme)
                                    st.button("⚠️ 強制忽略警告並執行分析", on_click=trigger_override)
                                    st.stop()
                                else:
                                    st.success("⚠️ 提示：已手動覆寫安全攔截設定，強制執行特徵數值分析。")

                            # ============ 3. 正常預測結果顯示 ============
                            detected_type = j.get('detected_milling')
                            display_type = "立銑 (End Milling)" if detected_type == "End_Milling" else "直銑 (躺銑) (Peripheral Milling)"

                            st.success(f"### ✨ 表面粗糙度估算值 (Ra): **{j.get('ra'):.4f} μm**")
                            st.info(f"⚙️ 系統當前調用之特徵萃取模型：**{display_type}**")

                            st.markdown("#### 🔬 影像預處理與特徵萃取可視化")
                            col_orig, col_bw = st.columns(2)
                            with col_orig: st.image(img, caption='1. 原始彩色輸入', width='stretch')
                            with col_bw: st.image(img.convert('L'), caption='2. 灰階紋理強化 (演算法分析特徵)', width='stretch')

                            st.info(f"📊 **系統狀態面板**：特徵置信度 (Confidence): **{ai_conf:.1f}%**")

                            if j.get('heatmap'): st.image(j.get('heatmap'), caption='Grad-CAM 表面紋理熱力圖', width='stretch')

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
                                st.image(img_with_boxes, caption='隨機取樣區域可視化 (綠框: 採用, 紅/黃框: 剔除極端值)', width='stretch')

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
# 🧪 模式 B：批量驗證與精度分析 (全新大分頁)
# ==========================================
elif tab == '🧪 批量驗證與精度分析 (Batch Evaluation)':
    st.subheader('🧪 批量測試與模型精準度評估看板')
    st.info('💡 **使用說明**：請直接全選並拖拽多張測試影像（需包含銑法與 Ra 數值，如：`立銑_7000-7_1.6492.jpg`）。系統將自動解析真實數據，與預測結果進行比對。')

    batch_files = st.file_uploader('📸 批量上傳測試影像 (可按 Ctrl+A 全選上傳)', type=['png','jpg','jpeg'], accept_multiple_files=True)

    if batch_files:
        st.success(f"📂 已成功載入 **{len(batch_files)}** 張待測影像！")

        if st.button('🚀 執行批量特徵分析與精度對照', use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []

            for idx, file in enumerate(batch_files):
                status_text.text(f"⏳ 運算中：第 ({idx+1}/{len(batch_files)}) 筆影像：{file.name}...")

                gt_type_code, gt_type_text, gt_ra = parse_filename_gt(file.name)

                files_payload = {'file': (file.name, file.getvalue(), 'image/jpeg')}
                api_params = {'milling_type': 'Auto'}

                try:
                    r = requests.post(f'{API_URL}/predict', files=files_payload, params=api_params, timeout=15)
                    if r.status_code == 200:
                        j = r.json()
                        ai_conf = j.get('ai_confidence', 1.0) * 100
                        pred_type_code = j.get('detected_milling')
                        pred_type_text = "立銑" if pred_type_code == "End_Milling" else "直銑 (躺銑)" if pred_type_code == "Peripheral_Milling" else "未知"
                        pred_ra = j.get('ra')
                        is_anomaly = j.get('is_anomaly', False)

                        type_correct = (gt_type_code == pred_type_code) if gt_type_code else None

                        abs_err = abs(pred_ra - gt_ra) if (pred_ra is not None and gt_ra is not None) else None
                        pct_err = (abs_err / gt_ra * 100.0) if (abs_err is not None and gt_ra) else None

                        results.append({
                            "圖片檔名": file.name,
                            "真實銑法": gt_type_text,
                            "系統判定銑法": pred_type_text,
                            "特徵置信度 (%)": round(ai_conf, 1),
                            "銑法辨識": "✅ 正確" if type_correct else "❌ 誤判" if type_correct is False else "❓ 未知",
                            "真實 Ra (μm)": round(gt_ra, 4) if gt_ra is not None else np.nan,
                            "預測 Ra (μm)": round(pred_ra, 4) if pred_ra is not None else np.nan,
                            "絕對誤差 (μm)": round(abs_err, 4) if abs_err is not None else np.nan,
                            "偏差率 (%)": round(pct_err, 2) if pct_err is not None else np.nan,
                            "影像狀態": "🚨 異常影像" if is_anomaly else "✅ 正常"
                        })
                    else:
                        results.append({"圖片檔名": file.name, "影像狀態": f"❌ API 錯誤 ({r.status_code})"})
                except Exception as e:
                    results.append({"圖片檔名": file.name, "影像狀態": f"❌ 錯誤: {type(e).__name__}"})

                progress_bar.progress((idx + 1) / len(batch_files))

            status_text.text("✅ 所有影像批量分析完畢！")

            df_res = pd.DataFrame(results)

            # ==========================================
            # 📊 第一區塊：核心 KPI 指標統計儀表板
            # ==========================================
            st.markdown("---")
            st.markdown("### 🎯 系統綜合效能 KPI 統計")

            if '真實 Ra (μm)' in df_res.columns and '預測 Ra (μm)' in df_res.columns:
                valid_df = df_res.dropna(subset=['真實 Ra (μm)', '預測 Ra (μm)'])
            else:
                valid_df = pd.DataFrame()

            c1, c2, c3, c4 = st.columns(4)

            if '銑法辨識' in df_res.columns:
                type_checked = df_res[df_res['銑法辨識'] != "❓ 未知"]
                acc = (type_checked['銑法辨識'] == "✅ 正確").mean() * 100 if not type_checked.empty else 0.0
                c1.metric("👁️ 銑法辨識正確率", f"{acc:.1f} %")
            else:
                c1.metric("👁️ 銑法辨識正確率", "N/A")

            mae = valid_df['絕對誤差 (μm)'].mean() if not valid_df.empty else 0.0
            c2.metric("📏 平均絕對誤差 (MAE)", f"{mae:.4f} μm")

            mape = valid_df['偏差率 (%)'].mean() if not valid_df.empty else 0.0
            c3.metric("📉 平均相對偏差率 (MAPE)", f"{mape:.2f} %")

            c4.metric("📸 測試影像總數", f"{len(df_res)} 張")

            # ==========================================
            # 📈 第二區塊：視覺化圖表分析
            # ==========================================
            st.markdown("---")
            st.markdown("### 📈 預測結果圖表分析")

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("**1. 真實 Ra vs 預測 Ra 擬合散佈圖 (越接近紅線越準確)**")
                if not valid_df.empty:
                    scatter = alt.Chart(valid_df).mark_circle(size=80).encode(
                        x=alt.X('真實 Ra (μm)', scale=alt.Scale(zero=False)),
                        y=alt.Y('預測 Ra (μm)', scale=alt.Scale(zero=False)),
                        color=alt.Color('銑法辨識', scale=alt.Scale(domain=['✅ 正確', '❌ 誤判'], range=['#28a745', '#dc3545'])),
                        tooltip=['圖片檔名', '真實銑法', '系統判定銑法', '真實 Ra (μm)', '預測 Ra (μm)', '偏差率 (%)']
                    )
                    min_val = min(valid_df['真實 Ra (μm)'].min(), valid_df['預測 Ra (μm)'].min())
                    max_val = max(valid_df['真實 Ra (μm)'].max(), valid_df['預測 Ra (μm)'].max())
                    line_df = pd.DataFrame({'x': [min_val, max_val], 'y': [min_val, max_val]})
                    ref_line = alt.Chart(line_df).mark_line(color='red', strokeDash=[5, 5]).encode(x='x', y='y')

                    st.altair_chart((scatter + ref_line).properties(height=350), use_container_width=True)

            with col_chart2:
                st.markdown("**2. 每張照片之偏差率 (%) 長條圖**")
                if not valid_df.empty:
                    bar = alt.Chart(valid_df).mark_bar().encode(
                        x=alt.X('圖片檔名', sort=None, axis=alt.Axis(labels=False), title="測試樣本"),
                        y=alt.Y('偏差率 (%)', title="偏差率 (%)"),
                        color=alt.condition(
                            alt.datum['偏差率 (%)'] < 10.0,
                            alt.value('#28a745'),
                            alt.value('#dc3545')
                        ),
                        tooltip=['圖片檔名', '真實 Ra (μm)', '預測 Ra (μm)', '偏差率 (%)']
                    ).properties(height=350)
                    st.altair_chart(bar, use_container_width=True)

            # ==========================================
            # 📋 第三區塊：詳細對照數據表格與匯出
            # ==========================================
            st.markdown("---")
            st.markdown("### 📋 完整比對數據明細表")
            st.dataframe(df_res, use_container_width=True)

            csv_out = df_res.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 匯出完整測試驗證報告 (CSV)",
                data=csv_out,
                file_name="AI_Milling_Batch_Evaluation_Report.csv",
                mime="text/csv"
            )

# ==========================================
# 👑 模式 C：系統管理員 (MLOps 中控台)
# ==========================================
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    st.subheader('👑 系統管理與模型控制台')

    if 'token' not in st.session_state: st.session_state['token'] = ''

    with st.expander("🔑 系統管理員登入", expanded=not bool(st.session_state['token'])):
        username = st.text_input('帳號 (Username)', value='admin')
        password = st.text_input('密碼 (Password)', type='password')
        if st.button('登入系統', use_container_width=True):
            try:
                r = requests.post(f'{API_URL}/login', params={'username': username, 'password': password})
                if r.status_code == 200:
                    st.session_state['token'] = r.json().get('access_token')
                    st.success('登入成功！')
                    st.rerun()
                else: st.error('登入失敗，帳號或密碼錯誤。')
            except Exception as e: st.error(f"連線錯誤：{str(e)}")

    token = st.session_state.get('token','')

    if token:
        headers = {'Authorization': f'Bearer {token}'}

        tab_train, tab_model, tab_data, tab_history, tab_stats = st.tabs([
            "🚀 訓練與終端機", "🤖 模型熱切換", "📊 資料集分析", "📜 預測紀錄與稽核", "🖥️ 硬體監控"
        ])

        with tab_train:
            st.markdown("#### 🚀 啟動模型訓練管線 (Training Pipeline)")
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("⚙️ 啟動【立銑】回歸模型訓練", use_container_width=True, type="primary"):
                    try:
                        res = requests.post(f"{API_URL}/train?milling_type=End_Milling&token=admin-token", headers=headers)
                        st.success(res.json().get("message", "指令發送成功"))
                    except Exception as e: st.error(str(e))
            with col_btn2:
                if st.button("⚙️ 啟動【直銑】回歸模型訓練", use_container_width=True, type="primary"):
                    try:
                        res = requests.post(f"{API_URL}/train?milling_type=Peripheral_Milling&token=admin-token", headers=headers)
                        st.success(res.json().get("message", "指令發送成功"))
                    except Exception as e: st.error(str(e))
            with col_btn3:
                if st.button("📊 啟動【銑法分類器】模型訓練", use_container_width=True):
                    try:
                        res = requests.post(f"{API_URL}/train?milling_type=Classifier&token=admin-token", headers=headers)
                        st.success(res.json().get("message", "指令發送成功"))
                    except Exception as e: st.error(str(e))

            st.markdown("---")
            try:
                r = requests.get(f'{API_URL}/train_logs', headers=headers)
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
                            r_prog = requests.get(f'{API_URL}/train_progress/{sel}', headers=headers, timeout=3)
                            progress = r_prog.json().get('progress', []) if r_prog.status_code == 200 else []
                            if progress:
                                df = pd.DataFrame(progress).set_index('epoch')
                                chart_placeholder.line_chart(df[['train_loss','val_loss']])
                        except: pass

                        try:
                            r_text = requests.get(f'{API_URL}/train_logs/{sel}', headers=headers, timeout=3)
                            if r_text.status_code == 200:
                                log_text = r_text.json().get('log', '')
                                lines = log_text.split('\n')
                                tail_text = '\n'.join(lines[-25:])
                                term_placeholder.code(tail_text, language='bash')
                        except: pass
                        time.sleep(2)

        with tab_model:
            st.markdown("#### 🔄 模型版本控制 (Rollback)")
            try:
                r_models = requests.get(f'{API_URL}/models')
                if r_models.status_code == 200:
                    model_list = [m['file'] for m in r_models.json().get('models', [])]
                    if model_list:
                        selected_model = st.selectbox("選擇要載入的歷史模型檔案", model_list)
                        if st.button("🌟 設為上線模型 (Deploy)", type="primary"):
                            res = requests.post(f'{API_URL}/admin/set_active_model', params={'model_file': selected_model}, headers=headers)
                            if res.status_code == 200: st.success(res.json().get('msg'))
                            else: st.error(res.json().get('error'))
            except Exception as e: st.error(f"獲取模型清單失敗: {e}")

        with tab_data:
            st.markdown("#### 📊 當前訓練資料集分佈")
            csv_path = os.path.join(BASE_DIR, 'data', 'final_training_manifest.csv')
            if os.path.exists(csv_path):
                df_data = pd.read_csv(csv_path)
                st.success(f"目前資料庫中共有 **{len(df_data)}** 張有效訓練影像。")
                col1, col2 = st.columns(2)
                with col1: st.bar_chart(df_data['speed'].value_counts())
                with col2: st.dataframe(df_data.head())

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
                    s = requests.get(f'{API_URL}/admin/stats', headers=headers)
                    if s.status_code == 200:
                        stats = s.json()
                        col1, col2, col3 = st.columns(3)
                        col1.metric("CPU 使用率", f"{stats['cpu']} %")
                        col2.metric("記憶體使用率", f"{stats['mem']['percent']} %")
                        col3.metric("GPU 狀態", "✅ 啟動" if stats['gpu']['available'] else "❌ 未偵測到")
                except Exception as e: st.error(str(e))
    else:
        st.warning("⚠️ 請先於上方登入系統，以解鎖完整 MLOps 功能。")
