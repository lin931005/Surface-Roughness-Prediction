import streamlit as st
import requests
import pandas as pd
import time
from PIL import Image, ImageDraw, ImageFont
import io
import os
import random
import altair as alt

# ==========================================
# 🌐 全域變數設定
# ==========================================
API_URL = "http://127.0.0.1:2578"

# ==========================================
# 🎨 網頁基礎與標題設定
# ==========================================
st.set_page_config(page_title='CNC 表面粗糙度預測系統', page_icon="⚙️", layout='centered')

st.title('🚀 CNC 加工表面粗糙度 AI 預測系統')
st.markdown("本系統採用 **ResNet-50 雙專家深度學習架構**，為您提供高精準度的 Ra 值即時預測。")

tab = st.sidebar.radio('切換操作模式', ['👨‍🔧 一般使用者 (現場檢測)', '👑 系統管理員'])

# ==========================================
# 👨‍🔧 模式 A：一般使用者 (現場檢測)
# ==========================================
if tab == '👨‍🔧 一般使用者 (現場檢測)':
    st.info("💡 **操作說明**：請上傳工件照片。系統將自動辨識加工法，您也可以手動指定以獲得最高精準度。")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        # 💡 核心新增：讓使用者選擇加工法，預設交給 AI 自動判斷
        milling_type_selection = st.selectbox(
            "選擇加工法 (預設 AI 自動辨識)",
            ("自動辨識 (Auto)", "立銑 (End Milling)", "直銑 (Peripheral Milling)")
        )
    with col_opt2:
        has_params = st.checkbox("⚙️ 附加主軸轉速 (提升準確度)")
        speed_rpm = 5000
        if has_params:
            speed_rpm = st.number_input("主軸轉速 (RPM)", min_value=1000, max_value=10000, value=5000, step=100)

    # 映射中文選項到後端 API 期待的參數
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
        st.image(img, caption='您上傳的待測影像', width='stretch')

        use_gc = st.checkbox('顯示 AI 視覺熱力圖 (Grad-CAM)', value=True)

        if 'force_override' not in st.session_state:
            st.session_state['force_override'] = False
        if 'do_predict' not in st.session_state:
            st.session_state['do_predict'] = False

        if 'last_file' not in st.session_state or st.session_state['last_file'] != uploaded.name:
            st.session_state['last_file'] = uploaded.name
            st.session_state['force_override'] = False
            st.session_state['do_predict'] = False

        def trigger_override():
            st.session_state['force_override'] = True
            st.session_state['do_predict'] = True

        def trigger_next_meme():
            st.session_state['do_predict'] = True

        predict_btn = st.button('🔮 開始 AI 預測', use_container_width=True)

        if predict_btn or st.session_state['do_predict']:
            st.session_state['do_predict'] = False

            files = {'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')}
            params = {'milling_type': selected_type_api} # 💡 傳遞選擇的加工法
            if use_gc:
                params['gradcam'] = 'true'
            if has_params:
                params['speed'] = speed_rpm

            with st.spinner('神經網路高速推論中...'):
                try:
                    r = requests.post(f'{API_URL}/predict', files=files, params=params, timeout=30)
                    if r.status_code == 200:
                        j = r.json()
                        if 'error' in j:
                            st.error(f"預測失敗：{j['error']}")
                        else:
                            if j.get('is_anomaly'):
                                if not st.session_state['force_override']:
                                    st.error(f"🚨 **異常影像警告：拒絕執行分析**")
                                    st.warning(f"系統偵測到此圖片缺乏均勻的金屬切削紋理 (色彩變化度 Color Std 高達 {j.get('preds_std'):.2f})！這顯然不是一張標準的顯微鏡工件照片。")

                                    meme_folder = os.path.join("data", "Utfg2026")

                                    if os.path.exists(meme_folder):
                                        valid_exts = ('.png', '.jpg', '.jpeg')
                                        all_images = [f for f in os.listdir(meme_folder) if f.lower().endswith(valid_exts)]

                                        if all_images:
                                            if 'meme_playlist' not in st.session_state or not st.session_state['meme_playlist']:
                                                shuffled_list = all_images.copy()
                                                random.shuffle(shuffled_list)
                                                st.session_state['meme_playlist'] = shuffled_list

                                            current_meme = st.session_state['meme_playlist'].pop(0)
                                            random_img_path = os.path.join(meme_folder, current_meme)

                                            st.image(random_img_path)
                                            st.markdown("<h4 style='text-align: center; color: #ff4b4b;'>您的照片似乎不符合標準，請參考上方照片(並沒有)</h4>", unsafe_allow_html=True)
                                            st.button("🔄 換一張參考照片", on_click=trigger_next_meme)
                                        else:
                                            st.image("https://http.cat/406", caption="Not Acceptable (毛毛照片不見啦！)")
                                    else:
                                        st.image("https://http.cat/406", caption="找不到 Utfg2026 資料夾，只好繼續派貓咪上場")

                                    st.button("真的是切削照片嗎?", on_click=trigger_override)
                                    st.stop()
                                else:
                                    st.success("好吧 我相信你... 系統已解除防禦，強制執行粗糙度檢測。")

                            # 💡 顯示 AI 最終決定使用的模型
                            detected_type = j.get('detected_milling')
                            display_type = "立銑 (End Milling)" if detected_type == "End_Milling" else "直銑 (Peripheral Milling)"

                            st.success(f"### ✨ 預測表面粗糙度 (Ra): **{j.get('ra'):.4f} μm**")
                            st.info(f"🧠 本次推論啟用的專家大腦：**{display_type} 模型**")

                            st.markdown("#### 🔬 AI 視覺預處理可視化")
                            col_orig, col_bw = st.columns(2)
                            with col_orig:
                                st.image(img, caption='1. 原始彩色輸入', use_column_width=True)
                            with col_bw:
                                bw_img = img.convert('L')
                                st.image(bw_img, caption='2. 濾除色彩雜訊 (AI 實際分析之特徵)', use_column_width=True)

                            preds_std = j.get('preds_std', 0)
                            preds_edge = j.get('preds_edge', '未提供')
                            st.info(f"📊 **參數監控面板**：Color Std (色彩變異): **{preds_std:.2f}**  | Laplacian (紋理邊緣): **{preds_edge}**")

                            if j.get('used_default_params'):
                                st.warning("⚠️ 本次預測採用【純視覺分析】(套用基準轉速)。若輸入實際轉速，預測將更精準喔！")
                            else:
                                st.info("🎯 本次預測採用【影像 + 轉速 雙通道高精度分析】")

                            if j.get('heatmap'):
                                st.image(j.get('heatmap'), caption='Grad-CAM 刀痕解析熱力圖', width='stretch')

                            if 'xai_details' in j:
                                details = j['xai_details']
                                patches_info = details['patches_info']

                                st.markdown("---")
                                st.markdown("### 📊 AI 決策深度解析 (XAI) 檢測報告")
                                st.info(f"💡 **檢測原理說明**：本系統採用 **蒙地卡羅隨機取樣 (Monte Carlo TTA)** 技術。為了避免金屬表面的灰塵或異常反光干擾，AI 隨機在影像中擷取了 **{details['num_patches']}** 個局部區塊進行獨立分析，並自動剔除偏差最大的極端值，確保最終結果精準可靠。")

                                img_with_boxes = img.copy()
                                draw = ImageDraw.Draw(img_with_boxes)

                                for p in patches_info:
                                    coords = p['coords']
                                    if "保留" in p['status']:
                                        color = "green"
                                    elif "異常高值" in p['status']:
                                        color = "red"
                                    else:
                                        color = "yellow"

                                    draw.rectangle(
                                        [coords['left'], coords['top'], coords['right'], coords['bottom']],
                                        outline=color,
                                        width=4
                                    )

                                st.image(img_with_boxes, caption='隨機取樣區域可視化 (綠框: 採用, 紅/黃框: 剔除極端值)', width='stretch')

                                col1, col2, col3 = st.columns(3)
                                col1.metric("總取樣次數", f"{details['num_patches']} 次")
                                col2.metric("剔除極端值數量", f"{details['trim_count'] * 2} 次")
                                col3.metric("計算平均採用次數", f"{details['num_patches'] - (details['trim_count'] * 2)} 次")

                                chart_data = []
                                for p in patches_info:
                                    chart_data.append({
                                        "區塊編號": f"Patch {p['id']}",
                                        "預測粗糙度 (Ra)": p['ra'],
                                        "狀態": p['status']
                                    })
                                df = pd.DataFrame(chart_data)

                                chart = alt.Chart(df).mark_circle(size=100).encode(
                                    x=alt.X('區塊編號', sort=None, title='隨機取樣區塊 (依數值排序)'),
                                    y=alt.Y('預測粗糙度 (Ra)', scale=alt.Scale(zero=False), title='Ra 值 (μm)'),
                                    color=alt.Color('狀態', scale=alt.Scale(
                                        domain=['保留 (有效計算區間)', '剔除 (異常低值)', '剔除 (異常高值/可能含灰塵)'],
                                        range=['#28a745', '#ffc107', '#dc3545']
                                    )),
                                    tooltip=['區塊編號', '預測粗糙度 (Ra)', '狀態']
                                ).properties(
                                    height=300
                                ).interactive()

                                st.altair_chart(chart, use_container_width=True)

                    else:
                        st.error(f"後端報錯 (代碼 {r.status_code})：{r.text}")
                except Exception as e:
                    st.error(f"系統連線失敗，請確認 FastAPI (Port 2578) 是否已啟動。錯誤：{str(e)}")

# ==========================================
# 👑 模式 B：系統管理員 (MLOps 專業中控台)
# ==========================================
else:
    import os
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    st.subheader('👑 MLOps 系統管理員專業中控台')
    st.markdown("提供模型重練、終端機監控、熱切換、數據分析與系統稽核功能。")

    if 'token' not in st.session_state:
        st.session_state['token'] = ''

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
                else:
                    st.error('登入失敗，帳號或密碼錯誤。')
            except Exception as e:
                st.error(f"連線錯誤：{str(e)}")

    token = st.session_state.get('token','')

    if token:
        headers = {'Authorization': f'Bearer {token}'}

        tab_train, tab_model, tab_data, tab_history, tab_stats = st.tabs([
            "🚀 訓練與終端機", "🤖 模型熱切換", "📊 資料集分析", "📜 預測紀錄與校正", "🖥️ 硬體監控"
        ])

        # ----------------------------------------------------
        # 分頁 1：訓練與即時終端機 (💡 新增專家訓練按鈕)
        # ----------------------------------------------------
        with tab_train:
            st.markdown("#### 🚀 啟動模型訓練管線")
            st.info("💡 點擊下方按鈕，伺服器將會在背景自動讀取資料並啟動訓練。請在下方選擇 Log 檔監看進度。")

            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("🧠 訓練【立銑】大腦", use_container_width=True, type="primary"):
                    try:
                        res = requests.post(f"{API_URL}/train?milling_type=End_Milling&token=admin-token", headers=headers)
                        st.success(res.json().get("message", "指令發送成功"))
                    except Exception as e: st.error(str(e))
            with col_btn2:
                if st.button("🧠 訓練【直銑】大腦", use_container_width=True, type="primary"):
                    try:
                        res = requests.post(f"{API_URL}/train?milling_type=Peripheral_Milling&token=admin-token", headers=headers)
                        st.success(res.json().get("message", "指令發送成功"))
                    except Exception as e: st.error(str(e))
            with col_btn3:
                if st.button("👁️ 訓練【分類器】大腦", use_container_width=True):
                    try:
                        res = requests.post(f"{API_URL}/train?milling_type=Classifier&token=admin-token", headers=headers)
                        st.success(res.json().get("message", "指令發送成功"))
                    except Exception as e: st.error(str(e))

            st.markdown("---")
            try:
                r = requests.get(f'{API_URL}/train_logs', headers=headers)
                logs = r.json().get('logs', [])
            except Exception:
                logs = []

            sel = st.selectbox('📡 選擇要監聽的訓練日誌 (Log)', [''] + logs)

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

        # ----------------------------------------------------
        # 分頁 2：模型熱切換 (未更動)
        # ----------------------------------------------------
        with tab_model:
            st.markdown("#### 🔄 模型版本控制 (Rollback)")
            st.info("若新模型表現不佳，您可以在此一鍵退版。新的命名規則為：`model_YYYYMMDD_HHMMSS.pth`")
            try:
                r_models = requests.get(f'{API_URL}/models')
                if r_models.status_code == 200:
                    model_list = [m['file'] for m in r_models.json().get('models', [])]
                    if model_list:
                        selected_model = st.selectbox("選擇要載入的歷史模型檔案", model_list)
                        if st.button("🌟 設為上線模型 (Deploy)", type="primary"):
                            with st.spinner("正在將模型載入 GPU 記憶體..."):
                                res = requests.post(f'{API_URL}/admin/set_active_model', params={'model_file': selected_model}, headers=headers)
                                if res.status_code == 200: st.success(res.json().get('msg'))
                                else: st.error(res.json().get('error'))
                    else: st.warning("目前還沒有歷史模型。")
            except Exception as e: st.error(f"獲取模型清單失敗: {e}")

        # ----------------------------------------------------
        # 分頁 3：資料集分析 (更新路徑)
        # ----------------------------------------------------
        with tab_data:
            st.markdown("#### 📊 當前訓練資料集分佈")
            csv_path = os.path.join(BASE_DIR, 'data', 'final_training_manifest.csv')
            if os.path.exists(csv_path):
                df_data = pd.read_csv(csv_path)
                st.success(f"目前資料庫中共有 **{len(df_data)}** 張有效訓練影像。")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**不同主軸轉速 (Speed) 的資料量**")
                    st.bar_chart(df_data['speed'].value_counts())
                with col2:
                    st.markdown("**原始資料預覽 (前 5 筆)**")
                    st.dataframe(df_data.head())
            else:
                st.warning("尚未生成訓練清單 (final_training_manifest.csv)")

        # ----------------------------------------------------
        # 分頁 4：預測紀錄與校正 (未更動)
        # ----------------------------------------------------
        with tab_history:
            st.markdown("#### 📜 歷史預測稽核日誌")
            pred_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
            if os.path.exists(pred_path):
                df_pred = pd.read_csv(pred_path)
                df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'], unit='s')
                df_pred = df_pred.sort_values('timestamp', ascending=False).reset_index(drop=True)
                st.dataframe(df_pred, use_container_width=True)
                csv = df_pred.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="📥 匯出預測紀錄 (CSV)", data=csv, file_name='ai_predictions_log.csv', mime='text/csv')
            else: st.info("目前還沒有任何預測紀錄。")
            st.markdown("---")
            st.markdown("#### 🛠️ 人工預測校正 (Human-in-the-Loop)")
            col_a, col_b = st.columns(2)
            with col_a: fname = st.text_input('異常預測之影像檔名 (例如: upload_12345)')
            with col_b: true_ra = st.number_input('儀器真實量測 Ra 值', value=0.0, format='%.6f')
            if st.button('📤 提交校正數據'):
                try:
                    r = requests.post(f'{API_URL}/report_true', params={'filename': fname, 'ra': float(true_ra)}, headers=headers)
                    st.success("數據已記錄至後端標註資料庫！")
                except Exception as e: st.error(str(e))

        # ----------------------------------------------------
        # 分頁 5：硬體監控 (未更動)
        # ----------------------------------------------------
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
