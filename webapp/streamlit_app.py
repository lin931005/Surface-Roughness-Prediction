import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title='Surface Roughness', layout='centered')

st.title('Surface Roughness Prediction')

tab = st.sidebar.radio('Mode', ['User', 'Admin'])

if tab == 'User':
    uploaded = st.file_uploader('Upload image', type=['png','jpg','jpeg'])
    if uploaded is not None:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption='input', use_column_width=True)
        if st.button('Predict'):
            files = {'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')}
            use_gc = st.checkbox('Return Grad-CAM heatmap', value=True)
            params = {}
            if use_gc:
                params['gradcam'] = 'true'
            try:
                r = requests.post('http://localhost:8000/predict', files=files, params=params, timeout=30)
                if r.status_code == 200:
                    j = r.json()
                    st.success(f"Predicted Ra: {j.get('ra')}")
                    if j.get('heatmap'):
                        st.image(j.get('heatmap'), caption='Grad-CAM', use_column_width=True)
                    elif j.get('heatmap_error'):
                        st.warning(j.get('heatmap_error'))
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

else:
    st.subheader('Admin')
    if 'token' not in st.session_state:
        st.session_state['token'] = ''
    username = st.text_input('Username', value='admin')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        try:
            r = requests.post('http://localhost:8000/login', params={'username': username, 'password': password})
            if r.status_code == 200:
                st.session_state['token'] = r.json().get('access_token')
                st.success('Logged in')
            else:
                st.error('Login failed')
        except Exception as e:
            st.error(str(e))
    token = st.session_state.get('token','')
    if token:
        st.success('Token set')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Retrain model'):
            if not token:
                st.error('Need token')
            else:
                try:
                    headers = {'Authorization': f'Bearer {token}'}
                    r = requests.post('http://localhost:8000/retrain', headers=headers, timeout=5)
                    st.write(r.json())
                except Exception as e:
                    st.error(str(e))
        if st.button('List models'):
            try:
                r = requests.get('http://localhost:8000/models')
                st.json(r.json())
            except Exception as e:
                st.error(str(e))
        if st.button('Dashboard'):
            try:
                headers = {'Authorization': f'Bearer {token}'} if token else {}
                s = requests.get('http://localhost:8000/admin/stats', headers=headers)
                ps = requests.get('http://localhost:8000/predictions/stats', headers=headers)
                st.subheader('System Stats')
                if s.status_code == 200:
                    st.json(s.json())
                st.subheader('Prediction Stats')
                if ps.status_code == 200:
                    data = ps.json()
                    st.metric('Total predictions', data.get('total', 0))
                    last = data.get('last', [])
                    if last:
                        import pandas as _pd
                        df = _pd.DataFrame(last)
                        if 'ra' in df.columns:
                            df['ra'] = df['ra'].astype(float)
                            st.line_chart(df['ra'])
            except Exception as e:
                st.error(str(e))
    with col2:
        uploaded_zip = st.file_uploader('Upload ZIP for batch predict', type=['zip'])
        if uploaded_zip is not None and st.button('Run batch predict'):
            try:
                files = {'file': (uploaded_zip.name, uploaded_zip.getvalue(), 'application/zip')}
                headers = {'Authorization': f'Bearer {token}'} if token else {}
                r = requests.post('http://localhost:8000/predict_batch', files=files, headers=headers, timeout=120)
                if r.status_code == 200:
                    st.write(r.json())
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(str(e))

    st.markdown('---')
    st.subheader('Train logs')
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    try:
        r = requests.get('http://localhost:8000/train_logs', headers=headers)
        logs = r.json().get('logs', [])
    except Exception:
        logs = []

    sel = st.selectbox('Select log', [''] + logs)
    if sel:
        col_a, col_b = st.columns([3,1])
        with col_b:
            if 'monitor' not in st.session_state:
                st.session_state['monitor'] = ''
            if st.session_state['monitor'] == sel:
                if st.button('Stop monitoring'):
                    st.session_state['monitor'] = ''
            else:
                if st.button('Start monitoring'):
                    st.session_state['monitor'] = sel
        # function to fetch progress
        def fetch_progress(name):
            try:
                r = requests.get(f'http://localhost:8000/train_progress/{name}', headers=headers, timeout=5)
                if r.status_code == 200:
                    return r.json().get('progress', [])
            except Exception:
                return []
            return []

        placeholder = st.empty()
        chart = None
        # initial fetch
        progress = fetch_progress(sel)
        if progress:
            import pandas as _pd
            df = _pd.DataFrame(progress)
            df = df.set_index('epoch')
            placeholder.line_chart(df[['train_loss','val_loss']])

        # live monitoring loop (short polling)
        import time as _time
        while st.session_state.get('monitor','') == sel:
            progress = fetch_progress(sel)
            if progress:
                import pandas as _pd
                df = _pd.DataFrame(progress)
                df = df.set_index('epoch')
                placeholder.line_chart(df[['train_loss','val_loss']])
            _time.sleep(2)

    st.subheader('Report true Ra (Human-in-the-Loop)')
    fname = st.text_input('Filename')
    true_ra = st.number_input('True Ra', value=0.0, format='%.6f')
    if st.button('Submit true Ra'):
        try:
            headers = {'Authorization': f'Bearer {token}'} if token else {}
            r = requests.post('http://localhost:8000/report_true', params={'filename': fname, 'ra': float(true_ra)}, headers=headers)
            st.write(r.json())
        except Exception as e:
            st.error(str(e))
