# Webapp for Surface Roughness Prediction

Run FastAPI server:

```bash
pip install -r webapp/requirements.txt
uvicorn webapp.app.main:app --host 0.0.0.0 --port 2578
```

Run Streamlit UI (optional):

```bash
streamlit run webapp/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

執行測試：

```bash
pip install -r webapp/requirements.txt
pytest -q
```

Notes:
- Admin endpoints are available for model management and training logs under `/models`, `/train_logs`.
- Use the new JWT login endpoint: `/login?username=admin&password=adminpass` to receive an `access_token`.
- Default demo credentials are `admin` / `adminpass`, but you should set `ADMIN_PASSWORD` and `JWT_SECRET` in `.env` for production.
- Copy `.env.sample` to `.env` and configure secrets before deploying.
