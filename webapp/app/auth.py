import os
import time
import jwt
import bcrypt
from dotenv import load_dotenv
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env')))

SECRET = os.environ.get('JWT_SECRET', 'change-this-secret-please-change')
ALGORITHM = "HS256"

# simple in-memory user store; replace with DB in production
# For production, set ADMIN_PASSWORD to a strong secret and keep JWT_SECRET secret.
USERS = {
    "admin": {
        "password": os.environ.get('ADMIN_PASSWORD', 'adminpass')
    }
}

# 💡 核心修改 1：設定 auto_error=False
# 讓 FastAPI 不要在缺少 Header 時直接封殺，而是放行給下方的函式自行判斷
security = HTTPBearer(auto_error=False)

def create_token(username: str, expires_in: int = 3600):
    payload = {"sub": username, "exp": int(time.time() + expires_in)}
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return payload.get('sub')
    except Exception:
        return None

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    # 💡 核心修改 2：防止 creds 為空時引發 AttributeError
    if not creds or not creds.credentials:
        raise HTTPException(status_code=401, detail='Invalid token')

    token = creds.credentials
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail='Invalid token')
    return user


def verify_credentials(username: str, password: str) -> bool:
    u = USERS.get(username)
    if not u:
        return False
    stored = u.get('password')
    if stored is None:
        return False
    if stored.startswith('$2b$') or stored.startswith('$2a$'):
        return bcrypt.checkpw(password.encode('utf-8'), stored.encode('utf-8'))
    return password == stored


def admin_auth(creds: HTTPAuthorizationCredentials = Depends(security), token: str = None):
    # 💡 核心修改 3：先安全檢查 creds 是否存在，再驗證 Header
    if creds and creds.credentials:
        user = verify_token(creds.credentials)
        if user:
            return user

    # Fallback to query token (legacy / 給 Streamlit 內部 API 呼叫使用)[cite: 3]
    if token and token == 'admin-token':
        return 'admin'

    raise HTTPException(status_code=401, detail='Unauthorized')
