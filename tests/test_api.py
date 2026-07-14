import httpx

BASE = "http://127.0.0.1:8000"


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    assert 'status' in r.json()


def test_login():
    r = httpx.post(f"{BASE}/login", params={'username':'admin','password':'adminpass'}, timeout=5)
    assert r.status_code == 200
    assert 'access_token' in r.json()
 