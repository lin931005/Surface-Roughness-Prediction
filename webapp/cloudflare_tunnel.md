使用 Cloudflare Tunnel 將本地服務暴露到外網（建議）

1. 安裝 cloudflared：https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation

2. 建立 tunnel：

```powershell
cloudflared tunnel login
cloudflared tunnel create my-surface-tunnel
cloudflared tunnel route dns my-surface-tunnel example-subdomain.yourdomain.com
```

3. 啟動 tunnel（示範本地 8000 -> 外網）：

```powershell
cloudflared tunnel run my-surface-tunnel --url http://localhost:8000
```

備註：可以把上述指令放到系統服務或 Docker 中自動啟動。