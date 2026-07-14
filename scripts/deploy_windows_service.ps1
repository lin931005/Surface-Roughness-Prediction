# 範例 PowerShell 腳本：使用 NSSM 將 uvicorn 設為 Windows 服務（需要先安裝 NSSM）
# https://nssm.cc/

$serviceName = "SurfaceRoughnessAPI"
$pythonExe = "C:\\Python310\\python.exe" # 修改為你的 python 路徑
$workingDir = "C:\\Users\\tony9\\Desktop\\5000 - BETTER"
$script = "-m uvicorn webapp.app.main:app --host 0.0.0.0 --port 8000"

Write-Host "請先下載並安裝 NSSM，並將 nssm.exe 放在 PATH 或指定完整路徑。"
$nssm = "nssm"
& $nssm install $serviceName $pythonExe $script
& $nssm set $serviceName AppDirectory $workingDir
& $nssm start $serviceName

Write-Host "服務已安裝並啟動（如有錯誤，請檢查 NSSM 與路徑）"