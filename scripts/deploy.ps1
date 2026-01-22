$ErrorActionPreference = "Stop"
# 部署脚本：安装依赖并提示后续步骤

# 输出当前步骤
Write-Host "Installing dependencies into current Python environment..."

# 获取 Python 命令
$python = Get-Command python -ErrorAction Stop

# 安装依赖
& $python.Source -m pip install --upgrade pip
& $python.Source -m pip install -r "requirements.txt"

# 输出提示
Write-Host ""
Write-Host "Deployment complete."
Write-Host "Next steps:"
Write-Host "  1) Prepare config.yaml (see repo root)"
Write-Host "  2) Build index: python 01_index.py"
Write-Host "  3) Chat: python 02_chat.py"
Write-Host "  4) Stability eval: python 04_stability.py"
