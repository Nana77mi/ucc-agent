$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies into current Python environment..."

$python = Get-Command python -ErrorAction Stop

& $python.Source -m pip install --upgrade pip
& $python.Source -m pip install -r "requirements.txt"

Write-Host ""
Write-Host "Deployment complete."
Write-Host "Next steps:"
Write-Host "  1) Prepare config.yaml (see repo root)"
Write-Host "  2) Build index: python 01_index.py"
Write-Host "  3) Chat: python 02_chat.py"
