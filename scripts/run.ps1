# ============ scripts/run.ps1 ============
# 本地开发快捷脚本（Windows PowerShell 版）
# 用法: .\scripts\run.ps1 [stage]
#
# Examples:
#   .\scripts\run.ps1 all       # 完整训练管线
#   .\scripts\run.ps1 teacher   # 仅训练教师
#   .\scripts\run.ps1 check     # 检查环境
#   .\scripts\run.ps1 export    # 仅导出 + 量化

param(
    [string]$Stage = "all"
)

$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\.."

Write-Host "SkyEye — Weather Classification Pipeline"
Write-Host "========================================="
Write-Host "Stage: $Stage"
Write-Host ""

# 激活虚拟环境（如果存在）
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "Using .venv virtual environment"
    $Python = ".venv\Scripts\python.exe"
} else {
    Write-Host "Using system Python"
    $Python = "python"
}

& $Python scripts\local_train.py $Stage

Pop-Location
