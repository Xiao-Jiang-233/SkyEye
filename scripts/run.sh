#!/usr/bin/env bash
# ============ scripts/run.sh ============
# 本地开发快捷脚本（Linux / macOS / Git Bash）
# 用法: bash scripts/run.sh [stage]
#
# Examples:
#   bash scripts/run.sh all       # 完整训练管线
#   bash scripts/run.sh teacher   # 仅训练教师
#   bash scripts/run.sh export    # 仅导出 + 量化

set -e
cd "$(dirname "$0")/.."

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    # Windows Git Bash
    PYTHON=".venv/Scripts/python.exe"
else
    PYTHON="python"
fi

echo "SkyEye — Weather Classification Pipeline"
echo "========================================="
echo "Python: $($PYTHON --version)"
echo "Stage: ${1:-all}"
echo ""

$PYTHON scripts/local_train.py "${1:-all}"
