#!/usr/bin/env bash
# ============ scripts/run.sh ============
# 本地开发快捷脚本（Mo 平台 Linux 环境）
# 用法: bash scripts/run.sh [stage]
#
# Examples:
#   bash scripts/run.sh all       # 完整训练管线
#   bash scripts/run.sh teacher   # 仅训练教师
#   bash scripts/run.sh export    # 仅导出 + 量化

set -e
cd "$(dirname "$0")/.."

echo "SkyEye — Weather Classification Pipeline"
echo "========================================="
echo "Python: $(python --version)"
echo "Stage: ${1:-all}"
echo ""

python scripts/local_train.py "${1:-all}"
