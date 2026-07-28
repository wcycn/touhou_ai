#!/bin/sh
# 双击启动新的Touhou AI统一桌面控制中心。
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
exec python3 touhou_ai.py gui
