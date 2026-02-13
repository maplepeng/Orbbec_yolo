#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/orbbec_yolo_pose}"

# 336L defaults:
#   color: 640x480@30
#   depth: 640x480@30
DEFAULT_ARGS=(
  "./models/engine/yolo11n-pose_fp16.engine"
  "--color=640x480@30"
  "--depth=640x480@30"
  "--rotate=270"
)

exec "$BIN" "${DEFAULT_ARGS[@]}" "$@"
