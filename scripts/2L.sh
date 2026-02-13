#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/orbbec_yolo_pose}"

# 2L defaults:
#   color: 640x400@30
#   depth: 640x400@30
#   must disable HW noise removal => --no_hw_noise
DEFAULT_ARGS=(
  "./models/engine/yolo11n-pose_fp16.engine"
  "--color=640x400@30"
  "--depth=640x400@30"
  "--rotate=270"
  "--no_hw_noise"
)

exec "$BIN" "${DEFAULT_ARGS[@]}" "$@"
