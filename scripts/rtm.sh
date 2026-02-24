#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/orbbec_rtm_pose}"

# 336L defaults:
#   color: 640x400@30
#   depth: 640x400@30
DEFAULT_ARGS=(
  "--rotate=270"
  "--yolo_kpt_min_count=0"
  "--yolo_edge_kpt_min_count=5"
  "--person_expand=1.10"
)

exec "$BIN" "${DEFAULT_ARGS[@]}" "$@"
