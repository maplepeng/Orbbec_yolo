#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-models/yolo11n-pose.onnx}"
ENGINE_PATH="${2:-models/yolo11n-pose_fp16.engine}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "[ERROR] trtexec not found."
  echo "Install TensorRT (JetPack) or locate trtexec under /usr."
  exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
  echo "[ERROR] ONNX file not found: $ONNX_PATH"
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

echo "[INFO] Building TensorRT FP16 engine"
echo "  ONNX   : $ONNX_PATH"
echo "  ENGINE : $ENGINE_PATH"

trtexec --onnx="$ONNX_PATH" \
        --saveEngine="$ENGINE_PATH" \
        --fp16 \
        --skipInference

echo "[DONE] Engine created: $ENGINE_PATH"