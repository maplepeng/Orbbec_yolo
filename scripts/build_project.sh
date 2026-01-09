#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BUILD_TYPE="${2:-Release}"

echo "[INFO] Build dir : ${BUILD_DIR}"
echo "[INFO] Build type: ${BUILD_TYPE}"

mkdir -p "${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" -j

echo "[DONE] Binaries:"
ls -lh "${BUILD_DIR}/orbbec_yolo_pose" 2>/dev/null || true
