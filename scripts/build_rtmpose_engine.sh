#!/usr/bin/env bash
set -euo pipefail

# RTMPose-s only (ONNX -> TensorRT engine)
#
# Optional positional args:
#   $1 : POSE_ONNX
#   $2 : POSE_ENGINE
#
# Optional env vars:
#   WORKSPACE_MB, FP16, SKIP_INFERENCE, TIMING_CACHE_DIR, POSE_TIMING_CACHE

POSE_ONNX="${1:-./models/onnx/mmpose_halpe26/rtmpose-s_8xb1024-700e_body8-halpe26-256x192/end2end.onnx}"
POSE_ENGINE="${2:-./models/engine/rtmpose_s_fp16.engine}"

POSE_INPUT_NAME="input"
POSE_INPUT_SHAPE="1x3x256x192"

WORKSPACE_MB="${WORKSPACE_MB:-2048}"
FP16="${FP16:-1}"
SKIP_INFERENCE="${SKIP_INFERENCE:-1}"

TIMING_CACHE_DIR="${TIMING_CACHE_DIR:-/tmp/trt_cache}"
POSE_TIMING_CACHE="${POSE_TIMING_CACHE:-${TIMING_CACHE_DIR}/rtmpose.cache}"

die() { echo "[ERROR] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Command not found: $1"; }
ensure_file() { [ -f "$1" ] || die "File not found: $1"; }
ensure_dir_of() { mkdir -p "$(dirname "$1")"; }

# TRT version compatibility: workspace flag
workspace_flag() {
  if trtexec --help 2>/dev/null | grep -q "memPoolSize"; then
    echo "--memPoolSize=workspace:${WORKSPACE_MB}"
  else
    echo "--workspace=${WORKSPACE_MB}"
  fi
}

timing_cache_flag() {
  if trtexec --help 2>/dev/null | grep -q "timingCacheFile"; then
    echo "--timingCacheFile=${POSE_TIMING_CACHE}"
  else
    echo ""
  fi
}

skip_inference_flag() {
  if [ "${SKIP_INFERENCE}" = "1" ]; then
    echo "--skipInference"
  else
    echo ""
  fi
}

precision_flag() {
  if [ "${FP16}" = "1" ]; then
    echo "--fp16"
  else
    echo ""
  fi
}

need_cmd trtexec
ensure_file "${POSE_ONNX}"
ensure_dir_of "${POSE_ENGINE}"
mkdir -p "${TIMING_CACHE_DIR}"

WFLAG="$(workspace_flag)"
CFLAG="$(timing_cache_flag)"
SFLAG="$(skip_inference_flag)"
PFLAG="$(precision_flag)"

echo "[INFO] Build RTMPose engine"
echo "  ONNX        : ${POSE_ONNX}"
echo "  ENGINE      : ${POSE_ENGINE}"
echo "  INPUT_NAME  : ${POSE_INPUT_NAME} (log only)"
echo "  INPUT_SHAPE : ${POSE_INPUT_SHAPE} (log only)"
echo "  WORKSPACE_MB: ${WORKSPACE_MB}"
[ -n "${CFLAG}" ] && echo "  TIMING_CACHE: ${POSE_TIMING_CACHE}"
echo "  FP16        : ${FP16}"
echo "  SKIP_INFER  : ${SKIP_INFERENCE}"
echo

set +e
trtexec \
  --onnx="${POSE_ONNX}" \
  --saveEngine="${POSE_ENGINE}" \
  ${PFLAG} \
  ${WFLAG} \
  ${CFLAG} \
  ${SFLAG}
ret=$?
set -e

if [ ${ret} -ne 0 ]; then
  echo "[ERROR] trtexec failed (ret=${ret})."
  echo "        This script does NOT pass explicit shape profiles."
  echo "        If the ONNX is dynamic-shape, export a static-shape ONNX and retry."
  exit ${ret}
fi

[ -s "${POSE_ENGINE}" ] || die "Engine not created or empty: ${POSE_ENGINE}"
echo "[DONE] ${POSE_ENGINE}"
ls -lh "${POSE_ENGINE}"
