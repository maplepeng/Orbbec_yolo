#!/usr/bin/env bash
set -euo pipefail

# [EDIT HERE] 경로/옵션

# (1) 입력 ONNX / 출력 engine 경로
POSE_ONNX="./models/onnx/mmpose_halpe26/rtmpose-s_8xb1024-700e_body8-halpe26-256x192/end2end.onnx"
POSE_ENGINE="./models/engine/rtmpose_s_fp16.engine"

# RTMDet은 패치된 ONNX를 지정
DET_ONNX="./models/onnx/mmdet_rtmdet/rtmdet_s_8xb32-300e_coco/end2end_topk3840.onnx"
DET_ENGINE="./models/engine/rtmdet_s_fp16.engine"

# (2) 입력 텐서 이름/shape (참고용 로그 — trtexec에 전달하지 않음)
POSE_INPUT_NAME="input"
POSE_INPUT_SHAPE="1x3x256x192"

DET_INPUT_NAME="input"
DET_INPUT_SHAPE="1x3x640x640"

# (3) 빌드 옵션
WORKSPACE_MB=2048
FP16=1

# 엔진 빌드만 할지(1) / 빌드+벤치까지 할지(0)
SKIP_INFERENCE=1

# (4) (선택) timing cache (빌드 반복 시 빨라짐)
TIMING_CACHE_DIR="/tmp/trt_cache"
POSE_TIMING_CACHE="${TIMING_CACHE_DIR}/rtmpose.cache"
DET_TIMING_CACHE="${TIMING_CACHE_DIR}/rtmdet.cache"

# internal helpers
die() { echo "[ERROR] $*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Command not found: $1"; }

ensure_file() { [ -f "$1" ] || die "File not found: $1"; }

ensure_dir_of() { mkdir -p "$(dirname "$1")"; }

# TRT 버전에 따라 workspace 플래그가 다를 수 있어 자동 선택
workspace_flag() {
  if trtexec --help 2>/dev/null | grep -q "memPoolSize"; then
    echo "--memPoolSize=workspace:${WORKSPACE_MB}"
  else
    echo "--workspace=${WORKSPACE_MB}"
  fi
}

timing_cache_flag() {
  local cache_file="$1"
  mkdir -p "$(dirname "$cache_file")" 2>/dev/null || true
  if trtexec --help 2>/dev/null | grep -q "timingCacheFile"; then
    echo "--timingCacheFile=${cache_file}"
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

build_engine() {
  local onnx="$1"
  local engine="$2"
  local input_name="$3"   # log only
  local input_shape="$4"  # log only
  local cache="$5"

  ensure_file "$onnx"
  ensure_dir_of "$engine"
  mkdir -p "$TIMING_CACHE_DIR"

  local WFLAG; WFLAG="$(workspace_flag)"
  local CFLAG; CFLAG="$(timing_cache_flag "$cache")"
  local SFLAG; SFLAG="$(skip_inference_flag)"
  local PFLAG; PFLAG="$(precision_flag)"

  echo "[INFO] Build engine"
  echo "  ONNX        : $onnx"
  echo "  ENGINE      : $engine"
  echo "  INPUT_NAME  : $input_name (log only)"
  echo "  INPUT_SHAPE : $input_shape (log only)"
  echo "  WORKSPACE_MB: $WORKSPACE_MB"
  [ -n "$CFLAG" ] && echo "  TIMING_CACHE: $cache"
  echo "  FP16        : $FP16"
  echo "  SKIP_INFER  : $SKIP_INFERENCE"
  echo

  # IMPORTANT: No explicit shapes here.
  # This works for static-shape ONNX (your RTMPose / patched RTMDet end2end).
  set +e
  trtexec \
    --onnx="$onnx" \
    --saveEngine="$engine" \
    $PFLAG \
    $WFLAG \
    $CFLAG \
    $SFLAG
  local ret=$?
  set -e

  if [ $ret -ne 0 ]; then
    echo "[ERROR] trtexec failed (ret=$ret)."
    echo "        This script does NOT pass explicit shape profiles."
    echo "        If the ONNX is dynamic-shape, export a static-shape ONNX (recommended) and retry."
    exit $ret
  fi

  [ -s "$engine" ] || die "Engine not created or empty: $engine"
  echo "[DONE] $engine"
  echo
}

# main
need_cmd trtexec

ensure_file "$POSE_ONNX"
ensure_file "$DET_ONNX"
mkdir -p "$TIMING_CACHE_DIR"

echo "[INFO] Sequential TRT build: RTMPose -> RTMDet"
echo

# 1) RTMPose
echo "[STEP 1/2] RTMPose"
build_engine "$POSE_ONNX" "$POSE_ENGINE" "$POSE_INPUT_NAME" "$POSE_INPUT_SHAPE" "$POSE_TIMING_CACHE"

# 2) RTMDet (patched)
echo "[STEP 2/2] RTMDet (patched ONNX)"
build_engine "$DET_ONNX" "$DET_ENGINE" "$DET_INPUT_NAME" "$DET_INPUT_SHAPE" "$DET_TIMING_CACHE"

echo "[ALL DONE]"
ls -lh "$POSE_ENGINE" "$DET_ENGINE"
