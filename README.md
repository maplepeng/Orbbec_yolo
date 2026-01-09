# CameraCppOrbbecYoloPose (Jetson)

Jetson(aarch64)에서 Orbbec 카메라 영상을 입력으로 받아 YOLO11n-pose TensorRT 엔진(FP16)으로 추론하는 C++ 프로젝트입니다.

## 핵심 정책
- CUDA / TensorRT / OpenCV / Orbbec SDK는 Jetson 시스템(=JetPack) 설치를 전제합니다.
- `models/*.onnx`는 repo에 포함합니다.
- `models/*.engine`은 Jetson에서 로컬로 생성하며 repo에 커밋하지 않습니다.

## 빠른 시작 (Jetson)
1) 시스템 요구사항 확인  
- CUDA Toolkit (system)
Verify:
```bash
nvcc --version || true
```

- TensorRT (system)
Verify:
```bash
which trtexec || true
trtexec --help | head -n 5 || true
ldconfig -p | grep -E 'libnvinfer\.so|libnvinfer_plugin\.so' || true
```

- OpenCV for C++
Verify:
```bash
pkg-config --modversion opencv4
```

- Orbbec SDK (installed by .deb)
Verify:
```bash
test -f /opt/OrbbecSDK_v2.5.5/lib/libOrbbecSDK.so \
  && echo "Orbbec SDK OK" \
  || echo "Orbbec SDK missing"
```

- 설치 가이드
./SYSTEM_REQUIREMENTS.md

2) TensorRT FP16 엔진 생성(ONNX → engine)

```bash
cd ~/Orbbec_yolo
chmod +x scripts/*.sh
bash scripts/build_engine_fp16.sh models/yolo11n-pose.onnx models/yolo11n-pose_fp16.engine
```

3) 빌드

```bash
bash scripts/build_project.sh
```

4) 실행 예시

- GUI ON
```bash
./build/orbbec_yolo_pose ./models/yolo11n-pose_fp16.engine --rotate=270
```

- GUI OFF
```bash
./build/orbbec_yolo_pose ./models/yolo11n-pose_fp16.engine --rotate=270 --no_gui
```

- 종료
CTRL+C

## 디렉토리 구조
.
├─ models/
│  ├─ yolo11n-pose.onnx
├─ scripts/
│  ├─ build_engine_fp16.sh
│  ├─ build_project.sh
│  └─ check_deps.sh
├─ src/
│  ├─ main.cpp
│  ├─ trt_runner.cpp
│  └─ trt_runner.hpp
├─ CMakeLists.txt
├─ README.md
├─ SYSTEM_REQUIREMENTS.md
└─ versions_snapshot.txt

## 참고
- Orbbec SDK 경로는 CMake에서 /opt/OrbbecSDK_v2.5.5로 고정되어 있습니다.
- TensorRT 라이브러리는 /usr/lib/aarch64-linux-gnu에서 찾도록 구성되어 있습니다.