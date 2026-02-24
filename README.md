# Orbbec Pose (Jetson)

Jetson(aarch64)에서 Orbbec 카메라 영상을 입력으로 받아 TensorRT 엔진(FP16)으로 인체 2D 스켈레톤 추론하는 C++ 프로젝트입니다.

## 핵심 정책
- CUDA / TensorRT / OpenCV / Orbbec SDK는 Jetson 시스템(=JetPack) 설치를 전제합니다.
- Jetson에서 `*.onnx`를 `.engine`를 로컬로 변환하여 사용합니다.
- RTMdet/RTMpose onnx 파일은 용량이 커 repo에 포함하지 않습니다.

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

- 요구사항 미충족 시
[설치 가이드](SYSTEM_REQUIREMENTS.md)

2) TensorRT FP16 엔진 생성(ONNX → engine)

```bash
cd ~/Orbbec_yolo
chmod +x scripts/*.sh
```

- YOLO11n-pose TensorRT engine 변환
```bash
bash scripts/build_engine_fp16.sh models/onnx/yolo11n-pose.onnx models/engine/yolo11n-pose_fp16.engine
```

- RTMdet-s + RTMpose-s TensorRT engine 변환
```bash
bash scripts/build_rtm_engine.sh
```

- RTMPose-s TensorRT engine만 변환
```bash
bash scripts/build_rtmpose_engine.sh
```

3) 빌드

```bash
bash scripts/build_project.sh
```
(`orbbec_yolo_pose`, `orbbec_rtm_pose` 둘 다 빌드)

4) 실행 예시

(1) YOLO11n-pose 인식 스크립트 실행
- 2L : HW noise removal OFF
- 336L : HW noise removal ON

- GUI ON
```bash
bash scripts/2L.sh
bash scripts/336L.sh
```

- GUI OFF
```bash
bash scripts/2L.sh --no_gui
bash scripts/336L.sh --no_gui
```

- Cycle time check
```bash
bash scripts/2L.sh --time
bash scripts/336L.sh --time
```

(2) YOLO11n-pose -> RTMpose-s 인식 스크립트 실행
- GUI / time은 위와 동일 방식

- YOLO11n-pose에서 nose(0), wrists(9,10), ankles(15,16) 인식될 시 RTMpose-s로 전달
(기본: `--yolo_edge_kpt_min_count=5`, `--person_expand=1.10`)
```bash
bash scripts/rtm.sh
```

- 종료
CTRL+C

## 디렉토리 구조
```bash
.
├─ models/
│  ├─ engine/
│  │  └─ .gitkeep
│  └─ onnx/
│     └─ yolo11n-pose.onnx
├─ scripts/
│  ├─ 2L.sh
│  ├─ 336L.sh
│  ├─ rtm.sh
│  ├─ build_engine_fp16.sh
│  ├─ build_project.sh
│  ├─ build_rtm_engine.sh
│  └─ build_rtmpose_engine.sh
├─ src/
│  ├─ main_yolo.cpp
│  ├─ main_rtm.cpp
│  ├─ orbbec_utils.cpp
│  ├─ orbbec_utils.hpp
│  ├─ trt_runner.cpp
│  ├─ trt_runner.hpp
│  ├─ trt_runner_multi.cpp
│  └─ trt_runner_multi.hpp
├─ CMakeLists.txt
├─ README.md
├─ SYSTEM_REQUIREMENTS.md
└─ versions_snapshot.txt
```

## 참고
- Orbbec SDK 경로는 CMake에서 /opt/OrbbecSDK_v2.5.5로 고정되어 있습니다.
- TensorRT 라이브러리는 /usr/lib/aarch64-linux-gnu에서 찾도록 구성되어 있습니다.
