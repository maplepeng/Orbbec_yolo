# System Requirements (Jetson)

This project targets **Jetson (aarch64)** and assumes **system-installed** CUDA / TensorRT / OpenCV and **Orbbec SDK (.deb)**.
Conda is **not required** for building or running the C++ binaries.

## 1) Tested Stack (recorded)
Keep a stack snapshot in the repo to make the environment reproducible.
- See: `versions_snapshot.txt`

Recommended verification commands:
```bash
cat /etc/nv_tegra_release
nvcc --version || true
dpkg -l | grep -E 'nvinfer|tensorrt' || true
pkg-config --modversion opencv4 2>/dev/null || true
```

---

## 2) Build Tools (system)
You need a standard C++ build toolchain.

Install:
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config git
```

(Optional but recommended):
```bash
sudo apt install -y ninja-build
```

CMake minimum version: 3.16+

---

## 3) OpenCV for C++ (system)
CMake must be able to resolve:
- `find_package(OpenCV REQUIRED)`

Install (typical):
```bash
sudo apt update
sudo apt install -y libopencv-dev
```

Verify:
```bash
pkg-config --modversion opencv4
```

If you built OpenCV from source and CMake cannot find it, configure with:
```bash
cmake -S . -B build -DOpenCV_DIR=/path/to/opencv4/lib/cmake/opencv4
```

---

## 4) CUDA Toolkit (system)
CMake requires:
- `find_package(CUDAToolkit REQUIRED)`
and links against:
- `CUDA::cudart`

On Jetson, CUDA is normally provided by your JetPack/L4T stack.
Verify:
```bash
nvcc --version || true
```

If CUDA/CUDAToolkit is missing, use **Jetson’s NVIDIA apt repo packages**. Two common approaches:

### Option A (preferred when available): install JetPack meta-package
```bash
sudo apt update
sudo apt install -y nvidia-jetpack
```

### Option B: install CUDA toolkit packages directly (package names depend on your L4T repo)
Search what’s available:
```bash
apt-cache search cuda-toolkit | head -n 50
```
Then install the matching toolkit package for your repo.

---

## 5) TensorRT (system)
CMake links against:
- `libnvinfer.so`
- `libnvinfer_plugin.so`

And for engine building you need:
- `trtexec` (usually provided by TensorRT binary package)

Install (typical Jetson packages):
```bash
sudo apt update
sudo apt install -y \
  libnvinfer-bin \
  libnvinfer-dev \
  libnvinfer-plugin-dev \
  tensorrt-libs
```

Verify:
```bash
which trtexec || true
trtexec --help | head -n 5 || true
ldconfig -p | grep -E 'libnvinfer\.so|libnvinfer_plugin\.so' || true
```

If `trtexec` is not found:
```bash
dpkg -L libnvinfer-bin | grep trtexec || true
```

---

## 6) Orbbec SDK (installed by .deb)
[OrbbecSDK_v2](https://github.com/orbbec/OrbbecSDK_v2)
This project expects Orbbec SDK installed at:
- `/opt/OrbbecSDK_v2.5.5`
  - include: `/opt/OrbbecSDK_v2.5.5/include`
  - library: `/opt/OrbbecSDK_v2.5.5/lib/libOrbbecSDK.so`

Install:
1) Obtain the Orbbec SDK `.deb` for Jetson/aarch64 from Orbbec.
2) Install it:
```bash
sudo dpkg -i ./OrbbecSDK_*.deb
sudo apt -f install -y
```

Verify:
```bash
test -f /opt/OrbbecSDK_v2.5.5/lib/libOrbbecSDK.so \
  && echo "Orbbec SDK OK" \
  || echo "Orbbec SDK missing"
```

Note:
- This project sets RPATH so the runtime can find Orbbec libs under `/opt/.../lib` without `ldconfig`.

---

## 7) Runtime Sanity Check (recommended)
After building the binary, verify no missing shared libraries:
```bash
ldd ./build/orbbec_yolo_pose | grep "not found" || echo "(none)"
```

If something is missing:
- Confirm TensorRT libs exist under your system library paths.
- Confirm Orbbec SDK library exists under `/opt/OrbbecSDK_v2.5.5/lib`.
- Confirm OpenCV dev package is installed and discoverable by CMake.
