cmake_minimum_required(VERSION 3.16)
project(CameraCppOrbbecYoloPose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Threads ----
find_package(Threads REQUIRED)

# ---- CUDA Toolkit (for cuda runtime / events) ----
find_package(CUDAToolkit REQUIRED)

# ---- Orbbec SDK (installed by .deb) ----
set(ORBBEC_ROOT "/opt/OrbbecSDK_v2.5.5")
set(ORBBEC_INCLUDE_DIR "${ORBBEC_ROOT}/include")
set(ORBBEC_LIB "${ORBBEC_ROOT}/lib/libOrbbecSDK.so")

if(NOT EXISTS "${ORBBEC_LIB}")
  message(FATAL_ERROR "Cannot find Orbbec library: ${ORBBEC_LIB}")
endif()

# ---- TensorRT libs (Jetson) ----
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu REQUIRED)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu REQUIRED)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "Orbbec include: ${ORBBEC_INCLUDE_DIR}")
message(STATUS "Orbbec lib: ${ORBBEC_LIB}")
message(STATUS "TensorRT libs: ${NVINFER_LIB}, ${NVINFER_PLUGIN_LIB}")

# RPATH so runtime can find /opt/OrbbecSDK_v2.5.5/lib without ldconfig
set(CMAKE_BUILD_RPATH "${ORBBEC_ROOT}/lib")
set(CMAKE_INSTALL_RPATH "${ORBBEC_ROOT}/lib")

# ---- Targets ----
add_executable(orbbec_yolo_pose
  src/main_yolo.cpp
  src/orbbec_utils.cpp
  src/trt_runner.cpp
)

target_include_directories(orbbec_yolo_pose PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ORBBEC_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(orbbec_yolo_pose PRIVATE
  ${OpenCV_LIBS}
  Threads::Threads
  ${ORBBEC_LIB}
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  CUDA::cudart
)

add_executable(orbbec_rtm_pose
  src/main_rtm.cpp
  src/orbbec_utils.cpp
  src/trt_runner_multi.cpp
)

target_include_directories(orbbec_rtm_pose PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${ORBBEC_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(orbbec_rtm_pose PRIVATE
  ${OpenCV_LIBS}
  Threads::Threads
  ${ORBBEC_LIB}
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  CUDA::cudart
)
