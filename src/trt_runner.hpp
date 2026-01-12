#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Letterbox 정보(좌표 복원용)
struct LetterBoxInfo {
    float scale = 1.f;   // resize scale
    int pad_x = 0;       // left padding in pixels (on the 640x640 image)
    int pad_y = 0;       // top padding
    int src_w = 0;       // original width
    int src_h = 0;       // original height
    int dst_w = 0;       // model input width (e.g. 640)
    int dst_h = 0;       // model input height (e.g. 640)
};

class TrtRunner {
public:
    // engine_path: TensorRT plan
    // input_w/h: network input resolution (e.g. 640x640)
    explicit TrtRunner(const std::string& engine_path, int input_w = 640, int input_h = 640);
    ~TrtRunner();

    bool ok() const { return ok_; }
    const std::string& lastError() const { return last_err_; }
    const std::string& inputName() const { return input_name_; }
    const std::string& outputName() const { return output_name_; }

    // Output dims helpers (YOLO11-pose: typically [1,56,8400])
    const nvinfer1::Dims& outputDims() const { return output_dims_; }
    int outputC() const;
    int outputN() const;

    int inputW() const { return input_w_; }
    int inputH() const { return input_h_; }

    // bgr 입력(원본 컬러 프레임)을 받아:
    // 1) letterbox로 640x640 맞추고
    // 2) RGB, 0~1, NCHW float로 변환한 뒤
    // 3) enqueueV3 수행
    //
    // out_host_float: output0을 float로 담아 반환 (shape=[C, N]=[56,8400]을 1D로)
    // lb: letterbox 정보(좌표 복원용)
    // return: inference time (ms, cudaEvent 기반)
    float infer_pose(const cv::Mat& bgr, std::vector<float>& out_host_float, LetterBoxInfo& lb);

private:
    struct Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            // INFO 이하만 출력(원하면 조절)
            if (severity <= Severity::kWARNING) {
                fprintf(stderr, "[TRT] %s\n", msg);
            }
        }
    };

    struct TrtDeleter {
        template <typename T>
        void operator()(T* p) const {
            // TensorRT 10.x: destroy()가 제거되어 delete 사용
            delete p;
        }
    };

    static std::vector<uint8_t> readFile(const std::string& path);
    static int64_t volume(const nvinfer1::Dims& d);
    static size_t dtypeSize(nvinfer1::DataType t);

    // 내부: letterbox + preprocess(NCHW float RGB 0~1) 결과를 host_input_에 채움
    cv::Mat letterboxBGR(const cv::Mat& src, LetterBoxInfo& lb) const;
    void preprocessToNCHW_RGB_0to1(const cv::Mat& bgr_letterboxed);

private:
    bool ok_ = false;
    std::string last_err_;

    Logger logger_;

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> ctx_;

    std::string input_name_;
    std::string output_name_;

    int input_w_ = 640;
    int input_h_ = 640;

    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;

    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};

    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_end_ = nullptr;

    // Host buffers
    std::vector<float> host_input_;      // NCHW float32
    std::vector<uint8_t> host_output_;   // raw output bytes (dtype-dependent)

    // Device buffers
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;

    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
};
