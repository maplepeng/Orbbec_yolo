#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct LetterBoxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int src_w = 0;
    int src_h = 0;
    int dst_w = 0;
    int dst_h = 0;
};

class TrtRunnerMulti {
public:
    explicit TrtRunnerMulti(const std::string& engine_path, int input_w = 640, int input_h = 640);
    ~TrtRunnerMulti();

    bool ok() const { return ok_; }
    const std::string& lastError() const { return last_err_; }

    int inputW() const { return input_w_; }
    int inputH() const { return input_h_; }

    const std::string& inputName() const { return input_name_; }
    const std::vector<std::string>& outputNames() const { return output_names_; }

    const nvinfer1::Dims& inputDims() const { return input_dims_; }
    const nvinfer1::Dims& outputDims(const std::string& name) const;

    float inferLetterbox(const cv::Mat& bgr, LetterBoxInfo& lb);
    float inferResize(const cv::Mat& bgr);
    float inferNCHW(const std::vector<float>& input);
    size_t inputElems() const;

    const std::vector<float>& output(const std::string& name) const;

private:
    class OutputAllocatorImpl;

    struct Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                fprintf(stderr, "[TRT] %s\n", msg);
            }
        }
    };

    struct TrtDeleter {
        template <typename T>
        void operator()(T* p) const {
            delete p;
        }
    };

    struct TensorInfo {
        std::string name;
        nvinfer1::TensorIOMode mode = nvinfer1::TensorIOMode::kOUTPUT;
        nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
        nvinfer1::Dims dims{};
        size_t bytes = 0;
        void* device_ptr = nullptr;
        bool dynamic_output = false;
        size_t runtime_bytes = 0;
        std::vector<uint8_t> host_raw;
        std::vector<float> host_float;
    };

    static std::vector<uint8_t> readFile(const std::string& path);
    static int64_t volume(const nvinfer1::Dims& d);
    static size_t dtypeSize(nvinfer1::DataType t);

    cv::Mat letterboxBGR(const cv::Mat& src, LetterBoxInfo& lb) const;
    void preprocessToNCHW_RGB_0to1(const cv::Mat& bgr, std::vector<float>& dst) const;
    float inferFromNCHW(const std::vector<float>& input);
    void* reallocDynamicOutput(const std::string& tensor_name, void* current, uint64_t size, uint64_t alignment, cudaStream_t stream);
    void notifyDynamicOutputShape(const std::string& tensor_name, const nvinfer1::Dims& dims);

private:
    bool ok_ = false;
    std::string last_err_;

    Logger logger_;

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> ctx_;

    int input_w_ = 640;
    int input_h_ = 640;

    std::string input_name_;
    std::vector<std::string> output_names_;

    nvinfer1::Dims input_dims_{};
    nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;

    std::unordered_map<std::string, TensorInfo> tensors_;
    std::unordered_map<std::string, std::unique_ptr<OutputAllocatorImpl>> output_allocators_;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_end_ = nullptr;

    std::vector<float> host_input_;
    size_t input_bytes_ = 0;
};
