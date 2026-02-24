#include "trt_runner_multi.hpp"

#include <opencv2/imgproc.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
bool dimsSpecified(const nvinfer1::Dims& d) {
    if (d.nbDims <= 0) return false;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] <= 0) return false;
    }
    return true;
}

}  // namespace

class TrtRunnerMulti::OutputAllocatorImpl : public nvinfer1::IOutputAllocator {
public:
    OutputAllocatorImpl(TrtRunnerMulti& owner, std::string tensor_name)
        : owner_(owner), tensor_name_(std::move(tensor_name)) {}

    void* reallocateOutputAsync(
        char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream) noexcept override {
        return owner_.reallocDynamicOutput(tensor_name_, currentMemory, size, alignment, stream);
    }

    void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override {
        owner_.notifyDynamicOutputShape(tensor_name_, dims);
    }

private:
    TrtRunnerMulti& owner_;
    std::string tensor_name_;
};

std::vector<uint8_t> TrtRunnerMulti::readFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    ifs.seekg(0, std::ios::end);
    const size_t n = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(n);
    ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
    return buf;
}

int64_t TrtRunnerMulti::volume(const nvinfer1::Dims& d) {
    if (!dimsSpecified(d)) return -1;
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        const int64_t di = static_cast<int64_t>(d.d[i]);
        if (di <= 0) return -1;
        if (v > std::numeric_limits<int64_t>::max() / di) return -1;
        v *= di;
    }
    return v;
}

size_t TrtRunnerMulti::dtypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        default: return 0;
    }
}

TrtRunnerMulti::TrtRunnerMulti(const std::string& engine_path, int input_w, int input_h)
    : input_w_(input_w), input_h_(input_h) {
    const auto plan = readFile(engine_path);
    if (plan.empty()) {
        last_err_ = "cannot open engine file: " + engine_path;
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        last_err_ = "createInferRuntime failed";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) {
        last_err_ = "deserializeCudaEngine failed";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    ctx_.reset(engine_->createExecutionContext());
    if (!ctx_) {
        last_err_ = "createExecutionContext failed";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    const int nb_io = engine_->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        const auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT && input_name_.empty()) {
            input_name_ = name;
        }
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_names_.push_back(name);
        }
    }

    if (input_name_.empty()) {
        last_err_ = "cannot find input tensor";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    if (output_names_.empty()) {
        last_err_ = "cannot find output tensors";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    input_dims_ = engine_->getTensorShape(input_name_.c_str());
    if (input_dims_.nbDims == 4) {
        nvinfer1::Dims dims = input_dims_;
        dims.d[0] = (dims.d[0] > 0) ? dims.d[0] : 1;
        dims.d[1] = (dims.d[1] > 0) ? dims.d[1] : 3;
        dims.d[2] = input_h_;
        dims.d[3] = input_w_;
        (void)ctx_->setInputShape(input_name_.c_str(), dims);
    }

    input_dims_ = ctx_->getTensorShape(input_name_.c_str());
    input_dtype_ = engine_->getTensorDataType(input_name_.c_str());

    const int64_t in_count = volume(input_dims_);
    if (in_count <= 0) {
        last_err_ = "input tensor shape is not fully specified after setInputShape";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    const size_t in_dtype = dtypeSize(input_dtype_);
    if (in_dtype == 0) {
        last_err_ = "unsupported input dtype";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    input_bytes_ = static_cast<size_t>(in_count) * in_dtype;
    host_input_.resize(static_cast<size_t>(in_count));

    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);

        TensorInfo info;
        info.name = name;
        info.mode = engine_->getTensorIOMode(name);
        info.dtype = engine_->getTensorDataType(name);
        info.dims = ctx_->getTensorShape(name);
        const size_t t_dtype = dtypeSize(info.dtype);
        if (t_dtype == 0) {
            last_err_ = "unsupported tensor dtype: " + info.name;
            std::cerr << "TRT init failed: " << last_err_ << "\n";
            return;
        }

        const int64_t t_count = volume(info.dims);
        if (t_count > 0) {
            info.bytes = static_cast<size_t>(t_count) * t_dtype;
        } else if (info.mode == nvinfer1::TensorIOMode::kOUTPUT) {
            info.dynamic_output = true;
        } else {
            last_err_ = "input tensor shape unresolved: " + info.name;
            std::cerr << "TRT init failed: " << last_err_ << "\n";
            return;
        }

        if (info.mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (t_count > 0) {
                cudaMalloc(&info.device_ptr, info.bytes);
                info.host_raw.resize(info.bytes);
                info.host_float.resize(static_cast<size_t>(t_count));
                ctx_->setTensorAddress(name, info.device_ptr);
            } else {
                info.host_raw.clear();
                info.host_float.clear();
                info.bytes = 0;
                info.device_ptr = nullptr;

                auto alloc = std::make_unique<OutputAllocatorImpl>(*this, info.name);
                if (!ctx_->setOutputAllocator(name, alloc.get())) {
                    last_err_ = "setOutputAllocator failed for tensor: " + info.name;
                    std::cerr << "TRT init failed: " << last_err_ << "\n";
                    return;
                }
                ctx_->setTensorAddress(name, nullptr);
                output_allocators_[info.name] = std::move(alloc);
            }
        } else {
            cudaMalloc(&info.device_ptr, info.bytes);
            ctx_->setTensorAddress(name, info.device_ptr);
        }
        tensors_[info.name] = std::move(info);
    }

    cudaStreamCreate(&stream_);
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_end_);

    ok_ = true;

    auto dimsToStr = [](const nvinfer1::Dims& d) {
        std::string s = "[";
        for (int i = 0; i < d.nbDims; ++i) {
            s += std::to_string(d.d[i]);
            if (i + 1 < d.nbDims) s += ", ";
        }
        s += "]";
        return s;
    };

    std::cerr << "=== Engine I/O tensors ===\n";
    for (const auto& kv : tensors_) {
        const auto& t = kv.second;
        std::cerr << (t.mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT  " : "OUTPUT ")
                  << t.name << " dims=" << dimsToStr(t.dims) << " bytes=" << t.bytes << "\n";
    }
}

TrtRunnerMulti::~TrtRunnerMulti() {
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_end_) cudaEventDestroy(ev_end_);
    if (stream_) cudaStreamDestroy(stream_);

    for (auto& kv : tensors_) {
        if (kv.second.device_ptr) cudaFree(kv.second.device_ptr);
    }
}

const nvinfer1::Dims& TrtRunnerMulti::outputDims(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("output tensor not found: " + name);
    }
    return it->second.dims;
}

cv::Mat TrtRunnerMulti::letterboxBGR(const cv::Mat& src, LetterBoxInfo& lb) const {
    lb.src_w = src.cols;
    lb.src_h = src.rows;
    lb.dst_w = input_w_;
    lb.dst_h = input_h_;

    const float r = std::min(static_cast<float>(lb.dst_w) / lb.src_w, static_cast<float>(lb.dst_h) / lb.src_h);
    lb.scale = r;

    const int new_w = static_cast<int>(std::round(lb.src_w * r));
    const int new_h = static_cast<int>(std::round(lb.src_h * r));

    lb.pad_x = (lb.dst_w - new_w) / 2;
    lb.pad_y = (lb.dst_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(lb.dst_h, lb.dst_w, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(lb.pad_x, lb.pad_y, new_w, new_h)));
    return out;
}

void TrtRunnerMulti::preprocessToNCHW_RGB_0to1(const cv::Mat& bgr, std::vector<float>& dst) const {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32F, 1.0 / 255.0);

    const int H = rgb_f.rows;
    const int W = rgb_f.cols;
    const int C = 3;

    dst.resize(static_cast<size_t>(C * H * W));
    const float* p = reinterpret_cast<const float*>(rgb_f.data);

    for (int y = 0; y < H; ++y) {
        const float* row = p + y * W * C;
        for (int x = 0; x < W; ++x) {
            const float r = row[x * C + 0];
            const float g = row[x * C + 1];
            const float b = row[x * C + 2];
            dst[0 * H * W + y * W + x] = r;
            dst[1 * H * W + y * W + x] = g;
            dst[2 * H * W + y * W + x] = b;
        }
    }
}

float TrtRunnerMulti::inferFromNCHW(const std::vector<float>& input) {
    if (!ok_) return -1.0f;

    const auto in_it = tensors_.find(input_name_);
    if (in_it == tensors_.end()) {
        std::cerr << "Input tensor metadata missing\n";
        return -1.0f;
    }

    const auto in_elems = static_cast<size_t>(volume(input_dims_));
    if (input.size() != in_elems) {
        std::cerr << "Input size mismatch: got " << input.size() << ", expected " << in_elems << "\n";
        return -1.0f;
    }

    if (input_dtype_ == nvinfer1::DataType::kFLOAT) {
        const auto err = cudaMemcpyAsync(in_it->second.device_ptr, input.data(), input_bytes_, cudaMemcpyHostToDevice, stream_);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << "\n";
            return -1.0f;
        }
    } else if (input_dtype_ == nvinfer1::DataType::kHALF) {
        std::vector<__half> tmp(in_elems);
        for (size_t i = 0; i < in_elems; ++i) tmp[i] = __float2half(input[i]);
        const auto err = cudaMemcpyAsync(in_it->second.device_ptr, tmp.data(), input_bytes_, cudaMemcpyHostToDevice, stream_);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << "\n";
            return -1.0f;
        }
    } else {
        std::cerr << "Unsupported input dtype\n";
        return -1.0f;
    }

    cudaEventRecord(ev_start_, stream_);
    const bool ok = ctx_->enqueueV3(stream_);
    cudaEventRecord(ev_end_, stream_);

    if (!ok) {
        std::cerr << "TRT enqueueV3 failed\n";
        return -1.0f;
    }

    for (const auto& out_name : output_names_) {
        auto& t = tensors_[out_name];
        if (t.dynamic_output && (t.device_ptr == nullptr || t.runtime_bytes == 0)) {
            std::cerr << "Dynamic output allocator did not provide memory for tensor: " << out_name << "\n";
            return -1.0f;
        }

        t.dims = ctx_->getTensorShape(out_name.c_str());
        size_t need_bytes = t.dynamic_output ? t.runtime_bytes : t.bytes;
        if (need_bytes == 0) continue;

        int64_t cnt = volume(t.dims);
        if (cnt > 0) {
            need_bytes = static_cast<size_t>(cnt) * dtypeSize(t.dtype);
        } else {
            const size_t elem = dtypeSize(t.dtype);
            cnt = (elem > 0) ? static_cast<int64_t>(need_bytes / elem) : 0;
        }

        if (cnt <= 0) {
            std::cerr << "Cannot resolve output element count for tensor: " << out_name << "\n";
            return -1.0f;
        }

        if (t.host_raw.size() < need_bytes) t.host_raw.resize(need_bytes);
        if (t.host_float.size() != static_cast<size_t>(cnt)) t.host_float.resize(static_cast<size_t>(cnt));

        const auto err = cudaMemcpyAsync(t.host_raw.data(), t.device_ptr, need_bytes, cudaMemcpyDeviceToHost, stream_);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync D2H failed for " << out_name << ": " << cudaGetErrorString(err) << "\n";
            return -1.0f;
        }
    }
    {
        const auto err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << "\n";
            return -1.0f;
        }
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start_, ev_end_);

    for (const auto& out_name : output_names_) {
        auto& t = tensors_[out_name];
        int64_t cnt = volume(t.dims);
        if (cnt <= 0) {
            const size_t elem = dtypeSize(t.dtype);
            const size_t bytes = t.dynamic_output ? t.runtime_bytes : t.bytes;
            cnt = (elem > 0) ? static_cast<int64_t>(bytes / elem) : 0;
        }
        const size_t elem_size = dtypeSize(t.dtype);
        const size_t max_cnt = (elem_size > 0) ? (t.host_raw.size() / elem_size) : 0;
        if (static_cast<size_t>(cnt) > max_cnt) cnt = static_cast<int64_t>(max_cnt);
        if (cnt <= 0) continue;

        if (t.dtype == nvinfer1::DataType::kFLOAT) {
            const float* fp = reinterpret_cast<const float*>(t.host_raw.data());
            for (int64_t i = 0; i < cnt; ++i) t.host_float[static_cast<size_t>(i)] = fp[i];
        } else if (t.dtype == nvinfer1::DataType::kHALF) {
            const __half* hp = reinterpret_cast<const __half*>(t.host_raw.data());
            for (int64_t i = 0; i < cnt; ++i) t.host_float[static_cast<size_t>(i)] = __half2float(hp[i]);
        } else if (t.dtype == nvinfer1::DataType::kINT32) {
            const int32_t* ip = reinterpret_cast<const int32_t*>(t.host_raw.data());
            for (int64_t i = 0; i < cnt; ++i) t.host_float[static_cast<size_t>(i)] = static_cast<float>(ip[i]);
        } else if (t.dtype == nvinfer1::DataType::kINT8) {
            const int8_t* ip = reinterpret_cast<const int8_t*>(t.host_raw.data());
            for (int64_t i = 0; i < cnt; ++i) t.host_float[static_cast<size_t>(i)] = static_cast<float>(ip[i]);
        } else if (t.dtype == nvinfer1::DataType::kBOOL) {
            const bool* bp = reinterpret_cast<const bool*>(t.host_raw.data());
            for (int64_t i = 0; i < cnt; ++i) t.host_float[static_cast<size_t>(i)] = bp[i] ? 1.0f : 0.0f;
        } else {
            std::cerr << "Unsupported output dtype for tensor: " << t.name << "\n";
            return -1.0f;
        }
    }

    return ms;
}

void* TrtRunnerMulti::reallocDynamicOutput(
    const std::string& tensor_name, void* current, uint64_t size, uint64_t alignment, cudaStream_t stream) {
    (void)current;
    (void)alignment;
    (void)stream;

    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) return nullptr;
    auto& t = it->second;

    if (t.device_ptr != nullptr && t.bytes >= size) {
        t.runtime_bytes = static_cast<size_t>(size);
        return t.device_ptr;
    }

    if (t.device_ptr != nullptr) {
        cudaFree(t.device_ptr);
        t.device_ptr = nullptr;
        t.bytes = 0;
    }

    void* ptr = nullptr;
    const auto err = cudaMalloc(&ptr, static_cast<size_t>(size));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dynamic output " << tensor_name << ": " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }

    t.device_ptr = ptr;
    t.bytes = static_cast<size_t>(size);
    t.runtime_bytes = static_cast<size_t>(size);
    return t.device_ptr;
}

void TrtRunnerMulti::notifyDynamicOutputShape(const std::string& tensor_name, const nvinfer1::Dims& dims) {
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) return;
    it->second.dims = dims;
}

float TrtRunnerMulti::inferLetterbox(const cv::Mat& bgr, LetterBoxInfo& lb) {
    cv::Mat in = letterboxBGR(bgr, lb);
    preprocessToNCHW_RGB_0to1(in, host_input_);
    return inferFromNCHW(host_input_);
}

float TrtRunnerMulti::inferResize(const cv::Mat& bgr) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);
    preprocessToNCHW_RGB_0to1(resized, host_input_);
    return inferFromNCHW(host_input_);
}

float TrtRunnerMulti::inferNCHW(const std::vector<float>& input) {
    return inferFromNCHW(input);
}

size_t TrtRunnerMulti::inputElems() const {
    const int64_t v = volume(input_dims_);
    return (v > 0) ? static_cast<size_t>(v) : 0U;
}

const std::vector<float>& TrtRunnerMulti::output(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("output tensor not found: " + name);
    }
    return it->second.host_float;
}
