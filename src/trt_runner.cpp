#include "trt_runner.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cuda_fp16.h>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

std::vector<uint8_t> TrtRunner::readFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    ifs.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(n);
    ifs.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

int64_t TrtRunner::volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<int64_t>(d.d[i]);
    return v;
}

size_t TrtRunner::dtypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 0;
    }
}

TrtRunner::TrtRunner(const std::string& engine_path, int input_w, int input_h)
: input_w_(input_w), input_h_(input_h) {
    // 1) 엔진 파일 로드
    auto plan = readFile(engine_path);
    if (plan.empty()) {
        last_err_ = "cannot open engine file: " + engine_path;
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    // 2) Runtime/Engine/Context 생성
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

    // 3) I/O 텐서 이름 탐색 (TensorRT 10: binding API 대신 I/O tensor API)
    int nb_io = engine_->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)  input_name_ = name;
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) output_name_ = name;
    }
    if (input_name_.empty() || output_name_.empty()) {
        last_err_ = "cannot find input/output tensor names";
        std::cerr << "TRT init failed: " << last_err_ << "\n";
        return;
    }

    // 4) 입력/출력 shape, dtype
    input_dtype_  = engine_->getTensorDataType(input_name_.c_str());
    output_dtype_ = engine_->getTensorDataType(output_name_.c_str());

    input_dims_  = engine_->getTensorShape(input_name_.c_str());
    output_dims_ = engine_->getTensorShape(output_name_.c_str());

    // 대부분 YOLO export는 [1,3,640,640] 고정. (동적이면 여기서 setInputShape 필요)
    if (input_dims_.nbDims == 4) {
        // 안전하게 H/W를 우리가 원하는 값으로 맞춰 setInputShape 시도
        nvinfer1::Dims dims = input_dims_;
        dims.d[0] = (dims.d[0] > 0) ? dims.d[0] : 1;
        dims.d[1] = (dims.d[1] > 0) ? dims.d[1] : 3;
        dims.d[2] = input_h_;
        dims.d[3] = input_w_;
        (void)ctx_->setInputShape(input_name_.c_str(), dims);
    }

    // runtime shape로 재조회
    input_dims_  = ctx_->getTensorShape(input_name_.c_str());
    output_dims_ = ctx_->getTensorShape(output_name_.c_str());

    // 5) 버퍼 크기 계산
    input_bytes_  = static_cast<size_t>(volume(input_dims_) * static_cast<int64_t>(dtypeSize(input_dtype_)));
    output_bytes_ = static_cast<size_t>(volume(output_dims_) * static_cast<int64_t>(dtypeSize(output_dtype_)));

    // host buffers
    host_input_.resize(static_cast<size_t>(volume(input_dims_)));     // float32 기준 (전처리 결과는 float로 고정)
    host_output_.resize(output_bytes_);

    // device buffers
    cudaMalloc(&d_input_, input_bytes_);
    cudaMalloc(&d_output_, output_bytes_);

    // 6) context에 tensor address 바인딩
    ctx_->setTensorAddress(input_name_.c_str(), d_input_);
    ctx_->setTensorAddress(output_name_.c_str(), d_output_);

    // 7) stream/event 생성
    cudaStreamCreate(&stream_);
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_end_);

    ok_ = true;

    // 디버그 출력(원하면 main에서 꺼도 됨)
    std::cerr << "=== Engine I/O tensors ===\n";
    auto printDims = [](const nvinfer1::Dims& d) {
        std::string s="[";
        for(int i=0;i<d.nbDims;i++){ s += std::to_string(d.d[i]); if(i+1<d.nbDims) s+=", "; }
        s += "]";
        return s;
    };
    std::cerr << "INPUT  " << input_name_  << " dims=" << printDims(input_dims_)  << " bytes=" << input_bytes_  << "\n";
    std::cerr << "OUTPUT " << output_name_ << " dims=" << printDims(output_dims_) << " bytes=" << output_bytes_ << "\n";
}

TrtRunner::~TrtRunner() {
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_end_)   cudaEventDestroy(ev_end_);
    if (stream_)   cudaStreamDestroy(stream_);
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
}

cv::Mat TrtRunner::letterboxBGR(const cv::Mat& src, LetterBoxInfo& lb) const {
    // YOLO 계열 표준 letterbox:
    // - 종횡비 유지하며 resize
    // - 남는 부분은 padding(114)
    lb.src_w = src.cols;
    lb.src_h = src.rows;
    lb.dst_w = input_w_;
    lb.dst_h = input_h_;

    float r = std::min(static_cast<float>(lb.dst_w) / lb.src_w,
                       static_cast<float>(lb.dst_h) / lb.src_h);
    lb.scale = r;

    int new_w = static_cast<int>(std::round(lb.src_w * r));
    int new_h = static_cast<int>(std::round(lb.src_h * r));

    int pad_w = lb.dst_w - new_w;
    int pad_h = lb.dst_h - new_h;

    // 중앙 패딩
    lb.pad_x = pad_w / 2;
    lb.pad_y = pad_h / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(lb.dst_h, lb.dst_w, src.type(), cv::Scalar(114,114,114));
    resized.copyTo(out(cv::Rect(lb.pad_x, lb.pad_y, new_w, new_h)));

    return out;
}

void TrtRunner::preprocessToNCHW_RGB_0to1(const cv::Mat& bgr_letterboxed) {
    // 입력은 BGR uint8(HWC). 이를:
    // - RGB로 변환
    // - float32로 변환 후 0~1 스케일
    // - NCHW로 재배열
    cv::Mat rgb;
    cv::cvtColor(bgr_letterboxed, rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32F, 1.0 / 255.0);

    const int H = rgb_f.rows;
    const int W = rgb_f.cols;
    const int C = 3;

    // host_input_은 NCHW 순서
    // index: c*H*W + y*W + x
    const float* p = reinterpret_cast<const float*>(rgb_f.data);

    for (int y = 0; y < H; ++y) {
        const float* row = p + y * W * C;
        for (int x = 0; x < W; ++x) {
            const float r = row[x*C + 0];
            const float g = row[x*C + 1];
            const float b = row[x*C + 2];
            host_input_[0*H*W + y*W + x] = r;
            host_input_[1*H*W + y*W + x] = g;
            host_input_[2*H*W + y*W + x] = b;
        }
    }
}

float TrtRunner::infer_pose(const cv::Mat& bgr, std::vector<float>& out_host_float, LetterBoxInfo& lb) {
    if (!ok_) return -1.f;

    // 1) letterbox
    cv::Mat in = letterboxBGR(bgr, lb);

    // 2) preprocess -> host_input_ 채움
    preprocessToNCHW_RGB_0to1(in);

    // 3) H2D
    cudaMemcpyAsync(d_input_, host_input_.data(), input_bytes_, cudaMemcpyHostToDevice, stream_);

    // 4) enqueueV3 + timing
    cudaEventRecord(ev_start_, stream_);
    bool ok = ctx_->enqueueV3(stream_);
    cudaEventRecord(ev_end_, stream_);

    if (!ok) {
        std::cerr << "TRT enqueueV3 failed\n";
        return -1.f;
    }

    // 5) D2H (raw bytes)
    cudaMemcpyAsync(host_output_.data(), d_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, ev_start_, ev_end_);

    // 6) output -> float vector로 변환
    //    (YOLO11-pose engine에서 output은 보통 FP32. 그래도 dtype에 따라 처리)
    const int64_t out_count = volume(output_dims_);
    out_host_float.resize(static_cast<size_t>(out_count));

    if (output_dtype_ == nvinfer1::DataType::kFLOAT) {
        const float* fp = reinterpret_cast<const float*>(host_output_.data());
        std::copy(fp, fp + out_count, out_host_float.begin());
    } else if (output_dtype_ == nvinfer1::DataType::kHALF) {
        const __half* hp = reinterpret_cast<const __half*>(host_output_.data());
        for (int64_t i = 0; i < out_count; ++i) {
            out_host_float[static_cast<size_t>(i)] = __half2float(hp[i]);
        }
    } else {
        std::cerr << "Unsupported output dtype\n";
        return -1.f;
    }

    return ms;
}

int TrtRunner::outputC() const {
    // YOLO11-pose: [1,56,8400] => C=56 at dim[1]
    if (output_dims_.nbDims >= 2) return output_dims_.d[1];
    return 0;
}

int TrtRunner::outputN() const {
    // YOLO11-pose: [1,56,8400] => N=8400 at last dim
    if (output_dims_.nbDims >= 1) return output_dims_.d[output_dims_.nbDims - 1];
    return 0;
}