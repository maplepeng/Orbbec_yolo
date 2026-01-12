// main.cpp
// Orbbec Color/Depth capture (FrameSync + D2C HW align) + TensorRT inference (YOLO11 Pose)
// - Print all stream profiles
// - Auto-pick "closest" profile to requested spec
// - FrameSync aggregation like Orbbec example (hw_d2c_align.cpp)
// - Letterbox to 640x640 (YOLO-style)
// - Decode YOLO11-pose output [1,56,8400] -> boxes + 17 keypoints
// - Restore coordinates to ORIGINAL color image and draw skeleton
//
// Build assumptions:
// - OrbbecSDK v2.5.5 installed at /opt/OrbbecSDK_v2.5.5
// - OpenCV available (system OpenCV recommended on Jetson)
// - CUDA + TensorRT headers/libs available (JetPack)
//
// Run example:
//   ./build/orbbec_yolo_pose ./yolo11n-pose_trt10p3_fp16.engine --rotate=270
//   ./build/orbbec_yolo_pose ./yolo11n-pose_trt10p3_fp16.engine --color=640x400@30 --depth=640x400@30 --conf=0.25 --nms=0.45

#include "orbbec_utils.hpp"

#include <libobsensor/ObSensor.hpp>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <limits>
#include <atomic>
#include <csignal>
#include <sys/select.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;

// ----------------------------- Small utilities -----------------------------

static inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }
static inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

// =========================
// Unified stop flag (Ctrl+C)
// =========================
static std::atomic<bool> g_running{true};

static void onSigInt(int) {
    // Ctrl+C 등으로 SIGINT가 들어오면 루프 종료
    g_running.store(false);
}

// no_gui 모드에서 "q + Enter"로도 종료
static int pollStdinKeyNonBlocking() {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;  // non-blocking

    const int rv = select(STDIN_FILENO + 1, &rfds, nullptr, nullptr, &tv);
    if (rv > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
        unsigned char c = 0;
        const ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n == 1) return static_cast<int>(c);
    }
    return -1;
}

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

static bool startsWith(const std::string& s, const std::string& p) {
    return s.rfind(p, 0) == 0;
}

static std::vector<uint8_t> readFileBytes(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    ifs.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(n);
    if (n) ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
    return buf;
}

static size_t dtypeSize(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 0;
    }
}

static int64_t volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

static std::string dimsToStr(const nvinfer1::Dims& d) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < d.nbDims; ++i) {
        oss << d.d[i];
        if (i + 1 < d.nbDims) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

// ----------------------------- CLI args -----------------------------

struct Spec {
    int w = 640;
    int h = 400;
    int fps = 30;
};

static bool parseWHFps(const std::string& s, Spec& out) {
    // format: "640x400@30"
    // allow "640x400" (fps keep)
    auto t = s;
    auto at = t.find('@');
    std::string wh = (at == std::string::npos) ? t : t.substr(0, at);
    std::string fps = (at == std::string::npos) ? "" : t.substr(at + 1);

    auto x = wh.find('x');
    if (x == std::string::npos) return false;
    out.w = std::stoi(wh.substr(0, x));
    out.h = std::stoi(wh.substr(x + 1));
    if (!fps.empty()) out.fps = std::stoi(fps);
    return true;
}

struct Args {
    std::string engine_path;

    Spec color{640, 400, 30};
    Spec depth{640, 400, 30}; // default: 640x400
    int rotate = 0;           // 0/90/180/270
    bool GUI = true; 

    bool enable_framesync = true;
    bool enable_d2c_align = true;  // use HW D2C align
    float conf_th = 0.25f;
    float nms_th = 0.45f;
    float kpt_th = 0.25f;

    int topk = 50;            // max candidates before NMS
    int max_draw = 5;         // draw up to N persons

    // depth visualization
    float depth_max_mm = 4000.0f; // colormap clipping
};

static void printUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " /path/to/model.engine [options]\n"
        << "Options:\n"
        << "  --color=640x400@30        Color target (default 640x400@30)\n"
        << "  --depth=640x400@30        Depth target (default 640x400@30)\n"
        << "  --rotate=0|90|180|270     Rotate BOTH color/depth for display+inference (default 0)\n"
        << "  --no_gui                  Disable GUI (default enabled)\n"
        << "  --no_sync                 Disable FrameSync (default: enabled)\n"
        << "  --no_align                Disable D2C align (default: enabled HW align)\n"
        << "  --conf=0.25               Detection confidence threshold\n"
        << "  --nms=0.45                NMS IoU threshold\n"
        << "  --kpt=0.25                Keypoint confidence threshold\n"
        << "  --depth_max_mm=4000       Depth colormap max (mm)\n";
}

static bool parseArgs(int argc, char** argv, Args& a) {
    if (argc < 2) return false;
    a.engine_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string s(argv[i]);
        if (s == "--help" || s == "-h") return false;

        if (startsWith(s, "--color=")) {
            parseWHFps(s.substr(8), a.color);
        } else if (startsWith(s, "--depth=")) {
            parseWHFps(s.substr(8), a.depth);
        } else if (startsWith(s, "--rotate=")) {
            a.rotate = std::stoi(s.substr(9));
        } else if (s == "--gui") {
            a.GUI = true;
        } else if (s == "--no_gui") {
            a.GUI = false;
        } else if (s == "--no_sync") {
            a.enable_framesync = false;
        } else if (s == "--no_align") {
            a.enable_d2c_align = false;
        } else if (startsWith(s, "--conf=")) {
            a.conf_th = std::stof(s.substr(7));
        } else if (startsWith(s, "--nms=")) {
            a.nms_th = std::stof(s.substr(6));
        } else if (startsWith(s, "--kpt=")) {
            a.kpt_th = std::stof(s.substr(6));
        } else if (startsWith(s, "--depth_max_mm=")) {
            a.depth_max_mm = std::stof(s.substr(15));
        } else {
            std::cerr << "Unknown option: " << s << "\n";
            return false;
        }
    }

    if (!(a.rotate == 0 || a.rotate == 90 || a.rotate == 180 || a.rotate == 270)) {
        std::cerr << "rotate must be 0/90/180/270\n";
        return false;
    }
    return true;
}

struct RunningStats {
    int64_t n = 0;
    double mean = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = 0.0;

    void add(double x) {
        ++n;
        mean = mean + ((x - mean) / n);
        if (n > 100) {
            if (x < min) min = x;
            if (x > max) max = x;
        };
    }
};

// ----------------------------- Frame -> cv::Mat converters -----------------------------

static cv::Mat decodeMJPGtoBGR(const uint8_t* data, size_t bytes) {
    // MJPG payload is typically a JPEG bitstream
    cv::Mat buf(1, static_cast<int>(bytes), CV_8UC1, const_cast<uint8_t*>(data));
    return cv::imdecode(buf, cv::IMREAD_COLOR); // returns BGR
}

static cv::Mat colorFrameToBGR(const std::shared_ptr<ob::ColorFrame>& cf) {
    if (!cf) return {};
    auto fmt = cf->getFormat();

    const uint8_t* data = reinterpret_cast<const uint8_t*>(cf->getData());
    size_t bytes = static_cast<size_t>(cf->getDataSize());
    int w = static_cast<int>(cf->getWidth());
    int h = static_cast<int>(cf->getHeight());

    // Most robust in practice: handle MJPG explicitly.
    if (fmt == OB_FORMAT_MJPG || fmt == OB_FORMAT_MJPEG) {
        return decodeMJPGtoBGR(data, bytes);
    }

    // Uncompressed RGB/BGR
    if (fmt == OB_FORMAT_RGB || fmt == OB_FORMAT_RGB888) {
        cv::Mat rgb(h, w, CV_8UC3, const_cast<uint8_t*>(data));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr.clone(); // clone because frame memory becomes invalid after loop iteration
    }
    if (fmt == OB_FORMAT_BGR) {
        cv::Mat bgr(h, w, CV_8UC3, const_cast<uint8_t*>(data));
        return bgr.clone();
    }

    // Common packed formats
    if (fmt == OB_FORMAT_YUYV) {
        cv::Mat yuyv(h, w, CV_8UC2, const_cast<uint8_t*>(data));
        cv::Mat bgr;
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
        return bgr.clone();
    }

    // Fallback: try decode as MJPG (some firmwares label differently)
    cv::Mat mj = decodeMJPGtoBGR(data, bytes);
    if (!mj.empty()) return mj;

    std::cerr << "[WARN] Unsupported color format: " << orbbec_utils::obFormatToStr(fmt)
              << " bytes=" << bytes << " w=" << w << " h=" << h << "\n";
    return {};
}

static cv::Mat depthFrameToMat16U(const std::shared_ptr<ob::DepthFrame>& df) {
    if (!df) return {};
    int w = static_cast<int>(df->getWidth());
    int h = static_cast<int>(df->getHeight());
    auto fmt = df->getFormat();

    if (fmt != OB_FORMAT_Y16) {
        std::cerr << "[WARN] Depth not Y16. fmt=" << orbbec_utils::obFormatToStr(fmt) << "\n";
        // Try anyway if it is 16-bit
    }

    cv::Mat d16(h, w, CV_16UC1, reinterpret_cast<void*>(df->getData()));
    return d16.clone();
}

static void rotateIfNeeded(cv::Mat& img, int rotate_deg) {
    if (img.empty()) return;
    if (rotate_deg == 0) return;
    if (rotate_deg == 90) cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
    else if (rotate_deg == 180) cv::rotate(img, img, cv::ROTATE_180);
    else if (rotate_deg == 270) cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
}

static cv::Mat depthToColorMap(const cv::Mat& depth16, float max_mm) {
    if (depth16.empty()) return {};
    // depth16 is in mm typically (Orbbec default). Clip and scale to 0..255.
    cv::Mat d = depth16.clone();
    // Replace 0 (invalid) with max for visualization so it becomes dark after inversion (optional).
    // Here we keep 0 as 0 to appear black.
    cv::Mat d32f;
    d.convertTo(d32f, CV_32F);
    cv::Mat clipped;
    cv::min(d32f, max_mm, clipped);
    cv::Mat u8;
    clipped.convertTo(u8, CV_8U, 255.0 / max_mm);
    cv::Mat cm;
    cv::applyColorMap(u8, cm, cv::COLORMAP_JET);
    return cm;
}

// ----------------------------- Letterbox (YOLO) -----------------------------

struct LetterboxInfo {
    float scale = 1.f;
    int pad_x = 0;
    int pad_y = 0;
    int in_w = 0, in_h = 0;
    int out_w = 0, out_h = 0;
};

static cv::Mat letterboxBGR(const cv::Mat& bgr, int out_w, int out_h, LetterboxInfo& info) {
    info.in_w = bgr.cols;
    info.in_h = bgr.rows;
    info.out_w = out_w;
    info.out_h = out_h;

    float r = std::min(out_w / (float)info.in_w, out_h / (float)info.in_h);
    int new_w = (int)std::round(info.in_w * r);
    int new_h = (int)std::round(info.in_h * r);

    info.scale = r;
    info.pad_x = (out_w - new_w) / 2;
    info.pad_y = (out_h - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(out_h, out_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, new_w, new_h)));
    return out;
}

// Convert BGR (uint8) -> CHW float32 RGB in [0,1]
static void bgrToCHW_RGB_0to1(const cv::Mat& bgr, int net_w, int net_h, std::vector<float>& out_chw) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Make sure size matches net input
    cv::Mat resized;
    if (rgb.cols != net_w || rgb.rows != net_h) {
        cv::resize(rgb, resized, cv::Size(net_w, net_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = rgb;
    }

    out_chw.resize((size_t)3 * net_w * net_h);
    const int H = net_h, W = net_w;
    for (int y = 0; y < H; ++y) {
        const cv::Vec3b* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; ++x) {
            const auto& p = row[x]; // RGB
            out_chw[0 * H * W + y * W + x] = p[0] / 255.0f;
            out_chw[1 * H * W + y * W + x] = p[1] / 255.0f;
            out_chw[2 * H * W + y * W + x] = p[2] / 255.0f;
        }
    }
}

// ----------------------------- TensorRT minimal runner (TRT10 enqueueV3) -----------------------------

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

class TrtEngine {
public:
    bool load(const std::string& engine_path) {
        auto plan = readFileBytes(engine_path);
        if (plan.empty()) {
            last_err_ = "cannot open engine file: " + engine_path;
            return false;
        }

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            last_err_ = "createInferRuntime failed";
            return false;
        }

        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine_) {
            last_err_ = "deserializeCudaEngine failed (bad engine / version mismatch)";
            return false;
        }

        ctx_.reset(engine_->createExecutionContext());
        if (!ctx_) {
            last_err_ = "createExecutionContext failed";
            return false;
        }

        // Enumerate IO tensors (TRT10 API)
        int nb = engine_->getNbIOTensors();
        if (nb <= 0) {
            last_err_ = "engine has no IO tensors";
            return false;
        }

        io_.clear();
        io_.reserve(nb);

        for (int i = 0; i < nb; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(name);
            auto dtype = engine_->getTensorDataType(name);
            auto shape = engine_->getTensorShape(name);

            Tensor t;
            t.name = name;
            t.is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
            t.dtype = dtype;
            t.engine_dims = shape;
            t.runtime_dims = shape; // will update after setInputShape if dynamic
            io_.push_back(t);
        }

        // Identify single input/output for this YOLO model
        for (auto& t : io_) {
            if (t.is_input && input_name_.empty()) input_name_ = t.name;
            if (!t.is_input && output_name_.empty()) output_name_ = t.name;
        }
        if (input_name_.empty() || output_name_.empty()) {
            last_err_ = "failed to identify input/output tensors";
            return false;
        }

        // For YOLO11 pose ONNX you showed: input is static [1,3,640,640]
        // Still, call setInputShape for robustness.
        auto in_dims = engine_->getTensorShape(input_name_.c_str());
        if (!ctx_->setInputShape(input_name_.c_str(), in_dims)) {
            last_err_ = "setInputShape failed";
            return false;
        }

        // After shape is set, query runtime shapes and allocate buffers
        for (auto& t : io_) {
            t.runtime_dims = ctx_->getTensorShape(t.name.c_str());
            size_t bytes = (size_t)volume(t.runtime_dims) * dtypeSize(t.dtype);
            t.bytes = bytes;

            // Allocate device buffer
            void* dptr = nullptr;
            cudaError_t ce = cudaMalloc(&dptr, bytes);
            if (ce != cudaSuccess) {
                last_err_ = "cudaMalloc failed for tensor " + t.name;
                return false;
            }
            t.dptr = dptr;

            // Bind address
            if (!ctx_->setTensorAddress(t.name.c_str(), t.dptr)) {
                last_err_ = "setTensorAddress failed for tensor " + t.name;
                return false;
            }
        }

        // Create stream + events
        cudaStreamCreate(&stream_);
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_end_);

        // Cache net input size (expect NCHW)
        auto id = ctx_->getTensorShape(input_name_.c_str());
        // id = [1,3,H,W]
        input_h_ = (id.nbDims >= 4) ? id.d[2] : 640;
        input_w_ = (id.nbDims >= 4) ? id.d[3] : 640;

        // Print IO
        std::cout << "=== Engine I/O tensors ===\n";
        for (auto& t : io_) {
            std::cout << (t.is_input ? "INPUT  " : "OUTPUT ")
                      << t.name
                      << " dims=" << dimsToStr(t.runtime_dims)
                      << " bytes=" << t.bytes
                      << "\n";
        }

        return true;
    }

    int inputW() const { return input_w_; }
    int inputH() const { return input_h_; }
    const std::string& lastError() const { return last_err_; }

    // Inference:
    // - takes ORIGINAL BGR frame
    // - produces output0 float vector (decoded to float regardless of fp16/fp32 output)
    // - returns GPU enqueue time (ms) measured by CUDA events
    float inferPose(const cv::Mat& bgr, std::vector<float>& out0, LetterboxInfo* lbinfo_out) {
        // 1) letterbox to net size
        LetterboxInfo lb;
        cv::Mat boxed = letterboxBGR(bgr, input_w_, input_h_, lb);
        if (lbinfo_out) *lbinfo_out = lb;

        // 2) preprocess -> CHW float RGB [0,1]
        bgrToCHW_RGB_0to1(boxed, input_w_, input_h_, host_in_);

        // 3) H2D
        Tensor* tin = findTensor(input_name_);
        Tensor* tout = findTensor(output_name_);
        if (!tin || !tout) return 0.f;

        cudaMemcpyAsync(tin->dptr, host_in_.data(), tin->bytes, cudaMemcpyHostToDevice, stream_);

        // 4) enqueueV3
        cudaEventRecord(ev_start_, stream_);
        if (!ctx_->enqueueV3(stream_)) {
            std::cerr << "[TRT] enqueueV3 failed\n";
            return 0.f;
        }
        cudaEventRecord(ev_end_, stream_);

        // 5) D2H output
        host_out_bytes_.resize(tout->bytes);
        cudaMemcpyAsync(host_out_bytes_.data(), tout->dptr, tout->bytes, cudaMemcpyDeviceToHost, stream_);

        cudaStreamSynchronize(stream_);

        // 6) convert output to float vector
        // YOLO11 pose output expected: float32 [1,56,8400] (as you observed via trt_bench)
        const int64_t out_elems = volume(tout->runtime_dims);
        out0.resize((size_t)out_elems);

        if (tout->dtype == nvinfer1::DataType::kFLOAT) {
            std::memcpy(out0.data(), host_out_bytes_.data(), (size_t)out_elems * sizeof(float));
        } else if (tout->dtype == nvinfer1::DataType::kHALF) {
            const __half* hp = reinterpret_cast<const __half*>(host_out_bytes_.data());
            for (int64_t i = 0; i < out_elems; ++i) out0[(size_t)i] = __half2float(hp[i]);
        } else {
            std::cerr << "[WARN] Unsupported output dtype for float conversion\n";
            std::fill(out0.begin(), out0.end(), 0.f);
        }

        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev_start_, ev_end_);
        return ms;
    }

    ~TrtEngine() {
        for (auto& t : io_) {
            if (t.dptr) cudaFree(t.dptr);
            t.dptr = nullptr;
        }
        if (ev_start_) cudaEventDestroy(ev_start_);
        if (ev_end_) cudaEventDestroy(ev_end_);
        if (stream_) cudaStreamDestroy(stream_);
    }

private:
    struct TRTDeleter {
        template <typename T>
        void operator()(T* p) const { if (p) delete p; }
    };

    // NOTE: In TRT10, these are still returned as raw pointers but are deletable by delete.
    // On Jetson TRT10, this pattern works in practice.
    std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> ctx_{nullptr};

    struct Tensor {
        std::string name;
        bool is_input = false;
        nvinfer1::DataType dtype{};
        nvinfer1::Dims engine_dims{};
        nvinfer1::Dims runtime_dims{};
        size_t bytes = 0;
        void* dptr = nullptr;
    };

    Tensor* findTensor(const std::string& name) {
        for (auto& t : io_) if (t.name == name) return &t;
        return nullptr;
    }

    TrtLogger logger_;
    std::vector<Tensor> io_;
    std::string input_name_;
    std::string output_name_;
    std::string last_err_;

    cudaStream_t stream_{nullptr};
    cudaEvent_t ev_start_{nullptr}, ev_end_{nullptr};

    int input_w_ = 640, input_h_ = 640;
    std::vector<float> host_in_;
    std::vector<uint8_t> host_out_bytes_;
};

// ----------------------------- YOLO11 pose decode + draw -----------------------------

struct Detection {
    float x1, y1, x2, y2; // in original image coords
    float conf;
    std::array<cv::Point2f, 17> kpt;
    std::array<float, 17> kpt_conf;
};

static float iou(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float uni = areaA + areaB - inter;
    return (uni <= 0.f) ? 0.f : (inter / uni);
}

static std::vector<Detection> nms(const std::vector<Detection>& dets, float nms_th) {
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j){ return dets[i].conf > dets[j].conf; });

    std::vector<Detection> out;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t _i = 0; _i < idx.size(); ++_i) {
        int i = idx[_i];
        if (suppressed[i]) continue;
        out.push_back(dets[i]);
        for (size_t _j = _i + 1; _j < idx.size(); ++_j) {
            int j = idx[_j];
            if (suppressed[j]) continue;
            if (iou(dets[i], dets[j]) > nms_th) suppressed[j] = true;
        }
    }
    return out;
}

static std::vector<Detection> decodeYolo11Pose(
    const std::vector<float>& out,   // [56*8400], layout [C, N]
    int N,                            // 8400
    int img_w, int img_h,             // original image size
    const LetterboxInfo& lb,
    float conf_th,
    int topk
) {
    const int C = 56;
    if ((int)out.size() != C * N) {
        std::cerr << "[WARN] unexpected output size: " << out.size() << " (expected " << (C * N) << ")\n";
        return {};
    }

    // channels:
    // 0:cx 1:cy 2:w 3:h 4:conf  5..: 17 keypoints * (x,y,score)
    std::vector<Detection> cand;
    cand.reserve(256);

    for (int i = 0; i < N; ++i) {
        float cx = out[0 * N + i];
        float cy = out[1 * N + i];
        float w  = out[2 * N + i];
        float h  = out[3 * N + i];
        float cf = out[4 * N + i];

        if (cf < conf_th) continue;

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // invert letterbox: input(640x640) -> original
        auto inv = [&](float x, float y) -> cv::Point2f {
            float ox = (x - lb.pad_x) / lb.scale;
            float oy = (y - lb.pad_y) / lb.scale;
            ox = clampf(ox, 0.f, (float)(img_w - 1));
            oy = clampf(oy, 0.f, (float)(img_h - 1));
            return cv::Point2f(ox, oy);
        };

        cv::Point2f p1 = inv(x1, y1);
        cv::Point2f p2 = inv(x2, y2);

        Detection d{};
        d.x1 = p1.x; d.y1 = p1.y;
        d.x2 = p2.x; d.y2 = p2.y;
        d.conf = cf;

        for (int k = 0; k < 17; ++k) {
            float kx = out[(5 + 3*k + 0) * N + i];
            float ky = out[(5 + 3*k + 1) * N + i];
            float ks = out[(5 + 3*k + 2) * N + i];
            cv::Point2f pk = inv(kx, ky);
            d.kpt[k] = pk;
            d.kpt_conf[k] = ks;
        }

        cand.push_back(d);
    }

    // Keep topK by confidence before NMS to reduce CPU load
    std::sort(cand.begin(), cand.end(), [](const Detection& a, const Detection& b){ return a.conf > b.conf; });
    if ((int)cand.size() > topk) cand.resize((size_t)topk);

    return cand;
}

static void drawPose(
    cv::Mat& bgr,
    const std::vector<Detection>& dets,
    float kpt_th,
    int max_draw
) {
    // COCO-17 skeleton edges
    static const std::vector<std::pair<int,int>> edges = {
        {0,1},{0,2},{1,3},{2,4},
        {5,6},{5,7},{7,9},{6,8},{8,10},
        {5,11},{6,12},{11,12},
        {11,13},{13,15},{12,14},{14,16}
    };

    int drawn = 0;
    for (const auto& d : dets) {
        if (drawn >= max_draw) break;
        drawn++;

        // bbox
        cv::rectangle(bgr, cv::Rect(cv::Point((int)d.x1,(int)d.y1), cv::Point((int)d.x2,(int)d.y2)),
                      cv::Scalar(0,255,0), 2);

        // conf
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << d.conf;
            cv::putText(bgr, oss.str(), cv::Point((int)d.x1, (int)std::max(0.f, d.y1 - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
        }

        // skeleton lines
        for (auto [a,b] : edges) {
            if (d.kpt_conf[a] >= kpt_th && d.kpt_conf[b] >= kpt_th) {
                cv::line(bgr, d.kpt[a], d.kpt[b], cv::Scalar(255, 255, 0), 2);
            }
        }

        // keypoints
        for (int k = 0; k < 17; ++k) {
            if (d.kpt_conf[k] < kpt_th) continue;
            cv::circle(bgr, d.kpt[k], 3, cv::Scalar(0, 0, 255), -1);
        }
    }
}

// ----------------------------- Main -----------------------------

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }

    // Ctrl+C로 언제든 종료되도록 SIGINT 핸들러 등록
    g_running.store(true);
    std::signal(SIGINT, onSigInt);

    // 1) Load TensorRT engine first (fail fast if engine mismatch)
    TrtEngine trt;
    if (!trt.load(args.engine_path)) {
        std::cerr << "TRT init failed: " << trt.lastError() << "\n";
        return 1;
    }

    // 2) Orbbec: create pipeline + config
    std::shared_ptr<ob::Pipeline> pipe;
    try {
        pipe = std::make_shared<ob::Pipeline>();
    } catch (const ob::Error& e) {
        std::cerr << "Failed to create ob::Pipeline: " << e.getMessage() << "\n";
        return 1;
    }

    auto cfg = std::make_shared<ob::Config>();

    // 3) List all profiles & choose closest
    try {

        // Frame aggregation: require both color+depth in same frameset (example code behavior)
        // cfg->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);



        // Stream profile selection + HW D2C compatibility matching is moved to orbbec_utils.
        // Default color preference: MJPG > BGR > RGB > YUYV, depth: Y16.
        auto sel = orbbec_utils::configureColorDepthStreams(
            pipe, cfg,
            args.color.w, args.color.h, args.color.fps,
            args.depth.w, args.depth.h, args.depth.fps,
            args.enable_d2c_align,
            /*print_lists=*/true
        );

        if (!sel.color) {
            std::cerr << "No color profile found\n";
            return 1;
        }
        if (!sel.depth) {
            std::cerr << "No depth profile found\n";
            return 1;
        }

        // Streams are already enabled in cfg inside configureColorDepthStreams().
        
        // Enable FrameSync (example code behavior)
        if (args.enable_framesync) {
            pipe->enableFrameSync();
        }

        pipe->start(cfg);

    } catch (const ob::Error& e) {
        std::cerr << "Orbbec error during config/start: " << e.getMessage() << "\n";
        return 1;
    }

    std::cout << "\n[OK] Pipeline started. rotate=" << args.rotate
              << " framesync=" << (args.enable_framesync ? "ON" : "OFF")
              << " d2c_align=" << (args.enable_d2c_align ? "HW" : "OFF")
              << "\n";

    // 4) Main loop timing stats
    auto to_ms = [](const Clock::time_point& a, const Clock::time_point& b) -> double {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    RunningStats cap_stats, infer_stats, loop_stats;

    while (g_running.load()) {
        const auto t0 = Clock::now(); // start time

        std::shared_ptr<ob::FrameSet> fs;
        try {
            fs = pipe->waitForFrameset(100);
        } catch (const ob::Error& e) {
            std::cerr << "waitForFrameset error: " << e.getMessage() << "\n";
            continue;
        }
        if (!fs) continue;

        auto c = fs->colorFrame();
        auto d = fs->depthFrame();
        if (!c || !d) continue;

        const auto t1 = Clock::now(); // captured time

        // Timestamp check: helps confirm sync quality in practice
        // (Unit depends on SDK; typical is ms or us. We'll still show delta.)
        double tc = static_cast<double>(c->getTimeStampUs());
        double td = static_cast<double>(d->getTimeStampUs());
        double dt = std::abs(tc - td) / 1000.0;

        cv::Mat color_bgr = colorFrameToBGR(c);
        cv::Mat depth16 = depthFrameToMat16U(d);
        if (color_bgr.empty() || depth16.empty()) continue;

        // Rotate both for consistent display/inference
        rotateIfNeeded(color_bgr, args.rotate);
        rotateIfNeeded(depth16, args.rotate);

        // Depth colormap for display
        cv::Mat depth_cm = depthToColorMap(depth16, args.depth_max_mm);

        const auto t2 = Clock::now(); // image processed time & infer start

        // 5) TensorRT inference
        LetterboxInfo lb{};
        std::vector<float> out0;
        float infer_ms = trt.inferPose(color_bgr, out0, &lb);

        const auto t3 = Clock::now(); // infer end

        // 6) Decode YOLO11 pose output [1,56,8400]
        // output dims are [1,56,8400] => flatten to [56,8400]
        // N=8400
        const int N = 8400;
        auto cand = decodeYolo11Pose(out0, N, color_bgr.cols, color_bgr.rows, lb, args.conf_th, args.topk);
        auto keep = nms(cand, args.nms_th);

        int key = -1;
        if (args.GUI == true) {
            // 7) Draw on ORIGINAL color image (NOT the letterbox crop)
            cv::Mat vis = color_bgr.clone();
            drawPose(vis, keep, args.kpt_th, args.max_draw);

            // 8) Show windows
            cv::imshow("color_pose", vis);
            if (!depth_cm.empty()) cv::imshow("depth_colormap", depth_cm);

            key = cv::waitKey(1) & 0xFF;
        } else {
            // no_gui에서도 동일 종료를 원하면:
            // 1) Ctrl+C는 항상 동작 (SIGINT)
            // 2) 추가로 "q + Enter"도 받고 싶으면 아래 폴링 사용
            key = pollStdinKeyNonBlocking();
        }

        if (key == 27 || key == 'q') {
            g_running.store(false);
        }

        // runtime stats
        const auto t4 = Clock::now(); // total process end

        const double capture_ms = to_ms(t0, t1);
        const double total_img_ms = to_ms(t0, t2);
        const double infer_process_ms = to_ms(t2, t3);
        const double loop_ms = to_ms(t0, t4);

        cap_stats.add(capture_ms);
        infer_stats.add(infer_ms);
        loop_stats.add(loop_ms);

        std::cout << "capture=" << std::fixed << std::setprecision(1) << capture_ms << " ms,";
        std::cout << "img_process=" << std::fixed << std::setprecision(1) << total_img_ms << " ms,";
        std::cout << "infer=" << std::fixed << std::setprecision(1) << infer_ms << " ms,";
        std::cout << "infer_process=" << std::fixed << std::setprecision(1) << infer_process_ms << " ms,";
        std::cout << "loop=" << std::fixed << std::setprecision(1) << loop_ms << " ms\n";
    }

    std::cout << "\n"
        << "total frame : " << cap_stats.n << "| GUI : " << args.GUI <<"\n"
        << "capture_mean=" << std::fixed << std::setprecision(1) << cap_stats.mean
        << " ms, capture_min=" << std::fixed << std::setprecision(1) << cap_stats.min << " ms\n"
        << "infer_mean=" << std::fixed << std::setprecision(1) << infer_stats.mean
        << " ms, infer_max=" << std::fixed << std::setprecision(1) << infer_stats.max << " ms\n"
        << "loop_mean=" << std::fixed << std::setprecision(1) << loop_stats.mean
        << " ms, loop_max=" << std::fixed << std::setprecision(1) << loop_stats.max << " ms\n";

    try {
        pipe->stop();
    } catch (...) {}

    return 0;
}   
