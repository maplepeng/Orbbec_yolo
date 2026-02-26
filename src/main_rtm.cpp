#include "orbbec_utils.hpp"
#include "trt_runner_multi.hpp"

#include <libobsensor/ObSensor.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/select.h>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;

static inline float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }

static std::atomic<bool> g_running{true};

static void onSigInt(int) { g_running.store(false); }

static int pollStdinKeyNonBlocking() {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    const int rv = select(STDIN_FILENO + 1, &rfds, nullptr, nullptr, &tv);
    if (rv > 0 && FD_ISSET(STDIN_FILENO, &rfds)) {
        unsigned char c = 0;
        const ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n == 1) return static_cast<int>(c);
    }
    return -1;
}

static bool startsWith(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static bool parseBoolString(const std::string& s, bool& out) {
    const std::string v = toLower(s);
    if (v == "1" || v == "true" || v == "on" || v == "yes") {
        out = true;
        return true;
    }
    if (v == "0" || v == "false" || v == "off" || v == "no") {
        out = false;
        return true;
    }
    return false;
}

struct Spec {
    int w = 640;
    int h = 400;
    int fps = 30;
};

static bool parseWHFps(const std::string& s, Spec& out) {
    auto t = s;
    const auto at = t.find('@');
    const std::string wh = (at == std::string::npos) ? t : t.substr(0, at);
    const std::string fps = (at == std::string::npos) ? "" : t.substr(at + 1);

    const auto x = wh.find('x');
    if (x == std::string::npos) return false;
    out.w = std::stoi(wh.substr(0, x));
    out.h = std::stoi(wh.substr(x + 1));
    if (!fps.empty()) out.fps = std::stoi(fps);
    return true;
}

static bool parseWH(const std::string& s, int& w, int& h) {
    const auto x = s.find('x');
    if (x == std::string::npos) return false;
    w = std::stoi(s.substr(0, x));
    h = std::stoi(s.substr(x + 1));
    return true;
}

struct Args {
    std::string yolo_engine_path = "./models/engine/yolo11n-pose_fp16.engine";
    std::string pose_engine_path = "./models/engine/rtmpose_s_fp16.engine";

    int yolo_input_w = 640;
    int yolo_input_h = 640;
    int pose_input_w = 192;
    int pose_input_h = 256;

    Spec color{640, 400, 30};
    Spec depth{640, 400, 30};
    int rotate = 0;

    bool GUI = true;
    bool time_check = false;
    bool debug = false;
    std::string debug_log_path = "./log/rtm_debug.jsonl";

    bool enable_framesync = true;
    bool enable_d2c_align = true;
    bool enable_hw_noise = true;
    float hw_noise_thresh = 0.2f;

    float yolo_conf_th = 0.50f;
    float yolo_nms_th = 0.45f;
    float yolo_kpt_conf_th = 0.25f;
    int yolo_kpt_min_count = 0;  // 0 disables YOLO keypoint-count gate.
    float yolo_edge_kpt_conf_th = 0.25f;
    int yolo_edge_kpt_min_count = 0;  // 0 disables YOLO edge-keypoint gate.
    float kpt_th = 0.25f;

    int topk = 50;
    int max_person = 3;
    float person_expand = 1.10f;

    float depth_max_mm = 4000.0f;
};

static void printUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --yolo_engine=/path/to/yolo.engine    (default ./models/engine/yolo11n-pose_fp16.engine)\n"
        << "  --pose_engine=/path/to/rtmpose.engine (default ./models/engine/rtmpose_s_fp16.engine)\n"
        << "  --yolo_input=640x640                  (default 640x640)\n"
        << "  --pose_input=192x256                  (default 192x256, WxH)\n"
        << "  --color=640x400@30                    (default 640x400@30)\n"
        << "  --depth=640x400@30                    (default 640x400@30)\n"
        << "  --rotate=0|90|180|270                 (default 0)\n"
        << "  --conf=0.50                           YOLOpose confidence threshold\n"
        << "  --nms=0.45                            YOLOpose NMS IoU threshold\n"
        << "  --yolo_kpt_conf=0.25                  YOLO keypoint confidence threshold for gate\n"
        << "  --yolo_kpt_min_count=0                Min # of YOLO keypoints above threshold (0=disable)\n"
        << "  --yolo_edge_kpt_conf=0.25             YOLO edge-keypoint confidence threshold for gate\n"
        << "  --yolo_edge_kpt_min_count=0           Min # of edge keypoints above threshold (0=disable)\n"
        << "                                        Edge keypoints: nose(0), wrists(9,10), ankles(15,16)\n"
        << "  --topk=50                             Keep top-K candidates before NMS\n"
        << "  --kpt=0.25                            Keypoint confidence threshold\n"
        << "  --max_person=3                        Max persons to run RTMPose\n"
        << "  --person_expand=1.10                  BBox expansion factor for top-down crop\n"
        << "  --no_gui                              Disable GUI\n"
        << "  --time                                Enable per-frame timing logs (requires --debug)\n"
        << "  --debug[=true|false]                  Enable realtime per-frame metrics logging\n"
        << "  --debug_log=/path/to/file.jsonl       Realtime metrics JSONL output path\n"
        << "  --no_sync | --no_align | --no_hw_noise\n"
        << "  --hw_noise_thresh=0.2\n"
        << "  --depth_max_mm=4000\n";
}

static bool parseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string s(argv[i]);
        if (s == "--help" || s == "-h") return false;

        if (startsWith(s, "--yolo_engine=")) {
            a.yolo_engine_path = s.substr(14);
        } else if (startsWith(s, "--det_engine=")) {  // backward compatible alias
            a.yolo_engine_path = s.substr(13);
        } else if (startsWith(s, "--pose_engine=")) {
            a.pose_engine_path = s.substr(14);
        } else if (startsWith(s, "--yolo_input=")) {
            if (!parseWH(s.substr(13), a.yolo_input_w, a.yolo_input_h)) return false;
        } else if (startsWith(s, "--det_input=")) {  // backward compatible alias
            if (!parseWH(s.substr(12), a.yolo_input_w, a.yolo_input_h)) return false;
        } else if (startsWith(s, "--pose_input=")) {
            if (!parseWH(s.substr(13), a.pose_input_w, a.pose_input_h)) return false;
        } else if (startsWith(s, "--color=")) {
            if (!parseWHFps(s.substr(8), a.color)) return false;
        } else if (startsWith(s, "--depth=")) {
            if (!parseWHFps(s.substr(8), a.depth)) return false;
        } else if (startsWith(s, "--rotate=")) {
            a.rotate = std::stoi(s.substr(9));
        } else if (s == "--gui") {
            a.GUI = true;
        } else if (s == "--no_gui") {
            a.GUI = false;
        } else if (s == "--time") {
            a.time_check = true;
        } else if (s == "--debug") {
            a.debug = true;
        } else if (startsWith(s, "--debug=")) {
            if (!parseBoolString(s.substr(8), a.debug)) {
                std::cerr << "Invalid --debug value: " << s.substr(8) << " (use true/false/1/0)\n";
                return false;
            }
        } else if (startsWith(s, "--debug_log=")) {
            a.debug_log_path = s.substr(12);
        } else if (s == "--no_sync") {
            a.enable_framesync = false;
        } else if (s == "--no_align") {
            a.enable_d2c_align = false;
        } else if (s == "--no_hw_noise") {
            a.enable_hw_noise = false;
        } else if (startsWith(s, "--hw_noise_thresh=")) {
            a.hw_noise_thresh = std::stof(s.substr(17));
        } else if (startsWith(s, "--conf=")) {
            a.yolo_conf_th = std::stof(s.substr(7));
        } else if (startsWith(s, "--nms=")) {
            a.yolo_nms_th = std::stof(s.substr(6));
        } else if (startsWith(s, "--yolo_kpt_conf=")) {
            a.yolo_kpt_conf_th = std::stof(s.substr(16));
        } else if (startsWith(s, "--yolo_kpt_min_count=")) {
            a.yolo_kpt_min_count = std::stoi(s.substr(21));
        } else if (startsWith(s, "--yolo_edge_kpt_conf=")) {
            a.yolo_edge_kpt_conf_th = std::stof(s.substr(std::string("--yolo_edge_kpt_conf=").size()));
        } else if (startsWith(s, "--yolo_edge_kpt_min_count=")) {
            a.yolo_edge_kpt_min_count = std::stoi(s.substr(std::string("--yolo_edge_kpt_min_count=").size()));
        } else if (startsWith(s, "--det_conf=")) {
            a.yolo_conf_th = std::stof(s.substr(11));
        } else if (startsWith(s, "--det_nms=")) {
            a.yolo_nms_th = std::stof(s.substr(10));
        } else if (startsWith(s, "--topk=")) {
            a.topk = std::stoi(s.substr(7));
        } else if (startsWith(s, "--kpt=")) {
            a.kpt_th = std::stof(s.substr(6));
        } else if (startsWith(s, "--max_person=")) {
            a.max_person = std::stoi(s.substr(13));
        } else if (startsWith(s, "--person_expand=")) {
            a.person_expand = std::stof(s.substr(16));
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
    if (a.yolo_input_w <= 0 || a.yolo_input_h <= 0 || a.pose_input_w <= 0 || a.pose_input_h <= 0) {
        std::cerr << "input size must be positive\n";
        return false;
    }
    if (a.topk <= 0) a.topk = 1;
    if (a.max_person <= 0) a.max_person = 1;
    if (a.person_expand <= 0.0f) a.person_expand = 1.0f;
    a.yolo_kpt_conf_th = clampf(a.yolo_kpt_conf_th, 0.0f, 1.0f);
    a.yolo_kpt_min_count = std::max(0, std::min(17, a.yolo_kpt_min_count));
    a.yolo_edge_kpt_conf_th = clampf(a.yolo_edge_kpt_conf_th, 0.0f, 1.0f);
    a.yolo_edge_kpt_min_count = std::max(0, std::min(5, a.yolo_edge_kpt_min_count));
    if (!a.debug && a.time_check) {
        std::cerr << "[WARN] --time is ignored when --debug is false\n";
        a.time_check = false;
    }
    if (a.debug && a.debug_log_path.empty()) {
        std::cerr << "--debug_log path must not be empty when --debug is enabled\n";
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
        }
    }

    bool hasExtrema() const { return n > 100; }
};

static uint64_t wallClockNowUs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

struct CaptureTimestampInfo {
    uint64_t global_us = 0;
    uint64_t system_us = 0;
    uint64_t hardware_us = 0;
    uint64_t capture_us = 0;
    uint64_t latency_base_us = 0;
    std::string source = "host_receive";
};

static CaptureTimestampInfo collectCaptureTimestampInfo(const std::shared_ptr<ob::Frame>& frame, uint64_t host_receive_us) {
    CaptureTimestampInfo info;

    try {
        info.global_us = frame ? frame->getGlobalTimeStampUs() : 0;
    } catch (...) {
        info.global_us = 0;
    }
    try {
        info.system_us = frame ? frame->getSystemTimeStampUs() : 0;
    } catch (...) {
        info.system_us = 0;
    }
    try {
        info.hardware_us = frame ? frame->getTimeStampUs() : 0;
    } catch (...) {
        info.hardware_us = 0;
    }

    if (info.global_us > 0) {
        info.source = "global_host";
        info.capture_us = info.global_us;
        info.latency_base_us = info.global_us;
        return info;
    }
    if (info.system_us > 0) {
        info.source = "system_host";
        info.capture_us = info.system_us;
        info.latency_base_us = info.system_us;
        return info;
    }
    if (info.hardware_us > 0) {
        info.source = "hardware_device";
        info.capture_us = info.hardware_us;
        info.latency_base_us = host_receive_us;
        return info;
    }

    info.source = "host_receive";
    info.capture_us = host_receive_us;
    info.latency_base_us = host_receive_us;
    return info;
}

static bool tryGetFrameMetadataValue(const std::shared_ptr<ob::Frame>& frame, OBFrameMetadataType type, int64_t& out) {
    if (!frame) return false;
    try {
        if (!frame->hasMetadata(type)) return false;
        out = frame->getMetadataValue(type);
        return true;
    } catch (...) {
        return false;
    }
}

static const char* syncReferenceToString(int32_t ref) {
    switch (ref) {
        case START_OF_EXPOSURE:
            return "START_OF_EXPOSURE";
        case MIDDLE_OF_EXPOSURE:
            return "MIDDLE_OF_EXPOSURE";
        case END_OF_EXPOSURE:
            return "END_OF_EXPOSURE";
        default:
            return "UNKNOWN";
    }
}

static double quantileLinear(std::vector<double> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    q = clampf(static_cast<float>(q), 0.0f, 1.0f);
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return v[lo];
    const double frac = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static cv::Mat decodeMJPGtoBGR(const uint8_t* data, size_t bytes) {
    cv::Mat buf(1, static_cast<int>(bytes), CV_8UC1, const_cast<uint8_t*>(data));
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}

static cv::Mat colorFrameToBGR(const std::shared_ptr<ob::ColorFrame>& cf) {
    if (!cf) return {};

    const auto fmt = cf->getFormat();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(cf->getData());
    const size_t bytes = static_cast<size_t>(cf->getDataSize());
    const int w = static_cast<int>(cf->getWidth());
    const int h = static_cast<int>(cf->getHeight());

    if (fmt == OB_FORMAT_MJPG || fmt == OB_FORMAT_MJPEG) {
        return decodeMJPGtoBGR(data, bytes);
    }

    if (fmt == OB_FORMAT_RGB || fmt == OB_FORMAT_RGB888) {
        cv::Mat rgb(h, w, CV_8UC3, const_cast<uint8_t*>(data));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr.clone();
    }

    if (fmt == OB_FORMAT_BGR) {
        cv::Mat bgr(h, w, CV_8UC3, const_cast<uint8_t*>(data));
        return bgr.clone();
    }

    if (fmt == OB_FORMAT_YUYV) {
        cv::Mat yuyv(h, w, CV_8UC2, const_cast<uint8_t*>(data));
        cv::Mat bgr;
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
        return bgr.clone();
    }

    cv::Mat mj = decodeMJPGtoBGR(data, bytes);
    if (!mj.empty()) return mj;

    std::cerr << "[WARN] Unsupported color format: " << orbbec_utils::obFormatToStr(fmt)
              << " bytes=" << bytes << " w=" << w << " h=" << h << "\n";
    return {};
}

static cv::Mat depthFrameToMat16U(const std::shared_ptr<ob::DepthFrame>& df) {
    if (!df) return {};
    const int w = static_cast<int>(df->getWidth());
    const int h = static_cast<int>(df->getHeight());

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
    cv::Mat d32f;
    depth16.convertTo(d32f, CV_32F);
    cv::Mat clipped;
    cv::min(d32f, max_mm, clipped);
    cv::Mat u8;
    clipped.convertTo(u8, CV_8U, 255.0 / max_mm);
    cv::Mat cm;
    cv::applyColorMap(u8, cm, cv::COLORMAP_JET);
    return cm;
}

struct CamInternal {
    bool ready = false;
    OBCameraIntrinsic color_intr{};
    float depth_value_scale = 0.0f;
};

static inline cv::Point2f unrotatePixelToOrig(const cv::Point2f& p_rot, int rotate_deg, int orig_w, int orig_h) {
    const float xr = p_rot.x;
    const float yr = p_rot.y;
    if (rotate_deg == 0) {
        return p_rot;
    } else if (rotate_deg == 90) {
        return cv::Point2f(yr, static_cast<float>(orig_h - 1) - xr);
    } else if (rotate_deg == 180) {
        return cv::Point2f(static_cast<float>(orig_w - 1) - xr, static_cast<float>(orig_h - 1) - yr);
    } else if (rotate_deg == 270) {
        return cv::Point2f(static_cast<float>(orig_w - 1) - yr, xr);
    }
    return p_rot;
}

static bool keypointToXYZ(const CamInternal& ci,
                          const cv::Mat& depth16_rot,
                          const cv::Point2f& p_rot,
                          int rotate_deg,
                          int orig_w, int orig_h,
                          float& X, float& Y, float& Z,
                          uint16_t& raw_depth) {
    if (!ci.ready) return false;
    if (depth16_rot.empty() || depth16_rot.type() != CV_16UC1) return false;

    const int u = static_cast<int>(std::lround(p_rot.x));
    const int v = static_cast<int>(std::lround(p_rot.y));
    if (u < 0 || u >= depth16_rot.cols || v < 0 || v >= depth16_rot.rows) return false;

    raw_depth = depth16_rot.at<uint16_t>(v, u);
    if (raw_depth == 0) return false;

    const float z_mm = raw_depth * ci.depth_value_scale;
    Z = z_mm * 0.001f;

    const cv::Point2f p_orig = unrotatePixelToOrig(p_rot, rotate_deg, orig_w, orig_h);
    X = (p_orig.x - ci.color_intr.cx) / ci.color_intr.fx * Z;
    Y = (p_orig.y - ci.color_intr.cy) / ci.color_intr.fy * Z;
    return true;
}

struct PersonDet {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float conf = 0.0f;
    int label = -1;
    int yolo_high_kpt_count = 0;
    int yolo_edge_kpt_count = 0;
    bool yolo_gate_pass = true;
    std::array<cv::Point2f, 17> yolo_kpt{};
    std::array<float, 17> yolo_kpt_conf{};
    float depth_mm = -1.0f;
};

static float robustTorsoDepthMm(const PersonDet& det, const cv::Mat& depth16, float depth_value_scale, float quantile = 0.65f) {
    if (depth16.empty() || depth16.type() != CV_16UC1) return -1.0f;

    const float w = std::max(1.0f, det.x2 - det.x1);
    const float h = std::max(1.0f, det.y2 - det.y1);

    // Torso-focused ROI to reduce foreground occluder bias from limbs/edges.
    const float rx1 = det.x1 + 0.35f * w;
    const float rx2 = det.x1 + 0.65f * w;
    const float ry1 = det.y1 + 0.25f * h;
    const float ry2 = det.y1 + 0.70f * h;

    const int x1 = std::max(0, static_cast<int>(std::floor(rx1)));
    const int y1 = std::max(0, static_cast<int>(std::floor(ry1)));
    const int x2 = std::min(depth16.cols - 1, static_cast<int>(std::ceil(rx2)));
    const int y2 = std::min(depth16.rows - 1, static_cast<int>(std::ceil(ry2)));
    if (x2 <= x1 || y2 <= y1) return -1.0f;

    const int roi_w = x2 - x1 + 1;
    const int roi_h = y2 - y1 + 1;
    const int roi_area = roi_w * roi_h;

    // Grid sampling to cap depth collection cost while keeping stable statistics.
    constexpr int target_samples = 400;
    int sample_step = 1;
    if (roi_area > target_samples) {
        sample_step = static_cast<int>(std::floor(std::sqrt(static_cast<float>(roi_area) / target_samples)));
        sample_step = std::max(1, std::min(sample_step, 8));
    }

    std::vector<uint16_t> vals;
    vals.reserve(static_cast<size_t>(roi_area / (sample_step * sample_step) + 4));
    for (int y = y1; y <= y2; y += sample_step) {
        const uint16_t* row = depth16.ptr<uint16_t>(y);
        for (int x = x1; x <= x2; x += sample_step) {
            const uint16_t d = row[x];
            if (d > 0) vals.push_back(d);
        }
    }
    if (vals.size() < 20U) return -1.0f;

    quantile = clampf(quantile, 0.50f, 0.90f);
    size_t qi = static_cast<size_t>(std::floor(quantile * static_cast<float>(vals.size() - 1)));
    if (qi >= vals.size()) qi = vals.size() - 1;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(qi), vals.end());
    const float scale = (depth_value_scale > 0.0f) ? depth_value_scale : 1.0f;
    return static_cast<float>(vals[qi]) * scale;
}

struct PoseResult {
    PersonDet det;
    cv::Rect2f expanded_box;
    cv::Rect2f pose_box;
    std::vector<cv::Point2f> kpt;
    std::vector<float> kpt_conf;
    float pose_mean_conf = 0.0f;
    float depth_mm = -1.0f;
};

static float iou(const PersonDet& a, const PersonDet& b) {
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = areaA + areaB - inter;
    return (uni <= 0.0f) ? 0.0f : (inter / uni);
}

static std::vector<PersonDet> nms(const std::vector<PersonDet>& dets, float nms_th) {
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return dets[i].conf > dets[j].conf; });

    std::vector<PersonDet> out;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t _i = 0; _i < idx.size(); ++_i) {
        const int i = idx[_i];
        if (suppressed[i]) continue;
        out.push_back(dets[i]);
        for (size_t _j = _i + 1; _j < idx.size(); ++_j) {
            const int j = idx[_j];
            if (suppressed[j]) continue;
            if (iou(dets[i], dets[j]) > nms_th) suppressed[j] = true;
        }
    }
    return out;
}

static std::string findTensorName(const std::vector<std::string>& names, const std::vector<std::string>& keys) {
    for (const auto& n : names) {
        const std::string ln = toLower(n);
        for (const auto& k : keys) {
            if (ln.find(k) != std::string::npos) return n;
        }
    }
    return {};
}

static cv::Point2f invYoloLetterbox(float x, float y, int img_w, int img_h, const LetterBoxInfo& lb) {
    float ox = x;
    float oy = y;

    if (x >= 0.0f && x <= 1.5f && y >= 0.0f && y <= 1.5f) {
        ox = x * static_cast<float>(lb.dst_w);
        oy = y * static_cast<float>(lb.dst_h);
    }

    ox = (ox - static_cast<float>(lb.pad_x)) / lb.scale;
    oy = (oy - static_cast<float>(lb.pad_y)) / lb.scale;

    ox = clampf(ox, 0.0f, static_cast<float>(img_w - 1));
    oy = clampf(oy, 0.0f, static_cast<float>(img_h - 1));
    return cv::Point2f(ox, oy);
}

static cv::Rect2f makeExpandedBox(const PersonDet& d, float padding) {
    const float cx = (d.x1 + d.x2) * 0.5f;
    const float cy = (d.y1 + d.y2) * 0.5f;
    float bw = std::max(1.0f, d.x2 - d.x1) * padding;
    float bh = std::max(1.0f, d.y2 - d.y1) * padding;

    float x = cx - bw * 0.5f;
    float y = cy - bh * 0.5f;
    return cv::Rect2f(x, y, bw, bh);
}

static cv::Rect2f makePoseBox(const cv::Rect2f& expanded_box, int img_w, int img_h, float aspect_w_over_h) {
    const float cx = expanded_box.x + expanded_box.width * 0.5f;
    const float cy = expanded_box.y + expanded_box.height * 0.5f;
    float bw = std::max(1.0f, expanded_box.width);
    float bh = std::max(1.0f, expanded_box.height);

    // Match MMPose TopDownGetBboxCenterScale aspect adjustment.
    if (bw > aspect_w_over_h * bh) {
        bh = bw / aspect_w_over_h;
    } else if (bw < aspect_w_over_h * bh) {
        bw = bh * aspect_w_over_h;
    }

    float x = cx - bw * 0.5f;
    float y = cy - bh * 0.5f;
    float w = bw;
    float h = bh;

    // Clamp to image bounds while preserving box center/size as much as possible.
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;
    if (x + w > img_w - 1.0f) x = std::max(0.0f, (img_w - 1.0f) - w);
    if (y + h > img_h - 1.0f) y = std::max(0.0f, (img_h - 1.0f) - h);
    return cv::Rect2f(x, y, w, h);
}

static void buildMMPoseInput(const cv::Mat& bgr,
                             const cv::Rect2f& pose_box,
                             int dst_w,
                             int dst_h,
                             std::vector<float>& nchw,
                             cv::Matx23f& M,
                             cv::Matx23f& Minv) {
    // Affine from padded person box -> pose input.
    cv::Point2f src_tri[3] = {
        cv::Point2f(pose_box.x, pose_box.y),
        cv::Point2f(pose_box.x + pose_box.width, pose_box.y),
        cv::Point2f(pose_box.x, pose_box.y + pose_box.height)
    };
    cv::Point2f dst_tri[3] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(dst_w - 1), 0.0f),
        cv::Point2f(0.0f, static_cast<float>(dst_h - 1))
    };
    M = cv::getAffineTransform(src_tri, dst_tri);
    Minv = cv::getAffineTransform(dst_tri, src_tri);

    cv::Mat warped;
    cv::warpAffine(bgr, warped, M, cv::Size(dst_w, dst_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // MMPose pipeline normalize: to_rgb=true with RGB mean/std (0..255 space).
    constexpr float mean[3] = {123.675f, 116.28f, 103.53f};  // R,G,B
    constexpr float stdv[3] = {58.395f, 57.12f, 57.375f};    // R,G,B

    nchw.resize(static_cast<size_t>(3 * dst_h * dst_w));
    for (int y = 0; y < dst_h; ++y) {
        const cv::Vec3b* row = warped.ptr<cv::Vec3b>(y);
        for (int x = 0; x < dst_w; ++x) {
            const float rch = (static_cast<float>(row[x][2]) - mean[0]) / stdv[0];
            const float g = (static_cast<float>(row[x][1]) - mean[1]) / stdv[1];
            const float b = (static_cast<float>(row[x][0]) - mean[2]) / stdv[2];
            nchw[0 * dst_h * dst_w + y * dst_w + x] = rch;
            nchw[1 * dst_h * dst_w + y * dst_w + x] = g;
            nchw[2 * dst_h * dst_w + y * dst_w + x] = b;
        }
    }
}

static std::vector<PersonDet> decodeYoloPoseBoxes(TrtRunnerMulti& yolo_runner,
                                                   const LetterBoxInfo& lb,
                                                   int img_w,
                                                   int img_h,
                                                   float conf_th,
                                                   int topk,
                                                   float yolo_kpt_conf_th,
                                                   int yolo_kpt_min_count,
                                                   float yolo_edge_kpt_conf_th,
                                                   int yolo_edge_kpt_min_count) {
    constexpr int kChannels = 56;  // YOLO11 pose output: [x,y,w,h,conf,17*(x,y,s)]
    const int topk_safe = std::max(1, topk);

    std::vector<PersonDet> out;
    const auto& names = yolo_runner.outputNames();
    if (names.empty()) return out;

    std::string yolo_name;
    size_t yolo_size = 0;
    for (const auto& n : names) {
        const auto& t = yolo_runner.output(n);
        if (t.size() >= kChannels && (t.size() % kChannels == 0) && t.size() > yolo_size) {
            yolo_name = n;
            yolo_size = t.size();
        }
    }
    if (yolo_name.empty()) {
        // Fallback: choose the largest output tensor.
        for (const auto& n : names) {
            const auto& t = yolo_runner.output(n);
            if (t.size() > yolo_size) {
                yolo_name = n;
                yolo_size = t.size();
            }
        }
    }
    if (yolo_name.empty() || yolo_size < kChannels) return out;

    const auto& t = yolo_runner.output(yolo_name);
    const auto dims = yolo_runner.outputDims(yolo_name);
    if (t.size() != yolo_size) return out;

    bool channel_first = true;  // [C,N]
    int N = 0;
    if (dims.nbDims >= 2) {
        const int a = dims.d[dims.nbDims - 2];
        const int b = dims.d[dims.nbDims - 1];
        if (a == kChannels && b > 0) {
            channel_first = true;
            N = b;
        } else if (b == kChannels && a > 0) {
            channel_first = false;  // [N,C]
            N = a;
        }
    }
    if (N <= 0) {
        if ((t.size() % kChannels) != 0) return out;
        N = static_cast<int>(t.size() / kChannels);
        channel_first = true;
    }

    auto at = [&](int c, int i) -> float {
        if (channel_first) return t[static_cast<size_t>(c) * static_cast<size_t>(N) + static_cast<size_t>(i)];
        return t[static_cast<size_t>(i) * kChannels + static_cast<size_t>(c)];
    };

    out.reserve(static_cast<size_t>(std::min(N, topk_safe * 2)));
    for (int i = 0; i < N; ++i) {
        const float cf = at(4, i);
        if (!std::isfinite(cf) || cf < conf_th) continue;

        const float cx = at(0, i);
        const float cy = at(1, i);
        const float w = at(2, i);
        const float h = at(3, i);
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h)) continue;

        const float x1 = cx - w * 0.5f;
        const float y1 = cy - h * 0.5f;
        const float x2 = cx + w * 0.5f;
        const float y2 = cy + h * 0.5f;

        const cv::Point2f p1 = invYoloLetterbox(x1, y1, img_w, img_h, lb);
        const cv::Point2f p2 = invYoloLetterbox(x2, y2, img_w, img_h, lb);
        if (p2.x <= p1.x || p2.y <= p1.y) continue;
        if ((p2.x - p1.x) * (p2.y - p1.y) < 64.0f) continue;

        PersonDet d;
        int yolo_high_kpt_count = 0;
        int yolo_edge_kpt_count = 0;
        auto isEdgeKpt = [](int k) -> bool {
            return k == 0 || k == 9 || k == 10 || k == 15 || k == 16;
        };
        for (int k = 0; k < 17; ++k) {
            const float kx = at(5 + 3 * k + 0, i);
            const float ky = at(5 + 3 * k + 1, i);
            const float ks = at(5 + 3 * k + 2, i);

            cv::Point2f pk = invYoloLetterbox(kx, ky, img_w, img_h, lb);
            if (!std::isfinite(pk.x) || !std::isfinite(pk.y)) pk = cv::Point2f(0.0f, 0.0f);
            d.yolo_kpt[static_cast<size_t>(k)] = pk;
            d.yolo_kpt_conf[static_cast<size_t>(k)] = std::isfinite(ks) ? ks : 0.0f;

            if (std::isfinite(ks) && ks >= yolo_kpt_conf_th) ++yolo_high_kpt_count;
            if (isEdgeKpt(k) && std::isfinite(ks) && ks >= yolo_edge_kpt_conf_th) ++yolo_edge_kpt_count;
        }

        d.x1 = p1.x;
        d.y1 = p1.y;
        d.x2 = p2.x;
        d.y2 = p2.y;
        d.conf = cf;
        d.label = 0;  // YOLOpose here is person-only.
        d.yolo_high_kpt_count = yolo_high_kpt_count;
        d.yolo_edge_kpt_count = yolo_edge_kpt_count;
        const bool pass_count_gate = (yolo_kpt_min_count <= 0) || (yolo_high_kpt_count >= yolo_kpt_min_count);
        const bool pass_edge_gate = (yolo_edge_kpt_min_count <= 0) || (yolo_edge_kpt_count >= yolo_edge_kpt_min_count);
        d.yolo_gate_pass = pass_count_gate && pass_edge_gate;
        out.push_back(d);
    }

    std::sort(out.begin(), out.end(), [](const PersonDet& a, const PersonDet& b) { return a.conf > b.conf; });
    if (static_cast<int>(out.size()) > topk_safe) out.resize(static_cast<size_t>(topk_safe));
    return out;
}

static bool decodePoseTensor(const std::vector<float>& t,
                             const nvinfer1::Dims& dims,
                             int pose_w,
                             int pose_h,
                             const cv::Matx23f& Minv,
                             std::vector<cv::Point2f>& out_kpt,
                             std::vector<float>& out_conf) {
    if (t.empty()) return false;

    int k = 0;
    enum Layout {
        UNKNOWN = 0,
        K3,
        C3K
    } layout = UNKNOWN;

    if (dims.nbDims >= 2) {
        const int a = dims.d[dims.nbDims - 2];
        const int b = dims.d[dims.nbDims - 1];
        if (b == 3 && a > 0) {
            layout = K3;
            k = a;
        } else if (a == 3 && b > 0) {
            layout = C3K;
            k = b;
        }
    }

    if (layout == UNKNOWN && t.size() % 3 == 0) {
        layout = K3;
        k = static_cast<int>(t.size() / 3);
    }

    if (k <= 0 || k > 300) return false;

    out_kpt.resize(static_cast<size_t>(k));
    out_conf.resize(static_cast<size_t>(k));

    float max_x = 0.0f;
    float max_y = 0.0f;

    for (int i = 0; i < k; ++i) {
        float x = 0.0f;
        float y = 0.0f;
        float s = 1.0f;

        if (layout == K3) {
            x = t[i * 3 + 0];
            y = t[i * 3 + 1];
            s = t[i * 3 + 2];
        } else {
            x = t[i + 0 * k];
            y = t[i + 1 * k];
            s = t[i + 2 * k];
        }

        max_x = std::max(max_x, std::fabs(x));
        max_y = std::max(max_y, std::fabs(y));

        out_kpt[static_cast<size_t>(i)] = cv::Point2f(x, y);
        out_conf[static_cast<size_t>(i)] = s;
    }

    const bool normalized = (max_x <= 1.5f && max_y <= 1.5f);

    for (int i = 0; i < k; ++i) {
        float px = out_kpt[static_cast<size_t>(i)].x;
        float py = out_kpt[static_cast<size_t>(i)].y;

        if (normalized) {
            px *= static_cast<float>(pose_w);
            py *= static_cast<float>(pose_h);
        }

        const float ox = Minv(0, 0) * px + Minv(0, 1) * py + Minv(0, 2);
        const float oy = Minv(1, 0) * px + Minv(1, 1) * py + Minv(1, 2);
        out_kpt[static_cast<size_t>(i)].x = ox;
        out_kpt[static_cast<size_t>(i)].y = oy;
    }

    return true;
}

static bool decodeRTMPose(TrtRunnerMulti& pose_runner,
                          int pose_w,
                          int pose_h,
                          const cv::Matx23f& Minv,
                          std::vector<cv::Point2f>& out_kpt,
                          std::vector<float>& out_conf) {
    const auto& names = pose_runner.outputNames();

    // 0) SimCC format: x=[1,K,Wbins], y=[1,K,Hbins], e.g., [1,26,384] & [1,26,512].
    //    Coordinates are decoded by argmax over bins.
    for (size_t i = 0; i < names.size(); ++i) {
        const auto dims_i = pose_runner.outputDims(names[i]);
        if (dims_i.nbDims != 3 || dims_i.d[0] != 1 || dims_i.d[1] <= 0 || dims_i.d[2] <= 0) continue;

        for (size_t j = i + 1; j < names.size(); ++j) {
            const auto dims_j = pose_runner.outputDims(names[j]);
            if (dims_j.nbDims != 3 || dims_j.d[0] != 1 || dims_j.d[1] != dims_i.d[1] || dims_j.d[2] <= 0) continue;

            const auto& a = pose_runner.output(names[i]);
            const auto& b = pose_runner.output(names[j]);
            const int K = dims_i.d[1];
            const int bins_a = dims_i.d[2];
            const int bins_b = dims_j.d[2];
            if (K <= 0 || bins_a <= 0 || bins_b <= 0) continue;
            if (a.size() != static_cast<size_t>(K * bins_a) || b.size() != static_cast<size_t>(K * bins_b)) continue;

            // x-axis bins should be closer to pose width * simcc_split_ratio.
            constexpr int simcc_split_ratio_int = 2;
            const int expect_x_bins = pose_w * simcc_split_ratio_int;
            const int dx_a = std::abs(bins_a - expect_x_bins);
            const int dx_b = std::abs(bins_b - expect_x_bins);
            const bool i_is_x = (dx_a <= dx_b);
            const auto& x_t = i_is_x ? a : b;
            const auto& y_t = i_is_x ? b : a;
            const int x_bins = i_is_x ? bins_a : bins_b;
            const int y_bins = i_is_x ? bins_b : bins_a;

            out_kpt.resize(static_cast<size_t>(K));
            out_conf.resize(static_cast<size_t>(K), 1.0f);

            constexpr float simcc_split_ratio = 2.0f;
            for (int k = 0; k < K; ++k) {
                int x_arg = 0;
                int y_arg = 0;
                float x_max = x_t[static_cast<size_t>(k) * static_cast<size_t>(x_bins)];
                float y_max = y_t[static_cast<size_t>(k) * static_cast<size_t>(y_bins)];

                for (int xb = 1; xb < x_bins; ++xb) {
                    const float v = x_t[static_cast<size_t>(k) * static_cast<size_t>(x_bins) + static_cast<size_t>(xb)];
                    if (v > x_max) { x_max = v; x_arg = xb; }
                }
                for (int yb = 1; yb < y_bins; ++yb) {
                    const float v = y_t[static_cast<size_t>(k) * static_cast<size_t>(y_bins) + static_cast<size_t>(yb)];
                    if (v > y_max) { y_max = v; y_arg = yb; }
                }

                const float px = (static_cast<float>(x_arg) + 0.5f) / simcc_split_ratio;
                const float py = (static_cast<float>(y_arg) + 0.5f) / simcc_split_ratio;
                const float ox = Minv(0, 0) * px + Minv(0, 1) * py + Minv(0, 2);
                const float oy = Minv(1, 0) * px + Minv(1, 1) * py + Minv(1, 2);
                out_kpt[static_cast<size_t>(k)].x = ox;
                out_kpt[static_cast<size_t>(k)].y = oy;
                // SimCC logits/probabilities peak-based confidence.
                const float xc = 1.0f / (1.0f + std::exp(-x_max));
                const float yc = 1.0f / (1.0f + std::exp(-y_max));
                out_conf[static_cast<size_t>(k)] = std::sqrt(std::max(0.0f, xc * yc));
            }
            return true;
        }
    }

    // 1) Prefer packed keypoint tensor [*,K,3] or [*,3,K].
    for (const auto& n : names) {
        const auto& t = pose_runner.output(n);
        const auto dims = pose_runner.outputDims(n);
        if (decodePoseTensor(t, dims, pose_w, pose_h, Minv, out_kpt, out_conf)) {
            return true;
        }
    }

    // 2) Fallback for separate outputs (x/y/score each length K).
    const std::string x_name = findTensorName(names, {"x"});
    const std::string y_name = findTensorName(names, {"y"});
    const std::string s_name = findTensorName(names, {"score", "conf"});

    if (!x_name.empty() && !y_name.empty()) {
        const auto& xs = pose_runner.output(x_name);
        const auto& ys = pose_runner.output(y_name);
        const std::vector<float>* ss = nullptr;
        if (!s_name.empty()) ss = &pose_runner.output(s_name);

        const size_t k = std::min(xs.size(), ys.size());
        if (k == 0 || k > 300) return false;

        out_kpt.resize(k);
        out_conf.resize(k, 1.0f);

        float max_x = 0.0f;
        float max_y = 0.0f;

        for (size_t i = 0; i < k; ++i) {
            out_kpt[i] = cv::Point2f(xs[i], ys[i]);
            if (ss && i < ss->size()) out_conf[i] = (*ss)[i];
            max_x = std::max(max_x, std::fabs(xs[i]));
            max_y = std::max(max_y, std::fabs(ys[i]));
        }

        const bool normalized = (max_x <= 1.5f && max_y <= 1.5f);
        for (size_t i = 0; i < k; ++i) {
            float px = out_kpt[i].x;
            float py = out_kpt[i].y;
            if (normalized) {
                px *= static_cast<float>(pose_w);
                py *= static_cast<float>(pose_h);
            }
            const float ox = Minv(0, 0) * px + Minv(0, 1) * py + Minv(0, 2);
            const float oy = Minv(1, 0) * px + Minv(1, 1) * py + Minv(1, 2);
            out_kpt[i].x = ox;
            out_kpt[i].y = oy;
        }
        return true;
    }

    return false;
}

static void drawPose(cv::Mat& bgr,
                     const std::vector<PoseResult>& results,
                     const std::vector<PersonDet>& yolo_only,
                     float rtm_kpt_th,
                     float yolo_kpt_th) {
    // HALPE26 (AlphaPose order): includes head/neck/hip-center and foot keypoints.
    static const std::vector<std::pair<int, int>> halpe26_edges = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {17, 18}, {18, 5}, {18, 6}, {18, 19},
        {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {11, 19}, {12, 19},
        {11, 13}, {13, 15}, {12, 14}, {14, 16},
        {15, 24}, {15, 20}, {15, 22},
        {16, 25}, {16, 21}, {16, 23}
    };

    for (const auto& r : results) {
        // YOLOpose detection bbox
        cv::rectangle(bgr,
                      cv::Rect(cv::Point(static_cast<int>(r.det.x1), static_cast<int>(r.det.y1)),
                               cv::Point(static_cast<int>(r.det.x2), static_cast<int>(r.det.y2))),
                      cv::Scalar(0, 255, 0),
                      2);

        // Draw only expanded ROI (1.10 right after expansion). RTM aspect-adjusted box is hidden.
        const cv::Rect2f img_bounds(0.0f, 0.0f, static_cast<float>(bgr.cols), static_cast<float>(bgr.rows));
        const cv::Rect2f pose_vis = r.expanded_box & img_bounds;
        if (pose_vis.width > 1.0f && pose_vis.height > 1.0f) {
            cv::rectangle(bgr,
                          cv::Rect(cv::Point(static_cast<int>(pose_vis.x), static_cast<int>(pose_vis.y)),
                                   cv::Point(static_cast<int>(pose_vis.x + pose_vis.width),
                                             static_cast<int>(pose_vis.y + pose_vis.height))),
                          cv::Scalar(0, 255, 255),
                          2);
        }

        std::ostringstream oss;
        oss << "d:" << std::fixed << std::setprecision(2) << r.det.conf
            << " yk:" << r.det.yolo_high_kpt_count
            << " ye:" << r.det.yolo_edge_kpt_count
            << " p:" << std::fixed << std::setprecision(2) << r.pose_mean_conf;
        if (r.depth_mm > 0.0f) {
            oss << " z:" << std::fixed << std::setprecision(0) << r.depth_mm;
        } else {
            oss << " z:NA";
        }
        cv::putText(bgr,
                    oss.str(),
                    cv::Point(static_cast<int>(r.det.x1), static_cast<int>(std::max(0.0f, r.det.y1 - 5))),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2);

        const size_t ksz = std::min(r.kpt.size(), r.kpt_conf.size());
        for (const auto& e : halpe26_edges) {
            if (static_cast<size_t>(e.first) >= ksz || static_cast<size_t>(e.second) >= ksz) continue;
            if (r.kpt_conf[e.first] >= rtm_kpt_th && r.kpt_conf[e.second] >= rtm_kpt_th) {
                cv::line(bgr, r.kpt[e.first], r.kpt[e.second], cv::Scalar(255, 255, 0), 2);
            }
        }

        for (size_t i = 0; i < ksz; ++i) {
            if (r.kpt_conf[i] < rtm_kpt_th) continue;
            cv::circle(bgr, r.kpt[i], 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    static const std::vector<std::pair<int, int>> coco17_edges = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {5, 11}, {6, 12}, {11, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };

    for (const auto& d : yolo_only) {
        cv::rectangle(bgr,
                      cv::Rect(cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                               cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2))),
                      cv::Scalar(0, 165, 255),
                      2);

        std::ostringstream oss;
        oss << "YOLO-only d:" << std::fixed << std::setprecision(2) << d.conf
            << " yk:" << d.yolo_high_kpt_count
            << " ye:" << d.yolo_edge_kpt_count;
        if (d.depth_mm > 0.0f) {
            oss << " z:" << std::fixed << std::setprecision(0) << d.depth_mm;
        } else {
            oss << " z:NA";
        }
        cv::putText(bgr,
                    oss.str(),
                    cv::Point(static_cast<int>(d.x1), static_cast<int>(std::max(0.0f, d.y1 - 5))),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(0, 165, 255),
                    2);

        for (const auto& e : coco17_edges) {
            if (d.yolo_kpt_conf[static_cast<size_t>(e.first)] >= yolo_kpt_th &&
                d.yolo_kpt_conf[static_cast<size_t>(e.second)] >= yolo_kpt_th) {
                cv::line(bgr,
                         d.yolo_kpt[static_cast<size_t>(e.first)],
                         d.yolo_kpt[static_cast<size_t>(e.second)],
                         cv::Scalar(0, 255, 255),
                         2);
            }
        }

        for (size_t k = 0; k < d.yolo_kpt.size(); ++k) {
            if (d.yolo_kpt_conf[k] < yolo_kpt_th) continue;
            cv::circle(bgr, d.yolo_kpt[k], 3, cv::Scalar(0, 255, 255), -1);
        }
    }
}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage(argv[0]);
        return 1;
    }

    g_running.store(true);
    std::signal(SIGINT, onSigInt);

    TrtRunnerMulti yolo_runner(args.yolo_engine_path, args.yolo_input_w, args.yolo_input_h);
    if (!yolo_runner.ok()) {
        std::cerr << "YOLOpose init failed: " << yolo_runner.lastError() << "\n";
        return 1;
    }

    TrtRunnerMulti pose_runner(args.pose_engine_path, args.pose_input_w, args.pose_input_h);
    if (!pose_runner.ok()) {
        std::cerr << "RTMPose init failed: " << pose_runner.lastError() << "\n";
        return 1;
    }

    std::shared_ptr<ob::Pipeline> pipe;
    try {
        pipe = std::make_shared<ob::Pipeline>();
    } catch (const ob::Error& e) {
        std::cerr << "Failed to create ob::Pipeline: " << e.getMessage() << "\n";
        return 1;
    }

    auto cfg = std::make_shared<ob::Config>();

    (void)orbbec_utils::tryConfigureHwNoiseRemoval(pipe, args.enable_hw_noise, args.hw_noise_thresh, true);

    try {
        auto sel = orbbec_utils::configureColorDepthStreams(
            pipe,
            cfg,
            args.color.w,
            args.color.h,
            args.color.fps,
            args.depth.w,
            args.depth.h,
            args.depth.fps,
            args.enable_d2c_align,
            true);

        if (!sel.color || !sel.depth) {
            std::cerr << "No valid color/depth profile found\n";
            return 1;
        }

        if (args.enable_framesync) pipe->enableFrameSync();
        pipe->start(cfg);
    } catch (const ob::Error& e) {
        std::cerr << "Orbbec error during config/start: " << e.getMessage() << "\n";
        return 1;
    }

    std::cout << "\n[OK] RTM pipeline started. rotate=" << args.rotate
              << " framesync=" << (args.enable_framesync ? "ON" : "OFF")
              << " d2c_align=" << (args.enable_d2c_align ? "HW" : "OFF")
              << "\n";

    auto to_ms = [](const Clock::time_point& a, const Clock::time_point& b) -> double {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    RunningStats cap_stats, yolo_infer_stats, rtm_infer_stats, infer_process_stats, pose_per_person_stats, loop_stats;
    std::ofstream realtime_log;
    std::vector<double> c_samples_ms;
    std::vector<double> latency_samples_ms;
    std::vector<double> aoi_samples_ms;
    uint64_t prev_stream_frame_id = 0;
    bool has_prev_stream_frame_id = false;
    int64_t frame_count = 0;
    int64_t deadline_miss_count = 0;
    int64_t total_skip_count = 0;
    int64_t debug_frame_count = 0;
    std::vector<uint64_t> deadline_miss_frame_nos;
    std::vector<uint64_t> deadline_miss_stream_frame_ids;
    const double deadline_ms = 45.0;
    int32_t sync_reference = -1;
    bool sync_reference_valid = false;
    std::string sync_reference_name = "UNKNOWN";

    if (args.debug) {
        try {
            auto dev = pipe->getDevice();
            if (dev && dev->isPropertySupported(OB_PROP_INTRA_CAMERA_SYNC_REFERENCE_INT, OB_PERMISSION_READ)) {
                sync_reference = dev->getIntProperty(OB_PROP_INTRA_CAMERA_SYNC_REFERENCE_INT);
                sync_reference_valid = true;
                sync_reference_name = syncReferenceToString(sync_reference);
            }
        } catch (...) {
            sync_reference = -1;
            sync_reference_valid = false;
            sync_reference_name = "UNKNOWN";
        }
    }

    if (args.debug) {
        realtime_log.open(args.debug_log_path, std::ios::out | std::ios::trunc);
        if (!realtime_log.is_open()) {
            std::cerr << "Failed to open debug log file: " << args.debug_log_path << "\n";
            return 1;
        }
        realtime_log << "{\"type\":\"meta\",\"fps\":" << args.color.fps
                     << ",\"deadline_ms\":" << std::fixed << std::setprecision(3) << deadline_ms
                     << ",\"sync_reference\":" << (sync_reference_valid ? std::to_string(sync_reference) : "null")
                     << ",\"sync_reference_name\":\"" << sync_reference_name << "\""
                     << ",\"log_version\":1}\n";
        std::cout << "[DEBUG] realtime metrics log: " << args.debug_log_path << "\n";
    }

    CamInternal cam{};

    while (!cam.ready) {
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

        cv::Mat color_bgr = colorFrameToBGR(c);
        if (color_bgr.empty()) continue;

        orbbec_utils::printDeviceInfo(pipe);
        const auto ci = orbbec_utils::getCameraInternalFromFrameset(fs);
        orbbec_utils::printCameraInternal(ci, true, true);

        cam.depth_value_scale = ci.depth_value_scale;
        cam.color_intr = ci.color_intr;
        cam.ready = true;
    }

    while (g_running.load()) {
        const auto t0 = args.debug ? Clock::now() : Clock::time_point{};

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
        const uint64_t stream_frame_id = c->getIndex();
        const uint64_t frame_no = static_cast<uint64_t>(frame_count + 1);

        const auto t1 = args.debug ? Clock::now() : Clock::time_point{};
        uint64_t t_start_process_us = 0;
        uint64_t t_capture_us = 0;
        uint64_t t_capture_for_latency_us = 0;
        std::string capture_ts_source;
        CaptureTimestampInfo capture_ts_info{};
        int64_t exposure_value = 0;
        bool has_exposure = false;
        if (args.debug) {
            t_start_process_us = wallClockNowUs();
            capture_ts_info = collectCaptureTimestampInfo(c, t_start_process_us);
            t_capture_us = capture_ts_info.capture_us;
            t_capture_for_latency_us = capture_ts_info.latency_base_us;
            capture_ts_source = capture_ts_info.source;
            has_exposure = tryGetFrameMetadataValue(c, OB_FRAME_METADATA_TYPE_EXPOSURE, exposure_value);
        }

        cv::Mat color_bgr = colorFrameToBGR(c);
        cv::Mat depth16 = depthFrameToMat16U(d);
        if (color_bgr.empty() || depth16.empty()) continue;

        rotateIfNeeded(color_bgr, args.rotate);
        rotateIfNeeded(depth16, args.rotate);

        const auto t2 = args.debug ? Clock::now() : Clock::time_point{};

        LetterBoxInfo yolo_lb{};
        const float yolo_ms = yolo_runner.inferLetterbox(color_bgr, yolo_lb);
        auto cand = decodeYoloPoseBoxes(yolo_runner,
                                        yolo_lb,
                                        color_bgr.cols,
                                        color_bgr.rows,
                                        args.yolo_conf_th,
                                        args.topk,
                                        args.yolo_kpt_conf_th,
                                        args.yolo_kpt_min_count,
                                        args.yolo_edge_kpt_conf_th,
                                        args.yolo_edge_kpt_min_count);
        auto keep = nms(cand, args.yolo_nms_th);

        std::vector<PersonDet> yolo_only;
        yolo_only.reserve(keep.size());
        std::vector<PersonDet> pose_candidates;
        pose_candidates.reserve(keep.size());

        for (const auto& det_in : keep) {
            PersonDet det = det_in;
            det.depth_mm = robustTorsoDepthMm(det, depth16, cam.depth_value_scale);

            if (!det.yolo_gate_pass) {
                yolo_only.push_back(std::move(det));
                continue;
            }
            pose_candidates.push_back(std::move(det));
        }

        struct PoseCandidate {
            PersonDet det;
            float rank_depth_mm = -1.0f;
            bool depth_valid = false;
        };

        std::vector<PoseCandidate> ranked;
        ranked.reserve(pose_candidates.size());
        for (const auto& det : pose_candidates) {
            PoseCandidate c;
            c.det = det;
            c.rank_depth_mm = c.det.depth_mm;
            c.depth_valid = (c.rank_depth_mm > 0.0f);
            ranked.push_back(std::move(c));
        }

        std::sort(ranked.begin(), ranked.end(), [](const PoseCandidate& a, const PoseCandidate& b) {
            if (a.depth_valid != b.depth_valid) return a.depth_valid > b.depth_valid;
            if (a.depth_valid && b.depth_valid && std::fabs(a.rank_depth_mm - b.rank_depth_mm) > 1e-3f) {
                return a.rank_depth_mm < b.rank_depth_mm;  // closer first
            }
            return a.det.conf > b.det.conf;  // fallback
        });

        std::vector<PersonDet> pose_inputs;
        pose_inputs.reserve(static_cast<size_t>(args.max_person));
        for (size_t i = 0; i < ranked.size(); ++i) {
            if (static_cast<int>(i) < args.max_person) {
                pose_inputs.push_back(ranked[i].det);
            } else {
                yolo_only.push_back(ranked[i].det);
            }
        }

        std::vector<PoseResult> results;
        results.reserve(pose_inputs.size());

        float pose_sum_ms = 0.0f;
        for (const auto& det : pose_inputs) {
            const float aspect = static_cast<float>(args.pose_input_w) / static_cast<float>(args.pose_input_h);
            const cv::Rect2f expanded_box = makeExpandedBox(det, args.person_expand);
            const cv::Rect2f pose_box = makePoseBox(expanded_box, color_bgr.cols, color_bgr.rows, aspect);
            if (pose_box.width <= 1.0f || pose_box.height <= 1.0f) continue;

            std::vector<float> pose_input;
            cv::Matx23f M{}, Minv{};
            buildMMPoseInput(color_bgr, pose_box, args.pose_input_w, args.pose_input_h, pose_input, M, Minv);
            const float pose_ms = pose_runner.inferNCHW(pose_input);
            if (pose_ms < 0.0f) {
                yolo_only.push_back(det);
                continue;
            }
            pose_sum_ms += pose_ms;
            pose_per_person_stats.add(pose_ms);

            PoseResult r;
            r.det = det;
            r.expanded_box = expanded_box;
            r.pose_box = pose_box;
            if (!decodeRTMPose(pose_runner, args.pose_input_w, args.pose_input_h, Minv, r.kpt, r.kpt_conf)) {
                yolo_only.push_back(det);
                continue;
            }
            if (!r.kpt_conf.empty()) {
                float sum = 0.0f;
                for (float s : r.kpt_conf) sum += s;
                r.pose_mean_conf = sum / static_cast<float>(r.kpt_conf.size());
            }
            r.depth_mm = det.depth_mm;
            results.push_back(std::move(r));
        }

        const float yolo_infer_ms = (yolo_ms > 0.0f ? yolo_ms : 0.0f);
        const float rtm_infer_ms = pose_sum_ms;
        const auto t3 = args.debug ? Clock::now() : Clock::time_point{};

        if (!results.empty()) {
            const auto& r0 = results.front();
            const size_t n = std::min(r0.kpt.size(), r0.kpt_conf.size());
            for (size_t k = 0; k < n; ++k) {
                if (r0.kpt_conf[k] < args.kpt_th) continue;
                float X = 0.0f, Y = 0.0f, Z = 0.0f;
                uint16_t raw = 0;
                (void)keypointToXYZ(cam,
                                    depth16,
                                    r0.kpt[k],
                                    args.rotate,
                                    args.color.w,
                                    args.color.h,
                                    X,
                                    Y,
                                    Z,
                                    raw);
                break;
            }
        }

        const uint64_t t_end_process_us = args.debug ? wallClockNowUs() : 0;

        int key = -1;
        if (args.GUI) {
            cv::Mat depth_cm = depthToColorMap(depth16, args.depth_max_mm);
            cv::Mat vis = color_bgr.clone();
            drawPose(vis, results, yolo_only, args.kpt_th, args.yolo_kpt_conf_th);
            cv::imshow("color_pose_rtm", vis);
            if (!depth_cm.empty()) cv::imshow("depth_colormap", depth_cm);
            key = cv::waitKey(1) & 0xFF;
        } else {
            key = pollStdinKeyNonBlocking();
        }

        if (key == 27 || key == 'q') g_running.store(false);

        const auto t4 = args.debug ? Clock::now() : Clock::time_point{};
        const uint64_t t_output_us = args.debug ? wallClockNowUs() : 0;
        ++frame_count;

        uint64_t frame_gap = 0;
        int64_t skip = 0;
        if (has_prev_stream_frame_id && stream_frame_id > prev_stream_frame_id) {
            frame_gap = stream_frame_id - prev_stream_frame_id;
            if (frame_gap > 1) {
                skip = static_cast<int64_t>(frame_gap - 1);
                std::cerr << "[WARN][SKIP] frame_no=" << frame_no
                          << " stream_frame_id prev=" << prev_stream_frame_id
                          << " curr=" << stream_frame_id
                          << " gap=" << frame_gap
                          << " skip=" << skip << "\n";
            }
        }
        prev_stream_frame_id = stream_frame_id;
        has_prev_stream_frame_id = true;

        if (args.debug) {
            const double capture_ms = to_ms(t0, t1);
            const double total_img_ms = to_ms(t0, t2);
            const double infer_process_ms = to_ms(t2, t3);
            const double loop_ms = to_ms(t0, t4);

            cap_stats.add(capture_ms);
            yolo_infer_stats.add(yolo_infer_ms);
            rtm_infer_stats.add(rtm_infer_ms);
            infer_process_stats.add(infer_process_ms);
            loop_stats.add(loop_ms);

            if (args.time_check) {
                std::cout << "capture=" << std::fixed << std::setprecision(1) << capture_ms << " ms,";
                std::cout << "img_process=" << std::fixed << std::setprecision(1) << total_img_ms << " ms,";
                std::cout << "yolo_infer=" << std::fixed << std::setprecision(1) << yolo_infer_ms << " ms,";
                std::cout << "rtm_infer=" << std::fixed << std::setprecision(1) << rtm_infer_ms << " ms,";
                std::cout << "infer_process=" << std::fixed << std::setprecision(1) << infer_process_ms << " ms,";
                std::cout << "loop=" << std::fixed << std::setprecision(1) << loop_ms << " ms";
                std::cout << " cand_count=" << cand.size();
                std::cout << " yolo_count=" << keep.size();
                std::cout << " pose_input_count=" << pose_inputs.size();
                std::cout << " yolo_only_count=" << yolo_only.size();
                std::cout << " pose_count=" << results.size() << "\n";
            }
        }

        if (args.debug) {
            const double c_ms = static_cast<double>(t_end_process_us - t_start_process_us) / 1000.0;
            const double latency_ms =
                (t_output_us >= t_capture_for_latency_us)
                    ? (static_cast<double>(t_output_us - t_capture_for_latency_us) / 1000.0)
                    : 0.0;
            const double aoi_ms = latency_ms;

            const bool deadline_miss = (deadline_ms > 0.0) && (latency_ms > deadline_ms);

            c_samples_ms.push_back(c_ms);
            latency_samples_ms.push_back(latency_ms);
            aoi_samples_ms.push_back(aoi_ms);
            total_skip_count += skip;
            deadline_miss_count += deadline_miss ? 1 : 0;
            if (deadline_miss) {
                deadline_miss_frame_nos.push_back(frame_no);
                deadline_miss_stream_frame_ids.push_back(stream_frame_id);
            }
            ++debug_frame_count;

            realtime_log << "{\"type\":\"frame\",\"frame_no\":" << frame_no
                         << ",\"frame_id\":" << stream_frame_id
                         << ",\"t_capture_us\":" << t_capture_us
                         << ",\"t_capture_source\":\"" << capture_ts_source << "\""
                         << ",\"system_ts_us\":" << capture_ts_info.system_us
                         << ",\"global_ts_us\":" << capture_ts_info.global_us
                         << ",\"sync_reference\":" << (sync_reference_valid ? std::to_string(sync_reference) : "null")
                         << ",\"sync_reference_name\":\"" << sync_reference_name << "\""
                         << ",\"exposure\":" << (has_exposure ? std::to_string(exposure_value) : "null")
                         << ",\"t_capture_for_latency_us\":" << t_capture_for_latency_us
                         << ",\"t_start_process_us\":" << t_start_process_us
                         << ",\"t_end_process_us\":" << t_end_process_us
                         << ",\"t_output_us\":" << t_output_us
                         << ",\"C_ms\":" << std::fixed << std::setprecision(3) << c_ms
                         << ",\"latency_ms\":" << latency_ms
                         << ",\"aoi_ms\":" << aoi_ms
                         << ",\"deadline_ms\":" << deadline_ms
                         << ",\"deadline_miss\":" << (deadline_miss ? "true" : "false")
                         << ",\"skip\":" << skip
                         << ",\"frame_gap\":" << frame_gap
                         << ",\"cand_count\":" << cand.size()
                         << ",\"yolo_count\":" << keep.size()
                         << ",\"pose_input_count\":" << pose_inputs.size()
                         << ",\"yolo_only_count\":" << yolo_only.size()
                         << ",\"pose_count\":" << results.size()
                         << "}\n";
            if ((debug_frame_count % 30) == 0) {
                realtime_log.flush();
            }
        }
    }

    if (args.debug) {
        if (cap_stats.n <= 100) {
            std::cout << "\nMax and Min values are calculated after the 100th frame.";
        }

        auto formatExtrema = [](const RunningStats& s, bool want_min) -> std::string {
            if (!s.hasExtrema()) return "N/A";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << (want_min ? s.min : s.max);
            return oss.str();
        };

        std::cout << "\n= SUMMARY =\n"
                  << "total frame : " << cap_stats.n << "| GUI : " << args.GUI << "\n"
                  << "capture_mean=" << std::fixed << std::setprecision(1) << cap_stats.mean
                  << " ms, capture_min=" << formatExtrema(cap_stats, true) << " ms\n"
                  << "yolo_infer_mean=" << std::fixed << std::setprecision(1) << yolo_infer_stats.mean
                  << " ms, yolo_infer_max=" << formatExtrema(yolo_infer_stats, false) << " ms\n"
                  << "rtm_infer_mean=" << std::fixed << std::setprecision(1) << rtm_infer_stats.mean
                  << " ms, rtm_infer_max=" << formatExtrema(rtm_infer_stats, false) << " ms\n"
                  << "infer_process_mean=" << std::fixed << std::setprecision(1) << infer_process_stats.mean
                  << " ms, infer_process_max=" << formatExtrema(infer_process_stats, false) << " ms\n"
                  << "pose_per_person_mean=" << std::fixed << std::setprecision(1) << pose_per_person_stats.mean
                  << " ms, pose_per_person_max=" << formatExtrema(pose_per_person_stats, false) << " ms\n"
                  << "loop_mean=" << std::fixed << std::setprecision(1) << loop_stats.mean
                  << " ms, loop_max=" << formatExtrema(loop_stats, false) << " ms\n";

        if (realtime_log.is_open()) {
            realtime_log.flush();
            realtime_log.close();
        }

        const auto fmt_ms = [](double v) -> std::string {
            if (!std::isfinite(v)) return "N/A";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << v;
            return oss.str();
        };

        const double miss_ratio = (debug_frame_count > 0)
                                      ? (100.0 * static_cast<double>(deadline_miss_count) / static_cast<double>(debug_frame_count))
                                      : 0.0;

        std::cout << "\n= REALTIME DEBUG SUMMARY =\n"
                  << "debug_log_path=" << args.debug_log_path << "\n"
                  << "frames_logged=" << debug_frame_count << "\n"
                  << "deadline_ms=" << fmt_ms(deadline_ms)
                  << ", deadline_miss_count=" << deadline_miss_count
                  << ", deadline_miss_ratio=" << fmt_ms(miss_ratio) << " %\n"
                  << "skip_total=" << total_skip_count << "\n"
                  << "C_ms_p50=" << fmt_ms(quantileLinear(c_samples_ms, 0.50))
                  << ", C_ms_p90=" << fmt_ms(quantileLinear(c_samples_ms, 0.90))
                  << ", C_ms_p99=" << fmt_ms(quantileLinear(c_samples_ms, 0.99)) << "\n"
                  << "latency_ms_p50=" << fmt_ms(quantileLinear(latency_samples_ms, 0.50))
                  << ", latency_ms_p90=" << fmt_ms(quantileLinear(latency_samples_ms, 0.90))
                  << ", latency_ms_p99=" << fmt_ms(quantileLinear(latency_samples_ms, 0.99)) << "\n"
                  << "AoI_ms_p50=" << fmt_ms(quantileLinear(aoi_samples_ms, 0.50))
                  << ", AoI_ms_p90=" << fmt_ms(quantileLinear(aoi_samples_ms, 0.90))
                  << ", AoI_ms_p99=" << fmt_ms(quantileLinear(aoi_samples_ms, 0.99)) << "\n";

        auto printCompactList = [](const char* label, const std::vector<uint64_t>& ids) {
            std::cout << label << "=";
            if (ids.empty()) {
                std::cout << "[]\n";
                return;
            }
            const size_t total_ids = ids.size();
            std::cout << "[";
            if (total_ids <= 40) {
                for (size_t i = 0; i < total_ids; ++i) {
                    if (i > 0) std::cout << ",";
                    std::cout << ids[i];
                }
            } else {
                for (size_t i = 0; i < 20; ++i) {
                    if (i > 0) std::cout << ",";
                    std::cout << ids[i];
                }
                std::cout << ",...,";
                for (size_t i = total_ids - 20; i < total_ids; ++i) {
                    if (i > total_ids - 20) std::cout << ",";
                    std::cout << ids[i];
                }
            }
            std::cout << "] (total=" << total_ids << ")\n";
        };
        printCompactList("deadline_miss_frame_no", deadline_miss_frame_nos);
        printCompactList("deadline_miss_stream_frame_id", deadline_miss_stream_frame_ids);
    } else {
        std::cout << "\n= SUMMARY =\n"
                  << "total frame : " << frame_count << "| GUI : " << args.GUI << "\n";
    }

    try {
        pipe->stop();
    } catch (...) {
    }

    return 0;
}
