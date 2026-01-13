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
//   ./build/orbbec_yolo_pose ./models/yolo11n-pose_fp16.engine --rotate=270
//   ./build/orbbec_yolo_pose ./models/yolo11n-pose_fp16.engine --rotate=270 --no_gui
//   ./build/orbbec_yolo_pose ./models/yolo11n-pose_fp16.engine --color=640x480@30 --depth=640x480@30 --rotate=270
//   ./build/orbbec_yolo_pose ./models/yolo11n-pose_fp16.engine --color=640x480@30 --depth=640x480@30 --rotate=270 --no_gui

#include "orbbec_utils.hpp"
#include "trt_runner.hpp"

#include <libobsensor/ObSensor.hpp>

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
    bool time_check = false;

    bool enable_framesync = true;
    bool enable_d2c_align = true;  // use HW D2C align
    bool enable_hw_noise = true;   // enable on-device depth denoise if supported
    float hw_noise_thresh = 0.2f;  // hardware noise removal threshold

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
        << "  --no_gui                  Disable GUI (default: enabled)\n"
        << "  --time                    Enable logging process time (default: disable)\n"
        << "  --no_sync                 Disable FrameSync (default: enabled)\n"
        << "  --no_align                Disable D2C align (default: enabled HW align)\n"
        << "  --no_hw_noise             Disable HW depth noise removal (default: enabled if supported)\n"
        << "  --hw_noise_thresh=0.2     HW noise removal threshold (default 0.2)\n"
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
        } else if (s == "--time") {
            a.time_check = true;
        } else if (s == "--no_sync") {
            a.enable_framesync = false;
        } else if (s == "--no_align") {
            a.enable_d2c_align = false;
        } else if (s == "--no_hw_noise") {
            a.enable_hw_noise = false;
        } else if (startsWith(s, "--hw_noise_thresh=")) {
            a.hw_noise_thresh = std::stof(s.substr(17));
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
    const LetterBoxInfo& lb,
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
    TrtRunner trt(args.engine_path);
    if (!trt.ok()) {
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

    // Optional: enable Orbbec on-device (hardware) depth noise removal when supported.
    // Per Orbbec docs, this is available on Gemini 330 series with firmware >= 1.4.60.
    (void)orbbec_utils::tryConfigureHwNoiseRemoval(pipe, args.enable_hw_noise, args.hw_noise_thresh, true);

    // 3) List all profiles & choose closest
    try {
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
        LetterBoxInfo lb{};
        std::vector<float> out0;
        float infer_ms = trt.infer_pose(color_bgr, out0, lb);

        const auto t3 = Clock::now(); // infer end

        // 6) Decode YOLO11 pose output (typically [1,56,8400])
        // N is read from the engine output dims when possible.
        int N = trt.outputN();
        if (N <= 0) {
            constexpr int C = 56;
            N = (out0.size() % C == 0) ? static_cast<int>(out0.size() / C) : 8400;
        }
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

        if (args.time_check == true) {
            std::cout << "capture=" << std::fixed << std::setprecision(1) << capture_ms << " ms,";
            std::cout << "img_process=" << std::fixed << std::setprecision(1) << total_img_ms << " ms,";
            std::cout << "infer=" << std::fixed << std::setprecision(1) << infer_ms << " ms,";
            std::cout << "infer_process=" << std::fixed << std::setprecision(1) << infer_process_ms << " ms,";
            std::cout << "loop=" << std::fixed << std::setprecision(1) << loop_ms << " ms\n";
        }
    }

    std::cout << "\n"
        << "= SUMMARY =" << "\n"
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
