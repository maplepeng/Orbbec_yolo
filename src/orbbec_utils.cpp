#include "orbbec_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <utility>

namespace orbbec_utils {

std::string obFormatToStr(OBFormat f) {
    switch (f) {
        case OB_FORMAT_MJPG: return "MJPG";
        case OB_FORMAT_RGB:  return "RGB";
        case OB_FORMAT_BGR:  return "BGR";
        case OB_FORMAT_YUYV: return "YUYV";
        case OB_FORMAT_Y16:  return "Y16";
        case OB_FORMAT_NV12: return "NV12";
        case OB_FORMAT_NV21: return "NV21";
        default: break;
    }
    return "FMT(" + std::to_string(static_cast<int>(f)) + ")";
}

void printVideoProfiles(const std::string& title, const std::shared_ptr<ob::StreamProfileList>& list) {
    if (!list) {
        std::cout << "\n=== " << title << " profiles (null) ===\n";
        return;
    }
    std::cout << "\n=== " << title << " profiles (" << list->getCount() << ") ===\n";
    for (uint32_t i = 0; i < list->getCount(); ++i) {
        auto p = list->getProfile(i)->as<ob::VideoStreamProfile>();
        if (!p) continue;
        std::cout << " [" << i << "] "
                  << p->getWidth() << "x" << p->getHeight()
                  << " @" << p->getFps()
                  << " format=" << obFormatToStr(p->getFormat())
                  << "\n";
    }
}

std::shared_ptr<ob::StreamProfile> pickClosestVideoProfile(
    const std::shared_ptr<ob::StreamProfileList>& list,
    int target_w, int target_h, int target_fps,
    const std::vector<OBFormat>& preferred_formats,
    const std::string& tag_for_logs
) {
    if (!list || list->getCount() == 0) return nullptr;

    struct Cand {
        uint32_t idx;
        int w, h, fps;
        OBFormat fmt;
        double score;
    };

    std::vector<Cand> cands;
    cands.reserve(list->getCount());

    for (uint32_t i = 0; i < list->getCount(); ++i) {
        auto vp = list->getProfile(i)->as<ob::VideoStreamProfile>();
        if (!vp) continue;

        int w = static_cast<int>(vp->getWidth());
        int h = static_cast<int>(vp->getHeight());
        int fps = static_cast<int>(vp->getFps());
        OBFormat fmt = vp->getFormat();

        double s = 0.0;
        s += std::abs(w - target_w) * 1000.0;
        s += std::abs(h - target_h) * 1000.0;
        s += std::abs(fps - target_fps) * 50.0;

        if (!preferred_formats.empty()) {
            auto it = std::find(preferred_formats.begin(), preferred_formats.end(), fmt);
            if (it == preferred_formats.end()) {
                s += 1e6;  // discourage non-preferred formats
            } else {
                s += (it - preferred_formats.begin()) * 100.0; // earlier is better
            }
        }

        cands.push_back({i, w, h, fps, fmt, s});
    }

    if (cands.empty()) return nullptr;
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.score < b.score; });

    const auto& best = cands.front();
    auto best_p = list->getProfile(best.idx);
    std::cout << "[ProfilePick][" << tag_for_logs << "] picked idx=" << best.idx
              << " " << best.w << "x" << best.h << "@" << best.fps
              << " fmt=" << obFormatToStr(best.fmt)
              << " (score=" << best.score << ")\n";
    return best_p;
}

static SelectedStreams configureColorDepthStreamsImpl(
    const std::shared_ptr<ob::Pipeline>& pipe,
    const std::shared_ptr<ob::Config>& cfg,
    int color_w, int color_h, int color_fps,
    int depth_w, int depth_h, int depth_fps,
    bool request_hw_d2c_align,
    bool print_lists,
    const std::vector<OBFormat>& color_pref,
    const std::vector<OBFormat>& depth_pref
) {
    SelectedStreams out{};

    if (!pipe || !cfg) return out;

    auto color_list = pipe->getStreamProfileList(OB_SENSOR_COLOR);
    auto depth_list = pipe->getStreamProfileList(OB_SENSOR_DEPTH);

    if (print_lists) {
        printVideoProfiles("COLOR", color_list);
        printVideoProfiles("DEPTH", depth_list);
    }

    auto pickColorSingleFmt = [&](OBFormat fmt, const char* tag) -> std::shared_ptr<ob::StreamProfile> {
        std::vector<OBFormat> one = {fmt};
        return pickClosestVideoProfile(color_list, color_w, color_h, color_fps, one, tag);
    };

    std::shared_ptr<ob::StreamProfile> color_profile;
    std::shared_ptr<ob::StreamProfile> depth_profile;

    if (request_hw_d2c_align) {
        bool hw_d2c_ok = false;

        for (OBFormat fmt : color_pref) {
            auto cand_color = pickColorSingleFmt(fmt, "COLOR(HW_CAND)");
            if (!cand_color) continue;

            auto cand_color_vp = cand_color->as<ob::VideoStreamProfile>();
            if (!cand_color_vp) continue;

            auto cand_d2c = pipe->getD2CDepthProfileList(cand_color_vp, ALIGN_D2C_HW_MODE);
            if (!cand_d2c || cand_d2c->getCount() == 0) {
                continue; // try next color format
            }

            if (print_lists) {
                printVideoProfiles("D2C_DEPTH(HW)", cand_d2c);
            }

            auto cand_depth = pickClosestVideoProfile(
                cand_d2c, depth_w, depth_h, depth_fps,
                depth_pref, "DEPTH(D2C_HW)"
            );

            if (!cand_depth) {
                try {
                    cand_depth = cand_d2c->getProfile(0);
                } catch (...) {
                    cand_depth.reset();
                }
            }

            if (cand_depth) {
                color_profile = cand_color;
                depth_profile = cand_depth;
                hw_d2c_ok = true;
                break;
            }
        }

        if (hw_d2c_ok) {
            cfg->setAlignMode(ALIGN_D2C_HW_MODE);
            out.hw_d2c_align = true;
        } else {
            std::cerr << "[WARN] No HW D2C-compatible (COLOR, DEPTH) pair found for requested spec. "
                         "Falling back to non-aligned streams.\n";
            cfg->setAlignMode(ALIGN_DISABLE);

            color_profile = pickClosestVideoProfile(color_list, color_w, color_h, color_fps, color_pref, "COLOR");
            depth_profile = pickClosestVideoProfile(depth_list, depth_w, depth_h, depth_fps, depth_pref, "DEPTH");
        }
    } else {
        cfg->setAlignMode(ALIGN_DISABLE);

        color_profile = pickClosestVideoProfile(color_list, color_w, color_h, color_fps, color_pref, "COLOR");
        depth_profile = pickClosestVideoProfile(depth_list, depth_w, depth_h, depth_fps, depth_pref, "DEPTH");
    }

    if (color_profile) cfg->enableStream(color_profile);
    if (depth_profile) cfg->enableStream(depth_profile);

    out.color = color_profile;
    out.depth = depth_profile;
    return out;
}

SelectedStreams configureColorDepthStreams(
    const std::shared_ptr<ob::Pipeline>& pipe,
    const std::shared_ptr<ob::Config>& cfg,
    int color_w, int color_h, int color_fps,
    int depth_w, int depth_h, int depth_fps,
    bool request_hw_d2c_align,
    bool print_lists
) {
    const std::vector<OBFormat> color_pref = {OB_FORMAT_MJPG, OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_YUYV};
    const std::vector<OBFormat> depth_pref = {OB_FORMAT_Y16};

    return configureColorDepthStreamsImpl(pipe, cfg,
        color_w, color_h, color_fps,
        depth_w, depth_h, depth_fps,
        request_hw_d2c_align,
        print_lists,
        color_pref,
        depth_pref
    );
}

SelectedStreams configureColorDepthStreams(
    const std::shared_ptr<ob::Pipeline>& pipe,
    const std::shared_ptr<ob::Config>& cfg,
    int color_w, int color_h, int color_fps,
    int depth_w, int depth_h, int depth_fps,
    bool request_hw_d2c_align,
    bool print_lists,
    const std::vector<OBFormat>& color_pref,
    const std::vector<OBFormat>& depth_pref
) {
    return configureColorDepthStreamsImpl(pipe, cfg,
        color_w, color_h, color_fps,
        depth_w, depth_h, depth_fps,
        request_hw_d2c_align,
        print_lists,
        color_pref,
        depth_pref
    );
}

HwNoiseRemovalResult tryConfigureHwNoiseRemoval(
    const std::shared_ptr<ob::Pipeline>& pipe,
    bool enable,
    float threshold,
    bool verbose
) {
    HwNoiseRemovalResult r;
    if (!pipe) {
        if (verbose) std::cerr << "[HWNoise] pipeline is null (skip)\n";
        return r;
    }

    try {
        auto dev = pipe->getDevice();
        if (!dev) {
            if (verbose) std::cerr << "[HWNoise] device is null (skip)\n";
            return r;
        }

        // Property IDs are defined by Orbbec SDK v2. If unsupported on the current device/firmware,
        // isPropertySupported() will return false and we simply skip.
        const auto kEnableProp = OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL;
        const auto kThreshProp = OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT;

        if (!dev->isPropertySupported(kEnableProp, OB_PERMISSION_WRITE)) {
            if (verbose) std::cerr << "[HWNoise] not supported on this device/firmware\n";
            return r;
        }
        r.supported = true;

        dev->setBoolProperty(kEnableProp, enable);
        r.enabled = enable;

        if (enable && dev->isPropertySupported(kThreshProp, OB_PERMISSION_WRITE)) {
            dev->setFloatProperty(kThreshProp, threshold);
            r.threshold = threshold;
        }

        if (verbose) {
            if (r.enabled) {
                std::cerr << "[HWNoise] enabled";
                if (r.threshold > 0.0f) std::cerr << " (threshold=" << r.threshold << ")";
                std::cerr << "\n";
            } else {
                std::cerr << "[HWNoise] disabled\n";
            }
        }
    } catch (const ob::Error& e) {
        if (verbose) {
           std::cerr << "[HWNoise][WARN] Orbbec error: " << e.getMessage()
                      << " (type=" << e.getExceptionType() << ")\n";
        }
    } catch (const std::exception& e) {
        if (verbose) std::cerr << "[HWNoise][WARN] exception: " << e.what() << "\n";
    }

    return r;
}

// ---------------------- Camera internal parameters helpers ----------------------
namespace {

template <typename T, typename = void>
struct has_name_method : std::false_type {};
template <typename T>
struct has_name_method<T, std::void_t<decltype(std::declval<T*>()->name())>> : std::true_type {};

template <typename T, typename = void>
struct has_getName_method : std::false_type {};
template <typename T>
struct has_getName_method<T, std::void_t<decltype(std::declval<T*>()->getName())>> : std::true_type {};

template <typename T, typename = void>
struct has_serialNumber_method : std::false_type {};
template <typename T>
struct has_serialNumber_method<T, std::void_t<decltype(std::declval<T*>()->serialNumber())>> : std::true_type {};

template <typename T, typename = void>
struct has_getSerialNumber_method : std::false_type {};
template <typename T>
struct has_getSerialNumber_method<T, std::void_t<decltype(std::declval<T*>()->getSerialNumber())>> : std::true_type {};

template <typename T, typename = void>
struct has_firmwareVersion_method : std::false_type {};
template <typename T>
struct has_firmwareVersion_method<T, std::void_t<decltype(std::declval<T*>()->firmwareVersion())>> : std::true_type {};

template <typename T, typename = void>
struct has_getFirmwareVersion_method : std::false_type {};
template <typename T>
struct has_getFirmwareVersion_method<T, std::void_t<decltype(std::declval<T*>()->getFirmwareVersion())>> : std::true_type {};

template <typename T, typename = void>
struct has_connectionType_method : std::false_type {};
template <typename T>
struct has_connectionType_method<T, std::void_t<decltype(std::declval<T*>()->connectionType())>> : std::true_type {};

template <typename T, typename = void>
struct has_getConnectionType_method : std::false_type {};
template <typename T>
struct has_getConnectionType_method<T, std::void_t<decltype(std::declval<T*>()->getConnectionType())>> : std::true_type {};

template <typename T, typename = void>
struct has_member_fx : std::false_type {};
template <typename T>
struct has_member_fx<T, std::void_t<decltype(std::declval<T>().fx)>> : std::true_type {};

template <typename T, typename = void>
struct has_member_k1 : std::false_type {};
template <typename T>
struct has_member_k1<T, std::void_t<decltype(std::declval<T>().k1)>> : std::true_type {};

template <typename T, typename = void>
struct has_member_rot : std::false_type {};
template <typename T>
struct has_member_rot<T, std::void_t<decltype(std::declval<T>().rot)>> : std::true_type {};

static void printIntrinsic(const OBCameraIntrinsic& K, const char* tag) {
    std::cout << "  - " << tag << " intrinsic: ";
    if constexpr (has_member_fx<OBCameraIntrinsic>::value) {
        std::cout << "w=" << K.width << " h=" << K.height
                  << " fx=" << K.fx << " fy=" << K.fy
                  << " cx=" << K.cx << " cy=" << K.cy << "\n";
    } else {
        std::cout << "(layout unknown in this SDK)\n";
    }
}

static void printDistortion(const OBCameraDistortion& D, const char* tag) {
    std::cout << "  - " << tag << " distortion: ";
    if constexpr (has_member_k1<OBCameraDistortion>::value) {
        std::cout << "k1=" << D.k1 << " k2=" << D.k2
                  << " p1=" << D.p1 << " p2=" << D.p2
                  << " k3=" << D.k3 << " k4=" << D.k4
                  << " k5=" << D.k5 << " k6=" << D.k6 << "\n";
    } else {
        std::cout << "(layout unknown in this SDK)\n";
    }
}

static void printExtrinsic(const OBExtrinsic& E, const char* tag) {
    std::cout << "  - " << tag << " extrinsic (R|t): ";
    if constexpr (has_member_rot<OBExtrinsic>::value) {
        // rot is 3x3 row-major, trans is 3x1 (meters)
        std::cout << "t=[" << E.trans[0] << ", " << E.trans[1] << ", " << E.trans[2] << "] "
                  << "R=[" << E.rot[0] << " " << E.rot[1] << " " << E.rot[2] << "; "
                          << E.rot[3] << " " << E.rot[4] << " " << E.rot[5] << "; "
                          << E.rot[6] << " " << E.rot[7] << " " << E.rot[8] << "]\n";
    } else {
        std::cout << "(layout unknown in this SDK)\n";
    }
}

} // anonymous namespace

float getDepthValueScaleSafe(const std::shared_ptr<ob::DepthFrame>& depth) {
    if (!depth) return 0.0f;
    try {
        return depth->getValueScale();
    } catch (...) {
        return 0.0f;
    }
}

CameraInternal getCameraInternalFromFrameset(const std::shared_ptr<ob::FrameSet>& fs) {
    CameraInternal out{};
    if (!fs) return out;

    try {
        auto c = fs->colorFrame();
        auto d = fs->depthFrame();
        if (!c || !d) return out;

        out.depth_value_scale = getDepthValueScaleSafe(d);

        auto c_vp = c->getStreamProfile()->as<ob::VideoStreamProfile>();
        auto d_vp = d->getStreamProfile()->as<ob::VideoStreamProfile>();
        if (c_vp) {
            out.color_intr  = c_vp->getIntrinsic();
            out.color_dist  = c_vp->getDistortion();
            out.has_color   = true;
        }
        if (d_vp) {
            out.depth_intr  = d_vp->getIntrinsic();
            out.depth_dist  = d_vp->getDistortion();
            out.has_depth   = true;
        }
        if (c_vp && d_vp) {
            out.depth_to_color = d_vp->getExtrinsicTo(c_vp);
            out.has_extrinsic_d2c = true;
        }
    } catch (...) {
        // best-effort: leave flags as-is
    }

    return out;
}

void printDeviceInfo(const std::shared_ptr<ob::Pipeline>& pipe) {
    if (!pipe) {
        std::cout << "[CamInfo] pipeline=null\n";
        return;
    }
    try {
        auto dev = pipe->getDevice();
        if (!dev) {
            std::cout << "[CamInfo] device=null\n";
            return;
        }
        auto info = dev->getDeviceInfo();
        if (!info) {
            std::cout << "[CamInfo] deviceInfo=null\n";
            return;
        }

        using InfoT = std::remove_reference_t<decltype(*info)>;

        std::ostringstream oss;
        oss << "[CamInfo] Device";
        if constexpr (has_name_method<InfoT>::value) {
            oss << " name=" << info->name();
        } else if constexpr (has_getName_method<InfoT>::value) {
            oss << " name=" << info->getName();
        }

        if constexpr (has_serialNumber_method<InfoT>::value) {
            oss << " sn=" << info->serialNumber();
        } else if constexpr (has_getSerialNumber_method<InfoT>::value) {
            oss << " sn=" << info->getSerialNumber();
        }

        if constexpr (has_firmwareVersion_method<InfoT>::value) {
            oss << " fw=" << info->firmwareVersion();
        } else if constexpr (has_getFirmwareVersion_method<InfoT>::value) {
            oss << " fw=" << info->getFirmwareVersion();
        }

        if constexpr (has_connectionType_method<InfoT>::value) {
            oss << " conn=" << info->connectionType();
        } else if constexpr (has_getConnectionType_method<InfoT>::value) {
            oss << " conn=" << info->getConnectionType();
        }

        std::cout << oss.str() << "\n";
    } catch (const ob::Error& e) {
        std::cout << "[CamInfo][WARN] Orbbec error: " << e.getMessage() << "\n";
    } catch (const std::exception& e) {
        std::cout << "[CamInfo][WARN] exception: " << e.what() << "\n";
    }
}

void printCameraInternal(const CameraInternal& ci,
                         bool print_distortion,
                         bool print_extrinsic) {
    std::cout << "[CamInfo] Calibration (from current streaming profiles)\n";
    if (ci.depth_value_scale > 0.0f) {
        std::cout << "  - depth value scale: " << std::fixed << std::setprecision(6)
                  << ci.depth_value_scale << " (meters per raw unit)\n";
    } else {
        std::cout << "  - depth value scale: (unavailable)\n";
    }

    if (ci.has_color) {
        printIntrinsic(ci.color_intr, "color");
        if (print_distortion) printDistortion(ci.color_dist, "color");
    } else {
        std::cout << "  - color intrinsic: (unavailable)\n";
    }

    if (ci.has_depth) {
        printIntrinsic(ci.depth_intr, "depth");
        if (print_distortion) printDistortion(ci.depth_dist, "depth");
    } else {
        std::cout << "  - depth intrinsic: (unavailable)\n";
    }

    if (print_extrinsic && ci.has_extrinsic_d2c) {
        printExtrinsic(ci.depth_to_color, "depth->color");
    } else if (print_extrinsic) {
        std::cout << "  - depth->color extrinsic: (unavailable)\n";
    }
}
} // namespace orbbec_utils