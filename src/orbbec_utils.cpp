#include "orbbec_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

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

} // namespace orbbec_utils