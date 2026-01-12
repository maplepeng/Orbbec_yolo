#pragma once

#include <libobsensor/ObSensor.hpp>

#include <memory>
#include <string>
#include <vector>

namespace orbbec_utils {

struct SelectedStreams {
    std::shared_ptr<ob::StreamProfile> color;
    std::shared_ptr<ob::StreamProfile> depth;
    bool hw_d2c_align = false;
};

std::string obFormatToStr(OBFormat f);

void printVideoProfiles(const std::string& title,
                        const std::shared_ptr<ob::StreamProfileList>& list);

std::shared_ptr<ob::StreamProfile> pickClosestVideoProfile(
    const std::shared_ptr<ob::StreamProfileList>& list,
    int target_w, int target_h, int target_fps,
    const std::vector<OBFormat>& preferred_formats,
    const std::string& tag_for_logs
);

// Select (color, depth) profiles and configure cfg:
// - If request_hw_d2c_align is true, it tries to find a HW D2C-compatible (color, depth) pair.
// - On success: cfg->setAlignMode(ALIGN_D2C_HW_MODE) and enable both streams.
// - On failure: cfg->setAlignMode(ALIGN_DISABLE) and enable best-effort non-aligned streams.
SelectedStreams configureColorDepthStreams(
    const std::shared_ptr<ob::Pipeline>& pipe,
    const std::shared_ptr<ob::Config>& cfg,
    int color_w, int color_h, int color_fps,
    int depth_w, int depth_h, int depth_fps,
    bool request_hw_d2c_align,
    bool print_lists = true
);

// Same as above, but caller can override format preference order.
SelectedStreams configureColorDepthStreams(
    const std::shared_ptr<ob::Pipeline>& pipe,
    const std::shared_ptr<ob::Config>& cfg,
    int color_w, int color_h, int color_fps,
    int depth_w, int depth_h, int depth_fps,
    bool request_hw_d2c_align,
    bool print_lists,
    const std::vector<OBFormat>& color_pref,
    const std::vector<OBFormat>& depth_pref
);

} // namespace orbbec_utils