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

// Result of trying to enable Orbbec on-device (hardware) depth noise removal.
// Supported on Gemini 330 series (firmware >= 1.4.60) per Orbbec docs.
struct HwNoiseRemovalResult {
    bool supported = false;
    bool enabled = false;
    float threshold = 0.0f;
};

// Try to enable/disable the hardware noise removal filter if the connected device supports it.
// Safe to call on any device: it checks property support first and logs a one-line summary.
HwNoiseRemovalResult tryConfigureHwNoiseRemoval(
    const std::shared_ptr<ob::Pipeline>& pipe,
    bool enable,
    float threshold,
    bool verbose = true
);

// ---------------------- Camera internal parameters helpers ----------------------
struct CameraInternal {
    bool has_color = false;
    bool has_depth = false;
    bool has_extrinsic_d2c = false;

    OBCameraIntrinsic  color_intr{};
    OBCameraDistortion color_dist{};
    OBCameraIntrinsic  depth_intr{};
    OBCameraDistortion depth_dist{};
    OBExtrinsic        depth_to_color{};

    // Depth value scale: z_meters = raw * depth_value_scale
    float depth_value_scale = 0.0f;
};

// Extract intrinsics/distortion/extrinsic and depth value scale from an actual frameset.
// This reflects the profiles currently streaming (resolution/FPS/format and D2C choice).
CameraInternal getCameraInternalFromFrameset(const std::shared_ptr<ob::FrameSet>& fs);

// Convenience: read depth value scale (returns 0.0 on failure).
float getDepthValueScaleSafe(const std::shared_ptr<ob::DepthFrame>& depth);

// Print device info (model/SN/FW if available) in a best-effort way.
// Implemented with compile-time detection to tolerate SDK API differences.
void printDeviceInfo(const std::shared_ptr<ob::Pipeline>& pipe);

// Print camera internal parameters.
void printCameraInternal(const CameraInternal& ci,
                         bool print_distortion = true,
                         bool print_extrinsic = true);

} // namespace orbbec_utils