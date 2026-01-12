#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

namespace bridge {

struct PreprocessResult {
    std::vector<float> data;  // Flattened CHW format [1, 3, H, W]
    float scale;
    float offset_x;
    float offset_y;
    uint32_t input_width;
    uint32_t input_height;
};

class PreProcessor {
public:
    explicit PreProcessor(uint32_t input_size = 640);

    /// Preprocess frame: BGR/RGB conversion, letterbox resize, normalize
    /// Returns flattened CHW tensor and transformation parameters
    PreprocessResult preprocess(
        const uint8_t* pixels,
        uint32_t width,
        uint32_t height,
        bool is_bgr
    );

private:
    uint32_t input_size_;
    static constexpr uint8_t LETTERBOX_COLOR = 114;
};

} // namespace bridge
