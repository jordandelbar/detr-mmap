#pragma once

#include "tensorrt_backend.hpp"
#include "detection_writer.hpp"
#include <vector>

namespace bridge {

struct TransformParams {
    uint32_t orig_width;
    uint32_t orig_height;
    float scale;
    float offset_x;
    float offset_y;
};

class PostProcessor {
public:
    explicit PostProcessor(float confidence_threshold = 0.5f);

    /// Parse and transform detections from model outputs
    /// Returns filtered and transformed bounding boxes
    std::vector<BoundingBox> parse_detections(
        const InferenceOutput& output,
        const TransformParams& transform
    );

private:
    float confidence_threshold_;
};

} // namespace bridge
