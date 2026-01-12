#include "postprocessing.hpp"
#include <algorithm>
#include <iostream>

namespace bridge {

PostProcessor::PostProcessor(float confidence_threshold)
    : confidence_threshold_(confidence_threshold) {}

std::vector<BoundingBox> PostProcessor::parse_detections(
    const InferenceOutput& output,
    const TransformParams& transform
) {
    std::vector<BoundingBox> detections;
    detections.reserve(output.num_detections);

    for (size_t i = 0; i < output.num_detections; ++i) {
        float confidence = output.scores[i];

        // Filter by confidence threshold
        if (confidence < confidence_threshold_) {
            continue;
        }

        // Get box coordinates (in letterbox space)
        float box_x1 = output.boxes[i * 4 + 0];
        float box_y1 = output.boxes[i * 4 + 1];
        float box_x2 = output.boxes[i * 4 + 2];
        float box_y2 = output.boxes[i * 4 + 3];

        // Transform coordinates back to original image space
        // Formula: original_coord = (letterbox_coord - offset) / scale
        float x1 = (box_x1 - transform.offset_x) / transform.scale;
        float y1 = (box_y1 - transform.offset_y) / transform.scale;
        float x2 = (box_x2 - transform.offset_x) / transform.scale;
        float y2 = (box_y2 - transform.offset_y) / transform.scale;

        // Clamp to image bounds
        x1 = std::clamp(x1, 0.0f, static_cast<float>(transform.orig_width));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(transform.orig_height));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(transform.orig_width));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(transform.orig_height));

        detections.push_back(BoundingBox{
            x1, y1, x2, y2,
            confidence,
            static_cast<uint32_t>(output.labels[i])
        });
    }

    return detections;
}

} // namespace bridge
