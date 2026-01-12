#include "preprocessing.hpp"
#include <iostream>

namespace bridge {

PreProcessor::PreProcessor(uint32_t input_size) : input_size_(input_size) {}

PreprocessResult PreProcessor::preprocess(
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height,
    bool is_bgr
) {
    // Create OpenCV matrix from raw pixel data
    cv::Mat src(height, width, CV_8UC3, const_cast<uint8_t*>(pixels));

    // Convert BGR to RGB if needed
    cv::Mat rgb;
    if (is_bgr) {
        cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = src;
    }

    // Calculate letterbox parameters
    float scale = std::min(
        static_cast<float>(input_size_) / width,
        static_cast<float>(input_size_) / height
    );
    uint32_t new_width = static_cast<uint32_t>(width * scale);
    uint32_t new_height = static_cast<uint32_t>(height * scale);
    uint32_t offset_x = (input_size_ - new_width) / 2;
    uint32_t offset_y = (input_size_ - new_height) / 2;

    // Resize image
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // Create letterboxed image with gray padding
    cv::Mat letterboxed(input_size_, input_size_, CV_8UC3,
                        cv::Scalar(LETTERBOX_COLOR, LETTERBOX_COLOR, LETTERBOX_COLOR));

    // Copy resized image to center of letterboxed image
    cv::Rect roi(offset_x, offset_y, new_width, new_height);
    resized.copyTo(letterboxed(roi));

    // Convert to float and normalize to [0, 1], then convert to CHW format
    // Output shape: [1, 3, input_size, input_size]
    size_t total_size = 3 * input_size_ * input_size_;
    std::vector<float> data(total_size);

    // Normalize and convert HWC -> CHW
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t y = 0; y < input_size_; ++y) {
            for (uint32_t x = 0; x < input_size_; ++x) {
                uint8_t pixel_val = letterboxed.at<cv::Vec3b>(y, x)[c];
                data[c * input_size_ * input_size_ + y * input_size_ + x] =
                    pixel_val / 255.0f;
            }
        }
    }

    return PreprocessResult{
        std::move(data),
        scale,
        static_cast<float>(offset_x),
        static_cast<float>(offset_y),
        input_size_,
        input_size_
    };
}

} // namespace bridge
