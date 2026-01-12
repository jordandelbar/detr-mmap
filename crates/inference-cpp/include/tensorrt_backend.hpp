#pragma once

#include <NvInfer.h>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace bridge {

struct InferenceOutput {
    std::vector<int64_t> labels;    // [num_detections]
    std::vector<float> boxes;       // [num_detections, 4]
    std::vector<float> scores;      // [num_detections]
    size_t num_detections;
};

class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class TensorRTBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend();

    // Disable copy, enable move
    TensorRTBackend(const TensorRTBackend&) = delete;
    TensorRTBackend& operator=(const TensorRTBackend&) = delete;
    TensorRTBackend(TensorRTBackend&&) noexcept;
    TensorRTBackend& operator=(TensorRTBackend&&) noexcept;

    /// Load TensorRT engine from file
    bool load_engine(const char* engine_path);

    /// Run inference
    /// inputs: images [1, 3, 640, 640], orig_sizes [1, 2]
    bool infer(const float* images, const int64_t* orig_sizes, InferenceOutput& output);

    /// Run inference with raw output pointers (FFI friendly)
    /// labels: [num_detections]
    /// boxes: [num_detections * 4]
    /// scores: [num_detections]
    bool infer_raw(
        const float* images,
        const int64_t* orig_sizes,
        int64_t* out_labels,
        float* out_boxes,
        float* out_scores
    );

    int get_num_detections() const { return num_detections_; }

private:
    bool allocate_buffers();
    void free_buffers();

    TensorRTLogger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // Device buffers
    void* d_images_ = nullptr;
    void* d_orig_sizes_ = nullptr;
    void* d_labels_ = nullptr;
    void* d_boxes_ = nullptr;
    void* d_scores_ = nullptr;

    // Buffer sizes
    size_t images_size_ = 0;
    size_t orig_sizes_size_ = 0;
    size_t labels_size_ = 0;
    size_t boxes_size_ = 0;
    size_t scores_size_ = 0;

    // Model info
    int num_detections_ = 300; // RT-DETR default
};

// Factory function for Rust FFI
std::unique_ptr<TensorRTBackend> new_tensorrt_backend();

} // namespace bridge
