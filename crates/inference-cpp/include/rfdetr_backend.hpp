#pragma once

#include <NvInfer.h>
#include <memory>
#include <vector>

namespace bridge {

struct RFDetrOutput {
    std::vector<float> dets;   // [num_queries, 4] - cxcywh normalized
    std::vector<float> logits; // [num_queries, num_classes] - class logits
    size_t num_queries;
    size_t num_classes;
};

class RFDetrLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class RFDetrBackend {
public:
    RFDetrBackend();
    ~RFDetrBackend();

    // Disable copy, enable move
    RFDetrBackend(const RFDetrBackend&)            = delete;
    RFDetrBackend& operator=(const RFDetrBackend&) = delete;
    RFDetrBackend(RFDetrBackend&&) noexcept;
    RFDetrBackend& operator=(RFDetrBackend&&) noexcept;

    /// Load TensorRT engine from file
    bool load_engine(const char* engine_path);

    /// Run inference with raw output pointers (FFI friendly)
    /// images: [1, 3, 512, 512] float32 on host
    /// out_dets: [num_queries * 4] float32 - cxcywh boxes (normalized 0-1)
    /// out_logits: [num_queries * num_classes] float32 - class logits
    bool infer_raw(const float* images, float* out_dets, float* out_logits);

    /// Run inference with device pointer input (zero-copy from GPU preprocessing)
    /// d_images: device pointer to [1, 3, 512, 512] float32 already on GPU
    /// out_dets: [num_queries * 4] float32 - cxcywh boxes (normalized 0-1) on host
    /// out_logits: [num_queries * num_classes] float32 - class logits on host
    bool infer_from_device(void* d_images, float* out_dets, float* out_logits);

    int get_num_queries() const { return num_queries_; }
    int get_num_classes() const { return num_classes_; }

private:
    bool allocate_buffers();
    void free_buffers();

    RFDetrLogger logger_;
    nvinfer1::IRuntime* runtime_          = nullptr;
    nvinfer1::ICudaEngine* engine_        = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    // Device buffers
    void* d_input_  = nullptr;
    void* d_dets_   = nullptr;
    void* d_logits_ = nullptr;

    // Buffer sizes
    size_t input_size_  = 0;
    size_t dets_size_   = 0;
    size_t logits_size_ = 0;

    // Model info
    int input_height_ = 512;
    int input_width_  = 512;
    int num_queries_  = 300;
    int num_classes_  = 91;
};

std::unique_ptr<RFDetrBackend> new_rfdetr_backend();
} // namespace bridge
