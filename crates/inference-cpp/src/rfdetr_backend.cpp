#include "rfdetr_backend.hpp"
#include "logging.hpp"
#include <cuda_runtime.h>
#include <fstream>

namespace bridge {

void RFDetrLogger::log(Severity severity, const char* msg) noexcept {
    // Suppress INFO messages, show warnings and errors
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
        LOG_ERROR(std::string("[TensorRT/RF-DETR] ") + msg);
    } else if (severity == Severity::kWARNING) {
        LOG_WARN(std::string("[TensorRT/RF-DETR] ") + msg);
    }
}

RFDetrBackend::RFDetrBackend() {}

RFDetrBackend::~RFDetrBackend() {
    free_buffers();
    if (context_)
        delete context_;
    if (engine_)
        delete engine_;
    if (runtime_)
        delete runtime_;
}

RFDetrBackend::RFDetrBackend(RFDetrBackend&& other) noexcept
    : runtime_(other.runtime_), engine_(other.engine_), context_(other.context_),
      d_input_(other.d_input_), d_dets_(other.d_dets_), d_logits_(other.d_logits_),
      input_size_(other.input_size_), dets_size_(other.dets_size_),
      logits_size_(other.logits_size_), input_height_(other.input_height_),
      input_width_(other.input_width_), num_queries_(other.num_queries_),
      num_classes_(other.num_classes_) {
    other.runtime_  = nullptr;
    other.engine_   = nullptr;
    other.context_  = nullptr;
    other.d_input_  = nullptr;
    other.d_dets_   = nullptr;
    other.d_logits_ = nullptr;
}

RFDetrBackend& RFDetrBackend::operator=(RFDetrBackend&& other) noexcept {
    if (this != &other) {
        free_buffers();
        if (context_)
            delete context_;
        if (engine_)
            delete engine_;
        if (runtime_)
            delete runtime_;

        runtime_      = other.runtime_;
        engine_       = other.engine_;
        context_      = other.context_;
        d_input_      = other.d_input_;
        d_dets_       = other.d_dets_;
        d_logits_     = other.d_logits_;
        input_size_   = other.input_size_;
        dets_size_    = other.dets_size_;
        logits_size_  = other.logits_size_;
        input_height_ = other.input_height_;
        input_width_  = other.input_width_;
        num_queries_  = other.num_queries_;
        num_classes_  = other.num_classes_;

        other.runtime_  = nullptr;
        other.engine_   = nullptr;
        other.context_  = nullptr;
        other.d_input_  = nullptr;
        other.d_dets_   = nullptr;
        other.d_logits_ = nullptr;
    }
    return *this;
}

bool RFDetrBackend::load_engine(const char* engine_path) {
    LOG_INFO(std::string("Loading RF-DETR TensorRT engine from: ") + engine_path);

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR(std::string("Failed to open engine file: ") + engine_path);
        return false;
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        LOG_ERROR("Failed to create TensorRT runtime");
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        LOG_ERROR("Failed to deserialize CUDA engine");
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        LOG_ERROR("Failed to create execution context");
        return false;
    }

    LOG_INFO("RF-DETR engine loaded successfully");

    // Allocate buffers
    if (!allocate_buffers()) {
        LOG_ERROR("Failed to allocate buffers");
        return false;
    }

    return true;
}

bool RFDetrBackend::allocate_buffers() {
    // Input: input [1, 3, 512, 512] float32
    input_size_ = 1 * 3 * input_height_ * input_width_ * sizeof(float);
    // Output: dets [1, num_queries, 4] float32
    dets_size_ = 1 * num_queries_ * 4 * sizeof(float);
    // Output: labels (logits) [1, num_queries, num_classes] float32
    logits_size_ = 1 * num_queries_ * num_classes_ * sizeof(float);

    if (cudaMalloc(&d_input_, input_size_) != cudaSuccess) {
        LOG_ERROR("Failed to allocate d_input_");
        return false;
    }
    if (cudaMalloc(&d_dets_, dets_size_) != cudaSuccess) {
        LOG_ERROR("Failed to allocate d_dets_");
        return false;
    }
    if (cudaMalloc(&d_logits_, logits_size_) != cudaSuccess) {
        LOG_ERROR("Failed to allocate d_logits_");
        return false;
    }

    LOG_INFO("RF-DETR CUDA buffers allocated");
    return true;
}

void RFDetrBackend::free_buffers() {
    if (d_input_)
        cudaFree(d_input_);
    if (d_dets_)
        cudaFree(d_dets_);
    if (d_logits_)
        cudaFree(d_logits_);
    d_input_  = nullptr;
    d_dets_   = nullptr;
    d_logits_ = nullptr;
}

bool RFDetrBackend::infer_raw(const float* images, float* out_dets, float* out_logits) {
    // Copy input to device
    if (cudaMemcpy(d_input_, images, input_size_, cudaMemcpyHostToDevice) != cudaSuccess) {
        LOG_ERROR("Failed to copy input to device");
        return false;
    }

    // Set input/output bindings
    // RF-DETR has: input -> dets, labels (logits)
    void* bindings[] = {
        d_input_, // input
        d_dets_,  // dets
        d_logits_ // labels (logits)
    };

    // Execute inference
    if (!context_->executeV2(bindings)) {
        LOG_ERROR("Failed to execute RF-DETR inference");
        return false;
    }

    // Copy outputs from device
    if (cudaMemcpy(out_dets, d_dets_, dets_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("Failed to copy dets from device");
        return false;
    }
    if (cudaMemcpy(out_logits, d_logits_, logits_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("Failed to copy logits from device");
        return false;
    }

    return true;
}

bool RFDetrBackend::infer_from_device(uint8_t* d_images, float* out_dets, float* out_logits) {
    // Use provided device pointer directly (zero-copy from GPU preprocessing)
    void* bindings[] = {
        d_images, // input already on device
        d_dets_,  // dets
        d_logits_ // labels (logits)
    };

    // Execute inference
    if (!context_->executeV2(bindings)) {
        LOG_ERROR("Failed to execute RF-DETR inference (from device)");
        return false;
    }

    // Copy outputs from device to host
    if (cudaMemcpy(out_dets, d_dets_, dets_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("Failed to copy dets from device");
        return false;
    }
    if (cudaMemcpy(out_logits, d_logits_, logits_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("Failed to copy logits from device");
        return false;
    }

    return true;
}

std::unique_ptr<RFDetrBackend> new_rfdetr_backend() {
    return std::make_unique<RFDetrBackend>();
}

} // namespace bridge
