#include "tensorrt_backend.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cstring>

namespace bridge {

void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    // Suppress INFO messages, show warnings and errors
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

TensorRTBackend::TensorRTBackend() {}

TensorRTBackend::~TensorRTBackend() {
    free_buffers();
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

TensorRTBackend::TensorRTBackend(TensorRTBackend&& other) noexcept
    : runtime_(other.runtime_),
      engine_(other.engine_),
      context_(other.context_),
      d_images_(other.d_images_),
      d_orig_sizes_(other.d_orig_sizes_),
      d_labels_(other.d_labels_),
      d_boxes_(other.d_boxes_),
      d_scores_(other.d_scores_),
      images_size_(other.images_size_),
      orig_sizes_size_(other.orig_sizes_size_),
      labels_size_(other.labels_size_),
      boxes_size_(other.boxes_size_),
      scores_size_(other.scores_size_),
      num_detections_(other.num_detections_) {
    other.runtime_ = nullptr;
    other.engine_ = nullptr;
    other.context_ = nullptr;
    other.d_images_ = nullptr;
    other.d_orig_sizes_ = nullptr;
    other.d_labels_ = nullptr;
    other.d_boxes_ = nullptr;
    other.d_scores_ = nullptr;
}

TensorRTBackend& TensorRTBackend::operator=(TensorRTBackend&& other) noexcept {
    if (this != &other) {
        free_buffers();
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;

        runtime_ = other.runtime_;
        engine_ = other.engine_;
        context_ = other.context_;
        d_images_ = other.d_images_;
        d_orig_sizes_ = other.d_orig_sizes_;
        d_labels_ = other.d_labels_;
        d_boxes_ = other.d_boxes_;
        d_scores_ = other.d_scores_;
        images_size_ = other.images_size_;
        orig_sizes_size_ = other.orig_sizes_size_;
        labels_size_ = other.labels_size_;
        boxes_size_ = other.boxes_size_;
        scores_size_ = other.scores_size_;
        num_detections_ = other.num_detections_;

        other.runtime_ = nullptr;
        other.engine_ = nullptr;
        other.context_ = nullptr;
        other.d_images_ = nullptr;
        other.d_orig_sizes_ = nullptr;
        other.d_labels_ = nullptr;
        other.d_boxes_ = nullptr;
        other.d_scores_ = nullptr;
    }
    return *this;
}

bool TensorRTBackend::load_engine(const std::string& engine_path) {
    std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;

    // Read engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create runtime and deserialize engine
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    std::cout << "Engine loaded successfully" << std::endl;

    // Allocate buffers
    if (!allocate_buffers()) {
        std::cerr << "Failed to allocate buffers" << std::endl;
        return false;
    }

    return true;
}

bool TensorRTBackend::allocate_buffers() {
    // Input: images [1, 3, 640, 640] float32
    images_size_ = 1 * 3 * 640 * 640 * sizeof(float);
    // Input: orig_target_sizes [1, 2] int64
    orig_sizes_size_ = 1 * 2 * sizeof(int64_t);
    // Output: labels [1, num_detections] int64
    labels_size_ = 1 * num_detections_ * sizeof(int64_t);
    // Output: boxes [1, num_detections, 4] float32
    boxes_size_ = 1 * num_detections_ * 4 * sizeof(float);
    // Output: scores [1, num_detections] float32
    scores_size_ = 1 * num_detections_ * sizeof(float);

    if (cudaMalloc(&d_images_, images_size_) != cudaSuccess) return false;
    if (cudaMalloc(&d_orig_sizes_, orig_sizes_size_) != cudaSuccess) return false;
    if (cudaMalloc(&d_labels_, labels_size_) != cudaSuccess) return false;
    if (cudaMalloc(&d_boxes_, boxes_size_) != cudaSuccess) return false;
    if (cudaMalloc(&d_scores_, scores_size_) != cudaSuccess) return false;

    std::cout << "CUDA buffers allocated" << std::endl;
    return true;
}

void TensorRTBackend::free_buffers() {
    if (d_images_) cudaFree(d_images_);
    if (d_orig_sizes_) cudaFree(d_orig_sizes_);
    if (d_labels_) cudaFree(d_labels_);
    if (d_boxes_) cudaFree(d_boxes_);
    if (d_scores_) cudaFree(d_scores_);
    d_images_ = nullptr;
    d_orig_sizes_ = nullptr;
    d_labels_ = nullptr;
    d_boxes_ = nullptr;
    d_scores_ = nullptr;
}

bool TensorRTBackend::infer(const float* images, const int64_t* orig_sizes, InferenceOutput& output) {
    // Copy inputs to device
    if (cudaMemcpy(d_images_, images, images_size_, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy images to device" << std::endl;
        return false;
    }
    if (cudaMemcpy(d_orig_sizes_, orig_sizes, orig_sizes_size_, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy orig_sizes to device" << std::endl;
        return false;
    }

    // Set input/output bindings
    void* bindings[] = {
        d_images_,      // images
        d_orig_sizes_,  // orig_target_sizes
        d_labels_,      // labels
        d_boxes_,       // boxes
        d_scores_       // scores
    };

    // Execute inference
    if (!context_->executeV2(bindings)) {
        std::cerr << "Failed to execute inference" << std::endl;
        return false;
    }

    // Allocate host memory for outputs
    output.labels.resize(num_detections_);
    output.boxes.resize(num_detections_ * 4);
    output.scores.resize(num_detections_);
    output.num_detections = num_detections_;

    // Copy outputs from device
    if (cudaMemcpy(output.labels.data(), d_labels_, labels_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy labels from device" << std::endl;
        return false;
    }
    if (cudaMemcpy(output.boxes.data(), d_boxes_, boxes_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy boxes from device" << std::endl;
        return false;
    }
    if (cudaMemcpy(output.scores.data(), d_scores_, scores_size_, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy scores from device" << std::endl;
        return false;
    }

    return true;
}

} // namespace bridge
