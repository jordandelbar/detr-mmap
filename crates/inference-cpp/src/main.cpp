#include "semaphore.hpp"
#include "frame_reader.hpp"
#include "detection_writer.hpp"
#include "preprocessing.hpp"
#include "tensorrt_backend.hpp"
#include "postprocessing.hpp"
#include "frame_generated.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>

using namespace bridge;

static const char* format_to_string(schema::ColorFormat format) {
    switch (format) {
        case schema::ColorFormat_BGR: return "BGR";
        case schema::ColorFormat_RGB: return "RGB";
        case schema::ColorFormat_GRAY: return "GRAY";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    std::cout << "=== C++ TensorRT Inference Starting ===" << std::endl;

    // Get model path from environment or use default
    const char* model_path_env = std::getenv("MODEL_PATH");
    std::string model_path = model_path_env ? model_path_env : "../../models/model_fp16.engine";

    std::cout << "Model path: " << model_path << std::endl;

    // Load TensorRT engine
    std::cout << "Loading TensorRT engine..." << std::endl;
    TensorRTBackend backend;
    if (!backend.load_engine(model_path)) {
        std::cerr << "Failed to load TensorRT engine" << std::endl;
        return 1;
    }
    std::cout << "✓ TensorRT engine loaded" << std::endl;

    // Initialize pre/post processors
    PreProcessor preprocessor(640);
    PostProcessor postprocessor(0.5f);

    // Connect to frame reader
    std::cout << "Connecting to frame buffer..." << std::endl;
    auto frame_reader_opt = FrameReader::build();
    if (!frame_reader_opt) {
        std::cerr << "Failed to connect to frame buffer. Is capture running?" << std::endl;
        std::cerr << "Retrying every 500ms..." << std::endl;

        while (!frame_reader_opt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            frame_reader_opt = FrameReader::build();
        }
    }
    auto frame_reader = std::move(*frame_reader_opt);
    std::cout << "✓ Frame buffer connected" << std::endl;

    // Connect to detection writer
    std::cout << "Connecting to detection buffer..." << std::endl;
    auto detection_writer_opt = DetectionWriter::build();
    if (!detection_writer_opt) {
        std::cerr << "Failed to connect to detection buffer" << std::endl;
        return 1;
    }
    auto detection_writer = std::move(*detection_writer_opt);
    std::cout << "✓ Detection buffer connected" << std::endl;

    // Open frame semaphore
    std::cout << "Opening frame inference semaphore..." << std::endl;
    auto frame_sem_opt = BridgeSemaphore::open(SemaphoreType::FrameCaptureToInference);
    if (!frame_sem_opt) {
        std::cerr << "Failed to open frame semaphore. Is capture running?" << std::endl;
        std::cerr << "Retrying every 500ms..." << std::endl;

        while (!frame_sem_opt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            frame_sem_opt = BridgeSemaphore::open(SemaphoreType::FrameCaptureToInference);
        }
    }
    auto frame_semaphore = std::move(*frame_sem_opt);
    std::cout << "✓ Frame semaphore connected" << std::endl;

    // Open or create controller semaphore
    std::cout << "Opening controller semaphore..." << std::endl;
    auto controller_sem_opt = BridgeSemaphore::open(SemaphoreType::DetectionInferenceToController);
    if (!controller_sem_opt) {
        std::cout << "Controller semaphore doesn't exist, creating..." << std::endl;
        controller_sem_opt = BridgeSemaphore::create(SemaphoreType::DetectionInferenceToController);
        if (!controller_sem_opt) {
            std::cerr << "Failed to create controller semaphore" << std::endl;
            return 1;
        }
    }
    auto controller_semaphore = std::move(*controller_sem_opt);
    std::cout << "✓ Controller semaphore connected" << std::endl;

    std::cout << "\n=== Starting TensorRT inference loop (event-driven) ===" << std::endl;
    std::cout << "Waiting for frames...\n" << std::endl;

    uint64_t frames_processed = 0;
    uint64_t frames_skipped = 0;
    uint64_t total_detections = 0;
    uint64_t total_inference_time_ms = 0;

    while (true) {
        // Wait for frame ready signal
        if (!frame_semaphore.wait()) {
            std::cerr << "Semaphore wait failed, sleeping..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Drain any additional pending signals to skip to the latest frame
        size_t skipped = frame_semaphore.drain();
        if (skipped > 0) {
            frames_skipped += skipped;
        }

        auto frame_start = std::chrono::high_resolution_clock::now();

        // Read frame
        const auto* frame = frame_reader.get_frame();
        if (!frame) {
            std::cerr << "Failed to read frame (torn read or no data)" << std::endl;
            continue;
        }

        // Extract frame metadata
        uint64_t frame_num = frame->frame_number();
        uint64_t timestamp_ns = frame->timestamp_ns();
        uint32_t camera_id = frame->camera_id();
        uint32_t width = frame->width();
        uint32_t height = frame->height();
        auto format = frame->format();
        auto pixels = frame->pixels();

        if (!pixels) {
            std::cerr << "Frame has no pixel data" << std::endl;
            continue;
        }

        // Preprocess frame
        bool is_bgr = (format == schema::ColorFormat_BGR);
        auto preprocess_result = preprocessor.preprocess(
            pixels->data(),
            width,
            height,
            is_bgr
        );

        // Prepare orig_sizes input [height, width]
        int64_t orig_sizes[2] = {
            static_cast<int64_t>(preprocess_result.input_height),
            static_cast<int64_t>(preprocess_result.input_width)
        };

        // Run inference
        InferenceOutput inference_output;
        if (!backend.infer(preprocess_result.data.data(), orig_sizes, inference_output)) {
            std::cerr << "Inference failed for frame " << frame_num << std::endl;
            continue;
        }

        // Post-process detections
        TransformParams transform{
            width,
            height,
            preprocess_result.scale,
            preprocess_result.offset_x,
            preprocess_result.offset_y
        };

        auto detections = postprocessor.parse_detections(inference_output, transform);

        // Write detections
        if (!detection_writer.write(frame_num, timestamp_ns, camera_id, detections)) {
            std::cerr << "Failed to write detections" << std::endl;
        }

        total_detections += detections.size();

        // Signal controller
        if (!controller_semaphore.post()) {
            std::cerr << "Failed to signal controller" << std::endl;
        }

        frames_processed++;
        frame_reader.mark_read();

        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        total_inference_time_ms += frame_time_ms;

        // Log every 10 frames
        if (frames_processed % 10 == 0) {
            std::cout << "[Frame " << frame_num << "] "
                      << width << "x" << height << " "
                      << format_to_string(format) << ", "
                      << "detections=" << detections.size() << ", "
                      << "time=" << frame_time_ms << "ms, "
                      << "skipped=" << frames_skipped
                      << std::endl;
        }

        // Periodic stats
        if (frames_processed % 100 == 0) {
            float avg_time = static_cast<float>(total_inference_time_ms) / frames_processed;
            float fps = 1000.0f / avg_time;
            std::cout << "\n>>> Stats: processed=" << frames_processed
                      << ", skipped=" << frames_skipped
                      << ", detections=" << total_detections
                      << ", avg_time=" << avg_time << "ms"
                      << ", fps=" << fps << " <<<\n" << std::endl;
        }
    }

    return 0;
}
