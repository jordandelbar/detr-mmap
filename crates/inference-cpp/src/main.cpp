#include "semaphore.hpp"
#include "frame_reader.hpp"
#include "detection_writer.hpp"
#include "frame_generated.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace bridge;

static const char* format_to_string(schema::ColorFormat format) {
    switch (format) {
        case schema::ColorFormat_BGR: return "BGR";
        case schema::ColorFormat_RGB: return "RGB";
        case schema::ColorFormat_GRAY: return "GRAY";
        default: return "UNKNOWN";
    }
}

int main() {
    std::cout << "=== C++ Inference POC Starting ===" << std::endl;

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

    std::cout << "\n=== Starting inference loop (event-driven) ===" << std::endl;
    std::cout << "Waiting for frames...\n" << std::endl;

    uint64_t frames_processed = 0;
    uint64_t frames_skipped = 0;
    uint64_t total_detections = 0;

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
        uint8_t channels = frame->channels();
        auto format = frame->format();

        // Log frame info (every 10 frames to reduce noise)
        if (frames_processed % 10 == 0) {
            std::cout << "[Frame " << frame_num << "] "
                      << width << "x" << height << "x" << (int)channels << " "
                      << format_to_string(format) << ", "
                      << "camera=" << camera_id << ", "
                      << "timestamp=" << timestamp_ns << "ns, "
                      << "skipped=" << frames_skipped
                      << std::endl;
        }

        // Create dummy detections (for POC: one fake bounding box or empty)
        std::vector<BoundingBox> detections;

        // Uncomment to add a dummy detection:
        // detections.push_back({100.0f, 100.0f, 200.0f, 200.0f, 0.95f, 0});

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

        // Periodic stats
        if (frames_processed % 100 == 0) {
            std::cout << "\n>>> Stats: processed=" << frames_processed
                      << ", skipped=" << frames_skipped
                      << ", detections=" << total_detections << " <<<\n" << std::endl;
        }
    }

    return 0;
}
