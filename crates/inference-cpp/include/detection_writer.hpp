#pragma once

#include <string>
#include <optional>
#include <vector>
#include <cstdint>
#include <atomic>

namespace bridge {

/// Simple bounding box structure for detections
struct BoundingBox {
    float x1, y1, x2, y2;
    float confidence;
    uint32_t class_id;
};

/// Memory-mapped detection writer with atomic sequence publishing
/// Mirrors the Rust DetectionWriter implementation
class DetectionWriter {
public:
    /// Open the default detection buffer
    static std::optional<DetectionWriter> build();

    /// Open a detection buffer at a custom path
    static std::optional<DetectionWriter> with_path(const std::string& path);

    ~DetectionWriter();

    // Disable copy, enable move
    DetectionWriter(const DetectionWriter&) = delete;
    DetectionWriter& operator=(const DetectionWriter&) = delete;
    DetectionWriter(DetectionWriter&& other) noexcept;
    DetectionWriter& operator=(DetectionWriter&& other) noexcept;

    /// Write detections to shared memory
    bool write(uint64_t frame_number, uint64_t timestamp_ns, uint32_t camera_id,
               const std::vector<BoundingBox>& detections);

    /// Get current sequence number
    uint64_t sequence() const;

private:
    DetectionWriter(void* mmap_ptr, size_t mmap_size, int fd);

    void* mmap_ptr_;
    size_t mmap_size_;
    int fd_;

    static constexpr size_t HEADER_SIZE = 8; // sizeof(AtomicU64)
};

} // namespace bridge
