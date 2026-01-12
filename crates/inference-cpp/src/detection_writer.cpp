#include "detection_writer.hpp"
#include "detection_generated.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

namespace bridge {

static const char* DEFAULT_DETECTION_BUFFER_PATH = "/dev/shm/bridge_detection_buffer";

DetectionWriter::DetectionWriter(void* mmap_ptr, size_t mmap_size, int fd)
    : mmap_ptr_(mmap_ptr), mmap_size_(mmap_size), fd_(fd) {}

DetectionWriter::~DetectionWriter() {
    if (mmap_ptr_ != MAP_FAILED && mmap_ptr_ != nullptr) {
        munmap(mmap_ptr_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

DetectionWriter::DetectionWriter(DetectionWriter&& other) noexcept
    : mmap_ptr_(other.mmap_ptr_),
      mmap_size_(other.mmap_size_),
      fd_(other.fd_) {
    other.mmap_ptr_ = nullptr;
    other.fd_ = -1;
}

DetectionWriter& DetectionWriter::operator=(DetectionWriter&& other) noexcept {
    if (this != &other) {
        if (mmap_ptr_ != MAP_FAILED && mmap_ptr_ != nullptr) {
            munmap(mmap_ptr_, mmap_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }

        mmap_ptr_ = other.mmap_ptr_;
        mmap_size_ = other.mmap_size_;
        fd_ = other.fd_;

        other.mmap_ptr_ = nullptr;
        other.fd_ = -1;
    }
    return *this;
}

std::optional<DetectionWriter> DetectionWriter::build() {
    return with_path(DEFAULT_DETECTION_BUFFER_PATH);
}

std::optional<DetectionWriter> DetectionWriter::with_path(const std::string& path) {
    static constexpr size_t DEFAULT_BUFFER_SIZE = 1024 * 1024; // 1MB

    // Try to open existing file first
    int fd = open(path.c_str(), O_RDWR);

    // If file doesn't exist, create it
    if (fd < 0 && errno == ENOENT) {
        std::cout << "Detection buffer doesn't exist, creating..." << std::endl;
        fd = open(path.c_str(), O_RDWR | O_CREAT, 0660);
        if (fd < 0) {
            std::cerr << "Failed to create detection buffer " << path << ": " << strerror(errno) << std::endl;
            return std::nullopt;
        }

        // Set file size
        if (ftruncate(fd, DEFAULT_BUFFER_SIZE) < 0) {
            std::cerr << "Failed to set detection buffer size: " << strerror(errno) << std::endl;
            close(fd);
            return std::nullopt;
        }
    } else if (fd < 0) {
        std::cerr << "Failed to open detection buffer " << path << ": " << strerror(errno) << std::endl;
        return std::nullopt;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "Failed to stat detection buffer: " << strerror(errno) << std::endl;
        close(fd);
        return std::nullopt;
    }

    // Memory-map the file
    void* ptr = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "Failed to mmap detection buffer: " << strerror(errno) << std::endl;
        close(fd);
        return std::nullopt;
    }

    // Initialize sequence to 0 if this is a new file
    auto* seq_ptr = static_cast<std::atomic<uint64_t>*>(ptr);
    if (st.st_size == DEFAULT_BUFFER_SIZE) {
        seq_ptr->store(0, std::memory_order_release);
    }

    return DetectionWriter(ptr, st.st_size, fd);
}

bool DetectionWriter::write(uint64_t frame_number, uint64_t timestamp_ns,
                             uint32_t camera_id, const std::vector<BoundingBox>& detections) {
    // Build FlatBuffer
    flatbuffers::FlatBufferBuilder builder(1024);

    // Convert BoundingBox vector to FlatBuffer format
    std::vector<flatbuffers::Offset<schema::BoundingBox>> detection_offsets;
    detection_offsets.reserve(detections.size());

    for (const auto& det : detections) {
        auto bbox = schema::CreateBoundingBox(builder, det.x1, det.y1, det.x2, det.y2,
                                              det.confidence, det.class_id);
        detection_offsets.push_back(bbox);
    }

    auto detections_vec = builder.CreateVector(detection_offsets);

    // Create DetectionResult
    auto detection_result = schema::CreateDetectionResult(
        builder, frame_number, timestamp_ns, camera_id, detections_vec);

    builder.Finish(detection_result);

    // Check if buffer fits
    size_t required_size = HEADER_SIZE + builder.GetSize();
    if (required_size > mmap_size_) {
        std::cerr << "Detection buffer too small: need " << required_size
                  << " bytes, have " << mmap_size_ << std::endl;
        return false;
    }

    // Write payload to shared memory (skip header)
    uint8_t* payload_ptr = static_cast<uint8_t*>(mmap_ptr_) + HEADER_SIZE;
    std::memcpy(payload_ptr, builder.GetBufferPointer(), builder.GetSize());

    // Ensure payload is visible before incrementing sequence (Release ordering)
    auto* seq_ptr = static_cast<std::atomic<uint64_t>*>(mmap_ptr_);
    uint64_t old_seq = seq_ptr->load(std::memory_order_relaxed);
    seq_ptr->store(old_seq + 1, std::memory_order_release);

    return true;
}

uint64_t DetectionWriter::sequence() const {
    const auto* seq_ptr = static_cast<const std::atomic<uint64_t>*>(mmap_ptr_);
    return seq_ptr->load(std::memory_order_acquire);
}

} // namespace bridge
