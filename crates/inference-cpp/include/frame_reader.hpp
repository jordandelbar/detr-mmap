#pragma once

#include <string>
#include <optional>
#include <cstdint>
#include <atomic>

// Forward declare FlatBuffers type
namespace bridge { namespace schema { struct Frame; } }

namespace bridge {

/// Memory-mapped frame reader with double-sequence-check for torn read protection
/// Mirrors the Rust FrameReader and MmapReader implementation
class FrameReader {
public:
    /// Open the default frame buffer
    static std::optional<FrameReader> build();

    /// Open a frame buffer at a custom path
    static std::optional<FrameReader> with_path(const std::string& path);

    ~FrameReader();

    // Disable copy, enable move
    FrameReader(const FrameReader&) = delete;
    FrameReader& operator=(const FrameReader&) = delete;
    FrameReader(FrameReader&& other) noexcept;
    FrameReader& operator=(FrameReader&& other) noexcept;

    /// Get current sequence number from shared memory
    uint64_t current_sequence() const;

    /// Get the current frame (returns nullptr if no frame or torn read detected)
    const schema::Frame* get_frame();

    /// Mark the current frame as read
    void mark_read();

private:
    FrameReader(void* mmap_ptr, size_t mmap_size, int fd);

    void* mmap_ptr_;
    size_t mmap_size_;
    int fd_;
    uint64_t last_sequence_;

    static constexpr size_t HEADER_SIZE = 8; // sizeof(AtomicU64)
};

} // namespace bridge
