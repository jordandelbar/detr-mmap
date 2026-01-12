#include "frame_reader.hpp"
#include "frame_generated.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

namespace bridge {

static const char* DEFAULT_FRAME_BUFFER_PATH = "/dev/shm/bridge_frame_buffer";

FrameReader::FrameReader(void* mmap_ptr, size_t mmap_size, int fd)
    : mmap_ptr_(mmap_ptr), mmap_size_(mmap_size), fd_(fd), last_sequence_(0) {}

FrameReader::~FrameReader() {
    if (mmap_ptr_ != MAP_FAILED && mmap_ptr_ != nullptr) {
        munmap(mmap_ptr_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

FrameReader::FrameReader(FrameReader&& other) noexcept
    : mmap_ptr_(other.mmap_ptr_),
      mmap_size_(other.mmap_size_),
      fd_(other.fd_),
      last_sequence_(other.last_sequence_) {
    other.mmap_ptr_ = nullptr;
    other.fd_ = -1;
}

FrameReader& FrameReader::operator=(FrameReader&& other) noexcept {
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
        last_sequence_ = other.last_sequence_;

        other.mmap_ptr_ = nullptr;
        other.fd_ = -1;
    }
    return *this;
}

std::optional<FrameReader> FrameReader::build() {
    return with_path(DEFAULT_FRAME_BUFFER_PATH);
}

std::optional<FrameReader> FrameReader::with_path(const std::string& path) {
    // Open shared memory file
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open frame buffer " << path << ": " << strerror(errno) << std::endl;
        return std::nullopt;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "Failed to stat frame buffer: " << strerror(errno) << std::endl;
        close(fd);
        return std::nullopt;
    }

    // Memory-map the file
    void* ptr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "Failed to mmap frame buffer: " << strerror(errno) << std::endl;
        close(fd);
        return std::nullopt;
    }

    return FrameReader(ptr, st.st_size, fd);
}

uint64_t FrameReader::current_sequence() const {
    // Read sequence with acquire ordering
    // In C++, std::atomic would be ideal, but we're directly accessing mmap
    // The Rust code uses Ordering::Acquire
    const auto* seq_ptr = static_cast<const std::atomic<uint64_t>*>(mmap_ptr_);
    return seq_ptr->load(std::memory_order_acquire);
}

const schema::Frame* FrameReader::get_frame() {
    // Double-sequence-check pattern for torn read protection
    uint64_t seq1 = current_sequence();

    // No frame available yet
    if (seq1 == 0) {
        return nullptr;
    }

    // Get pointer to FlatBuffer data (skip 8-byte header)
    const uint8_t* buffer = static_cast<const uint8_t*>(mmap_ptr_) + HEADER_SIZE;

    // Read sequence again to detect concurrent writes
    uint64_t seq2 = current_sequence();

    // Torn read detected - sequence changed during read
    if (seq1 != seq2) {
        return nullptr;
    }

    // Verify FlatBuffer validity
    flatbuffers::Verifier verifier(buffer, mmap_size_ - HEADER_SIZE);
    if (!verifier.VerifyBuffer<schema::Frame>()) {
        std::cerr << "Invalid FlatBuffer data" << std::endl;
        return nullptr;
    }

    return flatbuffers::GetRoot<schema::Frame>(buffer);
}

void FrameReader::mark_read() {
    last_sequence_ = current_sequence();
}

} // namespace bridge
