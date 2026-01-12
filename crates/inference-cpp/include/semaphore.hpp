#pragma once

#include <mqueue.h>
#include <string>
#include <optional>

namespace bridge {

enum class SemaphoreType {
    FrameCaptureToInference,
    FrameCaptureToGateway,
    DetectionInferenceToController
};

/// Wrapper around POSIX message queues for inter-process signaling
/// Mirrors the Rust BridgeSemaphore implementation
class BridgeSemaphore {
public:
    /// Open an existing message queue
    static std::optional<BridgeSemaphore> open(SemaphoreType type);

    /// Create a new message queue
    static std::optional<BridgeSemaphore> create(SemaphoreType type);

    ~BridgeSemaphore();

    // Disable copy, enable move
    BridgeSemaphore(const BridgeSemaphore&) = delete;
    BridgeSemaphore& operator=(const BridgeSemaphore&) = delete;
    BridgeSemaphore(BridgeSemaphore&& other) noexcept;
    BridgeSemaphore& operator=(BridgeSemaphore&& other) noexcept;

    /// Wait for a signal (blocking)
    bool wait();

    /// Try to consume a signal without blocking
    /// Returns true if signal was consumed, false if none available
    bool try_wait();

    /// Drain all pending signals, returning the count
    size_t drain();

    /// Post a signal
    bool post();

private:
    explicit BridgeSemaphore(mqd_t mqd);

    static const char* get_name(SemaphoreType type);

    mqd_t mqd_;
};

} // namespace bridge
