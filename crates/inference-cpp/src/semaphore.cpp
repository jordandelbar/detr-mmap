#include "semaphore.hpp"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>

namespace bridge {

const char* BridgeSemaphore::get_name(SemaphoreType type) {
    switch (type) {
        case SemaphoreType::FrameCaptureToInference:
            return "/bridge_frame_inference";
        case SemaphoreType::FrameCaptureToGateway:
            return "/bridge_frame_gateway";
        case SemaphoreType::DetectionInferenceToController:
            return "/bridge_detection_controller";
    }
    return nullptr;
}

BridgeSemaphore::BridgeSemaphore(mqd_t mqd) : mqd_(mqd) {}

BridgeSemaphore::~BridgeSemaphore() {
    if (mqd_ != (mqd_t)-1) {
        mq_close(mqd_);
    }
}

BridgeSemaphore::BridgeSemaphore(BridgeSemaphore&& other) noexcept
    : mqd_(other.mqd_) {
    other.mqd_ = (mqd_t)-1;
}

BridgeSemaphore& BridgeSemaphore::operator=(BridgeSemaphore&& other) noexcept {
    if (this != &other) {
        if (mqd_ != (mqd_t)-1) {
            mq_close(mqd_);
        }
        mqd_ = other.mqd_;
        other.mqd_ = (mqd_t)-1;
    }
    return *this;
}

std::optional<BridgeSemaphore> BridgeSemaphore::open(SemaphoreType type) {
    const char* name = get_name(type);

    mqd_t mqd = mq_open(name, O_RDWR);
    if (mqd == (mqd_t)-1) {
        std::cerr << "Failed to open queue " << name << ": " << strerror(errno) << std::endl;
        return std::nullopt;
    }

    return BridgeSemaphore(mqd);
}

std::optional<BridgeSemaphore> BridgeSemaphore::create(SemaphoreType type) {
    const char* name = get_name(type);

    // Try to unlink existing queue
    mq_unlink(name);

    // Set queue attributes: max 10 messages, 1 byte per message
    struct mq_attr attr;
    attr.mq_flags = 0;
    attr.mq_maxmsg = 10;
    attr.mq_msgsize = 1;
    attr.mq_curmsgs = 0;

    mqd_t mqd = mq_open(name, O_CREAT | O_EXCL | O_RDWR, 0660, &attr);
    if (mqd == (mqd_t)-1) {
        std::cerr << "Failed to create queue " << name << ": " << strerror(errno) << std::endl;
        return std::nullopt;
    }

    return BridgeSemaphore(mqd);
}

bool BridgeSemaphore::wait() {
    char buf[1];
    unsigned int prio;

    while (true) {
        ssize_t result = mq_receive(mqd_, buf, sizeof(buf), &prio);
        if (result >= 0) {
            return true;
        }

        if (errno == EINTR) {
            continue; // Retry on interrupt
        }

        std::cerr << "mq_receive failed: " << strerror(errno) << std::endl;
        return false;
    }
}

bool BridgeSemaphore::try_wait() {
    char buf[1];
    unsigned int prio;
    struct timespec timeout = {0, 0}; // Zero timeout for non-blocking

    ssize_t result = mq_timedreceive(mqd_, buf, sizeof(buf), &prio, &timeout);
    if (result >= 0) {
        return true;
    }

    if (errno == ETIMEDOUT) {
        return false; // No message available
    }

    std::cerr << "mq_timedreceive failed: " << strerror(errno) << std::endl;
    return false;
}

size_t BridgeSemaphore::drain() {
    size_t count = 0;
    while (try_wait()) {
        count++;
    }
    return count;
}

bool BridgeSemaphore::post() {
    char msg[1] = {1};
    if (mq_send(mqd_, msg, sizeof(msg), 0) == 0) {
        return true;
    }

    std::cerr << "mq_send failed: " << strerror(errno) << std::endl;
    return false;
}

} // namespace bridge
