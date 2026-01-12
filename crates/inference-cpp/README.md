# C++ Inference POC

This is a proof-of-concept C++ implementation of the inference service that integrates with the existing Rust bridge infrastructure.

## Purpose

Validate that a C++ process can:
- Read frames from shared memory with proper synchronization
- Use POSIX message queues for semaphore signaling
- Write detections back to shared memory
- Implement the same "drain" pattern for low-latency processing

## Architecture

The POC mirrors the Rust implementation structure:

```
Frame Producer (Rust capture)
    ↓ (shared memory + semaphore)
C++ Inference Service (this POC)
    ↓ (shared memory + semaphore)
Detection Consumer (Rust controller/gateway)
```

## Components

- **semaphore.hpp/cpp**: POSIX message queue wrapper for signaling
- **frame_reader.hpp/cpp**: Memory-mapped frame reader with double-sequence-check
- **detection_writer.hpp/cpp**: Memory-mapped detection writer
- **main.cpp**: Event-driven loop with drain pattern

## Key Features

1. **Double-Sequence-Check**: Torn read protection matching Rust implementation
2. **Drain Pattern**: Skips old frames to always process the latest
3. **FlatBuffers**: Uses same schemas as Rust services
4. **Atomic Operations**: Proper memory ordering (Acquire/Release)

## Building

```bash
cd crates/inference-cpp
mkdir build && cd build
cmake ..
make
```

## Running

Make sure the Rust capture service is running first:

```bash
# Terminal 1: Start capture
cd bridge-rt
cargo run --bin capture

# Terminal 2: Run C++ inference POC
./build/inference-cpp
```

## Current Status

**POC Stage**: Writes empty detections (no actual inference)

To add real inference:
1. Port pre-processing (RGB conversion, resize, normalize)
2. Integrate TensorRT
3. Port post-processing (NMS, coordinate transform)

## Dependencies

- FlatBuffers (for schema serialization)
- POSIX RT (for message queues)
- C++17 compiler

## Memory Layout

Shared memory structure (matches Rust):
```
[8 bytes: atomic sequence number]
[N bytes: FlatBuffer payload]
```

Sequence increments with Release ordering after payload write.
Readers use Acquire ordering to ensure visibility.
