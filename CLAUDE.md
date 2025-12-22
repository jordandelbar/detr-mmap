🚀 Project Overview

A high-performance, real-time object detection system built with Rust and K3s. It uses a multi-container sidecar pattern to bridge a Gateway and an Inference Service via Shared Memory and FlatBuffers for zero-copy data transfer.
    Primary Goal: Minimize latency by bypassing the network stack for inter-container communication.
    Target Model: Apache 2.0 compatible (e.g., YOLOX or RT-DETR).

🏗 Architecture & Tech Stack
    Language: Rust
    Serialization: FlatBuffers for zero-copy access.
    Transport: Shared Memory (/dev/shm) via POSIX memory-mapped files.
    Inference Engine: ONNX Runtime (ORT) with TensorRT Execution Provider.
    Orchestration: K3s (Kubernetes) using a single Pod for co-location of containers, in local devlopment kind is used. Kustomization adapts the difference between k3s and kind.

📦 Crate Structure (Workspace)
    crates/schema: FlatBuffer schemas (.fbs) and generated Rust bindings.
    crates/gateway: Ingress handler (HTTP/RTSP) -> Serializes to SHM.
    crates/inference: Reads SHM -> Runs TensorRT -> Writes results.
    crates/logic: Optional downstream decision engine (Zone monitoring/Alerts).

🛠 Development Guidelines
    Zero-Copy First: Avoid any unnecessary clone() or copy() of image data. Use raw pointers and lifetimes to map FlatBuffers directly from shared memory.
    Concurrency: Utilize Rust’s Send and Sync traits. The Gateway should handle multiple streams using tokio.
    TensorRT Engine Caching: Always use with_engine_cache_path in ORT to ensure the .engine files persist across Pod restarts via K3s volumes.
    Error Handling: Use thiserror for library errors and anyhow for binaries. No unwrap() in production code.
    Safety: Encapsulate unsafe shared memory blocks behind safe Rust abstractions.

📋 Useful Commands
    Build Workspace: cargo build --workspace
    Regenerate Schemas: flatc --rust -o crates/synapse-schema/src/ generated.fbs
    Run Local Tests: cargo test --workspace
    K3s Deploy: kubectl apply -f docker/deployment.yaml

💡 Important Context for AI
    SHM Bridge: When working on memory mapping, ensure /dev/shm is correctly mounted as an emptyDir with medium: Memory in Kubernetes.
    Lifetimes: When reading FlatBuffers, pay close attention to the buffer lifetime 'a.
    Model Agnostic: While currently focused on YOLOX/RT-DETR, the bridge logic should remain model-agnostic where possible.

Plan:

FlatBuffer Shared Memory Bridge Implementation Plan

Overview

Implement a zero-copy shared memory bridge between the gateway (camera capture) and inference service using FlatBuffers for serialization and POSIX named semaphores for synchronization.

Strategy: Drop-frames approach with single buffer - gateway always writes latest frame, inference always processes most recent data.

FlatBuffer Schema

File: crates/schema/frame.fbs

namespace bridge.schema;

enum ColorFormat : byte {
    BGR = 0,
    RGB = 1,
    GRAY = 2
}

table Frame {
    frame_number: uint64;
    timestamp_ns: uint64;
    camera_id: uint32;

    width: uint32;
    height: uint32;
    channels: uint8;
    format: ColorFormat;

    pixels: [ubyte];  // Raw BGR/RGB pixel data (last field for zero-copy)
}

root_type Frame;

Memory size: 8MB fixed allocation (supports up to 1920×1080×3 + overhead)

Architecture

Shared Memory Layout

- Location: /dev/shm/bridge_frame_buffer
- Size: 8 MB
- Content: Single FlatBuffer-serialized Frame

Synchronization Protocol (POSIX Named Semaphores)

- /bridge_writer_sem - Gateway controls write access
- /bridge_reader_sem - Signals when data is ready

Flow:
1. Gateway: Acquire writer_sem (non-blocking) → Write frame → Signal reader_sem
2. Inference: Wait on reader_sem → Read frame → Release writer_sem
3. If inference is slow, gateway overwrites (drops frames)

Implementation Steps

1. Schema Crate (crates/schema)

Files to create:
- frame.fbs - FlatBuffer schema definition
- build.rs - Auto-generate Rust bindings using flatc
- src/lib.rs - Re-export generated types + safe wrappers
- src/shm_bridge.rs - Shared memory and semaphore abstraction

Dependencies (add to Cargo.toml):
[dependencies]
flatbuffers = "24.3"

[build-dependencies]
flatc-rust = "0.2"

Key component: ShmBridge struct
- Encapsulates all unsafe operations (mmap, semaphores)
- Provides safe API: write_frame(), read_frame()
- Proper Drop implementation for cleanup

2. Gateway Integration (crates/gateway)

Files to modify/create:
- src/main.rs - Integrate ShmBridge into camera loop
- src/frame_writer.rs (new) - Convert OpenCV Mat → FlatBuffer
- Cargo.toml - Add dependencies

Dependencies:
schema = { path = "../schema" }
flatbuffers = "24.3"
memmap2 = "0.9"
nix = { version = "0.29", features = ["mman", "pthread"] }
anyhow = "1.0"

Changes to camera loop:
let mut shm = ShmBridge::create_writer("/bridge_frame", 8 * 1024 * 1024)?;

loop {
    cam.read(&mut frame)?;

    // Non-blocking write - drops frame if inference is busy
    if shm.try_acquire_writer_lock()? {
        let fb = build_flatbuffer_from_mat(&frame, frame_count)?;
        shm.write_frame(fb)?;
        frame_count += 1;
    } else {
        dropped_count += 1;
    }
}

3. Inference Service (crates/inference)

Files to modify/create:
- src/main.rs - Main loop reading from shared memory
- src/frame_processor.rs (new) - Process frames (stub for ONNX Runtime)
- Cargo.toml - Add dependencies

Dependencies: Same as gateway (schema, flatbuffers, memmap2, nix, anyhow, thiserror)

Main loop:
let shm = ShmBridge::create_reader("/bridge_frame")?;

loop {
    shm.acquire_reader_lock()?;  // Blocking wait for new frame

    let frame = shm.read_frame()?;  // Zero-copy deserialization
    let pixels: &[u8] = frame.pixels().unwrap();  // Direct access to mmap

    process_frame(&frame)?;  // Stub for ONNX Runtime

    shm.release_writer_lock()?;
}

Critical Files

| File                                    | Purpose                                      |
|-----------------------------------------|----------------------------------------------|
| crates/schema/frame.fbs                 | FlatBuffer schema - defines contract         |
| crates/schema/src/shm_bridge.rs         | Core bridge abstraction with all unsafe code |
| crates/schema/src/lib.rs                | Public API and safe wrappers                 |
| crates/gateway/src/main.rs              | Writer side integration                      |
| crates/gateway/src/frame_writer.rs      | Mat → FlatBuffer conversion                  |
| crates/inference/src/main.rs            | Reader side integration                      |
| crates/inference/src/frame_processor.rs | Frame processing stub                        |

Safety Considerations

All unsafe code isolated in shm_bridge.rs:
- Memory mapping: Proper size validation and alignment
- FlatBuffer access: Verification before deserialization
- Lifetime annotations: Frame data tied to mmap lifetime
- Drop safety: Clean up semaphores and unmap memory

Error Handling

- Use thiserror for library errors (schema crate)
- Use anyhow for binary errors (gateway, inference)
- Gateway: Log errors but continue (graceful degradation)
- Inference: Fatal on bridge errors, retry on processing errors
- No unwrap() in production code

Testing Strategy

1. Unit tests in schema crate (FlatBuffer serialization)
2. Gateway standalone test (write to SHM, verify with direct read)
3. Inference standalone test (mock frames in SHM)
4. Integration test (run both, verify transfer and latency)

Kubernetes Deployment Notes

Pod manifest needs shared /dev/shm:
volumes:
  - name: shm-bridge
    emptyDir:
      medium: Memory
      sizeLimit: 16Mi

containers:
  - name: gateway
    volumeMounts:
      - name: shm-bridge
        mountPath: /dev/shm
  - name: inference
    volumeMounts:
      - name: shm-bridge
        mountPath: /dev/shm
