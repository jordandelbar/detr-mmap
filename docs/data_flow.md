# Data Flow & Synchronization Deep Dive

The system operates as a high-speed, lossy pipeline. "Lossy" here is a feature, not a bug: it prioritizes latency (freshness) over completeness.
If the inference engine is slow, it skips frames rather than falling behind.

## 1. Frame Data Flow (The "Writer")
 * Component: capture crate
 * Mechanism: Shared Memory (mmap)
 * Architecture: Single-Slot Atomic Snapshot Buffer
     * The mmap file contains exactly one frame at a time.
     * The writer always overwrites the same memory region (starting at Header::SIZE, offset 8).
     * There is no ring buffer or frame history.
     * Each new frame completely replaces the previous frame in memory.
     * Crucial Detail: There is only one active writer for the frame buffer.
 * Concurrency Model (Torn Read Protection):
     * **Problem**: Writer can overwrite memory while a reader is mid-read.
     * **Solution**: Readers use double-sequence-check pattern:
         1. Read sequence number (seq1) with Acquire ordering
         2. Read frame data from shared memory
         3. Read sequence number again (seq2) with Acquire ordering
         4. If seq1 != seq2, data was overwritten mid-read → discard frame
     * **Trade-off**: Torn reads are possible but safely detected and discarded.
     * **Why acceptable**: Video frames are disposable. Losing occasional frames due to slow readers is better than processing corrupted data.
     * Implementation: `crates/bridge/src/mmap_reader.rs:58-70` (read_frame method)
 * Atomic Publishing:
     1. Write Payload: The writer copies the frame pixels into the shared memory.
     2. Memory Barrier: Executes a Release fence (implicit in atomic store).
     3. Update Sequence: Increments the atomic sequence counter in the file header.
     * This ensures that any reader seeing the new sequence number is guaranteed to see the fully written frame data (or will detect torn read via sequence mismatch).

## 2. Signaling (The "Semaphore")
 * Component: bridge::BridgeSemaphore
 * Mechanism: POSIX Message Queues (mq_overview(7))
     * Not a standard mutex/semaphore: It's a kernel-managed queue of messages.
     * Why?: Unlike standard semaphores, message queues allow for select()/poll()-like behavior (via mq_timedreceive), enabling the "drain" logic in inference.
 * Fan-Out Pattern:
     * The capture service holds two semaphores:
         1. /bridge_frame_inference
         2. /bridge_frame_gateway
     * After writing one frame, it posts a 1-byte message to both queues.
     * Each queue has a capacity of 10 messages (kernel limit).
     * This decouples the consumers. If the gateway is fast but inference is slow, the gateway processes all frames while the inference queue piles up (up to 10 messages).

## 3. Consumer Patterns: Inference vs Gateway

### 3.1 Inference: The "Drain" Pattern (Lossy, Low Latency)
 * Component: inference crate
 * Goal: Always process the newest frame available, skip old frames.
 * The Problem:
     * Capture runs at variable FPS (3-30 FPS depending on sentry mode, see Section 4).
     * Inference might take 100ms per frame.
     * By the time inference finishes one frame, the capture service has posted 3 more signals to the queue.
     * If the inference service processed them strictly in order, it would permanently lag 3-4 frames behind (100-130ms latency).
 * The Solution ("Drain"):
     1. Wait: The inference service calls semaphore.wait(). It blocks until at least one frame is ready.
     2. Wake Up: It wakes up.
     3. Drain: It immediately calls semaphore.drain().
         * This creates a loop calling try_wait() until the queue is empty.
         * It discards all the "old" signals (doesn't read the frames, just consumes the queue messages).
         * Returns the count of skipped signals for metrics.
     4. Read Latest: It checks the mmap header for the current global sequence number (e.g., seq=105).
     5. Skip: It skips reading frames 102, 103, 104 entirely. It reads frame 105 directly from shared memory.
     * Result: Latency is minimized to exactly the inference time + capture time, regardless of how slow the inference is.
     * Code: `crates/inference/src/service.rs:107-127`

### 3.2 Gateway: The "Process All" Pattern (Lossless, High Throughput)
 * Component: gateway crate
 * Goal: Stream all frames to WebSocket clients for smooth video playback.
 * Behavior:
     * Gateway does NOT use drain().
     * It calls semaphore.wait() for each signal and processes every frame.
     * Frame rate matches capture rate (3-30 FPS depending on sentry mode).
 * Why No Drain?:
     * Gateway is fast (JPEG encoding + WebSocket send takes ~5-10ms).
     * It keeps up with the 30 FPS capture rate easily.
     * WebSocket clients expect smooth, continuous video (not just latest frames).
     * If a client is slow, the tokio broadcast channel handles backpressure (slow clients get dropped frames at their end, not at the gateway).
 * Result: All frames are encoded and broadcast. Individual WebSocket clients may drop frames if they can't keep up, but the gateway itself processes everything.
 * Code: `crates/gateway/src/polling.rs:BufferPoller::run()`

## 4. Sentry Mode: Adaptive Frame Rate Control
 * Component: bridge::SentryControl
 * Mechanism: Shared Memory (single atomic byte in /dev/shm/bridge_sentry_control)
 * Purpose: Dynamically adjust camera capture frame rate based on detection state to conserve CPU and power.
 * Modes:
     1. **Standby Mode (Low FPS)**:
         * Frame rate: Configurable (default ~3 FPS via sentry_mode_fps config)
         * Activated when: No person detected for sustained period
         * Purpose: Conserve CPU, power, and bandwidth during idle periods
         * Inference still runs on every frame (just fewer frames per second)
     2. **Alarmed Mode (High FPS)**:
         * Frame rate: Camera's maximum rate (typically 30 FPS)
         * Activated when: Person is detected
         * Purpose: High temporal resolution for tracking and detection
         * Ensures responsive capture during activity

### 4.1 Control Flow
 * **Controller Service** (the "Brain"):
     1. Reads detections from shared memory via DetectionReader
     2. Runs state machine with debouncing:
         * **Standby**: No person detected, stays in low-FPS mode
         * **Validation**: Person detected, confirming for N frames before switching
         * **Tracking**: Person confirmed, maintains high-FPS mode
     3. Maps state to sentry mode:
         * Standby state → SentryMode::Standby
         * Validation/Tracking states → SentryMode::Alarmed
     4. Updates shared control: `sentry_control.set_mode(mode)`
     5. Code: `crates/controller/src/service.rs:67-85`

 * **Capture Service** (the "Executor"):
     1. Reads sentry mode every frame: `mode = sentry.get_mode()`
     2. Adjusts sleep duration between frames:
         * Standby: sleeps for 333ms (3 FPS)
         * Alarmed: sleeps for 33ms (30 FPS)
     3. Flushes stale V4L2 buffer frames on Standby→Alarmed transition:
         * V4L2 maintains an internal buffer queue that can hold old frames during standby
         * On mode switch, discards frames older than 50ms to ensure fresh data
         * Prevents processing stale buffered frames when responsiveness matters most
         * Code: `crates/capture/src/camera.rs:285-314`
     4. Logs mode transitions for observability
     5. Code: `crates/capture/src/camera.rs:345-363`

### 4.2 Implementation Details
 * **Atomic Operations**:
     * Writer (controller): `mode.store(value, Ordering::Release)`
     * Reader (capture): `mode.load(Ordering::Acquire)`
     * Ensures memory visibility without locks
 * **Zero-Copy**:
     * Single byte in shared memory (/dev/shm)
     * All services can access via `SentryControl::build()`
     * No serialization overhead
 * **State Machine Debouncing**:
     * Prevents rapid mode switching on transient detections
     * Requires sustained state (e.g., 5 consecutive frames) before transitioning
     * Configurable thresholds: validation_frames, tracking_exit_frames
     * Code: `crates/controller/src/state_machine.rs`

### 4.3 Third Semaphore: Detection Signaling
 * Component: /bridge_detection_controller
 * Purpose: Signal controller when new detections are available
 * Flow:
     1. Inference service writes detections to shared memory
     2. Posts to detection semaphore: `detection_semaphore.post()`
     3. Controller service waits on semaphore: `detection_semaphore.wait()`
     4. Controller reads detections and updates state machine
     5. State machine output determines sentry mode
 * Note: This completes the feedback loop: Capture → Inference → Controller → Capture

### 4.4 Latency Characteristics
 * Mode switch to Alarmed: Immediate on first detection (Validation state triggers Alarmed mode)
 * Detection-to-Tracking: First detection at standby FPS (~333ms worst case at 3 FPS), then validation frames at 30 FPS (~33ms each)
 * The validation delay is intentional to prevent false alarms from single-frame noise
