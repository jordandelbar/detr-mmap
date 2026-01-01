# Contributing Guide

## To Fix

- [ ] Magic numbers scattered (CONFIDENCE_THRESHOLD: 0.5, poll intervals)
- [ ] Could use more newtype patterns for domain concepts
- [x] Some code duplication:
  - [x] Dockerfiles
  - [x] config patterns
- [ ] Input validation beyond basic assertions
- [ ] Comprehensive test suite (unit, integration, property-based)
- [ ] Architecture documentation with diagrams
- [ ] GitHub Actions for CI/CD
- [ ] Health checks and metrics endpoints
- [ ] Benchmark suite for performance regression detection
- [ ] No automated image builds/publishing
- [ ] No Helm charts for production deployments
- [ ] Security scanning not integrated (Trivy, Dependabot)

## 📋 Complete Magic Numbers List

1. Image Processing Constants

| Magic Number | Location                                     | What It Is                          | Impact                                            |
|--------------|----------------------------------------------|-------------------------------------|---------------------------------------------------|
| 640, 640     | crates/inference/src/processing/pre.rs:4     | INPUT_SIZE (already constant!)      | Already defined, but hardcoded in config defaults |
| 114          | crates/inference/src/processing/pre.rs:5     | LETTERBOX_COLOR (already constant!) | Padding color for letterboxing                    |
| 255.0        | crates/inference/src/processing/pre.rs:94-96 | Pixel normalization divisor         | Converts u8 [0,255] → f32 [0.0, 1.0]              |
| 3            | Multiple locations                           | RGB channel count                   | Assumed everywhere                                |

2. ML/Inference Parameters

| Magic Number | Location                                        | What It Is                   | Impact                         |
|--------------|-------------------------------------------------|------------------------------|--------------------------------|
| 0.6          | crates/inference/src/config.rs:54, 79           | Default confidence threshold | Filters detections (critical!) |
| 0.5          | crates/inference/src/processing/post.rs:40-115  | Test confidence threshold    | Used in multiple tests         |
| 300          | crates/inference/src/processing/post.rs:333-378 | RT-DETR query count          | Model-specific constant        |
| 640, 640     | crates/inference/src/config.rs:39, 44, 77       | Default input dimensions     | Duplicated from pre.rs         |

3. Polling/Timing Intervals

| Magic Number | Location                                   | What It Is                       | Impact                     |
|--------------|--------------------------------------------|----------------------------------|----------------------------|
| 500          | crates/gateway/src/polling.rs:23, 36, 50     | Retry delay (ms) when connecting | Connection retry backoff   |
| 100          | crates/gateway/src/polling.rs:64, 70         | Error recovery delay (ms)        | Error handling sleep       |
| 100          | crates/inference/src/service.rs:48, 72, 87 | Inference poll interval (ms)     | Duplicated from config!    |
| 16           | crates/gateway/src/config.rs:29, 56          | Gateway poll interval (~60 FPS)    | Default frame polling rate |

4. Memory/Buffer Sizes

| Magic Number     | Location                              | What It Is                      | Impact                      |
|------------------|---------------------------------------|---------------------------------|-----------------------------|
| 8                | crates/bridge/src/mmap_writer.rs:7    | DATA_OFFSET (already constant!) | Atomic u64 sequence header  |
| 1024 * 1024      | crates/inference/src/config.rs:34, 76 | 1MB detection buffer default    | Mmap size                   |
| 32 * 1024 * 1024 | crates/capture/src/main.rs:15         | 32MB frame buffer               | Hardcoded, not from config! |
| 10               | crates/gateway/src/config.rs:37, 59     | Broadcast channel capacity      | WebSocket channel buffer    |

5. Network Configuration

| Magic Number   | Location                          | What It Is                | Impact         |
|----------------|-----------------------------------|---------------------------|----------------|
| "0.0.0.0:8080" | crates/gateway/src/config.rs:32, 57 | Default WebSocket address | Hardcoded port |
