# Bridge-RT Test Strategy & TODO List

## Testing Philosophy

**Priority Order:**
1. 🔴 High-Risk, High-Value → Test first (IPC synchronization, coordinate transformations)
2. 🟡 Pure Functions → Easiest to test, highest ROI (preprocessing, postprocessing)
3. 🟢 Integration Points → Where services communicate (mmap, WebSocket)
4. 🔵 Performance-Critical Paths → Benchmark to prevent regressions
5. ⚪ Edge Cases → Error handling, boundary conditions

---

## Phase 1: Critical Path Tests (HIGHEST PRIORITY)

### 🔴 Crate: `bridge` (IPC Primitives)

**File:** `crates/bridge/tests/integration_test.rs`
- [x] test_writer_reader_synchronization - Basic sync verification
- [x] test_concurrent_producer_consumer - Race condition detection
- [x] test_write_fails_when_data_exceeds_buffer - Buffer overflow protection
- [x] test_reader_handles_stale_data - Edge case: no writes yet
- [x] test_multiple_concurrent_readers - Multi-consumer scenario

**File:** `crates/bridge/src/mmap_reader.rs` (inline unit tests)
- [ ] test_new_reader_starts_with_zero_sequence
- [ ] test_has_new_data_returns_false_when_sequence_zero
- [ ] test_has_new_data_returns_true_after_sequence_increment
- [ ] test_mark_read_updates_last_sequence
- [ ] test_buffer_skips_header_bytes
- [ ] test_concurrent_reads_are_safe

**File:** `crates/bridge/src/mmap_writer.rs` (inline unit tests)
- [ ] test_new_writer_initializes_sequence_to_zero
- [ ] test_write_increments_sequence_atomically
- [ ] test_write_copies_data_before_sequence_update (memory ordering)
- [ ] test_flush_persists_to_disk
- [ ] test_buffer_mut_allows_direct_writes

**File:** `crates/bridge/src/errors.rs` (inline unit tests)
- [ ] test_error_display_formatting
- [ ] test_error_conversion_from_io_error

---

### 🔴 Crate: `inference` (Coordinate Math - CRITICAL)

**File:** `crates/inference/src/processing/post.rs` (inline unit tests)
- [x] test_confidence_threshold_filtering (0.49 vs 0.50)
- [x] test_coordinate_inverse_transformation (model → original coords)
- [x] test_coordinates_clamped_to_image_bounds (no overflow)
- [x] test_zero_detections_when_all_below_threshold
- [x] test_class_id_conversion_from_i64_to_u32
- [x] test_empty_input (edge case: 0 queries)
- [x] test_realistic_rtdetr_output (300 queries → 3 detections)

**File:** `crates/inference/src/processing/pre.rs` (inline unit tests)
- [ ] test_bgr_to_rgb_conversion (most cameras output BGR!)
- [ ] test_rgb_to_rgb_passthrough
- [ ] test_gray_format_returns_error
- [ ] test_buffer_size_mismatch_detection
- [ ] test_letterboxing_preserves_aspect_ratio
- [ ] test_letterboxing_uses_gray_padding (RGB 114,114,114)
- [ ] test_preprocessing_output_shape ([1, 3, 640, 640])
- [ ] test_pixel_normalization_to_0_1_range
- [ ] test_scale_calculation_for_tall_images
- [ ] test_scale_calculation_for_wide_images

---

## Phase 2: Integration Tests

### 🟢 Cross-Service Communication

**File:** `tests/gateway_to_inference_test.rs` (workspace-level)
- [ ] test_frame_serialization_deserialization_pipeline
  - Serialize frame (gateway) → Write mmap → Read (inference) → Preprocess
  - Assert: No data corruption, dimensions match

**File:** `tests/inference_to_logic_test.rs` (workspace-level)
- [ ] test_detection_serialization_via_mmap
  - Create detections → Serialize FlatBuffers → Write mmap → Logic reads
  - Assert: All detections present, coordinates correct

---

### 🟢 Crate: `gateway` (Serialization)

**File:** `crates/gateway/src/serialization.rs` (inline unit tests)
- [ ] test_serialize_frame_includes_metadata
- [ ] test_serialize_frame_includes_pixel_data
- [ ] test_deserialize_serialized_frame_roundtrip
- [ ] test_flatbuffers_zero_copy_access

**File:** `crates/gateway/src/camera.rs` (inline unit tests - requires mocking)
- [ ] test_camera_initialization_with_index
- [ ] test_frame_capture_returns_expected_dimensions
- [ ] test_camera_handles_capture_errors_gracefully

---

### 🟢 Crate: `logic` (WebSocket Streaming)

**File:** `crates/logic/src/polling.rs` (inline unit tests)
- [ ] test_jpeg_encoding_quality
- [ ] test_polling_detects_new_frames
- [ ] test_polling_combines_frame_and_detections
- [ ] test_polling_handles_mismatched_sequences

**File:** `crates/logic/src/ws.rs` (inline unit tests)
- [ ] test_websocket_handler_sends_frames
- [ ] test_websocket_handles_client_disconnect
- [ ] test_broadcast_to_multiple_clients
- [ ] test_slow_client_drops_frames (backpressure)

---

### 🟢 Crate: `common` (Logging)

**File:** `crates/common/src/logging.rs` (inline unit tests)
- [ ] test_development_uses_pretty_logging
- [ ] test_production_uses_json_logging
- [ ] test_rust_log_filter_applied

---

## Phase 3: Advanced Testing

### 🔵 Property-Based Tests (Optional but Recommended)

**File:** `crates/inference/tests/property_tests.rs`

Setup:
```toml
# Add to Cargo.toml
[dev-dependencies]
proptest = "1.4"
```

Tests:
- [ ] letterboxing_always_fits_in_640x640 (random dimensions)
- [ ] coordinate_transform_is_reversible (floating point precision)
- [ ] detection_parsing_never_panics (fuzz testing)
- [ ] test_detection_coordinates_never_negative (property test)
- [ ] test_detection_coordinates_within_bounds (property test)

---

### 🔵 Performance Benchmarks

**File:** `crates/inference/benches/preprocessing.rs`

Setup:
```toml
# Add to Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "preprocessing"
harness = false
```

Benchmarks:
- [ ] benchmark_preprocessing (1920x1080 frame)
- [ ] benchmark_postprocessing (300 detections)
- [ ] benchmark_bgr_to_rgb_conversion

**File:** `crates/bridge/benches/mmap_throughput.rs`
- [ ] benchmark_mmap_write (1KB, 10KB, 100KB, 1MB)
- [ ] benchmark_mmap_read

**File:** `crates/logic/benches/jpeg_encoding.rs`
- [ ] benchmark_jpeg_encoding_quality_80
- [ ] benchmark_jpeg_encoding_quality_95

---

### 🟡 Contract Tests (Schema Validation)

**File:** `crates/schema/tests/schema_compatibility_tests.rs`
- [ ] test_frame_schema_roundtrip
- [ ] test_detection_schema_backwards_compatibility
- [ ] test_frame_schema_supports_all_color_formats (RGB, BGR, GRAY)
- [ ] test_detection_supports_max_detections (300 queries)

---

## Phase 4: End-to-End & Resilience

### 🟢 End-to-End Tests

**File:** `tests/e2e/full_pipeline_test.rs`
- [ ] test_full_pipeline_with_synthetic_frame (requires model + /dev/shm)
- [ ] test_websocket_client_receives_frames

---

### ⚪ Chaos & Fault Injection Tests

**File:** `tests/resilience/fault_injection_test.rs`
- [ ] test_writer_handles_out_of_space (/dev/shm full)
- [ ] test_reader_handles_corrupted_mmap
- [ ] test_inference_handles_malformed_input
- [ ] test_websocket_handles_network_interruption
- [ ] test_services_handle_slow_consumer (30 FPS → 5 FPS)

---

### 🔵 Backend Tests

**File:** `crates/inference/src/backend/ort.rs` (inline unit tests)
- [ ] test_backend_initialization_with_valid_model
- [ ] test_backend_fails_with_invalid_model_path
- [ ] test_infer_output_shapes

---

## Infrastructure Setup

### Test Dependencies

**File:** `Cargo.toml` (workspace root)

```toml
[workspace.dependencies]
# Testing
proptest = "1.4"
criterion = { version = "0.5", features = ["html_reports"] }
tokio-test = "0.4"
tempfile = "3.8"
mockall = "0.12"
serial_test = "3.0"
test-log = "0.2"
wiremock = "0.6"
```

- [x] tempfile (added to bridge crate)
- [ ] proptest
- [ ] criterion
- [ ] tokio-test
- [ ] mockall
- [ ] serial_test
- [ ] test-log
- [ ] wiremock

---

### CI/CD Setup

**File:** `.github/workflows/test.yml`
- [ ] unit-tests job
- [ ] integration-tests job
- [ ] property-tests job
- [ ] benchmarks job
- [ ] coverage job (cargo-llvm-cov)

**File:** `.github/workflows/benchmark.yml`
- [ ] Run benchmarks on PR
- [ ] Regression detection (fail if 20%+ slower)

---

## Coverage Goals

| Crate     | Target | Current | Priority Tests                    |
|-----------|--------|---------|-----------------------------------|
| bridge    | 90%+   | ~30%    | ✅ IPC synchronization            |
| inference | 80%+   | ~25%    | ✅ Coordinate transforms (done!)  |
| gateway   | 60%+   | 0%      | ⚠️  Serialization                  |
| logic     | 75%+   | 0%      | WebSocket handling                |
| common    | 90%+   | 0%      | Logging configuration             |
| schema    | 100%   | 0%      | Schema roundtrips                 |

---

## Quick Reference: Test Commands

```bash
# Run all tests
cargo test --workspace

# Run specific crate
cargo test -p bridge
cargo test -p inference
cargo test -p gateway
cargo test -p logic

# Run with output
cargo test -- --nocapture

# Run integration tests only
cargo test --workspace --tests

# Run benchmarks
cargo bench --workspace

# Generate coverage report
cargo llvm-cov --workspace --html
```

---

## Test Data Setup

**Directory:** `test_data/`

```
test_data/
├── models/
│   └── rtdetr_minimal.onnx      # Tiny model for testing
├── frames/
│   ├── synthetic_640x480.bin
│   ├── real_1920x1080.bin
│   └── edge_cases/
│       ├── all_black.bin
│       ├── all_white.bin
│       └── checkerboard.bin
└── expected_outputs/
    └── detections_for_frame_1.json
```

- [ ] Create test_data directory structure
- [ ] Generate synthetic test frames
- [ ] Create minimal ONNX model for testing
- [ ] Generate golden output files

---

## Progress Summary

**Total Tests Planned:** ~80 tests
**Tests Implemented:** 12 ✅
**Completion:** 15%

**By Priority:**
- 🔴 Critical (Phase 1): 12/35 = 34% ⬆️
- 🟢 Integration (Phase 2): 0/25 = 0%
- 🔵 Advanced (Phase 3): 0/15 = 0%
- ⚪ Resilience (Phase 4): 0/5 = 0%

**Next Up:**
1. [ ] BGR→RGB conversion test (inference/pre.rs) ⚠️ CRITICAL
2. [ ] Preprocessing tests (letterboxing, normalization)
3. [ ] Serialization roundtrip tests (gateway)

---

## Notes

- Tests marked with ⚠️  are **blocking** for production
- Tests marked with 🔵 are **nice-to-have** but not critical
- Focus on Phase 1 & 2 before Phase 3 & 4
- Keep test runtime < 10 seconds for fast iteration
