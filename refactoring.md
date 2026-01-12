  1. Resource Connection Boilerplate (High Impact - DRY)
  Every service (capture, inference, gateway, controller) duplicates the logic for connecting to shared memory and semaphores. They all implement a variation of "loop indefinitely with sleep until resource is available".

   * Current State: ~20-30 lines of identical looping/sleeping code in the main or new function of every service.
   * Refactoring: Introduce a wait_for_resource helper in crates/common (or bridge) that handles the infinite retry loop with logging and backoff.
   * Benefit: Removes ~100 lines of duplicated boilerplate and standardizes startup behavior.

  Example of duplication to fix (`inference/src/service.rs`):

   1 let frame_semaphore = loop {
   2     match BridgeSemaphore::open(SemaphoreType::FrameCaptureToInference) {
   3         Ok(sem) => break sem,
   4         Err(_) => thread::sleep(Duration::from_millis(self.config.poll_interval_ms)),
   5     }
   6 };

  2. PreProcessor Memory Allocations (High Impact - Performance)
  The PreProcessor in crates/inference/src/processing/pre.rs allocates significant memory per frame, causing unnecessary heap churn in the hot path.

   * Current State: rgb_data, dst_image, letterboxed, and the final input Array are allocated fresh for every single frame call.
   * Refactoring: Change PreProcessor to hold reusable buffers (struct fields). Since video resolution rarely changes at runtime, these buffers can be allocated once.
   * Benefit: Reduces memory allocator pressure and improves frame throughput.

  3. Bridge Writer/Reader Duplication (Medium Impact - Maintainability)
  FrameWriter vs DetectionWriter (and their Reader counterparts) share 90% of their logic. They only differ in the specific FlatBuffer schema used.

   * Current State: Identical build_with_path, sequence, and underlying MmapWriter usage.
   * Refactoring: Introduce a generic MmapBackedWriter<T> or a macro to generate these classes.
   * Benefit: Reduces code duplication in crates/bridge.

  4. Semaphore "Open or Create" Pattern (Low Impact - Cleanliness)
  Services often try to open a semaphore, and if it fails, create it (or vice versa).

   * Current State: Ad-hoc match blocks in service initialization.
   * Refactoring: Add a BridgeSemaphore::ensure(type) method that encapsulates this "open existing or create new" logic.

  5. Configuration Loading (Low Impact - Cleanliness)
  All Config::from_env methods manually parse environment variables with unwrap_or.

   * Current State: Repetitive env::var("...").ok().and_then(...) chains.
   * Refactoring: A simple common::config::get_env(key, default) helper would clean this up significantly.
