// Core modules (always available)
pub mod errors;
pub mod paths;
pub mod types;

// Trace context for distributed tracing (requires tracing feature)
#[cfg(feature = "tracing")]
pub mod trace_context;

// Utility modules (internal)
#[cfg(any(
    feature = "frame-writer",
    feature = "detection-writer",
    feature = "frame-reader",
    feature = "detection-reader"
))]
pub(crate) mod macros;
#[cfg(any(feature = "frame-reader", feature = "detection-reader"))]
pub(crate) mod utils;

// Conditionally compiled modules
#[cfg(feature = "detection-reader")]
pub mod detection_reader;
#[cfg(feature = "detection-writer")]
pub mod detection_writer;
#[cfg(feature = "frame-reader")]
pub mod frame_reader;
#[cfg(feature = "frame-writer")]
pub mod frame_writer;
#[cfg(any(feature = "mmap-reader", feature = "mmap-writer"))]
pub(crate) mod header;
#[cfg(feature = "mmap-reader")]
pub(crate) mod mmap_reader;
#[cfg(feature = "mmap-writer")]
pub(crate) mod mmap_writer;
#[cfg(feature = "semaphores")]
pub mod semaphore;
#[cfg(feature = "sentry")]
pub mod sentry_control;

// Public re-exports
#[cfg(feature = "detection-reader")]
pub use detection_reader::DetectionReader;
#[cfg(feature = "detection-writer")]
pub use detection_writer::DetectionWriter;
pub use errors::BridgeError;
#[cfg(feature = "frame-reader")]
pub use frame_reader::FrameReader;
#[cfg(feature = "frame-writer")]
pub use frame_writer::FrameWriter;
#[cfg(feature = "semaphores")]
pub use semaphore::{BridgeSemaphore, SemaphoreType};
#[cfg(feature = "sentry")]
pub use sentry_control::{SentryControl, SentryMode};
#[cfg(feature = "tracing")]
pub use trace_context::TraceContext;
pub use types::{Detection, TraceMetadata};

// Re-export schema types that services need
pub use schema::ColorFormat;
