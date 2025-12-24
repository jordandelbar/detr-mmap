pub mod backend;
pub mod preprocessing;
pub mod postprocessing;

// Re-export commonly used types for convenience
pub use backend::{InferenceBackend, InferenceOutput};
pub use postprocessing::Detection;
