pub mod backend;
pub mod config;
pub mod postprocessing;
pub mod preprocessing;
pub mod service;

// Re-export commonly used types for convenience
pub use backend::{InferenceBackend, InferenceOutput};
pub use config::InferenceConfig;
pub use postprocessing::Detection;
pub use service::InferenceService;
