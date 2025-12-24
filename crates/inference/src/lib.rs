pub mod backend;
pub mod config;
pub mod logging;
pub mod processing;
pub mod serialization;
pub mod service;

pub use backend::{InferenceBackend, InferenceOutput};
pub use config::InferenceConfig;
pub use processing::post::Detection;
pub use serialization::DetectionSerializer;
pub use service::InferenceService;
