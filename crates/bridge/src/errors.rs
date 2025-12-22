use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("FlatBuffer verification failed")]
    InvalidFlatBuffer,

    #[error("Buffer size mismatch")]
    SizeMismatch,

    #[error("No data available yet")]
    NoDataAvailable,
}
