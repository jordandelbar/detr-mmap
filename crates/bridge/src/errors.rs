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

    #[error("Semaphore error: {0}")]
    SemaphoreError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_formatting() {
        // Test IoError display
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let bridge_err = BridgeError::IoError(io_err);
        assert_eq!(
            bridge_err.to_string(),
            "IO error: file not found",
            "IoError should display with 'IO error:' prefix"
        );

        // Test InvalidFlatBuffer display
        let err = BridgeError::InvalidFlatBuffer;
        assert_eq!(
            err.to_string(),
            "FlatBuffer verification failed",
            "InvalidFlatBuffer should display correct message"
        );

        // Test SizeMismatch display
        let err = BridgeError::SizeMismatch;
        assert_eq!(
            err.to_string(),
            "Buffer size mismatch",
            "SizeMismatch should display correct message"
        );

        // Test NoDataAvailable display
        let err = BridgeError::NoDataAvailable;
        assert_eq!(
            err.to_string(),
            "No data available yet",
            "NoDataAvailable should display correct message"
        );

        // Test SemaphoreError display
        let err = BridgeError::SemaphoreError("lock failed".to_string());
        assert_eq!(
            err.to_string(),
            "Semaphore error: lock failed",
            "SemaphoreError should display with custom message"
        );
    }

    #[test]
    fn test_error_conversion_from_io_error() {
        // Test automatic conversion from io::Error using the From trait
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "access denied");
        let bridge_err: BridgeError = io_err.into();

        match bridge_err {
            BridgeError::IoError(e) => {
                assert_eq!(e.kind(), io::ErrorKind::PermissionDenied);
                assert_eq!(e.to_string(), "access denied");
            }
            _ => panic!("Expected IoError variant"),
        }

        // Test that the #[from] attribute enables ? operator
        fn returns_io_error() -> Result<(), io::Error> {
            Err(io::Error::other("test error"))
        }

        fn uses_question_mark() -> Result<(), BridgeError> {
            returns_io_error()?;
            Ok(())
        }

        let result = uses_question_mark();
        assert!(result.is_err(), "Should propagate io::Error as BridgeError");
        match result.unwrap_err() {
            BridgeError::IoError(e) => assert_eq!(e.to_string(), "test error"),
            _ => panic!("Expected IoError variant"),
        }
    }
}
