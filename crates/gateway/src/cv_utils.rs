use opencv::core::Mat;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CvUtilsError {
    #[error("Failed to encode frame: {0}")]
    EncodeFrameFailed(opencv::Error),
    #[error("OpenCV error: {0}")]
    OpenCvError(opencv::Error),
    #[error("OpenCV decode error: {0}")]
    OpenCvDecodeError(opencv::Error),
}

impl From<opencv::Error> for CvUtilsError {
    fn from(err: opencv::Error) -> Self {
        CvUtilsError::OpenCvError(err)
    }
}

pub struct CvImage {
    pub mat: Mat,
}

impl CvImage {
    fn new() -> Self {
        let mat = Mat::default();
        Self { mat }
    }
}
