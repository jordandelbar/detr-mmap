use anyhow::{Context, Result};
use v4l::{
    Device,
    buffer::{Metadata, Type},
    io::{mmap::Stream, traits::CaptureStream},
};

const BUFFER_COUNT: u32 = 4;

/// Number of frames to discard on mode transition to flush stale buffers
const FLUSH_FRAME_COUNT: usize = 4;

pub struct FrameSource<'a> {
    stream: Stream<'a>,
}

impl<'a> FrameSource<'a> {
    pub fn new(device: &'a Device) -> Result<Self> {
        let stream = Stream::with_buffers(device, Type::VideoCapture, BUFFER_COUNT)
            .context("Failed to create capture stream")?;
        Ok(Self { stream })
    }

    /// Discard buffered frames to ensure fresh captures after mode transition.
    pub fn flush(&mut self) -> usize {
        (0..FLUSH_FRAME_COUNT)
            .take_while(|_| self.stream.next().is_ok())
            .count()
    }

    pub fn next_frame(&mut self) -> Result<(&[u8], Metadata)> {
        self.stream
            .next()
            .map(|(data, meta)| (data, *meta))
            .map_err(Into::into)
    }
}
