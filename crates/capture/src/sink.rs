use anyhow::Result;
use bridge::{BridgeSemaphore, FrameWriter, SemaphoreType, TraceMetadata};

pub struct FrameSink {
    writer: FrameWriter,
    inference: BridgeSemaphore,
    gateway: BridgeSemaphore,
}

impl FrameSink {
    pub fn new() -> Result<Self> {
        Ok(Self {
            writer: FrameWriter::build()?,
            inference: BridgeSemaphore::ensure(SemaphoreType::FrameCaptureToInference)?,
            gateway: BridgeSemaphore::ensure(SemaphoreType::FrameCaptureToGateway)?,
        })
    }

    pub fn write(
        &mut self,
        rgb: &[u8],
        camera_id: u32,
        frame_no: u64,
        width: u32,
        height: u32,
        trace: Option<&TraceMetadata>,
    ) -> Result<()> {
        self.writer
            .write(camera_id, rgb, frame_no, width, height, trace)?;
        self.inference.post().ok();
        self.gateway.post().ok();
        Ok(())
    }

    pub fn sequence(&self) -> u64 {
        self.writer.sequence()
    }
}
