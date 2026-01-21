use serde::{Deserialize, Serialize};

/// Trace context bytes for serialization into FlatBuffers.
/// Contains W3C trace context fields for distributed tracing across IPC boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceContextBytes {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub trace_flags: u8,
}

#[cfg(feature = "tracing")]
impl From<&crate::trace_context::TraceContext> for TraceContextBytes {
    fn from(ctx: &crate::trace_context::TraceContext) -> Self {
        Self {
            trace_id: ctx.trace_id,
            span_id: ctx.span_id,
            trace_flags: ctx.trace_flags,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

impl From<&schema::BoundingBox<'_>> for Detection {
    fn from(bbox: &schema::BoundingBox) -> Self {
        Self {
            x1: bbox.x1(),
            y1: bbox.y1(),
            x2: bbox.x2(),
            y2: bbox.y2(),
            confidence: bbox.confidence(),
            class_id: bbox.class_id(),
        }
    }
}

pub struct FrameMetadata {
    pub frame_number: u64,
    pub timestamp_ns: u64,
    pub width: u32,
    pub height: u32,
    pub camera_id: u32,
}

impl From<&schema::Frame<'_>> for FrameMetadata {
    fn from(frame: &schema::Frame) -> Self {
        Self {
            frame_number: frame.frame_number(),
            timestamp_ns: frame.timestamp_ns(),
            width: frame.width(),
            height: frame.height(),
            camera_id: frame.camera_id(),
        }
    }
}
