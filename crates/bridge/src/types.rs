use serde::{Deserialize, Serialize};

/// Trace context bytes for serialization into FlatBuffers.
/// Contains W3C trace context fields for distributed tracing across IPC boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceMetadata {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub trace_flags: u8,
}

impl TraceMetadata {
    /// Convert this trace context into an OpenTelemetry Context for span linking.
    #[cfg(feature = "tracing")]
    fn as_context(&self) -> opentelemetry::Context {
        use opentelemetry::trace::{
            SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState,
        };

        let span_context = SpanContext::new(
            TraceId::from_bytes(self.trace_id),
            SpanId::from_bytes(self.span_id),
            TraceFlags::new(self.trace_flags),
            true, // remote = true since this came from another process
            TraceState::default(),
        );

        opentelemetry::Context::new().with_remote_span_context(span_context)
    }

    /// Set this trace context as the parent of the given span.
    ///
    /// Use this at IPC boundaries to link spans across processes.
    /// All child spans created while inside this span will inherit the trace.
    ///
    /// # Example
    /// ```ignore
    /// let span = tracing::info_span!("process_frame");
    /// if let Some(ctx) = trace_ctx.as_ref() {
    ///     ctx.set_parent(&span);
    /// }
    /// let _guard = span.entered();
    /// // All #[instrument] functions called here become children
    /// ```
    #[cfg(feature = "tracing")]
    pub fn set_parent(&self, span: &tracing::Span) {
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        let _ = span.set_parent(self.as_context());
    }
}

#[cfg(feature = "tracing")]
impl From<&crate::trace_context::TraceContext> for TraceMetadata {
    fn from(ctx: &crate::trace_context::TraceContext) -> Self {
        Self {
            trace_id: ctx.trace_id,
            span_id: ctx.span_id,
            trace_flags: ctx.trace_flags,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

impl From<&schema::BoundingBox<'_>> for BoundingBox {
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

pub struct Frame {
    pub frame_number: u64,
    pub timestamp_ns: u64,
    pub width: u32,
    pub height: u32,
}

impl From<&schema::Frame<'_>> for Frame {
    fn from(frame: &schema::Frame) -> Self {
        Self {
            frame_number: frame.frame_number(),
            timestamp_ns: frame.timestamp_ns(),
            width: frame.width(),
            height: frame.height(),
        }
    }
}
