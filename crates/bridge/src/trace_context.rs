//! W3C Trace Context propagation for distributed tracing across IPC boundaries.
//!
//! This module provides utilities to capture the current OpenTelemetry span context
//! and restore it in another process after deserialization from FlatBuffers.

use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// W3C Trace Context for propagation through IPC/FlatBuffers.
///
/// Contains the essential fields needed to reconstruct a span context:
/// - `trace_id`: 16-byte trace identifier
/// - `span_id`: 8-byte span identifier
/// - `trace_flags`: 1-byte flags (sampling decision)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceContext {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub trace_flags: u8,
}

impl TraceContext {
    /// Capture the current span's context for injection into an IPC message.
    ///
    /// Returns `None` if there is no active span or the span context is invalid.
    pub fn from_current() -> Option<Self> {
        let span = tracing::Span::current();
        let otel_context = span.context();
        let span_ref = otel_context.span();
        let span_context = span_ref.span_context();

        if !span_context.is_valid() {
            return None;
        }

        Some(Self {
            trace_id: span_context.trace_id().to_bytes(),
            span_id: span_context.span_id().to_bytes(),
            trace_flags: span_context.trace_flags().to_u8(),
        })
    }

    /// Create a TraceContext from raw bytes extracted from FlatBuffers.
    ///
    /// Returns `None` if the input slices have incorrect lengths.
    pub fn from_bytes(trace_id: &[u8], span_id: &[u8], trace_flags: u8) -> Option<Self> {
        if trace_id.len() != 16 || span_id.len() != 8 {
            return None;
        }

        let mut tid = [0u8; 16];
        let mut sid = [0u8; 8];
        tid.copy_from_slice(trace_id);
        sid.copy_from_slice(span_id);

        Some(Self {
            trace_id: tid,
            span_id: sid,
            trace_flags,
        })
    }

    /// Convert this trace context into an OpenTelemetry Context that can be used
    /// as a parent for new spans.
    ///
    /// The returned context can be used with `tracing::Span::set_parent()` to
    /// link spans across process boundaries.
    pub fn into_context(&self) -> opentelemetry::Context {
        let span_context = SpanContext::new(
            TraceId::from_bytes(self.trace_id),
            SpanId::from_bytes(self.span_id),
            TraceFlags::new(self.trace_flags),
            true, // remote = true since this came from another process
            TraceState::default(),
        );

        opentelemetry::Context::new().with_remote_span_context(span_context)
    }

    /// Check if this trace context represents a sampled trace.
    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bytes_valid() {
        let trace_id = [1u8; 16];
        let span_id = [2u8; 8];
        let trace_flags = 0x01;

        let ctx = TraceContext::from_bytes(&trace_id, &span_id, trace_flags);
        assert!(ctx.is_some());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.trace_id, trace_id);
        assert_eq!(ctx.span_id, span_id);
        assert_eq!(ctx.trace_flags, trace_flags);
        assert!(ctx.is_sampled());
    }

    #[test]
    fn from_bytes_invalid_trace_id_length() {
        let trace_id = [1u8; 15]; // Wrong length
        let span_id = [2u8; 8];

        let ctx = TraceContext::from_bytes(&trace_id, &span_id, 0x01);
        assert!(ctx.is_none());
    }

    #[test]
    fn from_bytes_invalid_span_id_length() {
        let trace_id = [1u8; 16];
        let span_id = [2u8; 7]; // Wrong length

        let ctx = TraceContext::from_bytes(&trace_id, &span_id, 0x01);
        assert!(ctx.is_none());
    }

    #[test]
    fn into_context_creates_valid_span_context() {
        use opentelemetry::trace::TraceContextExt;

        let ctx = TraceContext {
            trace_id: [0x01; 16],
            span_id: [0x02; 8],
            trace_flags: 0x01,
        };

        let otel_ctx = ctx.into_context();
        let span_ref = otel_ctx.span();
        let span_ctx = span_ref.span_context();

        assert!(span_ctx.is_valid());
        assert!(span_ctx.is_remote());
        assert!(span_ctx.is_sampled());
        assert_eq!(span_ctx.trace_id().to_bytes(), ctx.trace_id);
        assert_eq!(span_ctx.span_id().to_bytes(), ctx.span_id);
    }

    #[test]
    fn is_sampled_flag() {
        let sampled = TraceContext {
            trace_id: [1u8; 16],
            span_id: [2u8; 8],
            trace_flags: 0x01,
        };
        assert!(sampled.is_sampled());

        let not_sampled = TraceContext {
            trace_id: [1u8; 16],
            span_id: [2u8; 8],
            trace_flags: 0x00,
        };
        assert!(!not_sampled.is_sampled());
    }
}
