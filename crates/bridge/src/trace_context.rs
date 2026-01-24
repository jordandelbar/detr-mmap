//! W3C Trace Context propagation for distributed tracing across IPC boundaries.
//!
//! This module provides utilities to capture the current OpenTelemetry span context
//! and restore it in another process after deserialization from FlatBuffers.
//!
//! The trace context is stored directly in the FlatBuffers schema (`schema::TraceContext`),
//! and these utilities work with that type to avoid duplication.

use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Capture the current span's trace context for injection into an IPC message.
///
/// Returns `None` if there is no active span or the span context is invalid.
pub fn capture_current_trace() -> Option<schema::TraceContext> {
    let span = tracing::Span::current();
    let otel_context = span.context();
    let span_ref = otel_context.span();
    let span_context = span_ref.span_context();

    if !span_context.is_valid() {
        return None;
    }

    Some(schema::TraceContext::new(
        &span_context.trace_id().to_bytes(),
        &span_context.span_id().to_bytes(),
        span_context.trace_flags().to_u8(),
    ))
}

/// Set the given trace context as the parent of the provided span.
///
/// Use this at IPC boundaries to link spans across processes.
/// All child spans created while inside this span will inherit the trace.
///
/// # Example
/// ```ignore
/// let span = tracing::info_span!("process_frame");
/// if let Some(trace) = frame.trace() {
///     set_trace_parent(trace, &span);
/// }
/// let _guard = span.entered();
/// // All #[instrument] functions called here become children
/// ```
pub fn set_trace_parent(trace: &schema::TraceContext, span: &tracing::Span) {
    let trace_id: [u8; 16] = std::array::from_fn(|i| trace.trace_id().get(i));
    let span_id: [u8; 8] = std::array::from_fn(|i| trace.span_id().get(i));

    let span_context = SpanContext::new(
        TraceId::from_bytes(trace_id),
        SpanId::from_bytes(span_id),
        TraceFlags::new(trace.trace_flags()),
        true, // remote = true since this came from another process
        TraceState::default(),
    );

    let otel_context = opentelemetry::Context::new().with_remote_span_context(span_context);
    let _ = span.set_parent(otel_context);
}

#[cfg(test)]
mod tests {
    #[test]
    fn set_trace_parent_creates_valid_span_context() {
        let trace = schema::TraceContext::new(&[0x01; 16], &[0x02; 8], 0x01);

        // Verify the trace context fields are accessible
        let trace_id: [u8; 16] = std::array::from_fn(|i| trace.trace_id().get(i));
        let span_id: [u8; 8] = std::array::from_fn(|i| trace.span_id().get(i));

        assert_eq!(trace_id, [0x01; 16]);
        assert_eq!(span_id, [0x02; 8]);
        assert_eq!(trace.trace_flags(), 0x01);
    }
}
