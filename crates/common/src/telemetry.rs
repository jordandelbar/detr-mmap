use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    Resource,
    metrics::{PeriodicReader, SdkMeterProvider},
    propagation::TraceContextPropagator,
    trace::{Sampler, SdkTracerProvider},
};
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initializes tracing and metrics providers on creation and shuts them down
/// gracefully when dropped.
///
/// # Example
/// ```ignore
/// let _telemetry = TelemetryGuard::init("my-service", "http://localhost:4317")?;
/// // ... application runs ...
/// // Telemetry is automatically flushed and shut down when guard is dropped
/// ```
pub struct TelemetryGuard {
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
}

impl TelemetryGuard {
    /// Initialize OpenTelemetry with OTLP export.
    ///
    /// Sets up both the OpenTelemetry providers and the tracing-opentelemetry layer
    /// to bridge `tracing` spans to OpenTelemetry.
    ///
    /// # Arguments
    /// * `service_name` - Name of this service (appears in traces/metrics)
    /// * `endpoint` - OTLP collector endpoint (e.g., "http://localhost:4317")
    pub fn init(service_name: &str, endpoint: &str) -> anyhow::Result<Self> {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let resource = Resource::builder()
            .with_attributes([
                KeyValue::new(
                    opentelemetry_semantic_conventions::attribute::SERVICE_NAME,
                    service_name.to_string(),
                ),
                KeyValue::new(
                    opentelemetry_semantic_conventions::attribute::SERVICE_VERSION,
                    env!("CARGO_PKG_VERSION"),
                ),
            ])
            .build();

        // OTLP Spans
        let span_exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()?;

        let tracer_provider = SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
            .with_batch_exporter(span_exporter)
            .build();

        global::set_tracer_provider(tracer_provider.clone());

        // OTLP Metrics
        let metric_exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()?;

        let reader = PeriodicReader::builder(metric_exporter)
            .with_interval(Duration::from_secs(10))
            .build();

        let meter_provider = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(reader)
            .build();

        global::set_meter_provider(meter_provider.clone());

        // Set up tracing-opentelemetry layer to bridge tracing spans to OpenTelemetry
        let otel_layer =
            tracing_opentelemetry::layer().with_tracer(global::tracer(service_name.to_string()));

        let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .with(otel_layer)
            .init();

        Ok(Self {
            tracer_provider,
            meter_provider,
        })
    }
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Err(e) = self.tracer_provider.shutdown() {
            eprintln!("Failed to shutdown tracer provider: {:?}", e);
        }
        if let Err(e) = self.meter_provider.shutdown() {
            eprintln!("Failed to shutdown meter provider: {:?}", e);
        }
    }
}

#[macro_export]
macro_rules! span {
    ($name:literal) => {
        tracing::info_span!($name).entered()
    };
}
