use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    metrics::{PeriodicReader, SdkMeterProvider},
    propagation::TraceContextPropagator,
    trace::{Sampler, SdkTracerProvider},
};
use std::time::Duration;

pub fn init_telemetry(
    _service_name: &str,
    endpoint: &str,
) -> anyhow::Result<(SdkTracerProvider, SdkMeterProvider)> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    // OTLP Spans
    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let tracer_provider = SdkTracerProvider::builder()
        //.with_resource(resource.clone()) // Use default resource
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
        .with_interval(Duration::from_secs(3))
        .build();

    let meter_provider = SdkMeterProvider::builder()
        //.with_resource(resource) // Use default resource
        .with_reader(reader)
        .build();

    global::set_meter_provider(meter_provider.clone());

    Ok((tracer_provider, meter_provider))
}

pub fn shutdown_telemetry(tracer_provider: &SdkTracerProvider, meter_provider: &SdkMeterProvider) {
    if let Err(e) = tracer_provider.shutdown() {
        eprintln!("Failed to shutdown tracer provider: {:?}", e);
    }
    if let Err(e) = meter_provider.shutdown() {
        eprintln!("Failed to shutdown meter provider: {:?}", e);
    }
}
