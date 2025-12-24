use flatbuffers::FlatBufferBuilder;

const CONFIDENCE_THRESHOLD: f32 = 0.5;

pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

pub fn parse_detections(
    labels: &ndarray::ArrayViewD<i64>,
    boxes: &ndarray::ArrayViewD<f32>,
    scores: &ndarray::ArrayViewD<f32>,
    orig_width: u32,
    orig_height: u32,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
) -> anyhow::Result<Vec<Detection>> {
    let mut detections = Vec::new();

    let num_queries = labels.shape()[1];

    for i in 0..num_queries {
        let class_id = labels[[0, i]];
        let confidence = scores[[0, i]];

        if confidence < CONFIDENCE_THRESHOLD {
            continue;
        }

        let x1 = ((boxes[[0, i, 0]] - offset_x) / scale)
            .max(0.0)
            .min(orig_width as f32);
        let y1 = ((boxes[[0, i, 1]] - offset_y) / scale)
            .max(0.0)
            .min(orig_height as f32);
        let x2 = ((boxes[[0, i, 2]] - offset_x) / scale)
            .max(0.0)
            .min(orig_width as f32);
        let y2 = ((boxes[[0, i, 3]] - offset_y) / scale)
            .max(0.0)
            .min(orig_height as f32);

        detections.push(Detection {
            x1,
            y1,
            x2,
            y2,
            confidence,
            class_id: class_id as u32,
        });
    }

    Ok(detections)
}

pub fn build_detection_flatbuffer(
    frame_number: u64,
    timestamp_ns: u64,
    camera_id: u32,
    detections: &[Detection],
) -> anyhow::Result<Vec<u8>> {
    let mut builder = FlatBufferBuilder::new();

    let bbox_vec: Vec<_> = detections
        .iter()
        .map(|d| {
            schema::BoundingBox::create(
                &mut builder,
                &schema::BoundingBoxArgs {
                    x1: d.x1,
                    y1: d.y1,
                    x2: d.x2,
                    y2: d.y2,
                    confidence: d.confidence,
                    class_id: d.class_id,
                },
            )
        })
        .collect();

    let detections_offset = builder.create_vector(&bbox_vec);

    let detection_result = schema::DetectionResult::create(
        &mut builder,
        &schema::DetectionResultArgs {
            frame_number,
            timestamp_ns,
            camera_id,
            detections: Some(detections_offset),
        },
    );

    builder.finish(detection_result, None);
    Ok(builder.finished_data().to_vec())
}
