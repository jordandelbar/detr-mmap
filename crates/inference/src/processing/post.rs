use flatbuffers::{FlatBufferBuilder, ForwardsUOffset, Vector, WIPOffset};

pub struct TransformParams {
    pub orig_width: u32,
    pub orig_height: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub scale: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

pub struct PostProcessor {
    pub confidence_threshold: f32,
}

impl PostProcessor {
    pub fn new(confidence_threshold: f32) -> Self {
        Self {
            confidence_threshold,
        }
    }

    /// Parse detections from RF-DETR output and build directly into FlatBuffer.
    #[tracing::instrument(skip(self, builder, dets, logits, transform))]
    pub fn parse_detections<'a>(
        &self,
        builder: &mut FlatBufferBuilder<'a>,
        dets: &ndarray::ArrayViewD<f32>, // [1, 300, 4] - boxes in cxcywh format (normalized 0-1)
        logits: &ndarray::ArrayViewD<f32>, // [1, 300, 91] - class logits
        transform: &TransformParams,
    ) -> anyhow::Result<(
        WIPOffset<Vector<'a, ForwardsUOffset<schema::Detection<'a>>>>,
        usize,
    )> {
        let num_queries = dets.shape()[1];
        let num_classes = logits.shape()[2];

        // First pass: collect detection offsets
        let mut detection_offsets = Vec::new();

        for i in 0..num_queries {
            // Find max logit and its index (argmax for class_id)
            // RF-DETR uses 1-indexed classes (0=background, 1=person, 2=bicycle, ...)
            // Skip index 0 (background) and convert to 0-indexed COCO IDs
            let mut max_logit = f32::NEG_INFINITY;
            let mut class_idx = 1usize;
            for c in 1..num_classes {
                let logit = logits[[0, i, c]];
                if logit > max_logit {
                    max_logit = logit;
                    class_idx = c;
                }
            }

            // Convert RF-DETR 1-indexed class to 0-indexed COCO class
            let class_id = (class_idx - 1) as u16;

            // Apply sigmoid to max logit for confidence
            let confidence = sigmoid(max_logit);

            if confidence < self.confidence_threshold {
                continue;
            }

            // Get box in cxcywh format (normalized 0-1)
            let cx = dets[[0, i, 0]];
            let cy = dets[[0, i, 1]];
            let w = dets[[0, i, 2]];
            let h = dets[[0, i, 3]];

            // Convert cxcywh to xyxy (still normalized)
            let (x1_norm, y1_norm, x2_norm, y2_norm) = cxcywh_to_xyxy(cx, cy, w, h);

            // Denormalize to input_size (e.g., 512x512)
            let x1_input = x1_norm * transform.input_width as f32;
            let y1_input = y1_norm * transform.input_height as f32;
            let x2_input = x2_norm * transform.input_width as f32;
            let y2_input = y2_norm * transform.input_height as f32;

            // Apply inverse letterbox transform to original image coordinates
            let x1 = ((x1_input - transform.offset_x) / transform.scale)
                .max(0.0)
                .min(transform.orig_width as f32);
            let y1 = ((y1_input - transform.offset_y) / transform.scale)
                .max(0.0)
                .min(transform.orig_height as f32);
            let x2 = ((x2_input - transform.offset_x) / transform.scale)
                .max(0.0)
                .min(transform.orig_width as f32);
            let y2 = ((y2_input - transform.offset_y) / transform.scale)
                .max(0.0)
                .min(transform.orig_height as f32);

            // Build Detection directly into FlatBuffer
            let bbox = schema::BoundingBox::new(x1, y1, x2, y2);
            let detection = schema::Detection::create(
                builder,
                &schema::DetectionArgs {
                    box_: Some(&bbox),
                    confidence,
                    class_id,
                },
            );
            detection_offsets.push(detection);
        }

        let count = detection_offsets.len();
        let detections_vector = builder.create_vector(&detection_offsets);

        Ok((detections_vector, count))
    }
}

/// Sigmoid activation function
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Convert bounding box from center-width-height format to corner format
#[inline]
fn cxcywh_to_xyxy(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let x1 = cx - w / 2.0;
    let y1 = cy - h / 2.0;
    let x2 = cx + w / 2.0;
    let y2 = cy + h / 2.0;
    (x1, y1, x2, y2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    /// Detection struct for test verification
    struct TestDetection {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        confidence: f32,
        class_id: u16,
    }

    /// Helper to create a default RF-DETR PostProcessor for tests
    fn test_postprocessor() -> PostProcessor {
        PostProcessor {
            confidence_threshold: 0.7,
        }
    }

    /// Helper to create a default TransformParams for 512x512 RF-DETR
    fn test_transform(
        orig_width: u32,
        orig_height: u32,
        scale: f32,
        offset_x: f32,
        offset_y: f32,
    ) -> TransformParams {
        TransformParams {
            orig_width,
            orig_height,
            input_width: 512,
            input_height: 512,
            scale,
            offset_x,
            offset_y,
        }
    }

    /// Helper to run parse_detections and extract results for verification
    fn run_parse_detections(
        post_processor: &PostProcessor,
        dets: &ndarray::ArrayViewD<f32>,
        logits: &ndarray::ArrayViewD<f32>,
        transform: &TransformParams,
    ) -> anyhow::Result<Vec<TestDetection>> {
        let mut builder = FlatBufferBuilder::new();
        let (detections_vector, count) =
            post_processor.parse_detections(&mut builder, dets, logits, transform)?;

        // Build a DetectionResult to finish the buffer
        let result = schema::DetectionResult::create(
            &mut builder,
            &schema::DetectionResultArgs {
                camera_id: 0,
                frame_number: 0,
                timestamp_ns: 0,
                detections: Some(detections_vector),
                trace: None,
            },
        );
        builder.finish(result, None);

        // Read back the detections
        let buf = builder.finished_data();
        let detection_result = flatbuffers::root::<schema::DetectionResult>(buf)?;

        let mut results = Vec::with_capacity(count);
        if let Some(detections) = detection_result.detections() {
            for det in detections {
                let bbox = det.box_().unwrap();
                results.push(TestDetection {
                    x1: bbox.x1(),
                    y1: bbox.y1(),
                    x2: bbox.x2(),
                    y2: bbox.y2(),
                    confidence: det.confidence(),
                    class_id: det.class_id(),
                });
            }
        }

        Ok(results)
    }

    /// Helper to create RF-DETR format test data from logits
    /// Creates dets [1, n, 4] and logits [1, n, num_classes] arrays
    ///
    /// Note: RF-DETR uses 1-indexed classes (0=background, 1=person, 2=bicycle, ...)
    /// The postprocessor converts these to 0-indexed COCO IDs (person=0, bicycle=1, ...)
    /// So pass rfdetr_class = coco_class + 1
    fn create_rfdetr_test_data(
        boxes_cxcywh: Vec<[f32; 4]>,
        class_logits: Vec<(usize, f32)>, // (rfdetr_class_idx, logit_value) - 1-indexed!
        num_classes: usize,
    ) -> (Array<f32, IxDyn>, Array<f32, IxDyn>) {
        let n = boxes_cxcywh.len();

        // Create dets array [1, n, 4]
        let mut dets_data = Vec::with_capacity(n * 4);
        for box_coords in &boxes_cxcywh {
            dets_data.extend_from_slice(box_coords);
        }
        let dets = Array::from_shape_vec(IxDyn(&[1, n, 4]), dets_data).unwrap();

        // Create logits array [1, n, num_classes]
        // Initialize with -10.0 (very low logit -> ~0 confidence after sigmoid)
        let mut logits_data = vec![-10.0f32; n * num_classes];
        for (i, (rfdetr_class_idx, logit_value)) in class_logits.iter().enumerate() {
            logits_data[i * num_classes + rfdetr_class_idx] = *logit_value;
        }
        let logits = Array::from_shape_vec(IxDyn(&[1, n, num_classes]), logits_data).unwrap();

        (dets, logits)
    }

    /// Test sigmoid function
    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    /// Test cxcywh to xyxy conversion
    #[test]
    fn test_cxcywh_to_xyxy() {
        let (x1, y1, x2, y2) = cxcywh_to_xyxy(0.5, 0.5, 0.4, 0.2);
        assert!((x1 - 0.3).abs() < 1e-6);
        assert!((y1 - 0.4).abs() < 1e-6);
        assert!((x2 - 0.7).abs() < 1e-6);
        assert!((y2 - 0.6).abs() < 1e-6);
    }

    /// Test that confidence threshold filters detections correctly
    #[test]
    fn test_confidence_threshold_filtering() {
        // RF-DETR: confidence comes from sigmoid(max_logit)
        // sigmoid(0.62) ≈ 0.65, sigmoid(0.85) ≈ 0.7, sigmoid(1.39) ≈ 0.8
        // RF-DETR uses 1-indexed classes, output is 0-indexed COCO
        // Default threshold is 0.7

        let boxes = vec![
            [0.1, 0.1, 0.1, 0.1], // Detection 1: will have ~0.65 confidence (filtered)
            [0.2, 0.2, 0.1, 0.1], // Detection 2: will have ~0.7 confidence (boundary)
            [0.3, 0.3, 0.1, 0.1], // Detection 3: will have ~0.8 confidence
        ];

        // RF-DETR class indices (1=person, 2=bicycle, 3=car)
        let class_logits = vec![
            (1, 0.62), // sigmoid(0.62) ≈ 0.65, RF-DETR class 1 -> COCO class 0
            (2, 0.85), // sigmoid(0.85) ≈ 0.7, RF-DETR class 2 -> COCO class 1
            (3, 1.39), // sigmoid(1.39) ≈ 0.8, RF-DETR class 3 -> COCO class 2
        ];

        let (dets, logits) = create_rfdetr_test_data(boxes, class_logits, 91);

        let post_processor = test_postprocessor();
        let transform = test_transform(512, 512, 1.0, 0.0, 0.0);
        let detections =
            run_parse_detections(&post_processor, &dets.view(), &logits.view(), &transform)
                .unwrap();

        // Should have 2 detections (0.7 and 0.8), not 0.65
        assert_eq!(detections.len(), 2, "Should filter out confidence < 0.7");
        assert!(
            (detections[0].confidence - 0.7).abs() < 0.02,
            "Boundary case: 0.7 included"
        );
        assert!(detections[1].confidence > 0.75, "High confidence included");
        // Output is 0-indexed COCO: RF-DETR 2 -> COCO 1, RF-DETR 3 -> COCO 2
        assert_eq!(detections[0].class_id, 1, "Class ID should match (bicycle)");
        assert_eq!(detections[1].class_id, 2, "Class ID should match (car)");
    }

    /// Test coordinate inverse transformation with known values
    /// RF-DETR uses normalized cxcywh coordinates
    #[test]
    fn test_coordinate_inverse_transformation() {
        // Test scenario:
        // Original image: 800x600
        // Input size: 512x512
        // Scale = min(512/800, 512/600) = 0.64 (width-limited)
        // New size: 512x384
        // Offset X = 0, Offset Y = (512-384)/2 = 64

        // Box in normalized cxcywh: cx=0.5, cy=0.5, w=0.2, h=0.2
        // In xyxy normalized: (0.4, 0.4, 0.6, 0.6)
        // In 512x512 space: (204.8, 204.8, 307.2, 307.2)
        // After inverse transform:
        //   x1 = (204.8 - 0) / 0.64 = 320
        //   y1 = (204.8 - 64) / 0.64 = 220
        //   x2 = (307.2 - 0) / 0.64 = 480
        //   y2 = (307.2 - 64) / 0.64 = 380

        let boxes = vec![[0.5, 0.5, 0.2, 0.2]];
        let class_logits = vec![(1, 5.0)]; // High confidence, RF-DETR class 1 (person)
        let (dets, logits) = create_rfdetr_test_data(boxes, class_logits, 91);

        let post_processor = test_postprocessor();
        let transform = test_transform(800, 600, 0.64, 0.0, 64.0);
        let detections =
            run_parse_detections(&post_processor, &dets.view(), &logits.view(), &transform)
                .unwrap();

        assert_eq!(detections.len(), 1);
        let det = &detections[0];

        // Verify coordinate transformation (with some tolerance for float math)
        assert!(
            (det.x1 - 320.0).abs() < 0.1,
            "x1 transformation incorrect: {}",
            det.x1
        );
        assert!(
            (det.y1 - 220.0).abs() < 0.1,
            "y1 transformation incorrect: {}",
            det.y1
        );
        assert!(
            (det.x2 - 480.0).abs() < 0.1,
            "x2 transformation incorrect: {}",
            det.x2
        );
        assert!(
            (det.y2 - 380.0).abs() < 0.1,
            "y2 transformation incorrect: {}",
            det.y2
        );
    }

    /// Test that coordinates are clamped to image bounds
    #[test]
    fn test_coordinates_clamped_to_image_bounds() {
        let boxes = vec![
            [0.05, 0.05, 0.2, 0.2], // Will result in negative coords after offset
            [0.95, 0.95, 0.2, 0.2], // Will exceed bounds
            [0.5, 0.5, 0.2, 0.2],   // Normal, within bounds
        ];

        // RF-DETR class indices (1=person, 2=bicycle, 3=car)
        let class_logits = vec![(1, 5.0), (2, 5.0), (3, 5.0)];
        let (dets, logits) = create_rfdetr_test_data(boxes, class_logits, 91);

        let post_processor = test_postprocessor();
        // Use offset to push first detection into negative territory
        let transform = test_transform(400, 400, 1.0, 50.0, 50.0);
        let detections =
            run_parse_detections(&post_processor, &dets.view(), &logits.view(), &transform)
                .unwrap();

        assert_eq!(detections.len(), 3);

        // Detection 1: Should be clamped to 0 on lower bound
        assert_eq!(detections[0].x1, 0.0, "Negative x1 should be clamped to 0");
        assert_eq!(detections[0].y1, 0.0, "Negative y1 should be clamped to 0");

        // Detection 2: Should be clamped to max bounds
        assert_eq!(
            detections[1].x2, 400.0,
            "x2 exceeding width should be clamped"
        );
        assert_eq!(
            detections[1].y2, 400.0,
            "y2 exceeding height should be clamped"
        );
    }

    /// Test that no detections are returned when all are below threshold
    #[test]
    fn test_zero_detections_when_all_below_threshold() {
        let boxes = vec![
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.1, 0.1],
            [0.3, 0.3, 0.1, 0.1],
        ];

        // All logits negative -> sigmoid < 0.5
        // RF-DETR class indices (1=person, 2=bicycle, 3=car)
        let class_logits = vec![
            (1, -2.0), // sigmoid(-2) ≈ 0.12
            (2, -1.0), // sigmoid(-1) ≈ 0.27
            (3, -0.1), // sigmoid(-0.1) ≈ 0.48
        ];
        let (dets, logits) = create_rfdetr_test_data(boxes, class_logits, 91);

        let post_processor = test_postprocessor();
        let transform = test_transform(512, 512, 1.0, 0.0, 0.0);
        let detections =
            run_parse_detections(&post_processor, &dets.view(), &logits.view(), &transform)
                .unwrap();

        assert_eq!(
            detections.len(),
            0,
            "Should return no detections when all below threshold"
        );
    }

    /// Test class ID extraction via argmax
    #[test]
    fn test_class_id_argmax() {
        let boxes = vec![
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.1, 0.1],
            [0.3, 0.3, 0.1, 0.1],
            [0.4, 0.4, 0.1, 0.1],
        ];

        // RF-DETR 1-indexed class -> COCO 0-indexed class
        // RF-DETR 1 (person) -> COCO 0
        // RF-DETR 40 (bottle) -> COCO 39 (note: RF-DETR skips some indices like COCO)
        // RF-DETR 80 (toothbrush) -> COCO 79
        // RF-DETR 2 (bicycle) -> COCO 1
        let class_logits = vec![
            (1, 5.0),  // RF-DETR person (1) -> COCO person (0)
            (40, 5.0), // RF-DETR bottle (40) -> COCO bottle (39)
            (80, 5.0), // RF-DETR toothbrush (80) -> COCO toothbrush (79)
            (2, 5.0),  // RF-DETR bicycle (2) -> COCO bicycle (1)
        ];
        let (dets, logits) = create_rfdetr_test_data(boxes, class_logits, 91);

        let post_processor = test_postprocessor();
        let transform = test_transform(512, 512, 1.0, 0.0, 0.0);
        let detections =
            run_parse_detections(&post_processor, &dets.view(), &logits.view(), &transform)
                .unwrap();

        assert_eq!(detections.len(), 4);

        // Verify COCO class IDs are correctly extracted via argmax
        // Output is RF-DETR index - 1
        assert_eq!(detections[0].class_id, 0, "Person class (0)");
        assert_eq!(detections[1].class_id, 39, "Bottle class (39)");
        assert_eq!(detections[2].class_id, 79, "Toothbrush class (79)");
        assert_eq!(detections[3].class_id, 1, "Bicycle class (1)");
    }

    /// Test edge case: Empty detections (0 queries)
    #[test]
    fn test_empty_input() {
        let dets = Array::from_shape_vec(IxDyn(&[1, 0, 4]), vec![]).unwrap();
        let logits = Array::from_shape_vec(IxDyn(&[1, 0, 91]), vec![]).unwrap();

        let post_processor = test_postprocessor();
        let transform = test_transform(512, 512, 1.0, 0.0, 0.0);
        let detections =
            run_parse_detections(&post_processor, &dets.view(), &logits.view(), &transform)
                .unwrap();

        assert_eq!(
            detections.len(),
            0,
            "Empty input should return no detections"
        );
    }

    /// Test realistic RF-DETR scenario with mixed confidences
    #[test]
    fn test_realistic_rfdetr_output() {
        // RF-DETR outputs 300 queries, but most have low confidence
        // Simulate 300 queries with only 3 high-confidence detections

        let num_queries = 300;
        let num_classes = 91;

        // Create dets [1, 300, 4]
        let mut dets_data = vec![0.0f32; num_queries * 4];
        // Detection 1: Person at top-left (normalized cxcywh)
        dets_data[0..4].copy_from_slice(&[0.2, 0.3, 0.2, 0.4]);
        // Detection 2: Dog in center
        dets_data[4..8].copy_from_slice(&[0.5, 0.5, 0.3, 0.3]);
        // Detection 3: Car at bottom-right
        dets_data[8..12].copy_from_slice(&[0.8, 0.8, 0.3, 0.3]);

        let dets = Array::from_shape_vec(IxDyn(&[1, num_queries, 4]), dets_data).unwrap();

        // Create logits [1, 300, 91]
        // Initialize with -10.0 (very low confidence)
        // RF-DETR uses 1-indexed classes: 1=person, 17=dog, 3=car
        // Output will be 0-indexed COCO: 0=person, 16=dog, 2=car
        let mut logits_data = vec![-10.0f32; num_queries * num_classes];
        // Set high logits for 3 detections (using RF-DETR 1-indexed classes)
        // Index = query_idx * num_classes + class_idx
        logits_data[1] = 5.0; // Query 0, RF-DETR Person (1) -> COCO (0), high confidence
        logits_data[num_classes + 17] = 3.5; // Query 1, RF-DETR Dog (17) -> COCO (16), medium-high confidence
        logits_data[2 * num_classes + 3] = 2.5; // Query 2, RF-DETR Car (3) -> COCO (2), medium confidence

        let logits =
            Array::from_shape_vec(IxDyn(&[1, num_queries, num_classes]), logits_data).unwrap();

        let post_processor = test_postprocessor();
        let transform = test_transform(512, 512, 1.0, 0.0, 0.0);
        let detections =
            run_parse_detections(&post_processor, &dets.view(), &logits.view(), &transform)
                .unwrap();

        // Only 3 detections should pass threshold
        assert_eq!(
            detections.len(),
            3,
            "Should filter 300 queries to 3 detections"
        );

        // Verify detections are in order (output is 0-indexed COCO)
        assert_eq!(detections[0].class_id, 0, "First detection: person");
        assert_eq!(detections[1].class_id, 16, "Second detection: dog");
        assert_eq!(detections[2].class_id, 2, "Third detection: car");

        // Verify confidences are reasonable
        assert!(
            detections[0].confidence > 0.99,
            "Person should have very high confidence"
        );
        assert!(
            detections[1].confidence > 0.95,
            "Dog should have high confidence"
        );
        assert!(
            detections[2].confidence > 0.90,
            "Car should have good confidence"
        );
    }
}
