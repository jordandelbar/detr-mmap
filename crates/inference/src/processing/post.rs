use bridge::types::Detection;

pub struct TransformParams {
    pub orig_width: u32,
    pub orig_height: u32,
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

    pub fn parse_detections(
        &self,
        labels: &ndarray::ArrayViewD<i64>,
        boxes: &ndarray::ArrayViewD<f32>,
        scores: &ndarray::ArrayViewD<f32>,
        transform: &TransformParams,
    ) -> anyhow::Result<Vec<Detection>> {
        let mut detections = Vec::new();

        let num_queries = labels.shape()[1];

        for i in 0..num_queries {
            let class_id = labels[[0, i]];
            let confidence = scores[[0, i]];

            if confidence < self.confidence_threshold {
                continue;
            }

            let x1 = ((boxes[[0, i, 0]] - transform.offset_x) / transform.scale)
                .max(0.0)
                .min(transform.orig_width as f32);
            let y1 = ((boxes[[0, i, 1]] - transform.offset_y) / transform.scale)
                .max(0.0)
                .min(transform.orig_height as f32);
            let x2 = ((boxes[[0, i, 2]] - transform.offset_x) / transform.scale)
                .max(0.0)
                .min(transform.orig_width as f32);
            let y2 = ((boxes[[0, i, 3]] - transform.offset_y) / transform.scale)
                .max(0.0)
                .min(transform.orig_height as f32);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    /// Test that confidence threshold filters detections correctly
    /// This is critical: 0.5 is the boundary, must be exact
    #[test]
    fn test_confidence_threshold_filtering() {
        // Create test data with 3 detections at different confidence levels
        let labels = Array::from_shape_vec(IxDyn(&[1, 3]), vec![0, 1, 2]).unwrap();
        let boxes = Array::from_shape_vec(
            IxDyn(&[1, 3, 4]),
            vec![
                // Detection 1: confidence 0.49 (should be filtered out)
                10.0, 10.0, 50.0, 50.0,
                // Detection 2: confidence 0.5 (should be included - boundary case)
                20.0, 20.0, 60.0, 60.0,
                // Detection 3: confidence 0.8 (should be included)
                30.0, 30.0, 70.0, 70.0,
            ],
        )
        .unwrap();
        let scores = Array::from_shape_vec(IxDyn(&[1, 3]), vec![0.49, 0.5, 0.8]).unwrap();

        let post_processor = PostProcessor {
            confidence_threshold: 0.5,
        };
        let transform = TransformParams {
            orig_width: 100,
            orig_height: 100,
            scale: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let detections = post_processor
            .parse_detections(&labels.view(), &boxes.view(), &scores.view(), &transform)
            .unwrap();

        // Should have 2 detections (0.5 and 0.8), not 0.49
        assert_eq!(detections.len(), 2, "Should filter out confidence < 0.5");
        assert_eq!(detections[0].confidence, 0.5, "Boundary case: 0.5 included");
        assert_eq!(detections[1].confidence, 0.8, "High confidence included");
        assert_eq!(detections[0].class_id, 1, "Class ID should match");
        assert_eq!(detections[1].class_id, 2, "Class ID should match");
    }

    /// Test coordinate inverse transformation with known values
    /// Formula: original_coord = (model_coord - offset) / scale
    #[test]
    fn test_coordinate_inverse_transformation() {
        // Test scenario:
        // Original image: 800x600
        // Letterboxed to: 640x640
        // Scale = 640/800 = 0.8
        // Offset X = (640 - 800*0.8) / 2 = 0
        // Offset Y = (640 - 600*0.8) / 2 = 80
        //
        // Model outputs box at: (100, 150, 200, 250) in 640x640 space
        // Original coords should be:
        //   x1 = (100 - 0) / 0.8 = 125
        //   y1 = (150 - 80) / 0.8 = 87.5
        //   x2 = (200 - 0) / 0.8 = 250
        //   y2 = (250 - 80) / 0.8 = 212.5

        let labels = Array::from_shape_vec(IxDyn(&[1, 1]), vec![0]).unwrap();
        let boxes =
            Array::from_shape_vec(IxDyn(&[1, 1, 4]), vec![100.0, 150.0, 200.0, 250.0]).unwrap();
        let scores = Array::from_shape_vec(IxDyn(&[1, 1]), vec![0.9]).unwrap();

        let post_processor = PostProcessor {
            confidence_threshold: 0.5,
        };
        let transform = TransformParams {
            orig_width: 800,
            orig_height: 600,
            scale: 0.8,
            offset_x: 0.0,
            offset_y: 80.0,
        };
        let detections = post_processor
            .parse_detections(&labels.view(), &boxes.view(), &scores.view(), &transform)
            .unwrap();

        assert_eq!(detections.len(), 1);
        let det = &detections[0];

        // Verify coordinate transformation
        assert_eq!(det.x1, 125.0, "x1 transformation incorrect");
        assert_eq!(det.y1, 87.5, "y1 transformation incorrect");
        assert_eq!(det.x2, 250.0, "x2 transformation incorrect");
        assert_eq!(det.y2, 212.5, "y2 transformation incorrect");
    }

    /// Test that coordinates are clamped to image bounds
    /// Prevents boxes from extending outside the image
    #[test]
    fn test_coordinates_clamped_to_image_bounds() {
        let labels = Array::from_shape_vec(IxDyn(&[1, 3]), vec![0, 1, 2]).unwrap();
        let boxes = Array::from_shape_vec(
            IxDyn(&[1, 3, 4]),
            vec![
                // Detection 1: Negative after transformation (offset > coord)
                5.0, 5.0, 50.0, 50.0,
                // Detection 2: Exceeds bounds after transformation
                500.0, 500.0, 800.0, 800.0, // Detection 3: Normal, within bounds
                100.0, 100.0, 200.0, 200.0,
            ],
        )
        .unwrap();
        let scores = Array::from_shape_vec(IxDyn(&[1, 3]), vec![0.9, 0.9, 0.9]).unwrap();
        let post_processor = PostProcessor {
            confidence_threshold: 0.5,
        };
        let transform = TransformParams {
            orig_width: 640,
            orig_height: 480,
            scale: 1.0,
            offset_x: 50.0,
            offset_y: 50.0,
        };
        let detections = post_processor
            .parse_detections(&labels.view(), &boxes.view(), &scores.view(), &transform)
            .unwrap();

        assert_eq!(detections.len(), 3);

        // Detection 1: Should be clamped to 0 on lower bound
        assert_eq!(detections[0].x1, 0.0, "Negative x1 should be clamped to 0");
        assert_eq!(detections[0].y1, 0.0, "Negative y1 should be clamped to 0");

        // Detection 2: Should be clamped to max bounds
        assert_eq!(
            detections[1].x2, 640.0,
            "x2 exceeding width should be clamped"
        );
        assert_eq!(
            detections[1].y2, 480.0,
            "y2 exceeding height should be clamped"
        );

        // Detection 3: Should be unchanged (within bounds)
        assert_eq!(detections[2].x1, 50.0);
        assert_eq!(detections[2].y1, 50.0);
    }

    /// Test that no detections are returned when all are below threshold
    #[test]
    fn test_zero_detections_when_all_below_threshold() {
        let labels = Array::from_shape_vec(IxDyn(&[1, 3]), vec![0, 1, 2]).unwrap();
        let boxes = Array::from_shape_vec(
            IxDyn(&[1, 3, 4]),
            vec![
                10.0, 10.0, 50.0, 50.0, 20.0, 20.0, 60.0, 60.0, 30.0, 30.0, 70.0, 70.0,
            ],
        )
        .unwrap();
        // All confidences below threshold
        let scores = Array::from_shape_vec(IxDyn(&[1, 3]), vec![0.1, 0.3, 0.49]).unwrap();

        let post_processor = PostProcessor {
            confidence_threshold: 0.5,
        };
        let transform = TransformParams {
            orig_width: 640,
            orig_height: 480,
            scale: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let detections = post_processor
            .parse_detections(&labels.view(), &boxes.view(), &scores.view(), &transform)
            .unwrap();

        assert_eq!(
            detections.len(),
            0,
            "Should return no detections when all below threshold"
        );
    }

    /// Test class ID conversion from i64 to u32
    /// Verifies that COCO class IDs are correctly converted
    #[test]
    fn test_class_id_conversion_from_i64_to_u32() {
        let labels = Array::from_shape_vec(IxDyn(&[1, 4]), vec![0, 39, 79, 1]).unwrap();
        let boxes = Array::from_shape_vec(
            IxDyn(&[1, 4, 4]),
            vec![
                10.0, 10.0, 50.0, 50.0, 20.0, 20.0, 60.0, 60.0, 30.0, 30.0, 70.0, 70.0, 40.0, 40.0,
                80.0, 80.0,
            ],
        )
        .unwrap();
        let scores = Array::from_shape_vec(IxDyn(&[1, 4]), vec![0.9, 0.8, 0.7, 0.95]).unwrap();

        let post_processor = PostProcessor {
            confidence_threshold: 0.5,
        };
        let transform = TransformParams {
            orig_width: 640,
            orig_height: 480,
            scale: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let detections = post_processor
            .parse_detections(&labels.view(), &boxes.view(), &scores.view(), &transform)
            .unwrap();

        assert_eq!(detections.len(), 4);

        // Verify COCO class IDs are correctly converted
        assert_eq!(detections[0].class_id, 0, "Person class (0)");
        assert_eq!(detections[1].class_id, 39, "Bottle class (39)");
        assert_eq!(detections[2].class_id, 79, "Last COCO class (79)");
        assert_eq!(detections[3].class_id, 1, "Bicycle class (1)");
    }

    /// Test edge case: Empty detections (0 queries)
    #[test]
    fn test_empty_input() {
        let labels = Array::from_shape_vec(IxDyn(&[1, 0]), vec![]).unwrap();
        let boxes = Array::from_shape_vec(IxDyn(&[1, 0, 4]), vec![]).unwrap();
        let scores = Array::from_shape_vec(IxDyn(&[1, 0]), vec![]).unwrap();

        let post_processor = PostProcessor {
            confidence_threshold: 0.5,
        };
        let transform = TransformParams {
            orig_width: 640,
            orig_height: 480,
            scale: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let detections = post_processor
            .parse_detections(&labels.view(), &boxes.view(), &scores.view(), &transform)
            .unwrap();

        assert_eq!(
            detections.len(),
            0,
            "Empty input should return no detections"
        );
    }

    /// Test realistic RT-DETR scenario with mixed confidences
    #[test]
    fn test_realistic_rtdetr_output() {
        // RT-DETR outputs 300 queries, but most have low confidence
        // Simulate 300 queries with only 3 high-confidence detections

        let mut label_data = vec![0i64; 300];
        label_data[0] = 0; // person
        label_data[1] = 16; // dog
        label_data[2] = 2; // car

        let mut box_data = vec![0.0f32; 300 * 4];
        // Detection 1: Person at top-left
        box_data[0..4].copy_from_slice(&[50.0, 50.0, 150.0, 250.0]);
        // Detection 2: Dog in center
        box_data[4..8].copy_from_slice(&[200.0, 200.0, 350.0, 400.0]);
        // Detection 3: Car at bottom-right
        box_data[8..12].copy_from_slice(&[400.0, 400.0, 600.0, 600.0]);

        let mut score_data = vec![0.01f32; 300]; // Low confidence for most
        score_data[0] = 0.95;
        score_data[1] = 0.87;
        score_data[2] = 0.76;

        let labels = Array::from_shape_vec(IxDyn(&[1, 300]), label_data).unwrap();
        let boxes = Array::from_shape_vec(IxDyn(&[1, 300, 4]), box_data).unwrap();
        let scores = Array::from_shape_vec(IxDyn(&[1, 300]), score_data).unwrap();

        let post_processor = PostProcessor {
            confidence_threshold: 0.5,
        };
        let transform = TransformParams {
            orig_width: 640,
            orig_height: 640,
            scale: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        };
        let detections = post_processor
            .parse_detections(&labels.view(), &boxes.view(), &scores.view(), &transform)
            .unwrap();

        // Only 3 detections should pass threshold
        assert_eq!(
            detections.len(),
            3,
            "Should filter 300 queries to 3 detections"
        );

        // Verify detections are in order
        assert_eq!(detections[0].class_id, 0, "First detection: person");
        assert_eq!(detections[1].class_id, 16, "Second detection: dog");
        assert_eq!(detections[2].class_id, 2, "Third detection: car");

        // Verify bounding boxes are correct
        assert_eq!(detections[0].x1, 50.0);
        assert_eq!(detections[0].y1, 50.0);
        assert_eq!(detections[0].x2, 150.0);
        assert_eq!(detections[0].y2, 250.0);
    }
}
