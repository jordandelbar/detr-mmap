/**
 * GPU Preprocessing Kernel for RF-DETR
 *
 * Performs fused operations:
 * 1. Bilinear resize
 * 2. Letterbox padding (gray 114)
 * 3. ImageNet normalization
 * 4. HWC -> CHW transpose
 *
 * Input:  RGB u8 image in HWC format [H, W, 3]
 * Output: Normalized f32 image in CHW format [3, target_H, target_W]
 */

extern "C" __global__ void preprocess_kernel(
    const unsigned char* __restrict__ input,  // Input RGB image [src_h, src_w, 3]
    float* __restrict__ output,               // Output CHW image [3, dst_h, dst_w]
    int src_w,                                // Source image width
    int src_h,                                // Source image height
    int dst_w,                                // Target/output width
    int dst_h,                                // Target/output height
    int resized_w,                            // Width after resize (before padding)
    int resized_h,                            // Height after resize (before padding)
    int offset_x,                             // X offset for letterbox centering
    int offset_y,                             // Y offset for letterbox centering
    float scale,                              // Scale factor applied during resize
    float mean_r,                             // ImageNet mean for R channel
    float mean_g,                             // ImageNet mean for G channel
    float mean_b,                             // ImageNet mean for B channel
    float std_r,                              // ImageNet std for R channel
    float std_g,                              // ImageNet std for G channel
    float std_b,                              // ImageNet std for B channel
    int letterbox_gray                        // Letterbox padding color (114)
) {
    // Each thread handles one output pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = dst_w * dst_h;

    if (idx >= total_pixels) return;

    // Convert linear index to 2D coordinates in output space
    int out_y = idx / dst_w;
    int out_x = idx % dst_w;

    float r, g, b;

    // Check if this pixel is in the letterbox padding region
    bool in_padding = (out_x < offset_x || out_x >= offset_x + resized_w ||
                       out_y < offset_y || out_y >= offset_y + resized_h);

    if (in_padding) {
        // Use letterbox gray value (normalized to 0-1)
        float gray_norm = letterbox_gray / 255.0f;
        r = gray_norm;
        g = gray_norm;
        b = gray_norm;
    } else {
        // Map output coordinates to source image coordinates (bilinear interpolation)
        // First, get position in resized image (without offset)
        float resized_x = (float)(out_x - offset_x);
        float resized_y = (float)(out_y - offset_y);

        // Map to source image coordinates
        float src_x = resized_x / scale;
        float src_y = resized_y / scale;

        // Bilinear interpolation
        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);
        x0 = max(0, min(x0, src_w - 1));
        y0 = max(0, min(y0, src_h - 1));

        float dx = src_x - (float)x0;
        float dy = src_y - (float)y0;

        // Sample 4 corners (input is HWC format)
        int idx00 = (y0 * src_w + x0) * 3;
        int idx01 = (y0 * src_w + x1) * 3;
        int idx10 = (y1 * src_w + x0) * 3;
        int idx11 = (y1 * src_w + x1) * 3;

        // Bilinear interpolation for each channel
        float r00 = input[idx00 + 0];
        float r01 = input[idx01 + 0];
        float r10 = input[idx10 + 0];
        float r11 = input[idx11 + 0];

        float g00 = input[idx00 + 1];
        float g01 = input[idx01 + 1];
        float g10 = input[idx10 + 1];
        float g11 = input[idx11 + 1];

        float b00 = input[idx00 + 2];
        float b01 = input[idx01 + 2];
        float b10 = input[idx10 + 2];
        float b11 = input[idx11 + 2];

        // Interpolate
        float w00 = (1.0f - dx) * (1.0f - dy);
        float w01 = dx * (1.0f - dy);
        float w10 = (1.0f - dx) * dy;
        float w11 = dx * dy;

        r = (r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11) / 255.0f;
        g = (g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11) / 255.0f;
        b = (b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11) / 255.0f;
    }

    // Apply ImageNet normalization
    r = (r - mean_r) / std_r;
    g = (g - mean_g) / std_g;
    b = (b - mean_b) / std_b;

    // Write to output in CHW format
    // Channel 0 (R): offset 0
    // Channel 1 (G): offset total_pixels
    // Channel 2 (B): offset 2 * total_pixels
    output[idx] = r;
    output[idx + total_pixels] = g;
    output[idx + 2 * total_pixels] = b;
}
