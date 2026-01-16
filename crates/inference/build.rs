use std::path::PathBuf;

fn main() {
    let build_trt = std::env::var("CARGO_FEATURE_TRT_BACKEND").is_ok();

    if build_trt {
        // C++ source files
        let cpp_root = PathBuf::from("../inference-cpp");
        let src_dir = cpp_root.join("src");
        let include_dir = cpp_root.join("include");

        let cuda_root =
            std::env::var("CUDA_ROOT").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let trt_root =
            std::env::var("TENSORRT_ROOT").unwrap_or_else(|_| "/usr/local/tensorrt".to_string());

        let cuda_include = PathBuf::from(&cuda_root).join("include");
        let mut cuda_lib = PathBuf::from(&cuda_root).join("lib64");
        if !cuda_lib.exists() {
            cuda_lib = PathBuf::from(&cuda_root).join("lib/x86_64-linux-gnu");
        }
        if !cuda_lib.exists() {
            cuda_lib = PathBuf::from(&cuda_root).join("lib");
        }

        let trt_include = PathBuf::from(&trt_root).join("include");
        let mut trt_lib = PathBuf::from(&trt_root).join("lib");
        if !trt_lib.exists() {
            trt_lib = PathBuf::from(&trt_root).join("lib/x86_64-linux-gnu");
        }
        if !trt_lib.exists() {
            // Fallback for some installations
            trt_lib = PathBuf::from(&trt_root).join("lib64");
        }

        // Build RF-DETR TensorRT backend
        cxx_build::bridge("src/backend/trt.rs")
            .file(src_dir.join("rfdetr_backend.cpp"))
            .file(src_dir.join("logging.cpp"))
            .include(&include_dir)
            .include(&cuda_include)
            .include(&trt_include)
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-Wno-unused-parameter")
            .compile("inference-trt");

        println!(
            "cargo:rerun-if-changed={}",
            src_dir.join("rfdetr_backend.cpp").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            include_dir.join("rfdetr_backend.hpp").display()
        );
        println!("cargo:rerun-if-changed=src/backend/trt.rs");

        // Link libraries
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-search=native={}", trt_lib.display());

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=nvinfer");
    }
}
