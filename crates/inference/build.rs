use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_FEATURE_TRT_BACKEND").is_ok() {
        // C++ source files
        let cpp_root = PathBuf::from("../inference-cpp");
        let src_dir = cpp_root.join("src");
        let include_dir = cpp_root.join("include");

        // CUDA and TensorRT paths (configurable via env, with defaults matching CMakeLists.txt)
        let cuda_root = std::env::var("CUDA_ROOT").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let trt_root = std::env::var("TENSORRT_ROOT").unwrap_or_else(|_| "/usr/local/tensorrt".to_string());

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

                // Build the C++ bridge
                cxx_build::bridge("src/backend/trt.rs")            .file(src_dir.join("tensorrt_backend.cpp"))
            .include(&include_dir)
            .include(&cuda_include)
            .include(&trt_include)
            .flag_if_supported("-std=c++17")
            // Suppress unused parameter warnings that might arise from the bridge
            .flag_if_supported("-Wno-unused-parameter")
            .compile("inference-trt");

        // Link libraries
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-search=native={}", trt_lib.display());

        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=nvinfer");
        // stdc++ is usually linked automatically by cxx/cc, but good to be safe if not
        // println!("cargo:rustc-link-lib=stdc++");

        // Rerun if C++ files change
        println!("cargo:rerun-if-changed={}", src_dir.join("tensorrt_backend.cpp").display());
        println!("cargo:rerun-if-changed={}", include_dir.join("tensorrt_backend.hpp").display());
        println!("cargo:rerun-if-changed=src/backend/trt.rs");
    }
}
