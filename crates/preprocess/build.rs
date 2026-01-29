#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::process::Command;

fn main() {
    // Only compile CUDA kernel when cuda feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda();

    // Always rerun if the CUDA source changes
    println!("cargo:rerun-if-changed=cuda/preprocess.cu");
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_file = PathBuf::from("cuda/preprocess.cu");
    let ptx_file = out_dir.join("preprocess.ptx");

    // Find nvcc
    let nvcc = find_nvcc().expect("nvcc not found. Please install CUDA toolkit.");

    // Compile CUDA to PTX
    let status = Command::new(&nvcc)
        .args([
            "--ptx",
            "-o",
            ptx_file.to_str().unwrap(),
            cuda_file.to_str().unwrap(),
            // Optimize for common GPU architectures
            "-arch=sm_50",
            "--generate-line-info",
            "-O3",
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed to compile CUDA kernel");
    }

    println!("cargo:rerun-if-changed=cuda/preprocess.cu");
}

#[cfg(feature = "cuda")]
fn find_nvcc() -> Option<PathBuf> {
    // Try CUDA_PATH environment variable first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // Try common CUDA installation paths
    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];

    for path in &common_paths {
        let nvcc = PathBuf::from(path);
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // Try to find nvcc in PATH
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some(PathBuf::from("nvcc"));
    }

    None
}
