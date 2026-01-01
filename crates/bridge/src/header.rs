use std::sync::atomic::AtomicU64;

/// SAFETY & MEMORY ORDERING:
///
/// This header defines the shared memory layout for mmap IPC.
///
/// Writer protocol:
/// 1. Write payload bytes to the data region
/// 2. Publish sequence with `Ordering::Release`
///
/// Reader protocol:
/// 1. Load sequence with `Ordering::Acquire`
/// 2. If sequence changed, payload is guaranteed visible
///
/// The Release-Acquire pair ensures:
/// - All payload writes happen-before the sequence store
/// - All sequence loads happen-before payload reads
/// - No torn reads on x86, ARM, or other architectures
///
/// Alignment:
/// The `#[repr(C, align(8))]` ensures AtomicU64 is always 8-byte aligned,
/// which is required for atomic operations. This prevents UB even if the
/// mmap offset changes.
#[repr(C, align(8))]
pub struct Header {
    /// Monotonically increasing sequence number.
    /// Starts at 0, increments on each write.
    /// 0 means "no data written yet"
    pub sequence: AtomicU64,
}

impl Header {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_alignment() {
        assert_eq!(
            std::mem::align_of::<Header>(),
            8,
            "Header must be 8-byte aligned for AtomicU64"
        );
    }

    #[test]
    fn test_header_size() {
        assert_eq!(
            Header::SIZE,
            8,
            "Header should be exactly 8 bytes (just the sequence)"
        );
    }
}
