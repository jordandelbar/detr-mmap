/// Safely verify and access flatbuffers data with bounds checking
///
/// This function ensures that:
/// 1. The buffer is large enough to contain FlatBuffers data (minimum 8 bytes)
/// 2. The FlatBuffers schema is valid and can be accessed
///
/// # Errors
///
/// Returns an error if:
/// - Buffer is too small (< 8 bytes)
/// - FlatBuffers verification fails
pub(crate) fn safe_flatbuffers_root<'a, T>(buffer: &'a [u8]) -> anyhow::Result<T::Inner>
where
    T: flatbuffers::Follow<'a> + flatbuffers::Verifiable + 'a,
{
    // Check minimum buffer size
    if buffer.len() < 8 {
        return Err(anyhow::anyhow!(
            "Buffer too small for flatbuffers: {} bytes",
            buffer.len()
        ));
    }

    // Attempt to get the root with proper error handling
    match flatbuffers::root::<T>(buffer) {
        Ok(root) => Ok(root),
        Err(e) => Err(anyhow::anyhow!(
            "Flatbuffers deserialization failed: {:?}, buffer size: {}",
            e,
            buffer.len()
        )),
    }
}
