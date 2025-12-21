#[allow(unused_imports, dead_code, clippy::all, unsafe_op_in_unsafe_fn)]
mod frame_generated;

#[allow(
    unused_imports,
    dead_code,
    clippy::all,
    unsafe_op_in_unsafe_fn,
    mismatched_lifetime_syntaxes
)]
mod detection_generated;

pub use detection_generated::bridge::schema::*;
pub use frame_generated::bridge::schema::*;
