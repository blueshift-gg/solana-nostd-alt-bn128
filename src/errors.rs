//! Error type for the alt_bn128 group and compression operations.

/// Re-export of the [`solana_program_error`] type that [`AltBn128Error`]
/// converts into, so callers can name it without an extra dependency.
pub use solana_program_error::ProgramError;

/// An alt_bn128 operation failure. Converts to
/// [`ProgramError::InvalidArgument`] so results are `?`-propagatable from a
/// program entrypoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AltBn128Error {
    /// A group operation (add / sub / mul / pairing) syscall failed, or an
    /// operand failed to deserialize on the host reference path.
    GroupError,
    /// A point (de)compression syscall failed, or an operand failed to
    /// deserialize on the host reference path.
    CompressionError,
}

impl core::fmt::Display for AltBn128Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            AltBn128Error::GroupError => "alt_bn128 group operation failed",
            AltBn128Error::CompressionError => "alt_bn128 (de)compression failed",
        };
        f.write_str(s)
    }
}

impl From<AltBn128Error> for ProgramError {
    fn from(_: AltBn128Error) -> Self {
        ProgramError::InvalidArgument
    }
}
