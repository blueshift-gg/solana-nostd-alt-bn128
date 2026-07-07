//! BN254 point (de)compression over the little-endian SVM wire format.
//!
//! Compression is done **manually** — pure byte re-encoding of the
//! x-coordinate plus a sign flag, no syscall — which is ~4x cheaper on-chain
//! than `sol_alt_bn128_compression` (whose per-call overhead dwarfs the work).
//! Decompression needs a field square root, so it stays on the syscall
//! (Arkworks off-chain). Both are covered by the on-chain `correctness` test
//! and host unit tests, including a byte-for-byte match against Arkworks'
//! `serialize_compressed` (point, negation, and infinity).

#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
use ark_bn254::{G1Affine, G2Affine};
#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{AltBn128Error, G1CompressedPoint, G1Point, G2CompressedPoint, G2Point};

// Compression operation selectors, LE-flagged (`| 0x80`) to match the
// little-endian wire format this crate uses end-to-end. Without the flag the
// syscall interprets operands as big-endian (EIP-197) and produces garbage.

/// G1 point compression (little-endian operand).
pub const ALT_BN128_G1_COMPRESS: u64 = 0x80;
/// G1 point decompression (little-endian operand).
pub const ALT_BN128_G1_DECOMPRESS: u64 = 0x01 | 0x80;
/// G2 point compression (little-endian operand).
pub const ALT_BN128_G2_COMPRESS: u64 = 0x02 | 0x80;
/// G2 point decompression (little-endian operand).
pub const ALT_BN128_G2_DECOMPRESS: u64 = 0x03 | 0x80;

#[cfg(all(
    any(target_os = "solana", target_arch = "bpf"),
    any(target_feature = "static-syscalls", feature = "static-syscalls")
))]
#[inline]
unsafe fn sol_alt_bn128_compression(
    op: u64,
    input: *const u8,
    input_size: u64,
    result: *mut u8,
) -> u64 {
    let f: unsafe extern "C" fn(u64, *const u8, u64, *mut u8) -> u64 = unsafe {
        core::mem::transmute(0x334fd5ed_usize) // murmur3_32("sol_alt_bn128_compression")
    };
    f(op, input, input_size, result)
}

#[cfg(all(
    any(target_os = "solana", target_arch = "bpf"),
    not(any(target_feature = "static-syscalls", feature = "static-syscalls"))
))]
unsafe extern "C" {
    fn sol_alt_bn128_compression(
        op: u64,
        input: *const u8,
        input_size: u64,
        result: *mut u8,
    ) -> u64;
}

/// `(q - 1) / 2`, little-endian. A coordinate is tagged "negative" when it
/// exceeds this — i.e. it is the larger of the two roots (`y > -y`) — matching
/// Arkworks' compressed `SWFlags` convention.
const HALF_Q_MINUS_1: [u8; 32] = [
    0xa3, 0x7e, 0x3e, 0x6c, 0x0b, 0x46, 0x10, 0x9e, 0x46, 0xe5, 0x38, 0xb4, 0x48, 0xb5, 0xc0, 0xcb,
    0x2e, 0xac, 0xc0, 0x40, 0xdb, 0x22, 0x28, 0xdc, 0x14, 0xd0, 0x98, 0x70, 0x39, 0x27, 0x32, 0x18,
];

/// `true` iff the 32-byte little-endian `Fq` element is `> (q - 1) / 2`.
const fn fq_gt_half(limb: &[u8]) -> bool {
    let mut i = 32;
    while i > 0 {
        i -= 1;
        if limb[i] != HALF_Q_MINUS_1[i] {
            return limb[i] > HALF_Q_MINUS_1[i];
        }
    }
    false
}

/// `true` iff every byte is zero. Inputs are always a multiple of 8 bytes, so
/// scan a `u64` at a time (8x fewer loads than byte-by-byte) and compare to 0.
fn is_all_zero(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(8)
        .all(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()) == 0)
}

/// Compress a G1 point to its 32-byte form: the x-coordinate, with the y-sign
/// flag in the high byte (`0x80` when `y > -y`, `0x40` for the point at
/// infinity), matching Arkworks' compressed `SWFlags` encoding.
///
/// This is pure byte arithmetic — no syscall — and ~7× cheaper on-chain than
/// `sol_alt_bn128_compression`. Like that syscall, it does *not* check the
/// input is on-curve; it re-encodes whatever bytes it is given.
pub fn g1_compress(g1: &G1Point) -> G1CompressedPoint {
    let p = &g1.0;
    if is_all_zero(p) {
        let mut out = [0u8; 32];
        out[31] = 0x40; // PointAtInfinity
        return out;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&p[..32]); // x
    if fq_gt_half(&p[32..64]) {
        out[31] |= 0x80; // YIsNegative
    }
    out
}

/// Decompress a 32-byte G1 point back to its 64-byte uncompressed form.
pub fn g1_decompress(g1: &G1CompressedPoint) -> Result<G1Point, AltBn128Error> {
    #[cfg(any(target_os = "solana", target_arch = "bpf"))]
    {
        let mut result = [0u8; 64];
        let status = unsafe {
            sol_alt_bn128_compression(
                ALT_BN128_G1_DECOMPRESS,
                g1.as_ptr(),
                g1.len() as u64,
                result.as_mut_ptr(),
            )
        };
        match status {
            0 => Ok(G1Point(result)),
            _ => Err(AltBn128Error::CompressionError),
        }
    }
    #[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
    {
        // `Validate::No` (via `_unchecked`) matches the syscall: decompression
        // reconstructs an on-curve point but does NOT subgroup-check.
        let point = G1Affine::deserialize_compressed_unchecked(&g1[..])
            .map_err(|_| AltBn128Error::CompressionError)?;
        let mut result = [0u8; 64];
        point
            .x
            .serialize_compressed(&mut result[..32])
            .map_err(|_| AltBn128Error::CompressionError)?;
        point
            .y
            .serialize_compressed(&mut result[32..])
            .map_err(|_| AltBn128Error::CompressionError)?;
        Ok(G1Point(result))
    }
}

/// Compress a G2 point to its 64-byte form: the `Fq2` x-coordinate, with the
/// y-sign flag in the high byte. The `Fq2` sign follows Arkworks' ordering —
/// compare `c1` first, then `c0` — so the flag is set from `y.c1` unless it is
/// zero, in which case `y.c0` decides. Pure byte arithmetic; same caveats as
/// [`g1_compress`].
pub fn g2_compress(g2: &G2Point) -> G2CompressedPoint {
    let p = &g2.0;
    if is_all_zero(p) {
        let mut out = [0u8; 64];
        out[63] = 0x40; // PointAtInfinity
        return out;
    }
    let mut out = [0u8; 64];
    out.copy_from_slice(&p[..64]); // x.c0 || x.c1
    // y = (c0 = [64..96], c1 = [96..128]); c1 is the leading coefficient.
    let negative = if is_all_zero(&p[96..128]) {
        fq_gt_half(&p[64..96])
    } else {
        fq_gt_half(&p[96..128])
    };
    if negative {
        out[63] |= 0x80; // YIsNegative
    }
    out
}

/// Decompress a 64-byte G2 point back to its 128-byte uncompressed form.
pub fn g2_decompress(g2: &G2CompressedPoint) -> Result<G2Point, AltBn128Error> {
    #[cfg(any(target_os = "solana", target_arch = "bpf"))]
    {
        let mut result = [0u8; 128];
        let status = unsafe {
            sol_alt_bn128_compression(
                ALT_BN128_G2_DECOMPRESS,
                g2.as_ptr(),
                g2.len() as u64,
                result.as_mut_ptr(),
            )
        };
        match status {
            0 => Ok(G2Point(result)),
            _ => Err(AltBn128Error::CompressionError),
        }
    }
    #[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
    {
        // `Validate::No` (via `_unchecked`) matches the syscall: decompression
        // reconstructs an on-curve point but does NOT subgroup-check, so an
        // on-curve non-subgroup G2 point round-trips on host exactly as on-chain.
        let point = G2Affine::deserialize_compressed_unchecked(&g2[..])
            .map_err(|_| AltBn128Error::CompressionError)?;
        let mut result = [0u8; 128];
        point
            .x
            .serialize_compressed(&mut result[..64])
            .map_err(|_| AltBn128Error::CompressionError)?;
        point
            .y
            .serialize_compressed(&mut result[64..])
            .map_err(|_| AltBn128Error::CompressionError)?;
        Ok(G2Point(result))
    }
}
