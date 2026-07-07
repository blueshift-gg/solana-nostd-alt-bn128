//! BN254 group operations over the **little-endian** wire format used by the
//! Solana SVM: G1 and G2 point add / sub / scalar-mul, plus the multi-pairing
//! check.
//!
//! On `target_os = "solana"` each call is a single `sol_alt_bn128_group_op`
//! syscall writing into a caller-owned stack buffer; off-Solana it falls
//! through to an Arkworks reference implementation so host tests and tooling
//! exercise the identical byte contracts.
//!
//! Note on subtraction: the syscall has **no** SUB opcode for either group
//! (the runtime dispatch only handles G1/G2 ADD and MUL plus PAIRING — the
//! `*_SUB` selectors in `solana-bn254` are never matched and return
//! `InvalidAttribute`). So `g1_sub`/`g2_sub` compute `a - b` as `a + (-b)`,
//! negating the operand's y-coordinate (which is just `q - y` per field limb)
//! and routing through the ADD syscall.

#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
use ark_bn254::Bn254;
#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
use ark_bn254::{G1Affine, G2Affine};
#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
use ark_ec::{AffineRepr, CurveGroup, pairing::Pairing};
#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
use ark_ff::One;
#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use core::ops::{Add, Deref, Mul, Neg, Sub};

use crate::AltBn128Error;

/// A 256-bit scalar for G1 scalar multiplication, little-endian.
pub type G1Scalar = [u8; 32];
/// A 256-bit scalar for G2 scalar multiplication, little-endian.
pub type G2Scalar = [u8; 32];
/// A G1 point in compressed form: the 32-byte x-coordinate plus a y-sign bit.
pub type G1CompressedPoint = [u8; 32];
/// A G2 point in compressed form: the 64-byte `Fq2` x-coordinate plus a y-sign bit.
pub type G2CompressedPoint = [u8; 64];

/// A G1 point in the SVM little-endian uncompressed encoding: `x || y`, each a
/// 32-byte little-endian `Fq` element.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct G1Point(pub [u8; 64]);

/// A G2 point in the SVM little-endian uncompressed encoding:
/// `x.c0 || x.c1 || y.c0 || y.c1`, each a 32-byte little-endian `Fq` element.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct G2Point(pub [u8; 128]);

impl Deref for G1Point {
    type Target = [u8; 64];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for G2Point {
    type Target = [u8; 128];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<[u8; 64]> for G1Point {
    fn from(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }
}

impl From<G1Point> for [u8; 64] {
    fn from(point: G1Point) -> Self {
        point.0
    }
}

impl From<[u8; 128]> for G2Point {
    fn from(bytes: [u8; 128]) -> Self {
        Self(bytes)
    }
}

impl From<G2Point> for [u8; 128] {
    fn from(point: G2Point) -> Self {
        point.0
    }
}

impl PartialEq<[u8; 64]> for G1Point {
    fn eq(&self, other: &[u8; 64]) -> bool {
        &self.0 == other
    }
}

impl PartialEq<[u8; 128]> for G2Point {
    fn eq(&self, other: &[u8; 128]) -> bool {
        &self.0 == other
    }
}

/// Set on the operation selector to request the little-endian variant of the
/// syscall. The SVM wire format this crate targets is LE end-to-end.
const LE_FLAG: u64 = 0x80;

/// G1 point addition (little-endian operands).
pub const ALT_BN128_ADD: u64 = LE_FLAG;
/// G1 scalar multiplication (little-endian operands).
pub const ALT_BN128_MUL: u64 = 0x02 | LE_FLAG;
/// Multi-pairing check (little-endian operands).
pub const ALT_BN128_PAIRING: u64 = 0x03 | LE_FLAG;
/// G2 point addition (little-endian operands).
pub const ALT_BN128_G2_ADD: u64 = 0x04 | LE_FLAG;
/// G2 scalar multiplication (little-endian operands).
pub const ALT_BN128_G2_MUL: u64 = 0x06 | LE_FLAG;

#[cfg(all(
    any(target_os = "solana", target_arch = "bpf"),
    any(target_feature = "static-syscalls", feature = "static-syscalls")
))]
#[inline]
unsafe fn sol_alt_bn128_group_op(
    group_op: u64,
    input: *const u8,
    input_size: u64,
    result: *mut u8,
) -> u64 {
    let f: unsafe extern "C" fn(u64, *const u8, u64, *mut u8) -> u64 =
        unsafe { core::mem::transmute(0xae0c318b_usize) }; // murmur3_32(b"sol_alt_bn128_group_op")
    f(group_op, input, input_size, result)
}

#[cfg(all(
    any(target_os = "solana", target_arch = "bpf"),
    not(any(target_feature = "static-syscalls", feature = "static-syscalls"))
))]
unsafe extern "C" {
    fn sol_alt_bn128_group_op(
        group_op: u64,
        input: *const u8,
        input_size: u64,
        result: *mut u8,
    ) -> u64;
}

/// BN254 base field modulus `q`, little-endian. Used to negate a point's
/// y-coordinate (the `Neg` impls, and the negate-then-add subtraction path).
const FIELD_MODULUS_LE: [u8; 32] = [
    0x47, 0xfd, 0x7c, 0xd8, 0x16, 0x8c, 0x20, 0x3c, 0x8d, 0xca, 0x71, 0x68, 0x91, 0x6a, 0x81, 0x97,
    0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8, 0x29, 0xa0, 0x31, 0xe1, 0x72, 0x4e, 0x64, 0x30,
];

/// Negate a single 32-byte little-endian `Fq` element: `c -> q - c`, with the
/// additive identity (`0`) mapping to itself (a `q - 0 = q` result would be an
/// unreduced, invalid field element).
const fn negate_fq(limb: [u8; 32]) -> [u8; 32] {
    let mut is_zero = true;
    let mut i = 0;
    while i < 32 {
        if limb[i] != 0 {
            is_zero = false;
            break;
        }
        i += 1;
    }
    if is_zero {
        return limb;
    }

    // q - c. Since 0 < c < q the result is already reduced; a single limb-wise
    // subtraction with borrow over the 32 little-endian bytes.
    let mut out = [0u8; 32];
    let mut borrow = 0u16;
    let mut j = 0;
    while j < 32 {
        let diff = FIELD_MODULUS_LE[j] as u16 + 256 - limb[j] as u16 - borrow;
        out[j] = diff as u8;
        borrow = 1 - (diff >> 8);
        j += 1;
    }
    out
}

/// Negate a G1 point: `(x, y) -> (x, -y)`. The point at infinity (all zeros)
/// is preserved. `y` is the 32-byte `Fq` limb at `[32..64]`.
const fn negate_g1(point: &G1Point) -> G1Point {
    let mut out = point.0;
    let mut y = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        y[i] = out[32 + i];
        i += 1;
    }
    let ny = negate_fq(y);
    let mut i = 0;
    while i < 32 {
        out[32 + i] = ny[i];
        i += 1;
    }
    G1Point(out)
}

/// Negate a G2 point: `(x, y) -> (x, -y)`. `y = (y.c0, y.c1)` is the `Fq2`
/// limb pair at `[64..96]` and `[96..128]`; each `Fq` component is negated
/// independently (either may be zero on its own for a valid point). The point
/// at infinity (all zeros) is preserved.
const fn negate_g2(point: &G2Point) -> G2Point {
    let mut out = point.0;
    let mut off = 64;
    while off < 128 {
        let mut limb = [0u8; 32];
        let mut i = 0;
        while i < 32 {
            limb[i] = out[off + i];
            i += 1;
        }
        let n = negate_fq(limb);
        let mut i = 0;
        while i < 32 {
            out[off + i] = n[i];
            i += 1;
        }
        off += 32;
    }
    G2Point(out)
}

impl G1Point {
    /// Negate the point: `(x, y) -> (x, -y)`. Pure coordinate arithmetic
    /// (`y -> q - y`, no syscall), so it is infallible and `const`-usable; the
    /// point at infinity is preserved. The `-` operator (`Neg`) delegates here.
    pub const fn negate(self) -> G1Point {
        negate_g1(&self)
    }

    /// Wrap little-endian (SVM wire) bytes — the internal storage format.
    pub const fn from_le_bytes(bytes: [u8; 64]) -> G1Point {
        G1Point(bytes)
    }

    /// From big-endian (EIP-197) bytes: reverses each 32-byte coordinate.
    pub const fn from_be_bytes(bytes: [u8; 64]) -> G1Point {
        G1Point(convert_endianness::<32, 64>(&bytes))
    }

    /// The little-endian (SVM wire) bytes — the internal storage.
    pub const fn to_le_bytes(self) -> [u8; 64] {
        self.0
    }

    /// Big-endian (EIP-197) bytes: reverses each 32-byte coordinate.
    pub const fn to_be_bytes(self) -> [u8; 64] {
        convert_endianness::<32, 64>(&self.0)
    }
}

impl G2Point {
    /// Negate the point: `(x, y) -> (x, -y)`. Pure coordinate arithmetic
    /// (`y -> q - y`, no syscall), so it is infallible and `const`-usable; the
    /// point at infinity is preserved. The `-` operator (`Neg`) delegates here.
    pub const fn negate(self) -> G2Point {
        negate_g2(&self)
    }

    /// Check the point is on-curve **and** in the prime-order subgroup — `Ok(())`
    /// if so. It is a membership *check* (it rejects), not a cofactor clearing
    /// (it does not transform the point).
    ///
    /// `g2_add`/`g2_sub` (and the underlying syscall) skip the subgroup check for
    /// performance — they only verify the curve equation — because the subgroup
    /// is closed under addition and a `pairing` already subgroup-checks its
    /// inputs. So for the common flow, where untrusted G2 points end up in a
    /// pairing, this is unnecessary.
    ///
    /// It *is* needed when you combine **untrusted** G2 points with `+`/`-` and
    /// your security model requires each summand to be a genuine subgroup element
    /// (e.g. BLS-style aggregation), where checking only the final pairing input
    /// is not equivalent to checking each point. The check is a G2 scalar
    /// multiplication by `1` — at ~15.7k CU, currently the cheapest way to run a
    /// prime-order subgroup check on-chain.
    pub fn check_prime_subgroup(&self) -> Result<(), AltBn128Error> {
        g2_mul(*self, SCALAR_ONE).map(|_| ())
    }

    /// Wrap little-endian (SVM wire) bytes — the internal storage format.
    pub const fn from_le_bytes(bytes: [u8; 128]) -> G2Point {
        G2Point(bytes)
    }

    /// From big-endian (EIP-197) bytes: reverses each 64-byte `Fq2` coordinate.
    pub const fn from_be_bytes(bytes: [u8; 128]) -> G2Point {
        G2Point(convert_endianness::<64, 128>(&bytes))
    }

    /// The little-endian (SVM wire) bytes — the internal storage.
    pub const fn to_le_bytes(self) -> [u8; 128] {
        self.0
    }

    /// Big-endian (EIP-197) bytes: reverses each 64-byte `Fq2` coordinate.
    pub const fn to_be_bytes(self) -> [u8; 128] {
        convert_endianness::<64, 128>(&self.0)
    }
}

// Private helpers behind the public `Add`/`Sub`/`Mul` operator impls below.

fn g1_add(a: G1Point, b: G1Point) -> Result<G1Point, AltBn128Error> {
    #[cfg(any(target_os = "solana", target_arch = "bpf"))]
    {
        let mut buffer = [0u8; 128];
        buffer[..64].copy_from_slice(&a[..]);
        buffer[64..].copy_from_slice(&b[..]);
        let mut result = core::mem::MaybeUninit::<[u8; 64]>::uninit();
        let status = unsafe {
            sol_alt_bn128_group_op(
                ALT_BN128_ADD,
                buffer.as_ptr(),
                buffer.len() as u64,
                result.as_mut_ptr() as *mut u8,
            )
        };
        match status {
            0 => Ok(G1Point(unsafe { result.assume_init() })),
            _ => Err(AltBn128Error::GroupError),
        }
    }
    #[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
    {
        let a =
            G1Affine::deserialize_uncompressed(&a[..]).map_err(|_| AltBn128Error::GroupError)?;
        let b =
            G1Affine::deserialize_uncompressed(&b[..]).map_err(|_| AltBn128Error::GroupError)?;
        let c = (a + b).into_affine();
        let mut result = [0u8; 64];
        c.x.serialize_uncompressed(&mut result[..32])
            .map_err(|_| AltBn128Error::GroupError)?;
        c.y.serialize_uncompressed(&mut result[32..])
            .map_err(|_| AltBn128Error::GroupError)?;
        Ok(G1Point(result))
    }
}

fn g1_sub(a: G1Point, b: G1Point) -> Result<G1Point, AltBn128Error> {
    // There is no SUB opcode, so a - b is computed as a + (-b). Negation is
    // pure coordinate arithmetic, so this works identically on host and on
    // chain and reuses the ADD path.
    g1_add(a, b.negate())
}

fn g1_mul(point: G1Point, scalar: G1Scalar) -> Result<G1Point, AltBn128Error> {
    #[cfg(any(target_os = "solana", target_arch = "bpf"))]
    {
        let mut buffer = [0u8; 96];
        buffer[..64].copy_from_slice(&point[..]);
        buffer[64..].copy_from_slice(&scalar);
        let mut result = core::mem::MaybeUninit::<[u8; 64]>::uninit();
        let status = unsafe {
            sol_alt_bn128_group_op(
                ALT_BN128_MUL,
                buffer.as_ptr(),
                buffer.len() as u64,
                result.as_mut_ptr() as *mut u8,
            )
        };
        match status {
            0 => Ok(G1Point(unsafe { result.assume_init() })),
            _ => Err(AltBn128Error::GroupError),
        }
    }
    #[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
    {
        let p = G1Affine::deserialize_uncompressed(&point[..])
            .map_err(|_| AltBn128Error::GroupError)?;
        // `mul_bigint` takes a `BigInt<4>` = `[u64; 4]` of little-endian limbs.
        // The scalar is already 32 little-endian bytes, so reinterpreting it as
        // four LE u64 limbs is correct on little-endian targets (Solana, x86,
        // arm64) — the only architectures this crate targets.
        let c = p
            .mul_bigint(unsafe { core::mem::transmute::<[u8; 32], [u64; 4]>(scalar) })
            .into_affine();
        let mut result = [0u8; 64];
        c.x.serialize_uncompressed(&mut result[..32])
            .map_err(|_| AltBn128Error::GroupError)?;
        c.y.serialize_uncompressed(&mut result[32..])
            .map_err(|_| AltBn128Error::GroupError)?;
        Ok(G1Point(result))
    }
}

fn g2_add(a: G2Point, b: G2Point) -> Result<G2Point, AltBn128Error> {
    #[cfg(any(target_os = "solana", target_arch = "bpf"))]
    {
        let mut buffer = [0u8; 256];
        buffer[..128].copy_from_slice(&a[..]);
        buffer[128..].copy_from_slice(&b[..]);
        let mut result = core::mem::MaybeUninit::<[u8; 128]>::uninit();
        let status = unsafe {
            sol_alt_bn128_group_op(
                ALT_BN128_G2_ADD,
                buffer.as_ptr(),
                buffer.len() as u64,
                result.as_mut_ptr() as *mut u8,
            )
        };
        match status {
            0 => Ok(G2Point(unsafe { result.assume_init() })),
            _ => Err(AltBn128Error::GroupError),
        }
    }
    #[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
    {
        // G2 addition validates the curve equation but NOT subgroup membership,
        // matching the syscall (`Validate::No` + an explicit `is_on_curve`); use
        // `G2Point::check_prime_subgroup` for the subgroup check. Every other op
        // uses the subgroup-checking `deserialize_uncompressed`.
        let a = g2_deserialize_on_curve(&a)?;
        let b = g2_deserialize_on_curve(&b)?;
        let c = (a + b).into_affine();
        let mut result = [0u8; 128];
        c.x.serialize_uncompressed(&mut result[..64])
            .map_err(|_| AltBn128Error::GroupError)?;
        c.y.serialize_uncompressed(&mut result[64..])
            .map_err(|_| AltBn128Error::GroupError)?;
        Ok(G2Point(result))
    }
}

/// Host helper: deserialize a G2 point checking the curve equation but *not*
/// subgroup membership (the validation level the G2-add syscall uses).
#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
fn g2_deserialize_on_curve(p: &G2Point) -> Result<G2Affine, AltBn128Error> {
    let point = G2Affine::deserialize_uncompressed_unchecked(&p[..])
        .map_err(|_| AltBn128Error::GroupError)?;
    if point.is_on_curve() {
        Ok(point)
    } else {
        Err(AltBn128Error::GroupError)
    }
}

fn g2_sub(a: G2Point, b: G2Point) -> Result<G2Point, AltBn128Error> {
    // a - b == a + (-b); see `g1_sub`.
    g2_add(a, b.negate())
}

fn g2_mul(point: G2Point, scalar: G2Scalar) -> Result<G2Point, AltBn128Error> {
    #[cfg(any(target_os = "solana", target_arch = "bpf"))]
    {
        let mut buffer = [0u8; 160];
        buffer[..128].copy_from_slice(&point[..]);
        buffer[128..].copy_from_slice(&scalar);
        let mut result = core::mem::MaybeUninit::<[u8; 128]>::uninit();
        let status = unsafe {
            sol_alt_bn128_group_op(
                ALT_BN128_G2_MUL,
                buffer.as_ptr(),
                buffer.len() as u64,
                result.as_mut_ptr() as *mut u8,
            )
        };
        match status {
            0 => Ok(G2Point(unsafe { result.assume_init() })),
            _ => Err(AltBn128Error::GroupError),
        }
    }
    #[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
    {
        let p = G2Affine::deserialize_uncompressed(&point[..])
            .map_err(|_| AltBn128Error::GroupError)?;
        // `mul_bigint` takes a `BigInt<4>` = `[u64; 4]` of little-endian limbs.
        // The scalar is already 32 little-endian bytes, so reinterpreting it as
        // four LE u64 limbs is correct on little-endian targets (Solana, x86,
        // arm64) — the only architectures this crate targets.
        let c = p
            .mul_bigint(unsafe { core::mem::transmute::<[u8; 32], [u64; 4]>(scalar) })
            .into_affine();
        let mut result = [0u8; 128];
        c.x.serialize_uncompressed(&mut result[..64])
            .map_err(|_| AltBn128Error::GroupError)?;
        c.y.serialize_uncompressed(&mut result[64..])
            .map_err(|_| AltBn128Error::GroupError)?;
        Ok(G2Point(result))
    }
}

/// The scalar `1`, little-endian.
const SCALAR_ONE: G2Scalar = {
    let mut s = [0u8; 32];
    s[0] = 1;
    s
};

/// Multi-pairing check over `N` (G1, G2) operand pairs. Returns a 32-byte
/// little-endian scalar: `1` iff the pairing product is the identity in GT,
/// `0` otherwise.
pub fn pairing<const N: usize>(input: &[(G1Point, G2Point); N]) -> Result<G1Scalar, AltBn128Error> {
    #[cfg(any(target_os = "solana", target_arch = "bpf"))]
    {
        // Flatten into a contiguous `[[u8; 192]; N]` (G1 || G2 per pair); a
        // `[(G1Point, G2Point); N]` has no guaranteed element layout, while
        // an array of `[u8; 192]` is contiguous by definition.
        let mut buffer = [[0u8; 192]; N];
        for (slot, (g1, g2)) in buffer.iter_mut().zip(input.iter()) {
            slot[..64].copy_from_slice(&g1[..]);
            slot[64..].copy_from_slice(&g2[..]);
        }
        let mut result = core::mem::MaybeUninit::<[u8; 32]>::uninit();
        let status = unsafe {
            sol_alt_bn128_group_op(
                ALT_BN128_PAIRING,
                buffer.as_ptr() as *const u8,
                (N * 192) as u64,
                result.as_mut_ptr() as *mut u8,
            )
        };
        match status {
            0 => Ok(unsafe { result.assume_init() }),
            _ => Err(AltBn128Error::GroupError),
        }
    }
    #[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
    {
        let mut g1s: [G1Affine; N] = [G1Affine::default(); N];
        let mut g2s: [G2Affine; N] = [G2Affine::default(); N];

        for (i, (g1_bytes, g2_bytes)) in input.iter().enumerate() {
            g1s[i] = G1Affine::deserialize_uncompressed(&g1_bytes[..])
                .map_err(|_| AltBn128Error::GroupError)?;
            g2s[i] = G2Affine::deserialize_uncompressed(&g2_bytes[..])
                .map_err(|_| AltBn128Error::GroupError)?;
        }

        let res = Bn254::multi_pairing(g1s, g2s);
        let mut result = [0u8; 32];
        if res.0 == ark_bn254::Fq12::one() {
            result[0] = 1;
        }
        Ok(result)
    }
}

// ---- Operator API ---------------------------------------------------------
//
// Point arithmetic is exposed solely through these operators. The group ops
// can fail (an invalid point encoding makes the syscall reject), so the
// operator `Output` is `Result<_, AltBn128Error>` — call sites write
// `(a + b)?`, `(a - b)?`, `(p * scalar)?`.

// Negation is pure coordinate arithmetic (`y -> q - y`, no syscall), so unlike
// the fallible add/sub/mul it is infallible — `Output` is the point itself.

impl Neg for G1Point {
    type Output = G1Point;
    fn neg(self) -> G1Point {
        self.negate()
    }
}

impl Neg for G2Point {
    type Output = G2Point;
    fn neg(self) -> G2Point {
        self.negate()
    }
}

// `a - b` lowers to `a + (-b)`, so the negation runs at runtime. When the
// subtrahend is known at compile time, bind its negation to a `const` (negation
// is a `const fn`) and add that instead — the negation folds away and the op
// costs the same as a bare add:
//
//     const NEG_B: G1Point = B.negate();  // computed at compile time
//     let diff = (a + NEG_B)?;            // a - b, with no runtime negation

impl Add for G1Point {
    type Output = Result<G1Point, AltBn128Error>;
    fn add(self, rhs: G1Point) -> Self::Output {
        g1_add(self, rhs)
    }
}

impl Sub for G1Point {
    type Output = Result<G1Point, AltBn128Error>;
    fn sub(self, rhs: G1Point) -> Self::Output {
        g1_sub(self, rhs)
    }
}

impl Mul<G1Scalar> for G1Point {
    type Output = Result<G1Point, AltBn128Error>;
    fn mul(self, rhs: G1Scalar) -> Self::Output {
        g1_mul(self, rhs)
    }
}

impl Add for G2Point {
    type Output = Result<G2Point, AltBn128Error>;
    fn add(self, rhs: G2Point) -> Self::Output {
        g2_add(self, rhs)
    }
}

impl Sub for G2Point {
    type Output = Result<G2Point, AltBn128Error>;
    fn sub(self, rhs: G2Point) -> Self::Output {
        g2_sub(self, rhs)
    }
}

impl Mul<G2Scalar> for G2Point {
    type Output = Result<G2Point, AltBn128Error>;
    fn mul(self, rhs: G2Scalar) -> Self::Output {
        g2_mul(self, rhs)
    }
}

/// Reverse the byte order within each `CHUNK_SIZE`-byte chunk of `bytes`,
/// converting a coordinate between big- and little-endian. `ARRAY_SIZE` must be
/// a multiple of `CHUNK_SIZE` (e.g. `<32, 64>` for G1, `<64, 128>` for G2). The
/// operation is its own inverse, so it serves both directions.
pub const fn convert_endianness<const CHUNK_SIZE: usize, const ARRAY_SIZE: usize>(
    bytes: &[u8; ARRAY_SIZE],
) -> [u8; ARRAY_SIZE] {
    let mut reversed = [0u8; ARRAY_SIZE];
    let mut chunk = 0;
    while chunk < ARRAY_SIZE {
        let mut i = 0;
        while i < CHUNK_SIZE {
            reversed[chunk + i] = bytes[chunk + CHUNK_SIZE - 1 - i];
            i += 1;
        }
        chunk += CHUNK_SIZE;
    }
    reversed
}
