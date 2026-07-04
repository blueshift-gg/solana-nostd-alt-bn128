# Solana NoStd AltBn128

[![CI](https://github.com/blueshift-gg/solana-nostd-alt-bn128/actions/workflows/ci.yml/badge.svg)](https://github.com/blueshift-gg/solana-nostd-alt-bn128/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/solana-nostd-alt-bn128.svg)](https://crates.io/crates/solana-nostd-alt-bn128)
[![docs.rs](https://docs.rs/solana-nostd-alt-bn128/badge.svg)](https://docs.rs/solana-nostd-alt-bn128)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/blueshift-gg/solana-nostd-alt-bn128/blob/master/LICENSE)

A more efficient, `no_std` implementation of the alt_bn128 (BN254) group and compression operations for the Solana SVM. Routes through the `sol_alt_bn128_group_op` / `sol_alt_bn128_compression` syscalls on-chain and falls through to the [`ark-bn254`](https://crates.io/crates/ark-bn254) (Arkworks) reference implementation off-chain, so the same APIs work in host code. The wire format is little-endian end-to-end, skipping the per-call byte-swapping the standard big-endian path performs.

## Quick start

```toml
[dependencies]
solana-nostd-alt-bn128 = "0.1.0"
```

```rust
use solana_nostd_alt_bn128::{pairing, G1Point, G2Point};

// Points are little-endian encoded; arithmetic is via operators.
let sum = (a + b)?;          // a, b: G1Point (or G2Point)
let diff = (a - b)?;         // a - b == a + (-b)
let scaled = (p * scalar)?;  // scalar: [u8; 32]
let neg = -a;                // negation is infallible

// Multi-pairing: returns a 32-byte LE scalar, `1` iff the product is the GT identity.
let one = pairing(&[(g1, g2), (h1, h2)])?;
```

`AltBn128Error` implements `Into<ProgramError>`, so every result is `?`-propagatable from any program entrypoint.

The crate is `#![no_std]` on the Solana target; no allocator setup required. Off-chain it builds as `std` for the Arkworks reference path.

## Features

- Operator-based API — `+`, `-`, `*`, and unary `-` on `G1Point` / `G2Point`. Fallible ops return `Result<_, AltBn128Error>`; negation is infallible
- Full G1 **and** G2 arithmetic (add / sub / scalar-mul), the multi-pairing check, and point (de)compression
- Little-endian wire format end-to-end — no per-call endianness conversion; `from_be_bytes` / `to_be_bytes` provide EIP-197 interop
- Subtraction without a SUB syscall (none exists): computed as `a + (-b)`, where negation is pure, `const`-usable coordinate arithmetic
- Uses `MaybeUninit` to skip zero-initializing syscall output buffers
- Byte-compatible with `ark-bn254` — the off-chain path round-trips through it
- **Point aggregation** — `aggregate_g1`/`aggregate_g2` sum a slice in one reused `[acc | addend]` buffer (calling ADD in place, so the accumulator is never re-copied); `aggregate_g1_in_place`/`aggregate_g2_in_place` slide the running sum through a mutable buffer with no per-step copy at all. **~16–18% fewer CU than folding with `+`** (≈112–129 CU/add saved), landing within ~11 CU of the add-syscall floor

## Benchmarks

On-chain compute unit cost per operation. These measure the operation only — the result is `black_box`'d, not compared — so they exclude the result-comparison overhead a correctness check would add (~25 CU).

| operation           | CU cost |
|---------------------|--------:|
| `g1_add`            |     474 |
| `g1_sub`            |    1126 |
| `g1_mul`            |    3955 |
| `g1_negate`         |     697 |
| `g2_add`            |     675 |
| `g2_sub`            |    2066 |
| `g2_mul`            |   15784 |
| `g2_negate`         |    1434 |
| `check_prime_subgroup` (G2) | 15731 |
| `pairing` (1 pair)  |   36739 |
| `pairing` (2 pairs) |   49092 |
| `pairing` (3 pairs) |   61445 |
| `g1_compress`       |      42 |
| `g1_decompress`     |     569 |
| `g2_compress`       |      52 |
| `g2_decompress`     |   13795 |

Subtraction is `add + negation` (there is no SUB syscall), so `g1_sub`/`g2_sub` cost roughly the add plus the matching `negate` row. When the subtrahend is known at compile time, binding its negation to a `const` folds the negation away and subtraction drops to the bare add cost.

Compression is implemented manually (byte re-encoding, no syscall): `g1_compress`/`g2_compress` are ~4x cheaper than the `sol_alt_bn128_compression` syscall, whose fixed per-call overhead dwarfs the trivial work. Decompression needs a field square root, so it stays on the syscall.

### Aggregation vs. folding with `+`

Summing N points with the `+` operator rebuilds the syscall input buffer and reconstructs a `Result<Point>` on every step, re-copying the running accumulator. `aggregate_g1`/`aggregate_g2` keep the accumulator in a single `[acc | addend]` buffer and call ADD in place; `aggregate_g1_in_place`/`aggregate_g2_in_place` slide the running sum through the caller's (mutable) buffer, removing the per-step copy entirely. Full-transaction CU for summing N G2 points (litesvm; the `+`-fold's 675 CU/add matches the `g2_add` row above, confirming the scale):

| N  | `+` fold | `aggregate_g2` | `aggregate_g2_in_place` |
|---:|---------:|---------------:|------------------------:|
| 2  |      873 |    786 (−10%)  |     815 (−7%)   |
| 4  |    2 223 |  1 912 (−14%)  |   1 907 (−14%)  |
| 8  |    4 923 |  4 164 (−15%)  |   4 091 (−17%)  |
| 16 |   10 323 |  8 668 (−16%)  |   8 459 (−18%)  |

Marginal cost per added point: `+` fold **675 CU**, `aggregate` **563**, `in_place` **546** — against the **535 CU** raw add-syscall floor. `aggregate` never clobbers its input and is best for tiny N; `in_place` is fastest for N ≳ 4 (the figures above include copying the points into a scratch buffer, so a caller that hands over a disposable buffer saves that copy and gets nearer the floor). The `aggregate_g1_4` / `aggregate_g2_4` / `aggregate_g2_in_place_4` sbpf benches report these under Mollusk when the suite is run.



To reproduce, install `cargo build-sbf` (Solana CLI) and run:

```sh
cargo test --test sbpf --jobs 1
```

The benchmarks compile each operation into its own SBPF program and run it through [Mollusk](https://github.com/anza-xyz/mollusk) via [`svm-unit-test`](https://crates.io/crates/svm-unit-test).

## License

Licensed under the [MIT License](https://github.com/blueshift-gg/solana-nostd-alt-bn128/blob/master/LICENSE). The license includes the standard "as-is" warranty disclaimer — use at your own risk.
