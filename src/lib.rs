//! A more efficient, `no_std` implementation of the alt_bn128 (BN254)
//! group and compression operations for the Solana SVM.
//!
//! On `target_os = "solana"` (or `target_arch = "bpf"`) every operation
//! routes straight through the `sol_alt_bn128_group_op` /
//! `sol_alt_bn128_compression` syscalls, writing into caller-owned stack
//! buffers — no heap, no `Vec`, no `std`. Off-Solana (host builds: tests,
//! off-chain tooling) the same APIs fall through to an Arkworks reference
//! implementation so call sites behave identically in both worlds.
//!
//! The crate is `no_std` *on the Solana target only*
//! (`cfg_attr(any(target_os = "solana", target_arch = "bpf"), no_std)`): there it links nothing but
//! `core`, so it can be pulled into the tiny `cdylib`s that
//! [`svm-unit-test`](https://crates.io/crates/svm-unit-test) generates for
//! each `#[svm_test]` without colliding with the test crate's own
//! `#[panic_handler]` — the collision that `std`-linked syscall crates
//! (e.g. `solana-bn254`) trigger as `E0152: duplicate lang item`. On host
//! builds it stays `std` so the Arkworks reference path (which allocates)
//! and the in-crate KAT unit tests compile normally. See `tests/sbpf.rs`
//! for the on-chain compute-unit harness.
#![cfg_attr(any(target_os = "solana", target_arch = "bpf"), no_std)]
#![deny(missing_docs)]

pub mod group_operations;
pub use group_operations::*;
pub mod compression;
pub use compression::*;
pub mod errors;
pub use errors::*;
#[cfg(test)]
mod tests;
