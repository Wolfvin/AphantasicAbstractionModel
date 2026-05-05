# Development Workflow

## Prerequisites
- Rust toolchain (stable)
- Python >= 3.9
- `pip install maturin pytest`

## Commands
From repository root:

- `make check-rust`
  - Runs Rust compile check with PyO3 forward-compatibility flag.
- `make test-rust`
  - Runs Rust test suite for `rsvs` crate.
- `make test-python`
  - Builds extension with maturin and runs Python tests under `python/tests`.
- `make test-all`
  - Runs rust + python test lanes.

## PyO3 Compatibility Policy
This project currently uses `pyo3 = 0.22`.
For Python versions newer than PyO3 max support, we run with:

`PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`

This is mandatory in CI and local checks until PyO3 is upgraded.
