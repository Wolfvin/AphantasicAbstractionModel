.PHONY: check-rust test-rust test-python test-all bridge-run audit

check-rust:
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check -p rsvs

test-rust:
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p rsvs

test-python:
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --manifest-path crates/rsvs-core/Cargo.toml
        PYTHONPATH=python python3 -m pytest python/tests -v

test-all: check-rust test-rust test-python

bridge-run:
        PYTHONPATH=python python3 -m rsvs.bridge_server

audit:
        cargo audit
