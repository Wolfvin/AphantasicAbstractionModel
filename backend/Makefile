.PHONY: check-rust test-rust test-python test-all serve audit

check-rust:
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check -p rsvs

test-rust:
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo test -p rsvs

test-python:
        PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --manifest-path crates/rsvs-core/Cargo.toml
        PYTHONPATH=python python3 -m pytest python/tests -v

test-all: check-rust test-rust test-python

serve:
        PYTHONPATH=python python3 -m rsvs.fastapi_server

audit:
        cargo audit
