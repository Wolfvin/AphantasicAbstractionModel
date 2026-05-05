# ──────────────────────────────────────────────────────────────────────
#  SymbolicPuzzle3D — Root Makefile
#  10k-star quality DevOps entry point
# ──────────────────────────────────────────────────────────────────────

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# ── Paths ────────────────────────────────────────────────────────────
BACKEND_DIR  := backend
FRONTEND_DIR := frontend

# ── Python ───────────────────────────────────────────────────────────
PYTHON       ?= python3
PIP          ?= pip3
VENV         ?= $(BACKEND_DIR)/.venv
PYO3_ABI     := PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# ── Docker ───────────────────────────────────────────────────────────
DOCKER_COMPOSE := docker compose

# ── Colours ──────────────────────────────────────────────────────────
CYAN  := \033[36m
BOLD  := \033[1m
RESET := \033[0m

# ──────────────────────────────────────────────────────────────────────
#  Targets
# ──────────────────────────────────────────────────────────────────────

.PHONY: help setup build test test-rust test-python test-frontend lint dev \
        docker-up docker-build docker-down clean

# ── help ─────────────────────────────────────────────────────────────
help: ## Show all Makefile targets
        @echo -e "$(BOLD)$(CYAN)SymbolicPuzzle3D — Available targets$(RESET)"
        @echo ""
        @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
                awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BOLD)%-18s$(RESET) %s\n", $$1, $$2}'
        @echo ""

# ── setup ────────────────────────────────────────────────────────────
setup: ## Install all dependencies (Rust, Python, Node)
        @echo -e "$(BOLD)→ Installing Rust toolchain…$(RESET)"
        rustup show active-toolchain || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        @echo -e "$(BOLD)→ Setting up Python virtual-env…$(RESET)"
        $(PYTHON) -m venv $(VENV) 2>/dev/null || true
        . $(VENV)/bin/activate && $(PIP) install --upgrade pip maturin ruff pytest pytest-cov mypy
        @echo -e "$(BOLD)→ Building Rust Python extension…$(RESET)"
        cd $(BACKEND_DIR) && $(PYO3_ABI) maturin develop --manifest-path crates/rsvs-core/Cargo.toml
        @echo -e "$(BOLD)→ Installing Node.js dependencies…$(RESET)"
        cd $(FRONTEND_DIR) && npm ci
        @echo -e "$(BOLD)✓ Setup complete.$(RESET)"

# ── build ────────────────────────────────────────────────────────────
build: ## Build Rust core + frontend
        @echo -e "$(BOLD)→ Building Rust core (release)…$(RESET)"
        cd $(BACKEND_DIR) && cargo build --release
        @echo -e "$(BOLD)→ Building Next.js frontend…$(RESET)"
        cd $(FRONTEND_DIR) && npm run build
        @echo -e "$(BOLD)✓ Build complete.$(RESET)"

# ── test ─────────────────────────────────────────────────────────────
test: test-rust test-python test-frontend ## Run all tests

test-rust: ## Run Rust tests (cargo test)
        @echo -e "$(BOLD)→ Running Rust tests…$(RESET)"
        cd $(BACKEND_DIR) && $(PYO3_ABI) cargo test --workspace

test-python: ## Run Python tests (pytest)
        @echo -e "$(BOLD)→ Building Rust extension for Python…$(RESET)"
        cd $(BACKEND_DIR) && $(PYO3_ABI) maturin develop --manifest-path crates/rsvs-core/Cargo.toml
        @echo -e "$(BOLD)→ Running Python tests…$(RESET)"
        cd $(BACKEND_DIR) && PYTHONPATH=python $(PYTHON) -m pytest python/tests -v --tb=short

test-frontend: ## Run frontend tests (npm test)
        @echo -e "$(BOLD)→ Running frontend tests…$(RESET)"
        cd $(FRONTEND_DIR) && npm test --if-present

# ── lint ─────────────────────────────────────────────────────────────
lint: ## Run all linters (Rust + Python + frontend)
        @echo -e "$(BOLD)→ Linting Rust (fmt + clippy)…$(RESET)"
        cd $(BACKEND_DIR) && cargo fmt --all -- --check
        cd $(BACKEND_DIR) && cargo clippy --all-targets -- -D warnings
        @echo -e "$(BOLD)→ Linting Python (ruff)…$(RESET)"
        cd $(BACKEND_DIR) && ruff check python/
        @echo -e "$(BOLD)→ Linting frontend (eslint)…$(RESET)"
        cd $(FRONTEND_DIR) && npm run lint
        @echo -e "$(BOLD)✓ Lint complete.$(RESET)"

# ── dev ──────────────────────────────────────────────────────────────
dev: ## Start development servers (bridge + frontend)
        @echo -e "$(BOLD)→ Starting Python bridge server…$(RESET)"
        @cd $(BACKEND_DIR) && PYTHONPATH=python $(PYTHON) -m rsvs.fastapi_server &
        @echo -e "$(BOLD)→ Starting Next.js dev server…$(RESET)"
        @cd $(FRONTEND_DIR) && npm run dev
        @wait

# ── Docker ───────────────────────────────────────────────────────────
docker-up: ## Start all services with Docker Compose
        $(DOCKER_COMPOSE) up -d

docker-build: ## Build all Docker images
        $(DOCKER_COMPOSE) build

docker-down: ## Stop all Docker services
        $(DOCKER_COMPOSE) down

# ── clean ────────────────────────────────────────────────────────────
clean: ## Clean all build artefacts
        @echo -e "$(BOLD)→ Cleaning Rust target…$(RESET)"
        cd $(BACKEND_DIR) && cargo clean 2>/dev/null || true
        @echo -e "$(BOLD)→ Cleaning Python cache…$(RESET)"
        cd $(BACKEND_DIR) && find python -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        cd $(BACKEND_DIR) && rm -rf .ruff_cache .mypy_cache .pytest_cache python/*.egg-info 2>/dev/null || true
        @echo -e "$(BOLD)→ Cleaning frontend build…$(RESET)"
        cd $(FRONTEND_DIR) && rm -rf .next node_modules/.cache 2>/dev/null || true
        @echo -e "$(BOLD)→ Cleaning Docker artefacts…$(RESET)"
        $(DOCKER_COMPOSE) down --rmi local --volumes 2>/dev/null || true
        @echo -e "$(BOLD)✓ Clean complete.$(RESET)"
