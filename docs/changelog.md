# Changelog

All notable changes to RSVS are documented here. For the complete changelog, see [CHANGELOG.md](https://github.com/Wolfvin/AphantasicAbstractionModel/blob/main/CHANGELOG.md) in the repository root.

---

## [8.3.0] - 2026-05-06

### Added

- **Python package restructured for PyPI publication**: `pip install rsvs` now works out of the box with maturin-compiled wheels
- **maturin build system**: Proper `pyproject.toml` with PEP 621 metadata
- **FastAPI/uvicorn made optional**: Server dependencies behind `rsvs[server]` extra
- **Type stubs (`__init__.pyi`)**: Full type annotations for IDE support
- **`_version.py` as single source of truth**: Eliminates version drift across files
- **`RsvsCoreProtocol`**: Typed protocol for the Rust core interface
- **GitHub CI/CD workflows**: Automated testing, release, and Docker builds
- **Professional documentation**: README, ARCHITECTURE, CONTRIBUTING, CHANGELOG, CITATION.cff
- **`examples/` directory**: Self-contained demo scripts

### Fixed

- **Version chaos unified to 8.3.0**: All files now read from `_version.py`
- **Validation rejecting `"composed"` compression_state**
- **`/similarity` null check on missing labels**
- **`pyproject.toml` dependencies format (PEP 621)**

---

## [8.2.0] - 2026-05-05

### Added

- **Convergence contributors in query results**
- **Convergence info in appraise scoring**
- **Convergence tendrils in 3D visualization**

---

## [8.1.0] - 2026-05-04

### Added

- **Cross-language sense alignment via `LanguageLink`**
- **Multi-language composition support with `lang` parameter**

---

## [8.0.0] - 2026-05-03

### Added

- **Convergence detection**: Automatic detection of structurally equivalent concepts across languages
- **Language links**: Cross-language sense alignment infrastructure
- **`ConvergenceInfo` in appraise results**
- **`ConvergenceContributors` in query results**

### Changed

- **Major version bump**: Convergence detection changes the scoring model

---

For older versions, see the [full changelog](https://github.com/Wolfvin/AphantasicAbstractionModel/blob/main/CHANGELOG.md).
