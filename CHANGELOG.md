# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project meta files: `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`.

### Removed
- Git submodules `research` and `secbox` (uninitialized; moved out of this repository).

### Changed
- Updated the clone instructions in `README.md` to the current repository URL and
  dropped the obsolete submodule step.

## [0.1.0]

### Added
- Reactive object-detection pipeline (`agent/`, `components/`, `main.py`) using YOLO
  and EasyOCR to detect and interact with in-game resources from screenshots.
- State-based reinforcement learning simulator (`state_sim/`) with a zone/biome
  environment, curriculum, and a PPO trainer with a mixture-of-experts actor-critic.
- V-JEPA world-model scaffolding (`world_model/`) adapted for mouse and keyboard control.
- Exploratory modules: SLAM navigation (`slam/`), a spatial world model, and a prototype.
- `uv` and `pyproject.toml` based dependency management with CPU and CUDA 11.8 extras.
- CodeQL scanning and Dependabot configuration under `.github/`.

[Unreleased]: https://github.com/cyborgx0x/nano-agent/compare/main...HEAD
[0.1.0]: https://github.com/cyborgx0x/nano-agent/releases/tag/v0.1.0
