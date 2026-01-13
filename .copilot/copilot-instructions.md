# Copilot Instructions for ML_Project_Template

## Project Overview
This is a modular machine learning project template for production workflows. The codebase is organized for clarity, reproducibility, and scalability. Key directories include:
- `src/`: Core source code, organized by pipeline stage (data, features, models, utils)
- `configs/`: Environment-specific YAML config files (`base.yaml`, `dev.yaml`, `prod.yaml`)
- `data/`: Data storage, split into `raw/`, `interim/`, and `processed/`
- `models/`: Model artifacts and outputs
- `notebooks/`: Jupyter notebooks for exploration and prototyping
- `tests/`: Unit tests for data and model logic

## Architecture & Patterns
- **Pipeline Structure:**
  - Data loading (`src/data/loader.py`), preprocessing (`src/data/preprocessor.py`), feature engineering (`src/features/engineering.py`), model training (`src/models/train.py`), prediction (`src/models/predict.py`), and evaluation (`src/models/evaluate.py`).
  - Utilities for config management, logging, and helpers are in `src/utils/`.
- **Config-Driven:**
  - All runs use YAML configs from `configs/`. Load configs via `src/utils/config.py`.
- **Artifacts:**
  - Models and outputs are saved in `models/artifacts/`.
- **Testing:**
  - Tests are in `tests/`, using pytest conventions. Example: `pytest tests/test_data.py`.

## Developer Workflows
- **Setup:**
  - Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
- **Run Pipeline:**
  - Main entry point is `main.py`. Pass config path as argument if needed.
    ```bash
    python main.py --config configs/dev.yaml
    ```
- **Testing:**
  - Run all tests:
    ```bash
    pytest
    ```
- **Build/Artifacts:**
  - No explicit build step; outputs are written to `models/artifacts/` and `data/processed/`.
- **Docker:**
  - Use `dockerfile` for containerization. Build with:
    ```bash
    docker build -t ml_project .
    ```

## Conventions & Patterns
- **Imports:** Use absolute imports from `src/`.
- **Logging:** Use `src/utils/logger.py` for unified logging.
- **Config Access:** Always load configs via `src/utils/config.py`.
- **Data Flow:** Raw data → interim → processed → model → artifacts.
- **Environment Switching:** Use different YAML files in `configs/` for dev/prod.

## Integration Points
- **External Dependencies:**
  - All Python dependencies are listed in `requirements.txt` and/or `pyproject.toml`.
  - Dockerfile for containerized runs.
- **Cross-Component Communication:**
  - Data and models are passed via file system (data/ and models/ folders).

## Examples
- To run a training pipeline with dev config:
  ```bash
  python main.py --config configs/dev.yaml
  ```
- To test data loader:
  ```bash
  pytest tests/test_data.py
  ```

## References
- See `README.md` for setup and basic usage.
- See `src/` for code structure and conventions.
- See `configs/` for environment-specific settings.

---
If any section is unclear or missing, please provide feedback for further refinement.
