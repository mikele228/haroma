#!/usr/bin/env bash
# Shared steps for scripts/setup_*.sh (Linux). Source after PROJECT_ROOT is set.
# shellcheck shell=bash

HAROMA_VENV="${PROJECT_ROOT}/.venv"

haroma_check_requirements_file() {
  if [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
    echo "[haroma] ERROR: requirements.txt not found at ${PROJECT_ROOT}/requirements.txt" >&2
    return 1
  fi
}

haroma_venv_and_pip() {
  haroma_check_requirements_file
  echo "[haroma] Creating virtualenv at ${HAROMA_VENV}"
  python3 -m venv "${HAROMA_VENV}"
  # shellcheck disable=SC1090
  source "${HAROMA_VENV}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  echo "[haroma] pip install -r requirements.txt (may build llama-cpp-python)..."
  # shellcheck disable=SC2086
  python -m pip install ${HAROMA_PIP_EXTRA_ARGS:-} -r "${PROJECT_ROOT}/requirements.txt"
}

haroma_spacy_en() {
  if [[ "${HAROMA_SETUP_SKIP_SPACY:-0}" == "1" ]]; then
    echo "[haroma] Skipping spaCy (HAROMA_SETUP_SKIP_SPACY=1)."
    return 0
  fi
  if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    python -m spacy download en_core_web_sm
  else
    echo "[haroma] spaCy en_core_web_sm already present."
  fi
}

haroma_data_dirs() {
  mkdir -p "${PROJECT_ROOT}/data/cognitive" "${PROJECT_ROOT}/data/downloads" \
    "${PROJECT_ROOT}/data/training" "${PROJECT_ROOT}/web"
}

haroma_training_data() {
  if [[ "${HAROMA_SETUP_SKIP_TRAINING_DATA:-0}" == "1" ]]; then
    echo "[haroma] Skipping download_training_data.py (HAROMA_SETUP_SKIP_TRAINING_DATA=1)."
    return 0
  fi
  python "${PROJECT_ROOT}/scripts/download_training_data.py"
}

# Generate soul/*.json from scripts/soul_defaults (interactive if TTY).
haroma_generate_soul() {
  local py="${HAROMA_VENV}/bin/python"
  if [[ ! -x "$py" ]]; then
    py="python3"
  fi
  echo "[haroma] Soul identity (scripts/generate_soul.py)..."
  "$py" "${PROJECT_ROOT}/scripts/generate_soul.py" "$@"
}

haroma_print_footer_linux() {
  local g=$'\033[0;32m' nc=$'\033[0m'
  echo ""
  echo -e "${g}=== HaromaX6 setup complete ===${nc}"
  echo ""
  echo "Virtualenv: ${HAROMA_VENV}"
  echo ""
  echo "  source ${HAROMA_VENV}/bin/activate"
  echo "  cd ${PROJECT_ROOT}"
  echo "  python main.py"
  echo ""
  echo "Open http://localhost:8193"
  echo ""
  echo "GGUF models: ${PROJECT_ROOT}/models or .env — see .env.example."
  echo "GPU: reinstall torch / llama-cpp-python with CUDA if needed."
  echo ""
}
