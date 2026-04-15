#!/usr/bin/env bash
# HaromaX6 — Alpine Linux (apk). Run as root.
#
# Note: Alpine uses musl libc. Some binary wheels (PyTorch, etc.) may differ from
# glibc distros; if pip fails, use Docker/Ubuntu or install torch from Alpine
# community testing. This script installs build deps and runs the same venv flow.
#
# Usage: sudo bash scripts/setup_alpine.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_common.sh"

CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${CYAN}[setup_alpine]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup_alpine]${NC} $*"; }
err() { echo -e "${RED}[setup_alpine]${NC} $*" >&2; }

if [[ "${EUID:-0}" -ne 0 ]]; then err "Run as root: sudo $0"; exit 1; fi
command -v apk &>/dev/null || { err "apk not found"; exit 1; }

warn "Alpine + musl: if pip install -r requirements.txt fails, try Ubuntu or Fedora, or use requirements-core.txt."
log "Installing packages (apk)..."
apk update
apk add --no-cache \
  bash build-base cmake ninja pkgconf \
  python3 py3-pip python3-dev \
  git curl wget unzip \
  linux-headers \
  openblas-dev \
  ffmpeg-dev ffmpeg-libs \
  libsndfile-dev portaudio-dev alsa-lib-dev \
  mesa-dev libsm-dev libxext-dev libxrender-dev glib-dev \
  v4l-utils \
  gpsd \
  iproute2 iw

haroma_venv_and_pip
haroma_generate_soul
haroma_spacy_en
haroma_data_dirs
haroma_training_data
haroma_print_footer_linux
