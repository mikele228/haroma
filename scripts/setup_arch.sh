#!/usr/bin/env bash
# HaromaX6 — Arch Linux / Manjaro (pacman). Run as root.
# Usage: sudo bash scripts/setup_arch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_common.sh"

CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${CYAN}[setup_arch]${NC} $*"; }
err() { echo -e "${RED}[setup_arch]${NC} $*" >&2; }

if [[ "${EUID:-0}" -ne 0 ]]; then err "Run as root: sudo $0"; exit 1; fi
command -v pacman &>/dev/null || { err "pacman not found"; exit 1; }

log "Syncing and installing packages (pacman)..."
pacman -Sy --needed --noconfirm \
  base-devel cmake ninja pkgconf \
  python python-pip \
  git curl wget unzip \
  openblas lapack \
  ffmpeg libsndfile portaudio alsa-lib \
  v4l-utils \
  mesa libsm libxext libxrender glib2 \
  gpsd \
  iproute2 iw

haroma_venv_and_pip
haroma_generate_soul
haroma_spacy_en
haroma_data_dirs
haroma_training_data
haroma_print_footer_linux
