#!/usr/bin/env bash
# HaromaX6 — Fedora / RHEL / Rocky / Alma (dnf). Run as root.
# Usage: sudo bash scripts/setup_fedora.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_common.sh"

CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${CYAN}[setup_fedora]${NC} $*"; }
err() { echo -e "${RED}[setup_fedora]${NC} $*" >&2; }

if [[ "${EUID:-0}" -ne 0 ]]; then err "Run as root: sudo $0"; exit 1; fi
command -v dnf &>/dev/null || { err "dnf not found"; exit 1; }

log "Installing RPM packages (dnf)..."
dnf install -y \
  gcc gcc-c++ cmake ninja-build pkgconf-pkg-config \
  python3 python3-pip python3-devel \
  git curl wget unzip \
  openblas-devel openssl-devel libffi-devel \
  ffmpeg libsndfile portaudio-devel alsa-lib-devel \
  v4l-utils \
  mesa-libGL libSM libXext libXrender glib2 \
  gpsd gpsd-clients \
  iproute iproute-tc iw

haroma_venv_and_pip
haroma_generate_soul
haroma_spacy_en
haroma_data_dirs
haroma_training_data
haroma_print_footer_linux
