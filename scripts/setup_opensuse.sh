#!/usr/bin/env bash
# HaromaX6 — openSUSE / SUSE (zypper). Run as root.
# Usage: sudo bash scripts/setup_opensuse.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_common.sh"

CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${CYAN}[setup_opensuse]${NC} $*"; }
err() { echo -e "${RED}[setup_opensuse]${NC} $*" >&2; }

if [[ "${EUID:-0}" -ne 0 ]]; then err "Run as root: sudo $0"; exit 1; fi
command -v zypper &>/dev/null || { err "zypper not found"; exit 1; }

log "Refreshing repositories and installing packages (zypper)..."
zypper --non-interactive refresh
zypper --non-interactive install -y \
  gcc gcc-c++ cmake ninja pkg-config \
  python3 python3-pip python3-devel \
  git curl wget unzip \
  libopenssl-devel libffi-devel \
  openblas-devel \
  ffmpeg libsndfile-devel portaudio-devel alsa-devel \
  v4l-utils \
  mesa libSM6 libXext6 glib2 \
  gpsd iproute2 iw

haroma_venv_and_pip
haroma_generate_soul
haroma_spacy_en
haroma_data_dirs
haroma_training_data
haroma_print_footer_linux
