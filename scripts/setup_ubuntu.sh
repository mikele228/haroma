#!/usr/bin/env bash
# HaromaX6 — Ubuntu / Debian full install (intended for root). See scripts/setup_common.sh.
# Usage: sudo bash scripts/setup_ubuntu.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_common.sh"

CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${CYAN}[setup_ubuntu]${NC} $*"; }
err() { echo -e "${RED}[setup_ubuntu]${NC} $*" >&2; }

if [[ "${EUID:-0}" -ne 0 ]]; then err "Run as root: sudo $0"; exit 1; fi
if [[ ! -f /etc/os-release ]]; then err "Missing /etc/os-release"; exit 1; fi
# shellcheck source=/dev/null
. /etc/os-release
log "Detected: ${PRETTY_NAME:-$NAME}"

export DEBIAN_FRONTEND=noninteractive
log "Installing APT packages..."
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  ca-certificates curl wget git unzip \
  build-essential cmake ninja-build pkg-config \
  python3 python3-pip python3-venv python3-dev \
  libffi-dev libssl-dev libopenblas-dev \
  ffmpeg libsndfile1 \
  portaudio19-dev libasound2-dev libportaudio2 \
  v4l-utils libglib2.0-0 libsm6 libxext6 libxrender1 \
  gpsd gpsd-clients iproute2 iw
apt-get install -y -qq python-is-python3 2>/dev/null || true

haroma_venv_and_pip
haroma_generate_soul
haroma_spacy_en
haroma_data_dirs
haroma_training_data
haroma_print_footer_linux
