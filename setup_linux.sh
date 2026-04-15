#!/usr/bin/env bash
# HaromaX6 — Linux entrypoint: detects distro and runs the matching installer.
#
# Prefer running the specific script if you know your distro:
#   sudo bash scripts/setup_ubuntu.sh      # Ubuntu, Debian, Mint, Raspberry Pi OS, Pop!_OS
#   sudo bash scripts/setup_fedora.sh      # Fedora, RHEL, Rocky, Alma
#   sudo bash scripts/setup_arch.sh        # Arch, Manjaro, EndeavourOS
#   sudo bash scripts/setup_alpine.sh      # Alpine (musl — see script notes)
#   sudo bash scripts/setup_opensuse.sh    # openSUSE, SUSE
#
# Usage:
#   chmod +x setup_linux.sh
#   sudo ./setup_linux.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -f /etc/os-release ]]; then
  echo "HaromaX6: /etc/os-release not found — install Python 3.10+ manually, then:"
  echo "  python3 -m venv .venv && source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

# shellcheck source=/dev/null
. /etc/os-release

ID_LC="${ID,,}"
VARIANT_ID_LC="${VARIANT_ID,,}"

route() {
  local script="$1"
  if [[ ! -f "${PROJECT_ROOT}/${script}" ]]; then
    echo "HaromaX6: missing ${script}" >&2
    exit 1
  fi
  echo "HaromaX6: detected ${PRETTY_NAME:-$ID} — running ${script}"
  exec bash "${PROJECT_ROOT}/${script}" "$@"
}

# Primary ID
case "${ID_LC}" in
  ubuntu|debian|raspbian|linuxmint|pop|elementary|zorin|kubuntu|xubuntu)
    route "scripts/setup_ubuntu.sh" "$@"
    ;;
  fedora|rocky|almalinux)
    route "scripts/setup_fedora.sh" "$@"
    ;;
  rhel|centos)
    # CentOS Stream / legacy: try Fedora script (dnf)
    if command -v dnf &>/dev/null; then
      route "scripts/setup_fedora.sh" "$@"
    fi
    ;;
  arch|manjaro|endeavouros|garuda|artix)
    route "scripts/setup_arch.sh" "$@"
    ;;
  alpine)
    route "scripts/setup_alpine.sh" "$@"
    ;;
esac

# openSUSE: ID can be opensuse-tumbleweed, opensuse-leap, opensuse
if [[ "${ID_LC}" == opensuse* ]] || [[ "${ID_LC}" == *suse* ]]; then
  route "scripts/setup_opensuse.sh" "$@"
fi

# ID_LIKE fallbacks (e.g. ID=linuxmint may still list debian)
for token in ${ID_LIKE:-}; do
  t="${token,,}"
  case "$t" in
    debian|ubuntu)
      route "scripts/setup_ubuntu.sh" "$@"
      ;;
    rhel|fedora)
      route "scripts/setup_fedora.sh" "$@"
      ;;
    arch)
      route "scripts/setup_arch.sh" "$@"
      ;;
  esac
done

echo "HaromaX6: distro '${ID:-unknown}' not auto-routed." >&2
echo "Install Python 3.10+, then pick one:" >&2
echo "  sudo bash scripts/setup_ubuntu.sh    # Debian-like" >&2
echo "  sudo bash scripts/setup_fedora.sh    # RPM/dnf" >&2
echo "  sudo bash scripts/setup_arch.sh      # pacman" >&2
echo "  sudo bash scripts/setup_alpine.sh    # apk" >&2
echo "  sudo bash scripts/setup_opensuse.sh  # zypper" >&2
exit 1
