#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# HaromaX6 / Elarion — Full Linux Setup
#
# Installs Python, all pip dependencies, pretrained models, and
# downloads public training data.  Safe to re-run (idempotent).
#
# Usage:
#   chmod +x setup_linux.sh
#   sudo ./setup_linux.sh          # needs sudo for system packages
# ─────────────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
PYTHON_MIN="3.10"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "\n${CYAN}=== HaromaX6 Setup for Linux ===${NC}"

# ─────────────────────────────────────────────────────────────────────
# 1. System packages + Python
# ─────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/6] Checking Python...${NC}"

if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    echo -e "  Found: ${GREEN}$PY_VER${NC}"
else
    echo -e "  ${YELLOW}Python not found. Installing...${NC}"
    if command -v apt-get &>/dev/null; then
        # Debian / Ubuntu
        apt-get update -qq
        apt-get install -y -qq python3 python3-pip python3-venv python3-dev \
            build-essential libffi-dev libssl-dev git curl wget \
            libasound2-dev portaudio19-dev  # for audio sensors
    elif command -v dnf &>/dev/null; then
        # Fedora / RHEL
        dnf install -y -q python3 python3-pip python3-devel \
            gcc gcc-c++ make libffi-devel openssl-devel git curl wget \
            alsa-lib-devel portaudio-devel
    elif command -v pacman &>/dev/null; then
        # Arch
        pacman -Sy --noconfirm python python-pip base-devel git curl wget \
            alsa-lib portaudio
    elif command -v apk &>/dev/null; then
        # Alpine
        apk add --no-cache python3 py3-pip python3-dev gcc musl-dev \
            libffi-dev openssl-dev git curl wget
    else
        echo "  ERROR: Unsupported package manager. Install Python 3.10+ manually."
        exit 1
    fi
    PY_VER=$(python3 --version 2>&1)
    echo -e "  Installed: ${GREEN}$PY_VER${NC}"
fi

# Ensure pip is available
python3 -m ensurepip --upgrade 2>/dev/null || true

# Symlink python -> python3 if needed
if ! command -v python &>/dev/null; then
    ln -sf "$(command -v python3)" /usr/local/bin/python 2>/dev/null || true
fi

# ─────────────────────────────────────────────────────────────────────
# 2. Pip upgrade
# ─────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/6] Upgrading pip...${NC}"
python3 -m pip install --upgrade pip --quiet 2>/dev/null

# ─────────────────────────────────────────────────────────────────────
# 3. Pip dependencies
# ─────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[3/6] Installing Python packages...${NC}"

python3 -m pip install --quiet \
    numpy \
    flask \
    requests \
    transformers \
    sentence-transformers \
    Pillow \
    spacy \
    librosa \
    soundfile \
    fastapi \
    uvicorn \
    opencv-python-headless

# PyTorch CPU (most compatible; users with NVIDIA GPU can re-install
# with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121)
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "  ${YELLOW}Installing PyTorch (CPU)...${NC}"
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
else
    echo -e "  ${GREEN}PyTorch already installed.${NC}"
fi

# Optional sensor libraries
python3 -m pip install --quiet pyserial sounddevice pynmea2 2>/dev/null || true

# I2C / GPIO for Raspberry Pi (fail silently on desktop)
python3 -m pip install --quiet smbus2 adafruit-circuitpython-bno055 \
    adafruit-circuitpython-amg88xx adafruit-circuitpython-mlx90640 2>/dev/null || true

echo -e "  ${GREEN}Core packages installed.${NC}"

# ─────────────────────────────────────────────────────────────────────
# 4. Pretrained models
# ─────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/6] Downloading pretrained models...${NC}"

# spaCy English model
if ! python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    echo "  Downloading spaCy en_core_web_sm..."
    python3 -m spacy download en_core_web_sm --quiet
else
    echo -e "  ${GREEN}spaCy en_core_web_sm: already installed.${NC}"
fi

# HuggingFace sentence-transformers model
echo "  Pre-caching sentence-transformers/all-MiniLM-L6-v2..."
python3 -c "
from transformers import AutoTokenizer, AutoModel
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('  Downloading model weights...')
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('  Model cached.')
"

echo -e "  ${GREEN}Pretrained models ready.${NC}"

# ─────────────────────────────────────────────────────────────────────
# 5. Create data directories
# ─────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/6] Setting up directories...${NC}"
mkdir -p "$DATA_DIR/cognitive" "$DATA_DIR/downloads" "$DATA_DIR/training" "$PROJECT_ROOT/web"
echo -e "  ${GREEN}Directories ready.${NC}"

# ─────────────────────────────────────────────────────────────────────
# 6. Download public training data
# ─────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[6/6] Downloading public training data...${NC}"
python3 "$PROJECT_ROOT/scripts/download_training_data.py"

echo -e "\n${CYAN}=== Setup Complete ===${NC}"
echo -e "To start Elarion:"
echo -e "  cd $PROJECT_ROOT"
echo -e "  python3 main.py"
echo -e "  Then open http://localhost:8193\n"
