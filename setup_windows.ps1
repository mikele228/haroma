#Requires -RunAsAdministrator
<#
.SYNOPSIS
    HaromaX6 / Elarion — Full Windows Setup
.DESCRIPTION
    Downloads and installs Python, all pip dependencies, pretrained models,
    and public training data.  Safe to re-run (idempotent).
.NOTES
    Run from an elevated PowerShell:
        Set-ExecutionPolicy Bypass -Scope Process -Force
        .\setup_windows.ps1
#>

$ErrorActionPreference = "Continue"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$DATA_DIR     = Join-Path $PROJECT_ROOT "data"
$PYTHON_VER   = "3.12.0"
$PYTHON_URL   = "https://www.python.org/ftp/python/$PYTHON_VER/python-$PYTHON_VER-amd64.exe"

Write-Host "`n=== HaromaX6 Setup for Windows ===" -ForegroundColor Cyan

# ─────────────────────────────────────────────────────────────────────
# 1. Python
# ─────────────────────────────────────────────────────────────────────
Write-Host "`n[1/6] Checking Python..." -ForegroundColor Yellow

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCmd) {
    $pyVer = & python --version 2>&1
    Write-Host "  Found: $pyVer" -ForegroundColor Green
} else {
    Write-Host "  Python not found. Downloading Python $PYTHON_VER..." -ForegroundColor Yellow
    $installer = Join-Path $env:TEMP "python-installer.exe"
    Invoke-WebRequest -Uri $PYTHON_URL -OutFile $installer -UseBasicParsing
    Write-Host "  Installing Python (this may take a minute)..."
    Start-Process -FilePath $installer -ArgumentList `
        "/quiet", "InstallAllUsers=1", "PrependPath=1", `
        "Include_pip=1", "Include_test=0" -Wait
    Remove-Item $installer -Force -ErrorAction SilentlyContinue

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + `
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $pyVer = & python --version 2>&1
    Write-Host "  Installed: $pyVer" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────────
# 2. Pip upgrade
# ─────────────────────────────────────────────────────────────────────
Write-Host "`n[2/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet 2>$null

# ─────────────────────────────────────────────────────────────────────
# 3. Pip dependencies
# ─────────────────────────────────────────────────────────────────────
Write-Host "`n[3/6] Installing Python packages..." -ForegroundColor Yellow

# Core requirements
python -m pip install --quiet `
    numpy `
    flask `
    requests `
    transformers `
    sentence-transformers `
    Pillow `
    spacy `
    librosa `
    soundfile `
    fastapi `
    uvicorn `
    opencv-python

# PyTorch CPU (most compatible on Windows; users with NVIDIA GPU
# can re-install with CUDA support afterwards)
$torchInstalled = python -c "import torch; print('ok')" 2>$null
if ($torchInstalled -ne "ok") {
    Write-Host "  Installing PyTorch (CPU)..." -ForegroundColor Yellow
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
} else {
    Write-Host "  PyTorch already installed." -ForegroundColor Green
}

# Optional sensor libraries (fail silently if hardware not present)
python -m pip install --quiet pyserial sounddevice pynmea2 2>$null

Write-Host "  Core packages installed." -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────
# 4. Pretrained models
# ─────────────────────────────────────────────────────────────────────
Write-Host "`n[4/6] Downloading pretrained models..." -ForegroundColor Yellow

# spaCy English model
$spacyModel = python -c "import spacy; spacy.load('en_core_web_sm'); print('ok')" 2>$null
if ($spacyModel -ne "ok") {
    Write-Host "  Downloading spaCy en_core_web_sm..."
    python -m spacy download en_core_web_sm --quiet
} else {
    Write-Host "  spaCy en_core_web_sm: already installed." -ForegroundColor Green
}

# HuggingFace sentence-transformers model (used by NeuralEncoder)
Write-Host "  Pre-caching sentence-transformers/all-MiniLM-L6-v2..."
python -c @"
from transformers import AutoTokenizer, AutoModel
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('  Downloading model weights...')
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('  Model cached.')
"@

Write-Host "  Pretrained models ready." -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────
# 5. Create data directories
# ─────────────────────────────────────────────────────────────────────
Write-Host "`n[5/6] Setting up directories..." -ForegroundColor Yellow
$dirs = @(
    (Join-Path $DATA_DIR "cognitive"),
    (Join-Path $DATA_DIR "downloads"),
    (Join-Path $DATA_DIR "training"),
    (Join-Path $PROJECT_ROOT "web")
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
}
Write-Host "  Directories ready." -ForegroundColor Green

# ─────────────────────────────────────────────────────────────────────
# 6. Download public training data
# ─────────────────────────────────────────────────────────────────────
Write-Host "`n[6/6] Downloading public training data..." -ForegroundColor Yellow
python (Join-Path $PROJECT_ROOT "scripts\download_training_data.py")

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "To start Elarion:" -ForegroundColor White
Write-Host "  cd $PROJECT_ROOT" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor White
Write-Host "  Then open http://localhost:8193`n" -ForegroundColor White
