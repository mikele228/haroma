#Requires -Version 5.1
<#
.SYNOPSIS
    HaromaX6 / Elarion — Windows setup (full requirements.txt + .venv).
.DESCRIPTION
    Creates a project virtual environment, installs dependencies from requirements.txt
    (including llama-cpp-python — needs Visual Studio Build Tools + CMake on PATH),
    spaCy data, optional HF cache warmup, and scripts/download_training_data.py.

    Run elevated if Python is missing (installer) or you use per-machine tools.
.NOTES
    Quick start (elevated PowerShell if needed):
        Set-ExecutionPolicy Bypass -Scope Process -Force
        cd C:\path\to\HaromaX6
        .\setup_windows.ps1

    For local GGUF builds: install "Desktop development with C++" workload and CMake,
    then re-run pip if llama-cpp-python failed.

    Environment:
        $env:HAROMA_SETUP_SKIP_TRAINING_DATA = "1"   — skip corpus download
        $env:HAROMA_SETUP_SKIP_SPACY = "1"           — skip en_core_web_sm
#>

$ErrorActionPreference = "Continue"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$DATA_DIR = Join-Path $PROJECT_ROOT "data"
$VENV = Join-Path $PROJECT_ROOT ".venv"
$PYEXE = Join-Path $VENV "Scripts\python.exe"
$PIP = Join-Path $VENV "Scripts\pip.exe"
$REQ = Join-Path $PROJECT_ROOT "requirements.txt"

function Write-Step($msg) { Write-Host "`n$msg" -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host "  $msg" -ForegroundColor Green }

Write-Host "`n=== HaromaX6 Setup for Windows ===" -ForegroundColor Cyan

# --- Python (prefer existing) ------------------------------------------------
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Step "[1/5] Python not on PATH — install from https://www.python.org/downloads/"
    Write-Host "  Enable 'Add python.exe to PATH', then re-run this script." -ForegroundColor Yellow
    exit 1
}
$pyVer = & python --version 2>&1
Write-Ok "Using $pyVer"

if (-not (Test-Path $REQ)) {
    Write-Host "ERROR: requirements.txt not found at $REQ" -ForegroundColor Red
    exit 1
}

# --- Virtual environment -----------------------------------------------------
Write-Step "[2/6] Creating virtualenv at .venv"
if (-not (Test-Path $PYEXE)) {
    & python -m venv $VENV
}
& $PYEXE -m pip install --upgrade pip setuptools wheel --quiet
Write-Ok "pip upgraded in venv"

# --- requirements.txt --------------------------------------------------------
Write-Step "[3/6] Installing from requirements.txt (may compile llama-cpp-python)..."
& $PYEXE -m pip install -r $REQ
if ($LASTEXITCODE -ne 0) {
    Write-Host "  pip install failed. Install Visual Studio Build Tools (C++ workload) and CMake, then run:" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor White
    Write-Host "  Or use: .\.venv\Scripts\pip install -r requirements-core.txt  (no llama-cpp-python)" -ForegroundColor Yellow
    exit $LASTEXITCODE
}
Write-Ok "Python packages installed"

# --- Soul identity (generated; not committed) --------------------------------
Write-Step "[4/6] Soul files (scripts/generate_soul.py)"
& $PYEXE (Join-Path $PROJECT_ROOT "scripts\generate_soul.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "  generate_soul.py failed — you can run later: .\.venv\Scripts\python scripts\generate_soul.py" -ForegroundColor Yellow
}

# --- spaCy -------------------------------------------------------------------
if ($env:HAROMA_SETUP_SKIP_SPACY -ne "1") {
    Write-Step "[5/6] spaCy en_core_web_sm"
    $spacyOk = & $PYEXE -c "import spacy; spacy.load('en_core_web_sm'); print('ok')" 2>$null
    if ($spacyOk -ne "ok") {
        & $PYEXE -m spacy download en_core_web_sm
    } else {
        Write-Ok "en_core_web_sm already present"
    }
} else {
    Write-Host "`n[5/6] Skipping spaCy (HAROMA_SETUP_SKIP_SPACY=1)" -ForegroundColor Yellow
}

# --- Dirs + training data ------------------------------------------------------
Write-Step "[6/6] Data directories and public training corpora"
foreach ($d in @(
        (Join-Path $DATA_DIR "cognitive"),
        (Join-Path $DATA_DIR "downloads"),
        (Join-Path $DATA_DIR "training"),
        (Join-Path $PROJECT_ROOT "web")
    )) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

if ($env:HAROMA_SETUP_SKIP_TRAINING_DATA -ne "1") {
    & $PYEXE (Join-Path $PROJECT_ROOT "scripts\download_training_data.py")
} else {
    Write-Host "  Skipping download_training_data.py (HAROMA_SETUP_SKIP_TRAINING_DATA=1)" -ForegroundColor Yellow
}

# --- Optional: warm sentence-transformers cache (same as legacy script) --------
try {
    Write-Host "`nPre-caching sentence-transformers/all-MiniLM-L6-v2 (optional)..." -ForegroundColor DarkGray
    & $PYEXE -c @'
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("  Model cache OK.")
'@
} catch {
    Write-Host "  (HF cache warmup skipped or failed — non-fatal)" -ForegroundColor DarkGray
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Activate and run:" -ForegroundColor White
Write-Host "  cd `"$PROJECT_ROOT`"" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor White
Write-Host "  http://localhost:8193`n" -ForegroundColor White
