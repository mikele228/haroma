# Install Python deps to maximize enabled training (encoder, backbone, VW, RL helpers).
# Run from repo root:  .\scripts\install_max_training.ps1
# Requires: Python 3.10+ on PATH.
#
# On Windows, llama-cpp-python may need VS Build Tools; if install fails, use:
#   pip install -r requirements-core.txt
# then: pip install -r requirements-training-extras.txt

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "[install_max_training] pip upgrade..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "[install_max_training] core requirements (full stack incl. llama-cpp-python)..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "[install_max_training] training extras (gymnasium, sklearn, vowpalwabbit)..." -ForegroundColor Cyan
python -m pip install -r requirements-training-extras.txt

Write-Host "[install_max_training] spaCy English model..." -ForegroundColor Cyan
python -m spacy download en_core_web_sm

Write-Host "[install_max_training] optional: download_training_data (network)..." -ForegroundColor Cyan
if (Test-Path "scripts\download_training_data.py") {
    python scripts\download_training_data.py
}

Write-Host "[install_max_training] done. Optional: pip install `"ray[rllib]`" for offline RLlib on JSONL." -ForegroundColor Green
