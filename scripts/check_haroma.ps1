# HaromaX6 — delegates to cross-platform ``scripts/run_checks.py`` (same as ``pixi run check_smoke``).
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
python (Join-Path $repoRoot "scripts\run_checks.py")
exit $LASTEXITCODE
