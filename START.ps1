param([switch]$Demo, [int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m pip install uv
    if ($LASTEXITCODE -ne 0) { throw 'Could not install uv' }
    python -m uv sync --python 3.13 --locked
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare Python 3.13 environment' }
}
$hreLibrary = if ($Demo) { 'demo-library' } else { 'library' }
if ($Demo) {
    & '.\.venv\Scripts\python.exe' -m hre --library $hreLibrary demo
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed' }
}
Write-Host "Open http://127.0.0.1:$Port in your browser. Ctrl+C stops the server."
& '.\.venv\Scripts\python.exe' -m hre --library $hreLibrary serve --port $Port
