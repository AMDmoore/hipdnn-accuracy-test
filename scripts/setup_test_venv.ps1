#requires -Version 5.1
<#
.SYNOPSIS
    Bootstrap a clean Python 3.14 venv that can run the HipDNN accuracy tests
    against a local OGA + MorphiZenEP build, with no global hacks.

.DESCRIPTION
    Creates a fresh venv at -VenvPath and wires up just enough so that
    ``python run_accuracy.py`` works against the user's local OGA runtime
    and gpu-test-package. Specifically:

      1. Creates the venv with -Python (default: 'py -3.14').
      2. pip-installs the test framework's runtime dependencies.
      3. Drops a single .pth file into the venv's site-packages that:
           - Adds -OgaRuntime to sys.path so 'import onnxruntime_genai'
             resolves to the local cp314 .pyd build, and
           - Calls os.add_dll_directory() for -OgaRuntime, -PackageDir/bin
             and -TherockDist/bin so the EP / ROCm DLLs load.

    EP registration itself (MorphiZenEP, etc.) is the repo's responsibility
    and lives in tests/_ep_bootstrap.py. This script does NOT install
    sitecustomize.py and does NOT monkey-patch __import__.

    All other env wiring (PATH prepend, HIP_CUSTOM_KERNELS_DIR, THEROCK_DIST)
    is performed at orchestrator start by config.setup_package_env using
    package_dir / therock_dist from test_config.json.

.PARAMETER VenvPath
    Where to create the new venv. Will refuse to overwrite unless -Force.

.PARAMETER OgaRuntime
    Directory containing onnxruntime_genai.cp314-win_amd64.pyd and
    onnxruntime-genai.dll.

.PARAMETER PackageDir
    Directory containing bin/onnxruntime_morphizen_ep.dll and lib/.

.PARAMETER TherockDist
    TheRock SDK root (must contain bin/ and lib/ for HIP / lld-link).

.PARAMETER Python
    Python launcher to bootstrap the venv. Must resolve to a 3.14 build
    matching the cp314 OGA .pyd ABI. Default: 'py -3.14'.

.PARAMETER Force
    Wipe an existing -VenvPath before creating.

.EXAMPLE
    .\setup_test_venv.ps1 `
        -VenvPath    C:\path\to\new\venv `
        -OgaRuntime  C:\path\to\oga-runtime `
        -PackageDir  C:\path\to\gpu-test-package `
        -TherockDist C:\path\to\therock-dist

.EXAMPLE
    .\setup_test_venv.ps1 -Force `
        -VenvPath    C:\tmp\venv `
        -OgaRuntime  C:\custom\oga `
        -PackageDir  C:\custom\pkg `
        -TherockDist C:\custom\therock
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$VenvPath,

    [Parameter(Mandatory = $true)]
    [string]$OgaRuntime,

    [Parameter(Mandatory = $true)]
    [string]$PackageDir,

    [Parameter(Mandatory = $true)]
    [string]$TherockDist,

    [string]$Python      = "py -3.14",

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Info($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------ validate
Step "Validating inputs"

foreach ($p in @($OgaRuntime, $PackageDir, $TherockDist)) {
    if (-not (Test-Path -LiteralPath $p -PathType Container)) {
        Fail "Directory does not exist: $p"
    }
}

$ogaPyd = Join-Path $OgaRuntime 'onnxruntime_genai.cp314-win_amd64.pyd'
if (-not (Test-Path -LiteralPath $ogaPyd)) {
    Fail "OGA .pyd not found: $ogaPyd (need a cp314 build for Python 3.14)"
}

$morphEp = Join-Path $PackageDir 'bin\onnxruntime_morphizen_ep.dll'
if (-not (Test-Path -LiteralPath $morphEp)) {
    Fail "MorphiZenEP DLL not found: $morphEp"
}

if (Test-Path -LiteralPath $VenvPath) {
    if ($Force) {
        Step "Removing existing venv at $VenvPath (-Force)"
        Remove-Item -LiteralPath $VenvPath -Recurse -Force
    } else {
        Fail "$VenvPath already exists. Re-run with -Force to overwrite."
    }
}

# ------------------------------------------------------------------ create venv
Step "Creating venv: $Python -m venv $VenvPath"

# $Python may be 'py -3.14' (two tokens) or 'C:\path\python.exe' (one token).
$pyTokens = $Python -split '\s+' | Where-Object { $_ }
$pyExe    = $pyTokens[0]
$pyArgs   = @($pyTokens | Select-Object -Skip 1) + @('-m', 'venv', $VenvPath)

& $pyExe @pyArgs
if ($LASTEXITCODE -ne 0) { Fail "venv creation failed (exit $LASTEXITCODE)" }

$venvPy = Join-Path $VenvPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPy)) { Fail "venv python missing: $venvPy" }

$venvVer = (& $venvPy -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if ($venvVer -ne '3.14') {
    Fail "venv python is $venvVer, but the OGA .pyd is cp314. Use -Python to point at a 3.14 build."
}
Info "venv python: $venvPy ($venvVer)"

# ------------------------------------------------------------------ pip install
Step "Upgrading pip"
& $venvPy -m pip install --upgrade pip --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed" }

Step "Installing test dependencies (pinned to current working venv)"
$pkgs = @(
    'transformers==5.6.2',
    'tokenizers==0.22.2',
    'huggingface_hub==1.11.0',
    'safetensors==0.7.0',
    'regex==2026.4.4',
    'numpy==2.4.4',
    'pandas==3.0.2',
    'colorama==0.4.6',
    'tqdm==4.67.3',
    'thefuzz==0.22.1',
    'onnx==1.21.0',
    'torch==2.11.0+cpu',
    'datasets'                  # PPL only; latest works
)
& $venvPy -m pip install --disable-pip-version-check `
    --index-url https://pypi.org/simple `
    --extra-index-url https://download.pytorch.org/whl/cpu `
    @pkgs
if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }

# ------------------------------------------------------------------ wire OGA via .pth
Step "Wiring local OGA build into venv (no sitecustomize, no __import__ patch)"

$siteDir = (& $venvPy -c "import site, sys; print(site.getsitepackages()[0])").Trim()
if (-not (Test-Path -LiteralPath $siteDir -PathType Container)) {
    Fail "Could not locate site-packages: $siteDir"
}
Info "site-packages: $siteDir"

# DLL search dir initializer. .pth file's 'import' line will trigger this
# at every Python startup. Scope is intentionally narrow: only adds DLL
# directories. EP registration is owned by tests/_ep_bootstrap.py.
$initPy = Join-Path $siteDir '_oga_dll_init.py'
$initContent = @"
"""Register Windows DLL search directories for the local OGA runtime + EP.

Imported at venv startup via _oga_dll_init.pth. Narrowly scoped: this file
ONLY makes the local OGA / MorphiZenEP / TheRock DLLs loadable. It does NOT
register any execution provider (that is handled by tests/_ep_bootstrap.py
when the test orchestrator launches subprocesses).
"""
import os

_DLL_DIRS = [
    r"$OgaRuntime",
    r"$PackageDir\bin",
    r"$TherockDist\bin",
]

for _d in _DLL_DIRS:
    if os.path.isdir(_d):
        try:
            os.add_dll_directory(_d)
        except (OSError, FileNotFoundError):
            pass
"@
Set-Content -LiteralPath $initPy -Value $initContent -Encoding UTF8
Info "wrote $initPy"

# .pth file: add OgaRuntime to sys.path AND trigger DLL init.
# Each non-comment line is either a path (added to sys.path) or
# 'import X' (executed at startup by site.py).
$pthFile = Join-Path $siteDir '_oga_dll_init.pth'
$pthContent = @"
$OgaRuntime
import _oga_dll_init
"@
Set-Content -LiteralPath $pthFile -Value $pthContent -Encoding UTF8
Info "wrote $pthFile"

# ------------------------------------------------------------------ smoke test
Step "Smoke-testing OGA import + EP DLL discovery"

$smoke = @"
import os, sys
print('python      :', sys.version.split()[0])
print('sys.path[0] :', sys.path[0])
# Make MorphiZenEP DLL discoverable on PATH for the registration test.
os.environ['PATH'] = r'$PackageDir\bin' + os.pathsep + os.environ.get('PATH','')
import onnxruntime_genai as og
print('og .pyd     :', og.__file__)
print('register fn :', hasattr(og, 'register_execution_provider_library'))
ep = r'$PackageDir\bin\onnxruntime_morphizen_ep.dll'
og.register_execution_provider_library('MorphiZenEP', ep)
print('registered  : MorphiZenEP ->', ep)
print('OK')
"@
$smokeFile = Join-Path $env:TEMP "_oga_smoke_$([guid]::NewGuid().ToString('N')).py"
Set-Content -LiteralPath $smokeFile -Value $smoke -Encoding UTF8
try {
    & $venvPy $smokeFile
    if ($LASTEXITCODE -ne 0) { Fail "smoke test failed (exit $LASTEXITCODE)" }
} finally {
    Remove-Item -LiteralPath $smokeFile -ErrorAction SilentlyContinue
}

# ------------------------------------------------------------------ done
Step "Done."
Write-Host ""
Write-Host "New venv ready: $VenvPath" -ForegroundColor Green
Write-Host ""
Write-Host "Activate and run the tests:" -ForegroundColor Yellow
Write-Host "  & '$venvPy' run_accuracy.py --tests MMLU"
Write-Host ""
Write-Host "Or activate first:" -ForegroundColor Yellow
Write-Host "  & '$VenvPath\Scripts\Activate.ps1'"
Write-Host "  python run_accuracy.py --tests MMLU"
