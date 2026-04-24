#requires -Version 5.1
<#
.SYNOPSIS
    One-line setup-and-run for the HipDNN accuracy test framework.
    Sets up a Python 3.14 venv (idempotent), patches test_config.json's
    three path fields, then runs `python run_accuracy.py` against the
    requested tests.

.DESCRIPTION
    Designed for the workflow:

        .\scripts\setup_test_venv.ps1 `
            -VenvPath    C:\work\venv314 `
            -OgaRuntime  C:\work\oga-runtime `
            -PackageDir  C:\work\gpu-test-package `
            -TherockDist C:\work\therock-dist `
            -ModelDir    C:\work\models\Llama-3.1-8B `
            -Tests       MMLU

    Re-running the same command later is fast: the venv is reused unless
    -Force is passed, so only the cheap steps (DLL wiring, config patch,
    smoke test, and the test itself) re-run.

    Specifically:

      1. If <VenvPath>\Scripts\python.exe doesn't exist (or -Force is
         passed), creates the venv with -Python (default: 'py -3.14') and
         pip-installs the framework's pinned runtime dependencies.
      2. Drops a single .pth file in the venv's site-packages that:
           - Adds -OgaRuntime to sys.path so 'import onnxruntime_genai'
             resolves to the local cp314 .pyd build, and
           - Calls os.add_dll_directory() for -OgaRuntime, -PackageDir\bin
             and -TherockDist\bin so EP / ROCm DLLs load.
      3. Patches the repo's existing test_config.json in place,
         substituting ONLY the three top-level path fields:
             model_dir    -> -ModelDir
             package_dir  -> -PackageDir
             therock_dist -> -TherockDist
         The genai_configs map and the per-test seq_lengths/params
         sections are left untouched.
      4. Runs a quick smoke test: imports onnxruntime_genai and registers
         MorphiZenEP against the local DLL.
      5. Execs `python run_accuracy.py --tests <Tests>` from the repo
         root and forwards its exit code.

    EP registration (MorphiZenEP, etc.) at test time is owned by
    tests/_ep_bootstrap.py. This script installs no sitecustomize.py and
    does not monkey-patch __import__.

.PARAMETER VenvPath
    Venv root. Reused if it already contains Scripts\python.exe; pass
    -Force to wipe and recreate.

.PARAMETER OgaRuntime
    Directory containing onnxruntime_genai.cp314-win_amd64.pyd and
    onnxruntime-genai.dll.

.PARAMETER PackageDir
    Directory containing bin\onnxruntime_morphizen_ep.dll and lib\.

.PARAMETER TherockDist
    TheRock SDK root (must contain bin\ and lib\ for HIP / lld-link).

.PARAMETER ModelDir
    OGA model directory containing model.onnx and the genai_config_*.json
    files referenced from test_config.json.

.PARAMETER Tests
    Tests to run, forwarded to `run_accuracy.py --tests`. If omitted,
    runs every test listed in test_config.json.

.PARAMETER Python
    Python launcher to bootstrap the venv. Must resolve to a 3.14 build
    matching the cp314 OGA .pyd ABI. Default: 'py -3.14'.

.PARAMETER Force
    Wipe the existing venv at -VenvPath before recreating.

.EXAMPLE
    # Fresh-machine setup AND run MMLU in one shot:
    .\scripts\setup_test_venv.ps1 `
        -VenvPath    C:\work\venv314 `
        -OgaRuntime  C:\work\oga-runtime `
        -PackageDir  C:\work\gpu-test-package `
        -TherockDist C:\work\therock-dist `
        -ModelDir    C:\work\models\Llama-3.1-8B `
        -Tests       MMLU

.EXAMPLE
    # Run all tests configured in test_config.json:
    .\scripts\setup_test_venv.ps1 `
        -VenvPath C:\work\venv314 -OgaRuntime ... -PackageDir ... `
        -TherockDist ... -ModelDir ...
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

    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [string[]]$Tests,

    [string]$Python      = "py -3.14",

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Info($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------ validate
Step "Validating inputs"

foreach ($p in @($OgaRuntime, $PackageDir, $TherockDist, $ModelDir)) {
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

# Locate the repo's test_config.json (script lives in <repo>/scripts/).
$repoRoot   = Split-Path -Parent $PSScriptRoot
$testConfig = Join-Path $repoRoot 'test_config.json'
if (-not (Test-Path -LiteralPath $testConfig)) {
    Fail "test_config.json not found at expected location: $testConfig"
}
Info "test_config.json: $testConfig"

$venvPy = Join-Path $VenvPath 'Scripts\python.exe'

if ($Force -and (Test-Path -LiteralPath $VenvPath)) {
    Step "Removing existing venv at $VenvPath (-Force)"
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

$venvAlreadyExists = Test-Path -LiteralPath $venvPy

# ------------------------------------------------------------------ create venv
if ($venvAlreadyExists) {
    Step "Reusing existing venv at $VenvPath (pass -Force to recreate)"
} else {
    Step "Creating venv: $Python -m venv $VenvPath"

    # $Python may be 'py -3.14' (two tokens) or 'C:\path\python.exe' (one token).
    $pyTokens = $Python -split '\s+' | Where-Object { $_ }
    $pyExe    = $pyTokens[0]
    $pyArgs   = @($pyTokens | Select-Object -Skip 1) + @('-m', 'venv', $VenvPath)

    & $pyExe @pyArgs
    if ($LASTEXITCODE -ne 0) { Fail "venv creation failed (exit $LASTEXITCODE)" }

    if (-not (Test-Path -LiteralPath $venvPy)) { Fail "venv python missing: $venvPy" }
}

$venvVer = (& $venvPy -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if ($venvVer -ne '3.14') {
    Fail "venv python is $venvVer, but the OGA .pyd is cp314. Use -Python to point at a 3.14 build (or -Force to recreate)."
}
Info "venv python: $venvPy ($venvVer)"

# ------------------------------------------------------------------ pip install
if (-not $venvAlreadyExists) {
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
} else {
    Info "Skipping pip install (venv already provisioned)"
}

# ------------------------------------------------------------------ wire OGA via .pth
Step "Wiring local OGA build into venv (no sitecustomize, no __import__ patch)"

# site.getsitepackages()[0] on Windows venvs returns the venv root, not
# the site-packages dir, so filter for the entry that actually ends in
# 'site-packages'.
$siteDir = (& $venvPy -c "import site; print(next(p for p in site.getsitepackages() if p.replace('\\','/').rstrip('/').endswith('site-packages')))").Trim()
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

# ------------------------------------------------------------------ patch test_config.json
Step "Patching test_config.json paths in place"

# Substitute ONLY the three top-level path keys, preserving the rest of the
# file (genai_configs map, per-test seq_lengths/params, formatting, comments,
# anything the user has customized) byte-for-byte. We use a regex on the raw
# text instead of round-tripping through ConvertFrom-Json/ConvertTo-Json,
# because PS 5.1's JSON serializer normalizes formatting and would clobber
# any non-default structure or comments.
function Set-JsonStringField {
    param([string]$Text, [string]$Key, [string]$Value)
    # Escape value for embedding inside a JSON string literal.
    $jsonEscaped = $Value.Replace('\', '\\').Replace('"', '\"')
    # Then escape '$' for [regex]::Replace's replacement-string syntax,
    # which treats $1, $&, etc. specially.
    $replSafe    = $jsonEscaped.Replace('$', '$$')
    $pattern     = '("' + [regex]::Escape($Key) + '"\s*:\s*)"[^"]*"'
    $rx          = [regex]::new($pattern)
    if (-not $rx.IsMatch($Text)) {
        Fail "test_config.json missing required top-level key: $Key"
    }
    $replacement = '${1}"' + $replSafe + '"'
    return $rx.Replace($Text, $replacement, 1)
}

# Normalize to forward slashes; that's what the orchestrator's existing
# config already uses and it sidesteps JSON-escaping headaches.
$mdJson = $ModelDir.Replace('\', '/')
$pdJson = $PackageDir.Replace('\', '/')
$tdJson = $TherockDist.Replace('\', '/')

$cfg = Get-Content -LiteralPath $testConfig -Raw -Encoding UTF8
$cfg = Set-JsonStringField -Text $cfg -Key 'model_dir'    -Value $mdJson
$cfg = Set-JsonStringField -Text $cfg -Key 'package_dir'  -Value $pdJson
$cfg = Set-JsonStringField -Text $cfg -Key 'therock_dist' -Value $tdJson

# Sanity-check it still parses, before we overwrite the file.
try {
    $null = $cfg | ConvertFrom-Json
} catch {
    Fail "Patched test_config.json failed to re-parse: $($_.Exception.Message)"
}

Set-Content -LiteralPath $testConfig -Value $cfg -Encoding UTF8 -NoNewline
Info "model_dir    = $mdJson"
Info "package_dir  = $pdJson"
Info "therock_dist = $tdJson"

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
try:
    og.register_execution_provider_library('MorphiZenEP', ep)
    print('registered  : MorphiZenEP ->', ep)
except RuntimeError as e:
    # On dev boxes with the legacy sitecustomize.py, MorphiZenEP is
    # already registered at venv startup. That's harmless for our
    # purposes -- it still proves the DLL is loadable.
    if 'already registered' in str(e).lower():
        print('registered  : MorphiZenEP (already registered, OK)')
    else:
        raise
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

# ------------------------------------------------------------------ run the test
$runArgs = @('run_accuracy.py')
if ($Tests -and $Tests.Count -gt 0) {
    $runArgs += '--tests'
    $runArgs += $Tests
    Step "Running: python $($runArgs -join ' ')   (cwd=$repoRoot)"
} else {
    Step "Running: python run_accuracy.py   (all tests in test_config.json, cwd=$repoRoot)"
}

Push-Location -LiteralPath $repoRoot
try {
    & $venvPy @runArgs
    $rc = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host ""
if ($rc -eq 0) {
    Write-Host "All done. Test run exit code: 0" -ForegroundColor Green
} else {
    Write-Host "Test run exit code: $rc" -ForegroundColor Red
}
Write-Host ""
Write-Host "Re-run anytime with the same command (venv is reused unless -Force):" -ForegroundColor Yellow
Write-Host "  .\scripts\setup_test_venv.ps1 -VenvPath $VenvPath -OgaRuntime $OgaRuntime ``"
Write-Host "      -PackageDir $PackageDir -TherockDist $TherockDist ``"
Write-Host "      -ModelDir $ModelDir -Tests <NAME>"
Write-Host ""
Write-Host "Or skip the bootstrapper entirely once the venv is set up:" -ForegroundColor Yellow
Write-Host "  cd $repoRoot"
Write-Host "  & '$venvPy' run_accuracy.py --tests <NAME>"

exit $rc
