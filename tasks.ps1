<#
.SYNOPSIS
    Windows task runner — the Makefile targets, without needing make.

.DESCRIPTION
    The project is developed on Windows where `make` is usually absent. Every target
    here mirrors the Makefile exactly, so the two cannot drift in what they do.

.EXAMPLE
    ./tasks.ps1 check
    ./tasks.ps1 train -Model mult -Seed 3
    ./tasks.ps1 experiments -Preset dev
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'install', 'data', 'train', 'eval', 'experiments', 'figures',
                 'report', 'serve', 'web', 'test', 'lint', 'typecheck', 'format',
                 'check', 'clean')]
    [string]$Target = 'help',

    [string]$Dataset = 'mosi',
    [string]$Model = 'mult',
    [int]$Seed = 0,
    [string]$Preset = 'smoke'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

switch ($Target) {
    'help' {
        Write-Host "Targets:" -ForegroundColor Cyan
        @(
            @{ n = 'install';     d = 'Create the environment and install pre-commit' },
            @{ n = 'data';        d = 'Build the feature cache and freeze the split' },
            @{ n = 'train';       d = 'Train one architecture (-Model -Dataset -Seed)' },
            @{ n = 'eval';        d = 'Sweep one checkpoint over the corruption grid' },
            @{ n = 'experiments'; d = 'Run a preset (-Preset smoke|dev|main|cross|mitigation)' },
            @{ n = 'figures';     d = 'Regenerate the paper figures' },
            @{ n = 'report';      d = 'Regenerate tables and the README headline' },
            @{ n = 'serve';       d = 'Run the API on http://localhost:8000' },
            @{ n = 'web';         d = 'Run the frontend on http://localhost:5173' },
            @{ n = 'test';        d = 'Run the test suite' },
            @{ n = 'lint';        d = 'ruff check' },
            @{ n = 'typecheck';   d = 'mypy --strict src' },
            @{ n = 'format';      d = 'ruff format + --fix' },
            @{ n = 'check';       d = 'lint + typecheck + test (what CI runs)' },
            @{ n = 'clean';       d = 'Remove caches and build artefacts' }
        ) | ForEach-Object { "  {0,-12} {1}" -f $_.n, $_.d | Write-Host }
    }
    'install'  { Invoke-Step 'uv sync' { uv sync --extra serve --extra viz }
                 Invoke-Step 'pre-commit install' { uv run pre-commit install } }
    'data'     { Invoke-Step 'wfb-data' { uv run wfb-data --dataset $Dataset } }
    'train'    { Invoke-Step 'wfb-train' { uv run wfb-train model=$Model data=$Dataset seed=$Seed } }
    'eval'     { Invoke-Step 'wfb-eval' { uv run wfb-eval model=$Model data=$Dataset seed=$Seed } }
    'experiments' { Invoke-Step 'run_all' { uv run python experiments/run_all.py --preset $Preset } }
    'figures'  { Invoke-Step 'figures' { uv run python -c "from wfb.reporting.figures import generate_all; [print('wrote', p) for p in generate_all()]" } }
    'report'   { Invoke-Step 'report' { uv run python -c "from pathlib import Path; from wfb.serving.results_store import ResultsStore; from wfb.reporting.tables import full_report, headline_table, update_readme; s = ResultsStore.load('experiments/results'); Path('experiments/results/REPORT.md').write_text(full_report(s), encoding='utf-8'); update_readme(Path('README.md'), headline_table(s)); print(headline_table(s))" } }
    'serve'    { Invoke-Step 'uvicorn' { uv run uvicorn wfb.serving.app:app --reload --port 8000 } }
    'web'      { Push-Location web; try { npm install; npm run dev } finally { Pop-Location } }
    'test'     { Invoke-Step 'pytest' { uv run pytest -q } }
    'lint'     { Invoke-Step 'ruff' { uv run ruff check src tests experiments } }
    'typecheck'{ Invoke-Step 'mypy' { uv run mypy --strict src } }
    'format'   { Invoke-Step 'ruff format' { uv run ruff format src tests experiments }
                 Invoke-Step 'ruff --fix' { uv run ruff check --fix src tests experiments } }
    'check'    { Invoke-Step 'ruff' { uv run ruff check src tests experiments }
                 Invoke-Step 'mypy' { uv run mypy --strict src }
                 Invoke-Step 'pytest' { uv run pytest -q } }
    'clean'    {
        foreach ($dir in '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov') {
            if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
        }
        Get-ChildItem -Recurse -Directory -Filter __pycache__ |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "cleaned" -ForegroundColor Green
    }
}
