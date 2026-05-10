param(
    [int]$BrandimarteRuns = 1,
    [int]$TaillardRuns = 1
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$solver = Join-Path $root "GOLDENMASTER2.py"
$resultsDir = Join-Path $root "results"

New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

Write-Host "Running Brandimarte benchmark set..."
python $solver --benchmark-dir (Join-Path $root "brandimarte_instances") --benchmark-set brandimarte --runs $BrandimarteRuns --output-csv (Join-Path $resultsDir "brandimarte_results.csv")

Write-Host "Running Taillard benchmark set..."
python $solver --benchmark-dir (Join-Path $root "taillard_instances") --benchmark-set taillard --runs $TaillardRuns --output-csv (Join-Path $resultsDir "taillard_results.csv")

Write-Host "Done. CSV files saved in $resultsDir"
