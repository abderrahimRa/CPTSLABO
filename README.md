# CPTS Lab Package

Main solver: `GOLDENMASTER2.py`

Included datasets:
- `brandimarte_instances/` for Brandimarte JSON instances
- `taillard_instances/` for Taillard JSPLIB instances

Requirement:
- `pip install -r requirements.txt`

Quick commands:

```powershell
python .\GOLDENMASTER2.py --benchmark-dir ".\brandimarte_instances" --benchmark-set brandimarte --runs 1 --output-csv ".\results\brandimarte_results.csv"
python .\GOLDENMASTER2.py --benchmark-dir ".\taillard_instances" --benchmark-set taillard --runs 1 --output-csv ".\results\taillard_results.csv"
```

Run all instances with many repetitions:

```powershell
python .\GOLDENMASTER2.py --benchmark-dir ".\brandimarte_instances" --benchmark-set brandimarte --runs 50 --output-csv ".\results\brandimarte_results.csv"
python .\GOLDENMASTER2.py --benchmark-dir ".\taillard_instances" --benchmark-set taillard --runs 50 --output-csv ".\results\taillard_results.csv"
```

Run both benchmark sets:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmarks.ps1 -BrandimarteRuns 1 -TaillardRuns 1
```

Run both benchmark sets with the same repetition count for every instance:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmarks.ps1 -Runs 50
```

Single instance examples:

```powershell
python .\GOLDENMASTER2.py .\brandimarte_instances\mk01.json --output-csv ".\results\mk01.csv"
python .\GOLDENMASTER2.py .\taillard_instances\ta01 --output-csv ".\results\ta01.csv"
```

Notes:
- The solver auto-loads `mkdata.json` for Brandimarte BKS values.
- The solver auto-loads `instances.json` for Taillard BKS and bounds.
- The built-in solver setting uses the current recommended time split: 60% CP and 40% tabu within the dynamic time budget.
- The CSV output contains one row per instance with averages across all runs.
- Printed and saved benchmark data includes BKS, CP average makespan, hybrid best makespan, hybrid average makespan, RPD, average time, average iterations, reached-BKS count, average time to BKS, and tabu improvement metrics.
