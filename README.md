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

Run both benchmark sets:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmarks.ps1 -BrandimarteRuns 1 -TaillardRuns 1
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
