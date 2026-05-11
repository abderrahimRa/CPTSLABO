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

Use all logical CPU cores on a stronger machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmarks.ps1 -Runs 50 -CpWorkers 0
```

Use multiple processes so repeated runs keep the CPU busy:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmarks.ps1 -Runs 50 -CpWorkers 4 -ParallelRuns 4
```

Safer command for a lab PC that is unstable with many CP threads:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmarks.ps1 -Runs 50 -CpWorkers 4
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
- The tabu phase no longer stops when BKS is reached; it keeps searching for better solutions.
- If a machine is unstable during repeated CP runs, lower the worker count with `--cp-workers 4` or `-CpWorkers 4`.
- `LB == UB` or an explicit metadata `optimum` is treated as proven-optimal, so that run stops early and the benchmark moves to the next run or instance.
- A non-proven BKS or upper bound is only used as a reference; the search continues trying to beat it.
- `-CpWorkers 0` or `--cp-workers 0` means use all logical CPU cores.
- `-ParallelRuns N` or `--parallel-runs N` runs multiple independent repetitions at once using separate Python processes.
- On a strong CPU, a balanced setup is usually better than using all cores inside every single run. Example: on a 16-thread machine, try `-CpWorkers 4 -ParallelRuns 4`.

Research-oriented controls:

```powershell
# Fixed budgets with richer logging and a CP-only baseline.
python .\GOLDENMASTER2.py .\brandimarte_instances\mk01.json --cp-budget 5 --tabu-budget 20 --cp-starts 4 --tabu-starts 2 --route-topk 3 --cp-only-baseline --log-interval 2

# TS-heavy split from a fixed total budget.
python .\GOLDENMASTER2.py .\brandimarte_instances\mk01.json --total-budget 30 --cp-fraction 0.10 --cp-starts 4 --tabu-starts 4 --route-topk 3

# Balanced split for benchmark sweeps.
python .\GOLDENMASTER2.py --benchmark-dir ".\brandimarte_instances" --benchmark-set brandimarte --runs 20 --total-budget 40 --cp-fraction 0.50 --cp-starts 4 --tabu-starts 2 --cp-only-baseline --cp-workers 8 --parallel-runs 2 --output-csv ".\results\brandimarte_research.csv"
```

Important research flags:
- `--cp-budget`, `--tabu-budget`: explicit hybrid time split.
- `--total-budget`, `--cp-fraction`: split a fixed total budget between CP and tabu.
- `--cp-starts`: number of diversified CP starts before handoff.
- `--tabu-starts`: number of top CP starts to continue with tabu.
- `--route-topk`: exact-evaluate the top K routing insertion positions per reassignment.
- `--cp-only-baseline`: run a CP-only baseline with the same total budget.
- `--continue-after-bks`: keep tabu running even after a proven BKS is reached.
- `--log-interval`: periodic tabu progress logging while the search runs.
