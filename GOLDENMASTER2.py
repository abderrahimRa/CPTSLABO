"""
FJSSP  CP + N5/N6 SOTA Tabu Search Hybrid Solver
=========================================================
Upgrades:
1. O(1) Makespan Evaluation (Head/Tail DAG Estimation).
2. True N6 Routing Insertion Optimization.
3. Dynamic Tenure & Plateau Hashing.
4. Fast O(V) Ruin-and-Recreate (removed O(N^2) bottleneck).
"""

from __future__ import annotations
import collections
import os
import re
import sys
import time
import json
import argparse
import csv
import random
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional

try:
    from ortools.sat.python import cp_model as _cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("  [warn] OR-Tools not found. Install with: pip install ortools")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Core Types & Parsers
# ─────────────────────────────────────────────────────────────────────────────
FJSSPInstance   = List[List[List[Tuple[int, float]]]]
MachineAssignment = List[List[int]]
DEFAULT_CP_WORKERS = min(os.cpu_count() or 8, 8)

def normalize_cp_workers(cp_workers: int) -> int:
    return (os.cpu_count() or 8) if cp_workers <= 0 else cp_workers

class Job:
    def __init__(self, job_id: str, flexible_operations: List[List[Tuple[int, float]]]):
        self.id = job_id
        self.flexible_operations = flexible_operations

def parse_brandimarte_json(path: str) -> Tuple[List[Job], Dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    machines = int(data.get("machines", 0))
    jobs: List[Job] = []
    for j_idx, job_ops in enumerate(data.get("jobs", [])):
        ops = [[(int(alt["machine"]), float(alt["processing"])) for alt in op] for op in job_ops if op]
        jobs.append(Job(f"J{j_idx + 1}", ops))
    return jobs, {m: f"M{m}" for m in range(machines)}

def parse_jsplib_jsp(path: str) -> Tuple[List[Job], Dict[int, str]]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)

    if not lines:
        raise ValueError(f"Empty JSP instance: {path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid JSP header in {path}")

    num_jobs = int(header[0])
    num_machines = int(header[1])
    if len(lines) < num_jobs + 1:
        raise ValueError(f"Incomplete JSP instance: expected {num_jobs} job rows in {path}")

    jobs: List[Job] = []
    for j_idx in range(num_jobs):
        values = [int(token) for token in lines[j_idx + 1].split()]
        expected_values = 2 * num_machines
        if len(values) != expected_values:
            raise ValueError(
                f"Invalid JSP row {j_idx + 1} in {path}: expected {expected_values} integers, found {len(values)}"
            )

        ops: List[List[Tuple[int, float]]] = []
        for op_idx in range(num_machines):
            machine_id = values[2 * op_idx]
            proc_time = float(values[2 * op_idx + 1])
            ops.append([(machine_id, proc_time)])
        jobs.append(Job(f"J{j_idx + 1}", ops))

    return jobs, {m: f"M{m}" for m in range(num_machines)}

def detect_instance_format(path: str) -> str:
    if path.lower().endswith(".json"):
        return "brandimarte"

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            header = line.split()
            if len(header) >= 2 and all(token.lstrip("+-").isdigit() for token in header[:2]):
                return "jsp"
            break

    raise ValueError(f"Could not detect instance format for {path}")

def parse_instance_file(path: str, instance_format: str = "auto") -> Tuple[List[Job], Dict[int, str], str]:
    actual_format = detect_instance_format(path) if instance_format == "auto" else instance_format
    if actual_format == "brandimarte":
        jobs, machine_map = parse_brandimarte_json(path)
    elif actual_format == "jsp":
        jobs, machine_map = parse_jsplib_jsp(path)
    else:
        raise ValueError(f"Unsupported instance format: {actual_format}")
    return jobs, machine_map, actual_format

def load_bks_from_metadata_json(path: str) -> Dict[str, float]:
    bks_map = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                name = entry.get("name", "").lower()
                bounds = entry.get("bounds") or {}
                val = entry.get("optimum")
                if val is None:
                    val = bounds.get("upper")
                if val is not None: bks_map[name] = float(val)
    return bks_map

def load_bks_from_mkdata(path: str) -> Dict[str, float]:
    return load_bks_from_metadata_json(path)

def load_instance_targets_from_metadata_json(path: str) -> Dict[str, Dict[str, Any]]:
    targets: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return targets

    with open(path, "r", encoding="utf-8") as f:
        for entry in json.load(f):
            name = entry.get("name", "").lower()
            if not name:
                continue

            bounds = entry.get("bounds") or {}
            optimum = entry.get("optimum")
            lower = bounds.get("lower")
            upper = bounds.get("upper")

            lower_val = float(lower) if lower is not None else None
            upper_val = float(upper) if upper is not None else None
            optimum_val = float(optimum) if optimum is not None else None

            proven_optimal = optimum_val is not None or (
                lower_val is not None and upper_val is not None and abs(lower_val - upper_val) < 1e-9
            )
            reference = optimum_val if optimum_val is not None else upper_val
            optimal_target = optimum_val if optimum_val is not None else (upper_val if proven_optimal else None)

            targets[name] = {
                "reference": reference,
                "optimal_target": optimal_target,
                "lower_bound": lower_val,
                "upper_bound": upper_val,
                "proven_optimal": proven_optimal,
            }

    return targets

def load_bks_from_csv(csv_path: str) -> Dict[str, float]:
    """Load BKS values from a CSV file with columns: instance_name, bks (or similar)."""
    bks_map = {}
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column names
                name = row.get("instance") or row.get("instance_name") or row.get("name") or ""
                bks_val = row.get("bks") or row.get("BKS") or row.get("optimum") or ""
                if name and bks_val:
                    try:
                        bks_map[name.lower()] = float(bks_val)
                    except ValueError:
                        pass
    return bks_map

def load_instance_targets_from_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "reference": value,
            "optimal_target": value,
            "lower_bound": None,
            "upper_bound": value,
            "proven_optimal": True,
        }
        for name, value in load_bks_from_csv(csv_path).items()
    }

def write_benchmark_summary_csv(output_csv: str, rows: List[Dict[str, Any]]):
    fieldnames = [
        "Instance_Name",
        "Lower_Bound",
        "Upper_Bound",
        "BKS",
        "Proven_Optimal",
        "CP_Avg_Makespan",
        "CP_Only_TotalBudget_Avg_Makespan",
        "Hybrid_Best_Makespan",
        "Hybrid_Avg_Makespan",
        "RPD_%",
        "Avg_Time_Sec",
        "Avg_Iterations",
        "Reached_BKS_Count",
        "Avg_Time_To_BKS_Sec",
        "TS_Improvement_Avg",
        "TS_Improvement_Pct_Avg",
        "First_Improvement_Time_Avg",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _jobs_to_instance(jobs: List[Job]) -> Tuple[FJSSPInstance, bool]:
    """Convert Job objects to FJSSP instance format."""
    instance = [[[(m, t) for m, t in op] for op in job.flexible_operations] for job in jobs]
    is_flexible = any(len(op) > 1 for job in instance for op in job)
    return instance, is_flexible

def collect_benchmark_files(benchmark_dir: str, benchmark_set: str) -> List[str]:
    entries = sorted(
        f for f in os.listdir(benchmark_dir)
        if os.path.isfile(os.path.join(benchmark_dir, f))
    )
    brandimarte_files = [f for f in entries if re.fullmatch(r"mk\d+\.json", f.lower())]
    taillard_files = [f for f in entries if re.fullmatch(r"ta\d+", f.lower())]

    if benchmark_set == "brandimarte":
        return brandimarte_files
    if benchmark_set == "taillard":
        return taillard_files
    if benchmark_set == "auto":
        if brandimarte_files:
            return brandimarte_files
        if taillard_files:
            return taillard_files
        return [
            f for f in entries
            if f.lower() not in {"mkdata.json", "instances.json"}
            and (f.lower().endswith(".json") or "." not in f)
        ]
    raise ValueError(f"Unsupported benchmark set: {benchmark_set}")

def load_bks_map(benchmark_dir: str, bks_csv: str = "", metadata_json: str = "") -> Dict[str, float]:
    return {
        name: info["reference"]
        for name, info in load_instance_targets(benchmark_dir, bks_csv=bks_csv, metadata_json=metadata_json).items()
        if info.get("reference") is not None
    }

def load_instance_targets(benchmark_dir: str, bks_csv: str = "", metadata_json: str = "") -> Dict[str, Dict[str, Any]]:
    if bks_csv and os.path.isfile(bks_csv):
        return load_instance_targets_from_csv(bks_csv)

    if metadata_json and os.path.isfile(metadata_json):
        return load_instance_targets_from_metadata_json(metadata_json)

    candidate_paths = [
        os.path.join(benchmark_dir, "mkdata.json"),
        os.path.join(benchmark_dir, "instances.json"),
        os.path.join(os.path.dirname(benchmark_dir), "mkdata.json"),
        os.path.join(os.path.dirname(benchmark_dir), "instances.json"),
    ]
    for candidate_path in candidate_paths:
        if os.path.isfile(candidate_path):
            return load_instance_targets_from_metadata_json(candidate_path)

    return {}

def compute_time_budgets(instance: FJSSPInstance) -> Tuple[float, float]:
    num_jobs = len(instance)
    num_machines = max((m for job in instance for op in job for m, _ in op), default=0) + 1
    t_max = 0.72 * (num_jobs * num_machines)
    return t_max * 0.60, t_max * 0.40

def resolve_time_budgets(
    instance: FJSSPInstance,
    cp_budget_override: Optional[float] = None,
    tabu_budget_override: Optional[float] = None,
    total_budget_override: Optional[float] = None,
    cp_fraction: Optional[float] = None,
) -> Tuple[float, float, float, str]:
    default_cp_budget, default_tabu_budget = compute_time_budgets(instance)
    default_total_budget = default_cp_budget + default_tabu_budget
    total_budget = default_total_budget if total_budget_override is None else max(0.0, total_budget_override)

    if cp_budget_override is not None and tabu_budget_override is not None:
        cp_budget = max(0.0, cp_budget_override)
        tabu_budget = max(0.0, tabu_budget_override)
        total_budget = cp_budget + tabu_budget
        budget_mode = "explicit cp+tabu"
    elif cp_budget_override is not None:
        cp_budget = max(0.0, cp_budget_override)
        tabu_budget = max(0.0, total_budget - cp_budget)
        budget_mode = "explicit cp"
    elif tabu_budget_override is not None:
        tabu_budget = max(0.0, tabu_budget_override)
        cp_budget = max(0.0, total_budget - tabu_budget)
        budget_mode = "explicit tabu"
    else:
        cp_share = 0.60 if cp_fraction is None else cp_fraction
        cp_budget = total_budget * cp_share
        tabu_budget = total_budget - cp_budget
        budget_mode = "fractional split" if cp_fraction is not None else "dynamic default"

    return cp_budget, tabu_budget, total_budget, budget_mode

def _fmt_metric(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "n/a"

def format_target_summary(target_info: Dict[str, Any]) -> str:
    return (
        f"LB={_fmt_metric(target_info.get('lower_bound'))}"
        f" | UB={_fmt_metric(target_info.get('upper_bound'))}"
        f" | BKS={_fmt_metric(target_info.get('optimal_target'))}"
        f" | Proven={'yes' if target_info.get('proven_optimal') else 'no'}"
    )

def build_summary_row(inst_name: str, target_info: Dict[str, Any], run_records: List[Dict[str, Any]], runs: int) -> Dict[str, str]:
    bks_val = target_info.get("optimal_target")
    lower_bound = target_info.get("lower_bound")
    upper_bound = target_info.get("upper_bound")
    proven_optimal = bool(target_info.get("proven_optimal"))
    avg_ms = sum(x["final_ms"] for x in run_records) / runs
    rpd = ((avg_ms - bks_val) / bks_val) * 100.0 if bks_val else 0.0
    avg_ts_improvement = sum(x.get("ts_improvement", 0.0) for x in run_records) / runs
    avg_ts_improvement_pct = sum(x.get("ts_improvement_pct", 0.0) for x in run_records) / runs
    time_to_bks_values = [x["time_to_bks"] for x in run_records if x["time_to_bks"] > 0]
    cp_only_values = [x["cp_only_ms"] for x in run_records if x.get("cp_only_ms") is not None]

    return {
        "Instance_Name": inst_name,
        "Lower_Bound": f"{lower_bound:.2f}" if lower_bound is not None else "",
        "Upper_Bound": f"{upper_bound:.2f}" if upper_bound is not None else "",
        "BKS": f"{bks_val:.2f}" if bks_val else "",
        "Proven_Optimal": "yes" if proven_optimal else "no",
        "CP_Avg_Makespan": f"{(sum(x['cp_ms'] for x in run_records) / runs):.2f}",
        "CP_Only_TotalBudget_Avg_Makespan": f"{(sum(cp_only_values) / len(cp_only_values)):.2f}" if cp_only_values else "",
        "Hybrid_Best_Makespan": f"{min(x['final_ms'] for x in run_records):.2f}",
        "Hybrid_Avg_Makespan": f"{avg_ms:.2f}",
        "RPD_%": f"{rpd:.2f}" if bks_val else "",
        "Avg_Time_Sec": f"{(sum(x['time_sec'] for x in run_records) / runs):.2f}",
        "Avg_Iterations": f"{(sum(x['iters'] for x in run_records) / runs):.0f}",
        "Reached_BKS_Count": f"{sum(1 for x in run_records if x.get('reached_bks', False))}/{runs}",
        "Avg_Time_To_BKS_Sec": f"{(sum(time_to_bks_values) / len(time_to_bks_values)):.2f}" if time_to_bks_values else "",
        "TS_Improvement_Avg": f"{avg_ts_improvement:.2f}",
        "TS_Improvement_Pct_Avg": f"{avg_ts_improvement_pct:.2f}%",
        "First_Improvement_Time_Avg": f"{(sum(x.get('first_improvement_time', 0.0) for x in run_records) / runs):.2f}",
    }

def run_single_experiment_task(task: Tuple[FJSSPInstance, Dict[str, Any], int, Dict[str, Any]]) -> Dict[str, Any]:
    instance, target_info, run_index, run_config = task
    return run_single_experiment(instance, target_info=target_info, run_index=run_index, **run_config)

def run_instance_benchmark(
    instance: FJSSPInstance,
    target_info: Dict[str, Any],
    runs: int,
    parallel_runs: int,
    inst_name: str,
    run_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actual_parallel_runs = max(1, parallel_runs)
    if actual_parallel_runs == 1 or runs == 1:
        run_records: List[Dict[str, Any]] = []
        for r in range(1, runs + 1):
            try:
                rec = run_single_experiment(instance, target_info=target_info, run_index=r - 1, **run_config)
                run_records.append(rec)
            except Exception as run_error:
                print(f"  [RUN-ERROR] {inst_name} | Run {r}/{runs} failed: {run_error}")
        return run_records

    print(f"  [Parallel] {inst_name}: running up to {actual_parallel_runs} runs at once")
    tasks = [
        (instance, target_info, r - 1, run_config)
        for r in range(1, runs + 1)
    ]
    run_records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_parallel_runs) as executor:
        future_to_run = {
            executor.submit(run_single_experiment_task, task): (idx + 1)
            for idx, task in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_to_run):
            run_number = future_to_run[future]
            try:
                run_records.append(future.result())
                print(f"  [RUN-COMPLETE] {inst_name} | Run {run_number}/{runs} finished")
            except Exception as run_error:
                print(f"  [RUN-ERROR] {inst_name} | Run {run_number}/{runs} failed: {run_error}")
    return run_records

# ─────────────────────────────────────────────────────────────────────────────
# CP Solver (Native Seed Injection)
# ─────────────────────────────────────────────────────────────────────────────
class _CPProgressCallback(_cp_model.CpSolverSolutionCallback):
    def __init__(self, time_limit: float, stagnation_window: float = 2.0):
        super().__init__()
        self._time_limit = time_limit
        self._stagnation_window = stagnation_window
        self._start_time = time.time()
        self._last_obj, self._last_improve_time = None, time.time()
        self._iteration = 0
        self._solutions_found = 0

    def on_solution_callback(self):
        now = time.time()
        obj = self.objective_value
        self._iteration += 1
        if self._last_obj is None or obj < self._last_obj - 1e-6:
            self._last_obj, self._last_improve_time = obj, now
            self._solutions_found += 1
        if (now - self._start_time) >= self._time_limit or (now - self._last_improve_time) > self._stagnation_window:
            self.stop_search()

def cp_initial_solution(
    instance: FJSSPInstance,
    time_limit: float = 10.0,
    stagnation_window: float = 2.0,
    scale_factor: int = 100,
    run_seed: int = 0,
    cp_workers: int = DEFAULT_CP_WORKERS,
):
    actual_workers = normalize_cp_workers(cp_workers)
    model = _cp_model.CpModel()
    horizon = sum(max(int(round(t * scale_factor)) for _, t in op) for job in instance for op in job)
    all_tasks, mach_intervals = {}, collections.defaultdict(list)

    for job_id, job in enumerate(instance):
        for op_id, op_options in enumerate(job):
            start_var = model.new_int_var(0, horizon, f"s_{job_id}_{op_id}")
            end_var   = model.new_int_var(0, horizon, f"e_{job_id}_{op_id}")
            option_list, presence_vars = [], []

            for opt_idx, (machine_id, proc_time) in enumerate(op_options):
                dur = max(1, int(round(proc_time * scale_factor)))
                if len(op_options) == 1:
                    iv, pv = model.new_interval_var(start_var, dur, end_var, f"iv_{job_id}_{op_id}"), None
                else:
                    pv = model.new_bool_var(f"pv_{job_id}_{op_id}_m{machine_id}")
                    iv = model.new_optional_interval_var(start_var, dur, end_var, pv, f"iv_{job_id}_{op_id}_m{machine_id}")
                    presence_vars.append(pv)
                mach_intervals[machine_id].append(iv)
                option_list.append((machine_id, pv, iv, dur, opt_idx))
            if len(presence_vars) > 1: model.add_exactly_one(presence_vars)
            all_tasks[job_id, op_id] = (start_var, end_var, option_list)

    for m_id, ivs in mach_intervals.items(): model.add_no_overlap(ivs)
    for job_id, job in enumerate(instance):
        for op_id in range(len(job) - 1):
            model.add(all_tasks[job_id, op_id + 1][0] >= all_tasks[job_id, op_id][1])

    obj_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(obj_var, [all_tasks[job_id, len(job) - 1][1] for job_id, job in enumerate(instance) if job])
    model.minimize(obj_var)

    solver = _cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = actual_workers
    solver.parameters.search_branching = _cp_model.PORTFOLIO_SEARCH
    solver.parameters.linearization_level = 2
    solver.parameters.random_seed = run_seed

    cb = _CPProgressCallback(time_limit, stagnation_window)
    print(
        f"  [CP] Initializing with Native Seed (Seed: {run_seed} | "
        f"Time Limit: {time_limit:.1f}s | Workers: {actual_workers}) ..."
    )
    status = solver.solve(model, cb)

    if status not in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
        status_name = solver.status_name(status)
        raise RuntimeError(f"CP-SAT failed to find a feasible solution (status={status_name}, seed={run_seed})")

    assign = [[0] * len(job) for job in instance]
    
    op_events = []
    for job_id, job in enumerate(instance):
        for op_id, op_options in enumerate(job):
            start_t = solver.value(all_tasks[job_id, op_id][0]) / scale_factor
            if len(op_options) > 1:
                for (m_id, pv, _, _, opt_idx) in all_tasks[job_id, op_id][2]:
                    if pv is not None and solver.value(pv):
                        assign[job_id][op_id] = opt_idx
                        op_events.append((start_t, m_id, job_id, op_id))
                        break
            else:
                m_id, _, _, _, opt_idx = all_tasks[job_id, op_id][2][0]
                assign[job_id][op_id] = opt_idx
                op_events.append((start_t, m_id, job_id, op_id))

    op_events.sort(key=lambda x: x[0])
    num_machines = max((m for job in instance for op in job for m, _ in op), default=0) + 1
    mach_sequences = [[] for _ in range(num_machines)]
    for _, m_id, j, o in op_events:
        mach_sequences[m_id].append((j, o))

    return mach_sequences, assign, solver.objective_value / scale_factor

def run_cp_multistart(
    instance: FJSSPInstance,
    total_budget: float,
    cp_starts: int,
    cp_workers: int,
    cp_stagnation_window: float,
    run_seed_base: int,
    label: str,
) -> List[Dict[str, Any]]:
    actual_starts = max(1, cp_starts)
    budget_per_start = max(0.01, total_budget / actual_starts)
    candidates: List[Dict[str, Any]] = []

    print(
        f"  [{label}] Launching {actual_starts} CP starts | Budget/start={budget_per_start:.2f}s"
        f" | Stagnation={cp_stagnation_window:.2f}s | Workers={normalize_cp_workers(cp_workers)}"
    )

    for start_idx in range(actual_starts):
        seed = run_seed_base + start_idx * 97
        cp_t0 = time.time()
        mach_seq, assign, cp_ms = cp_initial_solution(
            instance,
            time_limit=budget_per_start,
            stagnation_window=cp_stagnation_window,
            run_seed=seed,
            cp_workers=cp_workers,
        )
        cp_time = time.time() - cp_t0
        candidate = {
            "seed": seed,
            "cp_ms": cp_ms,
            "time_sec": cp_time,
            "mach_seq": mach_seq,
            "assign": assign,
        }
        candidates.append(candidate)
        print(
            f"  [{label}] Start {start_idx + 1}/{actual_starts} | Seed={seed}"
            f" | Makespan={cp_ms:.2f} | Time={cp_time:.2f}s"
        )

    best_candidate = min(candidates, key=lambda c: (c["cp_ms"], c["time_sec"]))
    print(
        f"  [{label}] Best CP start | Seed={best_candidate['seed']}"
        f" | Makespan={best_candidate['cp_ms']:.2f}"
    )
    return candidates

# ─────────────────────────────────────────────────────────────────────────────
# O(1) SOTA DAG Evaluator (Heads & Tails)
# ─────────────────────────────────────────────────────────────────────────────
class DAGEvaluator:
    def __init__(self, instance: FJSSPInstance, num_machines: int):
        self.instance = instance
        self.num_jobs = len(instance)
        self.num_machines = num_machines
        
    def evaluate(self, mach_seq: List[List[Tuple[int, int]]], assign: MachineAssignment) -> Tuple[float, Dict[Tuple[int,int], float], Dict[Tuple[int,int], float], List[Tuple[int,int]], bool]:
        in_degree: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = collections.defaultdict(list)
        
        # 1. Build Precedence Edges
        for j in range(self.num_jobs):
            for o in range(len(self.instance[j]) - 1):
                adj[(j, o)].append((j, o + 1))
                in_degree[(j, o + 1)] += 1
                
        # 2. Build Machine Sequence Edges
        for m in range(self.num_machines):
            seq = mach_seq[m]
            for i in range(len(seq) - 1):
                adj[seq[i]].append(seq[i+1])
                in_degree[seq[i+1]] += 1
                
        # 3. Topological Sort
        queue: collections.deque[Tuple[int, int]] = collections.deque([(j, o) for j in range(self.num_jobs) for o in range(len(self.instance[j])) if in_degree[(j,o)] == 0])
        topo_order: List[Tuple[int, int]] = []
        
        while queue:
            u: Tuple[int, int] = queue.popleft()
            topo_order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        total_ops = sum(len(job) for job in self.instance)
        if len(topo_order) < total_ops:
            return float('inf'), {}, {}, [], False # Cycle detected

        # 4. Forward Pass (Heads / Earliest Start)
        heads: Dict[Tuple[int, int], float] = {u: 0.0 for u in topo_order}
        p_times: Dict[Tuple[int, int], float] = {}
        for u in topo_order:
            j, o = u
            p_times[u] = self.instance[j][o][assign[j][o]][1]
            for v in adj[u]:
                if heads[u] + p_times[u] > heads[v]:
                    heads[v] = heads[u] + p_times[u]
                    
        # 5. Backward Pass (Tails / Time to Sink)
        tails: Dict[Tuple[int, int], float] = {u: 0.0 for u in topo_order}
        for u in reversed(topo_order):
            for v in adj[u]:
                if tails[v] + p_times[v] > tails[u]:
                    tails[u] = tails[v] + p_times[v]
                    
        # 6. Global Makespan & Critical Path
        makespan = 0.0
        for u in topo_order:
            ms = heads[u] + p_times[u] + tails[u]
            if ms > makespan:
                makespan = ms

        critical_ops: List[Tuple[int, int]] = [u for u in topo_order if abs((heads[u] + p_times[u] + tails[u]) - makespan) < 1e-5]

        return makespan, heads, tails, critical_ops, True

    def get_o1_insertion_estimate(self, j: int, o: int, p_new: float, m_seq: List[Tuple[int, int]], k: int, heads: Dict, tails: Dict, assign: MachineAssignment) -> float:
        """
        O(1) Evaluation of inserting operation (j,o) into position k of a machine sequence, 
        using cached Head and Tail arrays. Uses actual assigned processing times.
        """
        # Job precedence constraints: time after previous op in same job
        r_job = heads.get((j, o-1), 0.0) + self.instance[j][o-1][assign[j][o-1]][1] if o > 0 else 0.0 
        q_job = tails.get((j, o+1), 0.0) + self.instance[j][o+1][assign[j][o+1]][1] if o < len(self.instance[j])-1 else 0.0
        
        # Machine sequence constraint: time after predecessor on same machine
        if k > 0:
            pred_m = m_seq[k-1]
            pred_opt = assign[pred_m[0]][pred_m[1]]
            r_mach = heads.get(pred_m, 0.0) + self.instance[pred_m[0]][pred_m[1]][pred_opt][1] 
        else:
            r_mach = 0.0
            
        # Machine sequence constraint: time before successor on same machine
        if k < len(m_seq):
            succ_m = m_seq[k]
            succ_opt = assign[succ_m[0]][succ_m[1]]
            q_mach = tails.get(succ_m, 0.0) + self.instance[succ_m[0]][succ_m[1]][succ_opt][1]
        else:
            q_mach = 0.0
            
        r_est = max(r_job, r_mach)
        q_est = max(q_job, q_mach)
        return r_est + p_new + q_est
    
    def get_o1_swap_estimate(self, u_op: Tuple[int, int], v_op: Tuple[int, int], heads: Dict, tails: Dict, p_times: Dict, current_makespan: float) -> float:
        """
        O(1) Evaluation of swapping adjacent operations u and v on same machine.
        After swap: v executes first, then u, both on same machine.
        Returns estimated new makespan.
        """
        j1, o1 = u_op
        j2, o2 = v_op
        
        # v now takes u's slot (runs first after both jobs' predecessors)
        r_v_new = max(heads.get(u_op, 0.0), heads.get(v_op, 0.0))
        # u now runs after v
        r_u_new = r_v_new + p_times.get(v_op, 1.0)
        
        # Estimated max completion among these two ops
        # All other non-swapped ops remain unchanged
        est_ms = max(
            r_v_new + p_times.get(v_op, 1.0) + tails.get(v_op, 0.0),
            r_u_new + p_times.get(u_op, 1.0) + tails.get(u_op, 0.0),
        )
        return est_ms

def ruin_and_recreate_fast(mach_seq: List[List[Tuple[int, int]]], instance: FJSSPInstance, assign: MachineAssignment, num_machines: int) -> Tuple[List[List[Tuple[int, int]]], MachineAssignment]:
    """Fast structural disruption to break plateaus without O(N^2) overhead."""
    new_assign: MachineAssignment = [row[:] for row in assign]
    new_mach_seq: List[List[Tuple[int, int]]] = [[] for _ in range(num_machines)]
    
    for j in range(len(instance)):
        for o in range(len(instance[j])):
            if random.random() < 0.15 and len(instance[j][o]) > 1:
                new_assign[j][o] = random.randint(0, len(instance[j][o]) - 1)
            
            m_id = instance[j][o][new_assign[j][o]][0]
            new_mach_seq[m_id].append((j, o))
            
    for m in range(num_machines):
        random.shuffle(new_mach_seq[m]) 
        
    return new_mach_seq, new_assign

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-Optimized Tabu Core
# ─────────────────────────────────────────────────────────────────────────────
def run_sota_tabu_search(
    instance: FJSSPInstance,
    initial_mach_seq: List[List[Tuple[int,int]]],
    initial_assign: MachineAssignment,
    time_limit: float = 20.0,
    reference_target: float = -1.0,
    bks_target: float = -1.0,
    stop_on_bks: bool = True,
    route_topk: int = 1,
    log_interval: float = 5.0,
):
    num_machines = max((m for job in instance for op in job for m, _ in op), default=0) + 1
    evaluator = DAGEvaluator(instance, num_machines)
    
    mach_seq = [list(seq) for seq in initial_mach_seq]
    assign = [row[:] for row in initial_assign]
    
    best_ms, heads, tails, critical_ops, valid = evaluator.evaluate(mach_seq, assign)
    if not valid: return mach_seq, float('inf'), 0, 0, 0, []

    global_best_ms = best_ms
    global_best_seq = [list(seq) for seq in mach_seq]
    global_best_assign = [row[:] for row in assign]

    tabu_list_seq, tabu_list_route = {}, {}
    history = []
    
    # NEW: Plateau hashing to penalize repetitive makespans
    ms_plateau_counts = collections.Counter()
    
    it = 0
    start_time = time.time()
    last_drop_time = 0.0
    time_to_bks = 0.0
    bks_logged = False
    reference_logged = False
    no_improve = 0
    last_best_ms = best_ms
    improvements_list = []
    next_status_log = log_interval if log_interval > 0 else float("inf")

    while True:
        elapsed = time.time() - start_time
        if bks_target > 0 and global_best_ms <= bks_target:
            if not bks_logged:
                time_to_bks = elapsed
                bks_logged = True
                if stop_on_bks:
                    print(f"  [BKS-HIT] Reached proven BKS ({bks_target}) at {elapsed:.2f}s, moving to next run")
                else:
                    print(f"  [BKS-HIT] Reached proven BKS ({bks_target}) at {elapsed:.2f}s, continuing search (--continue-after-bks)")
            if stop_on_bks:
                break
        if reference_target > 0 and global_best_ms <= reference_target and not reference_logged:
            if bks_target <= 0 or abs(reference_target - bks_target) > 1e-9:
                print(f"  [REF-HIT] Reached reference upper bound ({reference_target}) at {elapsed:.2f}s, continuing search")
            reference_logged = True
        if elapsed >= time_limit: break

        # Dynamic Tenures
        t_seq = random.randint(10 + len(critical_ops)//4, 15 + len(critical_ops)//2)
        t_route = random.randint(10, 20)

        if no_improve > 1500:
            mach_seq, assign = ruin_and_recreate_fast(global_best_seq, instance, global_best_assign, num_machines)
            best_ms, heads, tails, critical_ops, valid = evaluator.evaluate(mach_seq, assign)
            if not valid:
                mach_seq = [list(seq) for seq in global_best_seq]
                assign = [row[:] for row in global_best_assign]
                best_ms, heads, tails, critical_ops, valid = evaluator.evaluate(mach_seq, assign)
            no_improve = 0
            tabu_list_seq.clear()
            tabu_list_route.clear()
            ms_plateau_counts.clear()
            print(f"    [TS-Ruin] Iter {it} @ {elapsed:.2f}s: Plateau break triggered, restarting from best={global_best_ms:.2f}")
            continue

        best_neighbor_penalized = float('inf')
        best_neighbor_actual = float('inf')
        best_neighbor_seq = None
        best_neighbor_assign = None
        best_move_hash_seq = None
        best_move_hash_route = None
        n5_candidates = 0
        n6_exact_evals = 0

        # Compute p_times for O(1) swap estimate
        p_times_curr = {(j, o): instance[j][o][assign[j][o]][1] for j in range(len(instance)) for o in range(len(instance[j]))}

        # N5 Neighborhood: Critical Block Swaps (O(1) using cached heads/tails)
        for m in range(num_machines):
            seq = mach_seq[m]
            for i in range(len(seq) - 1):
                u_op: Tuple[int, int] = seq[i]
                v_op: Tuple[int, int] = seq[i+1]
                if u_op in critical_ops or v_op in critical_ops:
                    if u_op[0] == v_op[0]: continue 
                    n5_candidates += 1
                    
                    move_hash = frozenset([u_op, v_op])
                    is_tabu = tabu_list_seq.get(move_hash, 0) > it
                    
                    # O(1) estimate: don't call full evaluate()
                    c_ms = evaluator.get_o1_swap_estimate(u_op, v_op, heads, tails, p_times_curr, best_ms)
                    
                    # Plateau Penalization
                    penalty = 1.05 if ms_plateau_counts[c_ms] > 20 else 1.0
                    penalized_ms = c_ms * penalty
                    
                    if (not is_tabu and penalized_ms < best_neighbor_penalized) or (c_ms < global_best_ms):
                        best_neighbor_penalized = penalized_ms
                        best_neighbor_actual = c_ms
                        best_neighbor_seq = [list(s) for s in mach_seq]
                        best_neighbor_seq[m][i], best_neighbor_seq[m][i+1] = v_op, u_op
                        best_move_hash_seq = move_hash

        # N6 Neighborhood: Optimal Routing Insertion (Fixes the Bug)
        for (j, o) in critical_ops:
            current_opt = assign[j][o]
            op_alts = instance[j][o]
            if len(op_alts) < 2: continue
            
            current_m = op_alts[current_opt][0]
            
            for opt_idx, (m_new, p_new) in enumerate(op_alts):
                if opt_idx == current_opt: continue
                
                route_hash = (j, o, m_new)
                is_tabu = tabu_list_route.get(route_hash, 0) > it

                temp_seq = mach_seq[m_new]
                route_candidates = []
                for k in range(len(temp_seq) + 1):
                    est = evaluator.get_o1_insertion_estimate(j, o, p_new, temp_seq, k, heads, tails, assign)
                    route_candidates.append((est, k))

                for _, best_k in sorted(route_candidates, key=lambda item: item[0])[:max(1, route_topk)]:
                    n_seq = [list(s) for s in mach_seq]
                    n_seq[current_m].remove((j, o))
                    n_seq[m_new].insert(best_k, (j, o))
                    
                    n_assign = [row[:] for row in assign]
                    n_assign[j][o] = opt_idx
                    
                    c_ms, _, _, _, c_valid = evaluator.evaluate(n_seq, n_assign)
                    n6_exact_evals += 1
                    
                    if c_valid:
                        penalty = 1.05 if ms_plateau_counts[c_ms] > 20 else 1.0
                        penalized_ms = c_ms * penalty
                        
                        if (not is_tabu and penalized_ms < best_neighbor_penalized) or (c_ms < global_best_ms):
                            best_neighbor_penalized = penalized_ms
                            best_neighbor_actual = c_ms
                            best_neighbor_seq = n_seq
                            best_neighbor_assign = n_assign
                            best_move_hash_seq = None
                            best_move_hash_route = route_hash

        # Move acceptance
        if best_neighbor_seq is not None:
            mach_seq = best_neighbor_seq
            if best_neighbor_assign is not None:
                assign = best_neighbor_assign
                
            if best_move_hash_seq: tabu_list_seq[best_move_hash_seq] = it + t_seq
            if best_move_hash_route: tabu_list_route[best_move_hash_route] = it + t_route
            
            # Recalculate full DAG state for the accepted move
            best_ms, heads, tails, critical_ops, _ = evaluator.evaluate(mach_seq, assign)
            ms_plateau_counts[best_ms] += 1
            
            if best_ms < global_best_ms:
                global_best_ms = best_ms
                global_best_seq = [list(s) for s in mach_seq]
                global_best_assign = [row[:] for row in assign]
                no_improve = 0
                last_drop_time = elapsed
                improvement = last_best_ms - global_best_ms
                improvements_list.append((it, elapsed, global_best_ms, improvement))
                history.append((elapsed, global_best_ms))
                last_best_ms = global_best_ms
                print(f"    [TS-Improve] Iter {it} @ {elapsed:.2f}s: -> {global_best_ms:.2f} (d={improvement:.2f})")
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        it += 1
        if elapsed >= next_status_log:
            print(
                f"    [TS-STATUS] t={elapsed:.2f}s | iter={it} | current={best_ms:.2f}"
                f" | best={global_best_ms:.2f} | no_improve={no_improve} | critical={len(critical_ops)}"
                f" | n5={n5_candidates} | n6_exact={n6_exact_evals}"
            )
            next_status_log += log_interval
        
    return global_best_seq, global_best_ms, it, last_drop_time, time_to_bks, history

def run_single_experiment(
    instance: FJSSPInstance,
    cp_budget: float,
    tabu_budget: float,
    target_info: Optional[Dict[str, Any]] = None,
    run_index: int = 0,
    cp_workers: int = DEFAULT_CP_WORKERS,
    cp_stagnation_window: float = 15.0,
    cp_starts: int = 1,
    tabu_starts: int = 1,
    route_topk: int = 1,
    stop_on_bks: bool = True,
    log_interval: float = 5.0,
    run_cp_only_baseline: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    target_info = target_info or {}
    bks = target_info.get("optimal_target")
    reference_target = target_info.get("reference")
    total_budget = cp_budget + tabu_budget
    actual_tabu_starts = max(1, min(max(1, tabu_starts), max(1, cp_starts))) if tabu_budget > 0 else 0

    print(f"\n  [RUN {run_index + 1}] Starting experiment...")
    print(
        f"  [RUN-CONFIG] Targets: {format_target_summary(target_info)}"
        f" | CP={cp_budget:.2f}s | TS={tabu_budget:.2f}s | Total={total_budget:.2f}s"
    )
    print(
        f"  [RUN-CONFIG] CP starts={max(1, cp_starts)} | TS starts={actual_tabu_starts}"
        f" | Route top-k={max(1, route_topk)} | Stop on BKS={stop_on_bks}"
        f" | CP stagnation={cp_stagnation_window:.2f}s | CP workers={normalize_cp_workers(cp_workers)}"
    )

    cp_candidates = run_cp_multistart(
        instance,
        total_budget=cp_budget,
        cp_starts=cp_starts,
        cp_workers=cp_workers,
        cp_stagnation_window=cp_stagnation_window,
        run_seed_base=run_index * 1000,
        label="CP-HANDOFF",
    )
    cp_time = sum(candidate["time_sec"] for candidate in cp_candidates)
    best_cp_candidate = min(cp_candidates, key=lambda candidate: (candidate["cp_ms"], candidate["time_sec"]))
    selected_cp_candidates = sorted(cp_candidates, key=lambda candidate: (candidate["cp_ms"], candidate["time_sec"]))[:max(1, actual_tabu_starts or 1)]
    print(
        f"  [CP->TS HANDOFF] Selected {len(selected_cp_candidates)}/{len(cp_candidates)} CP starts"
        f" for tabu search | Best CP={best_cp_candidate['cp_ms']:.2f}"
    )

    cp_only_ms: Optional[float] = None
    if run_cp_only_baseline:
        cp_only_candidates = run_cp_multistart(
            instance,
            total_budget=total_budget,
            cp_starts=cp_starts,
            cp_workers=cp_workers,
            cp_stagnation_window=cp_stagnation_window,
            run_seed_base=run_index * 1000 + 50000,
            label="CP-BASELINE",
        )
        cp_only_ms = min(candidate["cp_ms"] for candidate in cp_only_candidates)
        print(f"  [CP-BASELINE] Best CP-only makespan with total budget: {cp_only_ms:.2f}")

    ts_results: List[Dict[str, Any]] = []
    total_ts_time = 0.0
    if tabu_budget > 0:
        tabu_budget_per_start = max(0.01, tabu_budget / max(1, actual_tabu_starts))
        for start_idx, candidate in enumerate(selected_cp_candidates, start=1):
            print(
                f"  [TS-START] Candidate {start_idx}/{len(selected_cp_candidates)} | Seed={candidate['seed']}"
                f" | CP makespan={candidate['cp_ms']:.2f} | TS budget={tabu_budget_per_start:.2f}s"
            )
            ts_t0 = time.time()
            _, final_ms, total_iters, last_drop_time, time_to_bks, history = run_sota_tabu_search(
                instance,
                candidate["mach_seq"],
                candidate["assign"],
                time_limit=tabu_budget_per_start,
                reference_target=reference_target if reference_target else -1.0,
                bks_target=bks if bks else -1.0,
                stop_on_bks=stop_on_bks,
                route_topk=route_topk,
                log_interval=log_interval,
            )
            total_ts_time += time.time() - ts_t0
            ts_results.append(
                {
                    "seed": candidate["seed"],
                    "cp_ms": candidate["cp_ms"],
                    "final_ms": final_ms,
                    "iters": total_iters,
                    "last_drop_time": last_drop_time,
                    "time_to_bks": time_to_bks,
                    "history": history,
                }
            )

    if ts_results:
        best_result = min(ts_results, key=lambda result: (result["final_ms"], result["cp_ms"]))
        cp_ms = best_result["cp_ms"]
        final_ms = best_result["final_ms"]
        total_iters = best_result["iters"]
        last_drop_time = best_result["last_drop_time"]
        time_to_bks = best_result["time_to_bks"]
        history = best_result["history"]
        selected_seed = best_result["seed"]
    else:
        cp_ms = best_cp_candidate["cp_ms"]
        final_ms = cp_ms
        total_iters = 0
        last_drop_time = 0.0
        time_to_bks = 0.0
        history = []
        selected_seed = best_cp_candidate["seed"]

    total_time = time.time() - t0
    reached_bks = (bks is not None and final_ms <= bks)

    ts_improvement = cp_ms - final_ms
    ts_improvement_pct = (ts_improvement / cp_ms * 100.0) if cp_ms > 0 else 0
    first_improvement_time = history[0][0] if history else 0.0

    print(
        f"  [HYBRID_FINAL] Selected seed={selected_seed} | Initial CP: {cp_ms:.2f}"
        f" -> Final TS: {final_ms:.2f} | Best CP among starts: {best_cp_candidate['cp_ms']:.2f}"
    )
    print(f"  [IMPROVEMENT] TS Gain: {ts_improvement:.2f} ({ts_improvement_pct:.2f}%) | Total Time: {total_time:.2f}s | Total Iters: {total_iters}")
    if cp_only_ms is not None:
        hybrid_advantage = cp_only_ms - final_ms
        hybrid_advantage_pct = (hybrid_advantage / cp_only_ms * 100.0) if cp_only_ms > 0 else 0.0
        print(
            f"  [CP-ONLY VS HYBRID] CP-only total budget: {cp_only_ms:.2f}"
            f" | Hybrid delta: {hybrid_advantage:.2f} ({hybrid_advantage_pct:.2f}%)"
        )
    if bks:
        gap = ((final_ms - bks) / bks * 100.0)
        print(f"  [GAP-TO-BKS] BKS: {bks:.2f} | Gap: {gap:.2f}% | Reached: {reached_bks}")
    elif reference_target:
        reference_gap = ((final_ms - reference_target) / reference_target * 100.0)
        print(f"  [GAP-TO-UB] UB: {reference_target:.2f} | Gap: {reference_gap:.2f}% | Proven BKS: no")

    return {
        "cp_ms": cp_ms, "final_ms": final_ms, "time_sec": total_time,
        "iters": float(total_iters), "last_drop_time": last_drop_time,
        "time_to_bks": time_to_bks, "reached_bks": reached_bks, "history": history,
        "ts_improvement": ts_improvement, "ts_improvement_pct": ts_improvement_pct,
        "first_improvement_time": first_improvement_time, "ts_time": total_ts_time,
        "cp_only_ms": cp_only_ms,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Lab-ready FJSSP/JSP benchmark runner")
    parser.add_argument("instance", nargs="?", help="Single instance path (.json for Brandimarte or JSPLIB file for Taillard)")
    parser.add_argument("--benchmark-dir", help="Directory containing Brandimarte or Taillard instances")
    parser.add_argument("--benchmark-set", choices=["auto", "brandimarte", "taillard"], default="auto", help="Benchmark file selection preset")
    parser.add_argument("--format", choices=["auto", "brandimarte", "jsp"], default="auto", help="Single-file parser format")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cp-workers", type=int, default=DEFAULT_CP_WORKERS, help="CP-SAT worker threads per run; use 0 for all logical CPU cores")
    parser.add_argument("--parallel-runs", type=int, default=1, help="Number of independent runs to execute in parallel per instance")
    parser.add_argument("--cp-budget", type=float, default=None, help="Explicit CP budget in seconds")
    parser.add_argument("--tabu-budget", type=float, default=None, help="Explicit tabu budget in seconds")
    parser.add_argument("--total-budget", type=float, default=None, help="Explicit total budget in seconds before splitting")
    parser.add_argument("--cp-fraction", type=float, default=None, help="CP share of the total budget, from 0.0 to 1.0")
    parser.add_argument("--cp-stagnation-window", type=float, default=15.0, help="Stop a CP start after this many seconds without improvement")
    parser.add_argument("--cp-starts", type=int, default=1, help="Number of diversified CP starts before tabu handoff")
    parser.add_argument("--tabu-starts", type=int, default=1, help="Number of top CP starts to continue with tabu search")
    parser.add_argument("--route-topk", type=int, default=1, help="Exact-evaluate the top K routing insertion positions per alternative")
    parser.add_argument("--log-interval", type=float, default=5.0, help="Seconds between tabu status updates; use 0 to disable")
    parser.add_argument("--continue-after-bks", action="store_true", help="Keep tabu running even after reaching a proven BKS")
    parser.add_argument("--cp-only-baseline", action="store_true", help="Also run a CP-only baseline with the full hybrid time budget")
    parser.add_argument("--output-csv", default="benchmark_summary.csv")
    parser.add_argument("--bks-csv", default="", help="CSV file with BKS values (columns: instance_name, bks)")
    parser.add_argument("--metadata-json", default="", help="Metadata JSON such as mkdata.json or JSPLIB instances.json")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.parallel_runs < 1:
        parser.error("--parallel-runs must be at least 1")
    if args.cp_starts < 1:
        parser.error("--cp-starts must be at least 1")
    if args.tabu_starts < 1:
        parser.error("--tabu-starts must be at least 1")
    if args.route_topk < 1:
        parser.error("--route-topk must be at least 1")
    if args.cp_fraction is not None and not 0.0 <= args.cp_fraction <= 1.0:
        parser.error("--cp-fraction must be between 0.0 and 1.0")
    for name in ("cp_budget", "tabu_budget", "total_budget", "cp_stagnation_window"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    shared_run_config = {
        "cp_workers": args.cp_workers,
        "cp_stagnation_window": args.cp_stagnation_window,
        "cp_starts": args.cp_starts,
        "tabu_starts": args.tabu_starts,
        "route_topk": args.route_topk,
        "stop_on_bks": not args.continue_after_bks,
        "log_interval": args.log_interval,
        "run_cp_only_baseline": args.cp_only_baseline,
    }

    if args.benchmark_dir:
        benchmark_dir, runs = args.benchmark_dir, max(1, args.runs)
        if not os.path.isdir(benchmark_dir):
            parser.error(f"benchmark directory not found: {benchmark_dir}")

        target_map = load_instance_targets(benchmark_dir, bks_csv=args.bks_csv, metadata_json=args.metadata_json)
        benchmark_files = collect_benchmark_files(benchmark_dir, args.benchmark_set)
        summary_rows = []

        print(
            f"\n  [Benchmark] Set: {args.benchmark_set} | Instances: {len(benchmark_files)}"
            f" | Runs per instance: {runs} | Parallel Runs: {max(1, args.parallel_runs)}"
        )
        print(
            f"  [Global Config] CP starts={args.cp_starts} | TS starts={args.tabu_starts}"
            f" | Route top-k={args.route_topk} | Continue after BKS={args.continue_after_bks}"
            f" | CP-only baseline={args.cp_only_baseline} | CP workers={normalize_cp_workers(args.cp_workers)}"
        )

        for fname in benchmark_files:
            inst_name = os.path.splitext(fname)[0]
            try:
                inst_path = os.path.join(benchmark_dir, fname)
                jobs, _, detected_format = parse_instance_file(inst_path, args.format)
                instance, _ = _jobs_to_instance(jobs)
                cp_budget, tabu_budget, total_budget, budget_mode = resolve_time_budgets(
                    instance,
                    cp_budget_override=args.cp_budget,
                    tabu_budget_override=args.tabu_budget,
                    total_budget_override=args.total_budget,
                    cp_fraction=args.cp_fraction,
                )
                
                target_info = target_map.get(inst_name.lower(), {})
                run_config = dict(shared_run_config)
                run_config.update({"cp_budget": cp_budget, "tabu_budget": tabu_budget})
                
                print(f"\n  [Benchmark] Instance {inst_name} ({detected_format}) -> running {runs} times")
                print(f"  [Targets] {format_target_summary(target_info)}")
                print(
                    f"  [Budget Plan] Mode={budget_mode} | Total={total_budget:.2f}s"
                    f" | CP={cp_budget:.2f}s | TS={tabu_budget:.2f}s"
                )
                run_records = run_instance_benchmark(
                    instance,
                    target_info=target_info,
                    runs=runs,
                    parallel_runs=args.parallel_runs,
                    inst_name=inst_name,
                    run_config=run_config,
                )

                successful_runs = len(run_records)
                if successful_runs == 0:
                    print(f"  [ERROR] Instance {inst_name} produced no successful runs")
                    continue

                summary_row = build_summary_row(inst_name, target_info, run_records, successful_runs)
                summary_rows.append(summary_row)
                print(
                    f"  [Result] {inst_name} summary | Best={summary_row['Hybrid_Best_Makespan']}"
                    f" | Avg={summary_row['Hybrid_Avg_Makespan']} | RPD={summary_row['RPD_%'] or 'n/a'}%"
                    f" | CP-only={summary_row['CP_Only_TotalBudget_Avg_Makespan'] or 'n/a'}"
                    f" | Successful Runs={successful_runs}/{runs}"
                )
                
                write_benchmark_summary_csv(args.output_csv, summary_rows)
            except Exception as e:
                print(f"  [ERROR] Instance {inst_name} failed: {str(e)}")
        return

    if args.instance:
        if not os.path.isfile(args.instance):
            parser.error(f"instance not found: {args.instance}")

        jobs, _, detected_format = parse_instance_file(args.instance, args.format)
        instance, _ = _jobs_to_instance(jobs)
        cp_budget, tabu_budget, total_budget, budget_mode = resolve_time_budgets(
            instance,
            cp_budget_override=args.cp_budget,
            tabu_budget_override=args.tabu_budget,
            total_budget_override=args.total_budget,
            cp_fraction=args.cp_fraction,
        )
        inst_name = os.path.splitext(os.path.basename(args.instance))[0]
        target_map = load_instance_targets(os.path.dirname(args.instance) or ".", bks_csv=args.bks_csv, metadata_json=args.metadata_json)
        target_info = target_map.get(inst_name.lower(), {})
        run_config = dict(shared_run_config)
        run_config.update({"cp_budget": cp_budget, "tabu_budget": tabu_budget})

        print(f"\n  [Single Instance] {inst_name} ({detected_format})")
        print(f"  [Targets] {format_target_summary(target_info)}")
        print(
            f"  [Budgets] Mode={budget_mode} | Total={total_budget:.2f}s"
            f" | CP={cp_budget:.2f}s | TS={tabu_budget:.2f}s | CP Workers={normalize_cp_workers(args.cp_workers)}"
        )
        run_record = run_single_experiment(
            instance,
            target_info=target_info,
            run_index=0,
            **run_config,
        )
        summary_row = build_summary_row(inst_name, target_info, [run_record], 1)
        write_benchmark_summary_csv(args.output_csv, [summary_row])
        print(f"  [CSV] Wrote summary to: {args.output_csv}")
        return

    parser.error("provide either a single instance path or --benchmark-dir")

if __name__ == "__main__":
    main()
