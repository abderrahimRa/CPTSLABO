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

def write_benchmark_summary_csv(output_csv: str, rows: List[Dict[str, Any]]):
    fieldnames = ["Instance_Name", "BKS", "CP_Avg_Makespan", "Hybrid_Best_Makespan", "Hybrid_Avg_Makespan", 
                  "RPD_%", "Avg_Time_Sec", "Avg_Iterations", "Reached_BKS_Count", "Avg_Time_To_BKS_Sec",
                  "TS_Improvement_Avg", "TS_Improvement_Pct_Avg", "First_Improvement_Time_Avg"]
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
    if bks_csv and os.path.isfile(bks_csv):
        return load_bks_from_csv(bks_csv)

    if metadata_json and os.path.isfile(metadata_json):
        return load_bks_from_metadata_json(metadata_json)

    candidate_paths = [
        os.path.join(benchmark_dir, "mkdata.json"),
        os.path.join(benchmark_dir, "instances.json"),
        os.path.join(os.path.dirname(benchmark_dir), "mkdata.json"),
        os.path.join(os.path.dirname(benchmark_dir), "instances.json"),
    ]
    for candidate_path in candidate_paths:
        if os.path.isfile(candidate_path):
            return load_bks_from_metadata_json(candidate_path)

    return {}

def compute_time_budgets(instance: FJSSPInstance) -> Tuple[float, float]:
    num_jobs = len(instance)
    num_machines = max((m for job in instance for op in job for m, _ in op), default=0) + 1
    t_max = 0.72 * (num_jobs * num_machines)
    return t_max * 0.60, t_max * 0.40

def build_summary_row(inst_name: str, bks_val: Optional[float], run_records: List[Dict[str, Any]], runs: int) -> Dict[str, str]:
    avg_ms = sum(x["final_ms"] for x in run_records) / runs
    rpd = ((avg_ms - bks_val) / bks_val) * 100.0 if bks_val else 0.0
    avg_ts_improvement = sum(x.get("ts_improvement", 0.0) for x in run_records) / runs
    avg_ts_improvement_pct = sum(x.get("ts_improvement_pct", 0.0) for x in run_records) / runs
    time_to_bks_values = [x["time_to_bks"] for x in run_records if x["time_to_bks"] > 0]

    return {
        "Instance_Name": inst_name,
        "BKS": f"{bks_val:.2f}" if bks_val else "",
        "CP_Avg_Makespan": f"{(sum(x['cp_ms'] for x in run_records) / runs):.2f}",
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
    solver.parameters.num_search_workers = max(1, cp_workers)
    solver.parameters.search_branching = _cp_model.PORTFOLIO_SEARCH
    solver.parameters.linearization_level = 2
    solver.parameters.random_seed = run_seed

    cb = _CPProgressCallback(time_limit, stagnation_window)
    print(
        f"  [CP] Initializing with Native Seed (Seed: {run_seed} | "
        f"Time Limit: {time_limit:.1f}s | Workers: {max(1, cp_workers)}) ..."
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
            current_makespan
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
def run_sota_tabu_search(instance: FJSSPInstance, initial_mach_seq: List[List[Tuple[int,int]]], initial_assign: MachineAssignment, time_limit: float = 20.0, target_bks: float = -1.0):
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
    no_improve = 0
    last_best_ms = best_ms
    improvements_list = []

    while True:
        elapsed = time.time() - start_time
        if target_bks > 0 and global_best_ms <= target_bks:
            if not bks_logged:
                time_to_bks = elapsed
                bks_logged = True
                print(f"  [BKS-HIT] Reached BKS ({target_bks}) at {elapsed:.2f}s, continuing search for better solutions")
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

                best_k = -1
                best_k_est = float('inf')
                
                temp_seq = mach_seq[m_new]
                for k in range(len(temp_seq) + 1):
                    # Uses Heads/Tails + assign to evaluate insertion in O(1)
                    est = evaluator.get_o1_insertion_estimate(j, o, p_new, temp_seq, k, heads, tails, assign)
                    if est < best_k_est:
                        best_k_est = est
                        best_k = k
                        
                if best_k != -1:
                    n_seq = [list(s) for s in mach_seq]
                    n_seq[current_m].remove((j, o))
                    n_seq[m_new].insert(best_k, (j, o))
                    
                    n_assign = [row[:] for row in assign]
                    n_assign[j][o] = opt_idx
                    
                    c_ms, _, _, _, c_valid = evaluator.evaluate(n_seq, n_assign)
                    
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
        
    return global_best_seq, global_best_ms, it, last_drop_time, time_to_bks, history

def run_single_experiment(
    instance: FJSSPInstance,
    cp_budget: float,
    tabu_budget: float,
    bks: Optional[float] = None,
    run_index: int = 0,
    cp_workers: int = DEFAULT_CP_WORKERS,
) -> Dict[str, Any]:
    t0 = time.time()
    
    print(f"\n  [RUN {run_index + 1}] Starting experiment...")
    mach_seq, assign, cp_ms = cp_initial_solution(
        instance,
        time_limit=cp_budget,
        stagnation_window=15.0,
        run_seed=run_index * 13,
        cp_workers=cp_workers,
    )
    
    cp_time = time.time() - t0
    print(f"  [CP-DONE] Time: {cp_time:.2f}s | Initial Makespan: {cp_ms:.2f}")
    print(f"  [CP->TS HANDOFF] CP Solution: {cp_ms:.2f} | Passing to SOTA Tabu Search ({tabu_budget:.1f}s budget)...")
    
    _, final_ms, total_iters, last_drop_time, time_to_bks, history = run_sota_tabu_search(
        instance,
        mach_seq,
        assign,
        time_limit=tabu_budget,
        target_bks=bks if bks else -1.0,
    )
    total_time = time.time() - t0
    reached_bks = (bks is not None and final_ms <= bks)
    
    ts_improvement = cp_ms - final_ms
    ts_improvement_pct = (ts_improvement / cp_ms * 100.0) if cp_ms > 0 else 0
    first_improvement_time = history[0][0] if history else 0.0
    
    print(f"  [HYBRID_FINAL] Initial CP: {cp_ms:.2f} -> Final TS: {final_ms:.2f}")
    print(f"  [IMPROVEMENT] TS Gain: {ts_improvement:.2f} ({ts_improvement_pct:.2f}%) | Total Time: {total_time:.2f}s | Total Iters: {total_iters}")
    if bks:
        gap = ((final_ms - bks) / bks * 100.0)
        print(f"  [GAP-TO-BKS] BKS: {bks:.2f} | Gap: {gap:.2f}% | Reached: {reached_bks}")

    return {
        "cp_ms": cp_ms, "final_ms": final_ms, "time_sec": total_time,
        "iters": float(total_iters), "last_drop_time": last_drop_time,
        "time_to_bks": time_to_bks, "reached_bks": reached_bks, "history": history,
        "ts_improvement": ts_improvement, "ts_improvement_pct": ts_improvement_pct,
        "first_improvement_time": first_improvement_time, "ts_time": total_time - cp_time
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
    parser.add_argument("--cp-workers", type=int, default=DEFAULT_CP_WORKERS, help="CP-SAT worker threads per run (default: safer capped value)")
    parser.add_argument("--output-csv", default="benchmark_summary.csv")
    parser.add_argument("--bks-csv", default="", help="CSV file with BKS values (columns: instance_name, bks)")
    parser.add_argument("--metadata-json", default="", help="Metadata JSON such as mkdata.json or JSPLIB instances.json")
    args = parser.parse_args()

    if args.benchmark_dir:
        benchmark_dir, runs = args.benchmark_dir, max(1, args.runs)
        if not os.path.isdir(benchmark_dir):
            parser.error(f"benchmark directory not found: {benchmark_dir}")

        bks_map = load_bks_map(benchmark_dir, bks_csv=args.bks_csv, metadata_json=args.metadata_json)
        benchmark_files = collect_benchmark_files(benchmark_dir, args.benchmark_set)
        summary_rows = []

        print(f"\n  [Benchmark] Set: {args.benchmark_set} | Instances: {len(benchmark_files)} | Runs per instance: {runs}")

        for fname in benchmark_files:
            inst_name = os.path.splitext(fname)[0]
            try:
                inst_path = os.path.join(benchmark_dir, fname)
                jobs, _, detected_format = parse_instance_file(inst_path, args.format)
                instance, _ = _jobs_to_instance(jobs)
                cp_budget, tabu_budget = compute_time_budgets(instance)
                
                bks_val = bks_map.get(inst_name.lower())
                run_records = []
                
                print(f"\n  [Benchmark] Instance {inst_name} ({detected_format}) -> running {runs} times")
                for r in range(1, runs + 1):
                    try:
                        rec = run_single_experiment(
                            instance,
                            cp_budget,
                            tabu_budget,
                            bks=bks_val,
                            run_index=r-1,
                            cp_workers=args.cp_workers,
                        )
                        run_records.append(rec)
                    except Exception as run_error:
                        print(f"  [RUN-ERROR] {inst_name} | Run {r}/{runs} failed: {run_error}")

                successful_runs = len(run_records)
                if successful_runs == 0:
                    print(f"  [ERROR] Instance {inst_name} produced no successful runs")
                    continue

                summary_row = build_summary_row(inst_name, bks_val, run_records, successful_runs)
                summary_rows.append(summary_row)
                print(
                    f"  [Result] {inst_name} summary | Best={summary_row['Hybrid_Best_Makespan']}"
                    f" | Avg={summary_row['Hybrid_Avg_Makespan']} | RPD={summary_row['RPD_%'] or 'n/a'}%"
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
        cp_budget, tabu_budget = compute_time_budgets(instance)
        inst_name = os.path.splitext(os.path.basename(args.instance))[0]
        bks_map = load_bks_map(os.path.dirname(args.instance) or ".", bks_csv=args.bks_csv, metadata_json=args.metadata_json)
        bks_val = bks_map.get(inst_name.lower())

        print(f"\n  [Single Instance] {inst_name} ({detected_format})")
        print(f"  [Budgets] CP={cp_budget:.1f}s | TS={tabu_budget:.1f}s | CP Workers={args.cp_workers}")
        run_record = run_single_experiment(instance, cp_budget, tabu_budget, bks=bks_val, run_index=0, cp_workers=args.cp_workers)
        summary_row = build_summary_row(inst_name, bks_val, [run_record], 1)
        write_benchmark_summary_csv(args.output_csv, [summary_row])
        print(f"  [CSV] Wrote summary to: {args.output_csv}")
        return

    parser.error("provide either a single instance path or --benchmark-dir")

if __name__ == "__main__":
    main()
