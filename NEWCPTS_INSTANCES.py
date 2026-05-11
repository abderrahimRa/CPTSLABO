"""
Instance-compatible runner for the existing NEWCPTS solver.

This keeps NEWCPTS.py on its original .mch/.job workflow while exposing the
same uploaded Brandimarte and Taillard instance support used by GOLDENMASTER2.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

from GOLDENMASTER2 import (
    FJSSPInstance,
    build_summary_row,
    collect_benchmark_files,
    compute_time_budgets,
    load_instance_targets,
    parse_instance_file,
    write_benchmark_summary_csv,
)
from NEWCPTS import _jobs_to_instance, cp_initial_solution, run_n5_tabu_search


def run_single_experiment(
    instance: FJSSPInstance,
    cp_budget: float,
    tabu_budget: float,
    bks: Optional[float] = None,
    optimal_target: Optional[float] = None,
    run_index: int = 0,
) -> Dict[str, Any]:
    del optimal_target

    t0 = time.time()
    print(f"\n  [RUN {run_index + 1}] Starting NEWCPTS instance experiment...")

    seq, assign, cp_ms = cp_initial_solution(
        instance,
        time_limit=cp_budget,
        stagnation_window=15.0,
    )

    cp_time = time.time() - t0
    print(f"  [CP-DONE] Time: {cp_time:.2f}s | Initial Makespan: {cp_ms:.2f}")
    print(f"  [CP->TS HANDOFF] CP Solution: {cp_ms:.2f} | Passing to NEWCPTS tabu search ({tabu_budget:.1f}s budget)...")

    _, final_ms, total_iters = run_n5_tabu_search(
        instance,
        seq,
        assign,
        time_limit=tabu_budget,
    )

    total_time = time.time() - t0
    reached_bks = bks is not None and final_ms <= bks
    ts_improvement = cp_ms - final_ms
    ts_improvement_pct = (ts_improvement / cp_ms * 100.0) if cp_ms > 0 else 0.0

    print(f"  [HYBRID_FINAL] Initial CP: {cp_ms:.2f} -> Final TS: {final_ms:.2f}")
    print(f"  [IMPROVEMENT] TS Gain: {ts_improvement:.2f} ({ts_improvement_pct:.2f}%) | Total Time: {total_time:.2f}s | Total Iters: {total_iters}")
    if bks is not None:
        gap = ((final_ms - bks) / bks) * 100.0
        print(f"  [GAP-TO-BKS] BKS: {bks:.2f} | Gap: {gap:.2f}% | Reached: {reached_bks}")

    return {
        "cp_ms": cp_ms,
        "final_ms": final_ms,
        "time_sec": total_time,
        "iters": float(total_iters),
        "last_drop_time": 0.0,
        "time_to_bks": 0.0,
        "reached_bks": reached_bks,
        "history": [],
        "ts_improvement": ts_improvement,
        "ts_improvement_pct": ts_improvement_pct,
        "first_improvement_time": 0.0,
        "ts_time": total_time - cp_time,
    }


def run_instance_benchmark(
    instance: FJSSPInstance,
    cp_budget: float,
    tabu_budget: float,
    bks: Optional[float],
    optimal_target: Optional[float],
    runs: int,
    inst_name: str,
) -> List[Dict[str, Any]]:
    run_records: List[Dict[str, Any]] = []
    for run_number in range(1, runs + 1):
        try:
            run_records.append(
                run_single_experiment(
                    instance,
                    cp_budget,
                    tabu_budget,
                    bks=bks,
                    optimal_target=optimal_target,
                    run_index=run_number - 1,
                )
            )
        except Exception as run_error:
            print(f"  [RUN-ERROR] {inst_name} | Run {run_number}/{runs} failed: {run_error}")
    return run_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runner for NEWCPTS using uploaded benchmark instances")
    parser.add_argument("instance", nargs="?", help="Single instance path (.json for Brandimarte or JSPLIB file for Taillard)")
    parser.add_argument("--benchmark-dir", help="Directory containing Brandimarte or Taillard instances")
    parser.add_argument("--benchmark-set", choices=["auto", "brandimarte", "taillard"], default="auto", help="Benchmark file selection preset")
    parser.add_argument("--format", choices=["auto", "brandimarte", "jsp"], default="auto", help="Single-file parser format")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-csv", default="newcpts_benchmark_summary.csv")
    parser.add_argument("--bks-csv", default="", help="CSV file with BKS values")
    parser.add_argument("--metadata-json", default="", help="Metadata JSON such as mkdata.json or instances.json")
    args = parser.parse_args()

    if args.benchmark_dir:
        benchmark_dir = args.benchmark_dir
        runs = max(1, args.runs)
        if not os.path.isdir(benchmark_dir):
            parser.error(f"benchmark directory not found: {benchmark_dir}")

        target_map = load_instance_targets(benchmark_dir, bks_csv=args.bks_csv, metadata_json=args.metadata_json)
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

                target_info = target_map.get(inst_name.lower(), {})
                bks_val = target_info.get("reference")
                optimal_target = target_info.get("optimal_target")

                print(f"\n  [Benchmark] Instance {inst_name} ({detected_format}) -> running {runs} times")
                run_records = run_instance_benchmark(
                    instance,
                    cp_budget,
                    tabu_budget,
                    bks=bks_val,
                    optimal_target=optimal_target,
                    runs=runs,
                    inst_name=inst_name,
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
                    f" | Successful Runs={successful_runs}/{runs}"
                )
                write_benchmark_summary_csv(args.output_csv, summary_rows)
            except Exception as instance_error:
                print(f"  [ERROR] Instance {inst_name} failed: {instance_error}")
        return

    if args.instance:
        if not os.path.isfile(args.instance):
            parser.error(f"instance not found: {args.instance}")

        jobs, _, detected_format = parse_instance_file(args.instance, args.format)
        instance, _ = _jobs_to_instance(jobs)
        cp_budget, tabu_budget = compute_time_budgets(instance)
        inst_name = os.path.splitext(os.path.basename(args.instance))[0]
        target_map = load_instance_targets(os.path.dirname(args.instance) or ".", bks_csv=args.bks_csv, metadata_json=args.metadata_json)
        target_info = target_map.get(inst_name.lower(), {})
        bks_val = target_info.get("reference")
        optimal_target = target_info.get("optimal_target")

        print(f"\n  [Single Instance] {inst_name} ({detected_format})")
        print(f"  [Budgets] CP={cp_budget:.1f}s | TS={tabu_budget:.1f}s")
        run_record = run_single_experiment(
            instance,
            cp_budget,
            tabu_budget,
            bks=bks_val,
            optimal_target=optimal_target,
            run_index=0,
        )
        summary_row = build_summary_row(inst_name, target_info, [run_record], 1)
        write_benchmark_summary_csv(args.output_csv, [summary_row])
        print(f"  [CSV] Wrote summary to: {args.output_csv}")
        return

    parser.error("provide either a single instance path or --benchmark-dir")


if __name__ == "__main__":
    main()
