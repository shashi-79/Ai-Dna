"""
Unified CLI Benchmark Runner for AI DNA Experimental Suites.
Executes Experiments 1 to 6 and outputs validation metrics.
"""

import argparse
import sys
import os
import torch


# Add local package directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_dna.experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    run_experiment_5,
    run_experiment_6,
)


def main():
    parser = argparse.ArgumentParser(description="AI DNA Architecture Validation Benchmark Suite")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"],
        help="Experiment to execute (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in accelerated quick mode with reduced step count for fast validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cpu' or 'cuda')",
    )

    args = parser.parse_args()
    print(f"\n========================================================")
    print(f" Omni-Modal AI DNA Architecture Benchmark Harness")
    print(f" Device: {args.device} | Quick Mode: {args.quick}")
    print(f"========================================================\n")

    results = {}

    if args.experiment in ["all", "exp1"]:
        results["exp1"] = run_experiment_1(quick=args.quick, device_str=args.device)
        print("\n" + "-"*56 + "\n")

    if args.experiment in ["all", "exp2"]:
        results["exp2"] = run_experiment_2(quick=args.quick, device_str=args.device)
        print("\n" + "-"*56 + "\n")

    if args.experiment in ["all", "exp3"]:
        results["exp3"] = run_experiment_3(quick=args.quick, device_str=args.device)
        print("\n" + "-"*56 + "\n")

    if args.experiment in ["all", "exp4"]:
        results["exp4"] = run_experiment_4(quick=args.quick, device_str=args.device)
        print("\n" + "-"*56 + "\n")

    if args.experiment in ["all", "exp5"]:
        results["exp5"] = run_experiment_5(num_generations=2 if args.quick else 3, quick=args.quick, device_str=args.device)
        print("\n" + "-"*56 + "\n")

    if args.experiment in ["all", "exp6"]:
        results["exp6"] = run_experiment_6(quick=args.quick, device_str=args.device)
        print("\n" + "-"*56 + "\n")

    print("========================================================")
    print(" Benchmarks Completed Successfully!")
    print("========================================================\n")


if __name__ == "__main__":
    main()
