"""
Orchestrator: run all 9 model pipelines sequentially.

Each pipeline is launched as an isolated subprocess so that a C-level crash
(e.g. AutoARIMA's Fortran ARIMA code segfaulting on complex hourly data)
cannot kill the orchestrator or subsequent models. Python try/except cannot
catch SIGSEGV — subprocess isolation is the only reliable guard.

Usage:
    python pipelines/run_all.py                # Full run (all 9 models × 3 datasets)
    python pipelines/run_all.py --smoke-test   # Quick validation of all pipelines
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import subprocess

import time
from pathlib import Path

_PIPELINES_DIR = Path(__file__).resolve().parent

# Ordered list of (model_name, pipeline_script_path)
PIPELINES = [
    ("SeasonalNaive", _PIPELINES_DIR / "run_seasonal_naive.py"),
    ("AutoARIMA",     _PIPELINES_DIR / "run_auto_arima.py"),
    ("LightGBM",      _PIPELINES_DIR / "run_lightgbm.py"),
    ("PatchTST",      _PIPELINES_DIR / "run_patchtst.py"),
    ("NBEATS",        _PIPELINES_DIR / "run_nbeats.py"),
    ("TiDE",          _PIPELINES_DIR / "run_tide.py"),
    ("DeepAR",        _PIPELINES_DIR / "run_deepar.py"),
    ("DLinear",       _PIPELINES_DIR / "run_dlinear.py"),
    ("TimesNet",      _PIPELINES_DIR / "run_timesnet.py"),
]


def main(smoke_test: bool = False):
    mode = "SMOKE TEST" if smoke_test else "FULL RUN"
    print(f"\n{'#'*60}")
    print(f"# DSS5104 CA2 — Run All Models ({mode})")
    print(f"# {len(PIPELINES)} models × 3 datasets")
    print(f"{'#'*60}")

    total_start = time.time()
    statuses = []

    for model_name, script in PIPELINES:
        t0 = time.time()
        cmd = [sys.executable, str(script)]
        if smoke_test:
            cmd.append("--smoke-test")

        print(f"\n{'#'*60}")
        print(f"# Launching: {model_name}")
        print(f"{'#'*60}")

        # Run in a child process — isolates segfaults, OOM kills, and
        # any other fatal signal so the orchestrator keeps running.
        result = subprocess.run(cmd, check=False)

        elapsed = time.time() - t0
        
        # C-level teardown crashes (e.g. PyTorch DeepAR on Windows returning 3221226505 or -1073740791)
        # can happen during python exit AFTER the script completes successfully.
        # Verify success robustly by checking if the pipeline produced its final output file.
        final_file = Path(__file__).resolve().parents[1] / "results" / f"{model_name}_Traffic.csv"
        
        is_success = (result.returncode == 0)
        # Add a 1-second margin to t0 to avoid fractional second timing issues
        if not is_success and final_file.exists() and final_file.stat().st_mtime >= (t0 - 1.0):
            is_success = True

        if is_success:
            statuses.append((model_name, "✓", f"{elapsed:.1f}s"))
        else:
            rc = result.returncode
            reason = "SIGSEGV (C-level crash)" if rc == -11 or rc == 139 else f"exit code {rc}"
            statuses.append((model_name, "✗", f"FAILED — {reason} — {elapsed:.1f}s"))

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'#'*60}")
    print(f"# ALL DONE — Total: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'#'*60}")
    print("\nPipeline summary:")
    for name, status, info in statuses:
        print(f"  {status}  {name:<16} {info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all model pipelines sequentially.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run all pipelines with minimal settings")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
