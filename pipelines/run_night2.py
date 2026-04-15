"""
Night 2 orchestrator: runs all sensitivity / sweep / sanity experiments.

Order of operations (cheapest first, so any quick failure surfaces early):

  1. Sampling sanity check     (~10 min)
       LightGBM on M4 with stratified vs random sampling. Verifies that
       our stratification choice doesn't bias the model comparison.

  2. HP sensitivity            (~1.5 h)
       Four narrow studies on PatchTST, NBEATS, DLinear. Directly
       addresses "how sensitive are models to hyperparameters?" from
       the assignment's Key Question 3.

  3. Data-volume sweep         (~2.5 h)
       NBEATS / PatchTST / DLinear on M4 (4 sizes) and M5 (3 sizes).
       Directly addresses "do deep models overfit on smaller datasets?"
       — the headline plot for the report.

Total estimated runtime: ~4 hours on the 5070 Ti.

Each child pipeline is launched as a subprocess so a failure (or Windows
exit code 3221226505 false alarm on cleanup) cannot kill the orchestrator
or subsequent steps. Success is detected by checking whether the expected
output directory was populated, NOT by exit code — same Gemini-style fix
as the main run_all.py uses to handle the Windows process-cleanup bug.

Usage:
    python pipelines/run_night2.py
"""

import os
# DeepAR's MPS fallback flag (harmless on CUDA). Set BEFORE torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
# UTF-8 console: must run before any prints that contain ✓, ✗, ─, etc.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import subprocess
import time
from pathlib import Path

_REPO_ROOT     = Path(__file__).resolve().parents[1]
_PIPELINES_DIR = _REPO_ROOT / "pipelines"
_RESULTS_DIR   = _REPO_ROOT / "results"


# Each step: (name, script path, success_check_fn) where success_check_fn(results_dir) -> bool
def _has_sanity_outputs(results_dir: Path) -> bool:
    """Sanity check is successful if the comparison.csv exists and is non-trivial."""
    p = results_dir / "sampling_sanity" / "comparison.csv"
    return p.exists() and p.stat().st_size > 100


def _has_hp_sensitivity_outputs(results_dir: Path) -> bool:
    """HP sensitivity is successful if at least 3 of the 4 studies wrote a _combined.csv."""
    hp_dir = results_dir / "hp_sensitivity"
    if not hp_dir.exists():
        return False
    expected_studies = [
        "patchtst_patch_len_m4",
        "patchtst_lookback_m4",
        "nbeats_n_blocks_m4",
        "dlinear_lookback_traffic",
    ]
    found = sum(
        1 for s in expected_studies
        if (hp_dir / s / "_combined.csv").exists()
    )
    return found >= 3


def _has_sweep_outputs(results_dir: Path) -> bool:
    """Sweep is successful if at least one combined per-dataset file exists."""
    sweep_dir = results_dir / "data_volume"
    if not sweep_dir.exists():
        return False
    return (
        (sweep_dir / "M4_combined.csv").exists()
        or (sweep_dir / "M5_combined.csv").exists()
    )


PIPELINES = [
    ("Sampling Sanity", _PIPELINES_DIR / "run_sampling_sanity_check.py", _has_sanity_outputs),
    ("HP Sensitivity",  _PIPELINES_DIR / "run_hp_sensitivity.py",         _has_hp_sensitivity_outputs),
    ("Data-Volume Sweep", _PIPELINES_DIR / "run_data_volume_sweep.py",    _has_sweep_outputs),
]


def main():
    print(f"\n{'#'*60}")
    print(f"# DSS5104 CA2 — Night 2 Orchestrator")
    print(f"# {len(PIPELINES)} steps queued")
    print(f"{'#'*60}")

    total_start = time.time()
    statuses = []

    for name, script, check_fn in PIPELINES:
        if not script.exists():
            print(f"\n  Skipping {name}: script not found at {script}")
            statuses.append((name, "skip", "script missing"))
            continue

        t0 = time.time()
        cmd = [sys.executable, str(script)]

        print(f"\n{'#'*60}")
        print(f"# Launching: {name}")
        print(f"# Script   : {script.name}")
        print(f"{'#'*60}")

        # Subprocess isolation: a fatal Windows cleanup exit code or a
        # SIGSEGV-equivalent in the child cannot kill us.
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0

        # Gemini-style success detection: ignore exit code, check for outputs.
        # Windows sometimes returns 3221226505 (STATUS_STACK_BUFFER_OVERRUN)
        # on torch cleanup AFTER the actual work succeeded — exit code is
        # unreliable, file existence is not.
        ok_by_files = check_fn(_RESULTS_DIR)
        ok_by_code = (result.returncode == 0)

        if ok_by_files:
            tag = "OK" if ok_by_code else "OK (files exist; ignoring exit code)"
            statuses.append((name, "ok", f"{elapsed:.1f}s — {tag}"))
        else:
            rc = result.returncode
            statuses.append((name, "fail", f"FAILED — exit={rc} no outputs — {elapsed:.1f}s"))

    total_elapsed = time.time() - total_start

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# NIGHT 2 DONE — Total: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'#'*60}")
    print("\nPipeline summary:")
    for name, status, info in statuses:
        marker = {"ok": "[OK]", "fail": "[FAIL]", "skip": "[SKIP]"}.get(status, "[?]")
        print(f"  {marker:<8} {name:<22} {info}")


if __name__ == "__main__":
    main()
