"""
Night 3 orchestrator — runs all the analysis additions in sequence.

Sequence:
  1. Data-volume sweep extension (~30-45 min)
       AutoARIMA × M4 at 4 sizes (8 fits)
       Traffic × {NBEATS, PatchTST, DLinear} at 4 sizes (72 fits)

  2. SeasonalNaive M4 difficulty check (~2 min)
       Verifies the n=2000 jump hypothesis

  3. DM test on NBEATS vs AutoARIMA on M4 (~10 sec)
       Pairwise statistical test as a second piece of evidence
       backing up the Nemenyi {NBEATS, AutoARIMA} clique

  4. Re-aggregation and re-plotting (~1 min)
       Re-runs aggregate_results.py to fold in any new CSVs
       Re-runs plot_data_volume.py to refresh the curves

Total expected wall-clock time: ~40-50 minutes

If any individual step fails, the orchestrator continues with the
next step (so a single failure doesn't kill all the work). Failures
are reported at the end.

Usage:
    python pipelines/run_night3.py
"""

import sys
import subprocess
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]


STEPS = [
    {
        "name": "1. Data-volume sweep extensions (AutoARIMA M4 + Traffic 3-model sweep)",
        "script": REPO_ROOT / "pipelines" / "run_data_volume_sweep_extension.py",
        "estimated_min": 45,
    },
    {
        "name": "2. SeasonalNaive M4 difficulty check",
        "script": REPO_ROOT / "analysis" / "sn_difficulty_check.py",
        "estimated_min": 2,
    },
    {
        "name": "3. Diebold-Mariano test (NBEATS vs AutoARIMA on M4)",
        "script": REPO_ROOT / "analysis" / "dm_test_nbeats_vs_autoarima.py",
        "estimated_min": 1,
    },
    {
        "name": "4. Re-aggregate main results",
        "script": REPO_ROOT / "analysis" / "aggregate_results.py",
        "estimated_min": 1,
    },
    {
        "name": "5. Re-plot data-volume curves",
        "script": REPO_ROOT / "analysis" / "plot_data_volume.py",
        "estimated_min": 1,
    },
]


def run_step(step: dict) -> tuple[bool, float]:
    """Run one step. Returns (success, elapsed_seconds)."""
    print(f"\n{'#'*70}")
    print(f"# {step['name']}")
    print(f"# Estimated: {step['estimated_min']} min")
    print(f"# Script:    {step['script']}")
    print(f"{'#'*70}\n")

    if not step["script"].exists():
        print(f"  ✗ Script not found: {step['script']}")
        return False, 0.0

    t0 = time.time()
    try:
        # Use the same Python interpreter that's running this orchestrator
        result = subprocess.run(
            [sys.executable, str(step["script"])],
            cwd=str(REPO_ROOT),
            check=False,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0
        if success:
            print(f"\n  ✓ Step completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        else:
            print(f"\n  ✗ Step exited with code {result.returncode} after {elapsed:.1f}s")
        return success, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ✗ Step crashed: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def main():
    print(f"\n{'='*70}")
    print(f"  NIGHT 3 ORCHESTRATOR")
    print(f"  Running {len(STEPS)} steps")
    print(f"  Estimated total: {sum(s['estimated_min'] for s in STEPS)} min")
    print(f"{'='*70}")

    t0 = time.time()
    results = []
    for step in STEPS:
        success, elapsed = run_step(step)
        results.append({"name": step["name"], "success": success, "elapsed": elapsed})

    total_elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  NIGHT 3 SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status}  {r['name']:<60} {r['elapsed']/60:5.1f} min")
    print(f"\n  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    failed = [r for r in results if not r["success"]]
    if failed:
        print(f"\n  {len(failed)} step(s) failed:")
        for r in failed:
            print(f"    - {r['name']}")
        sys.exit(1)
    else:
        print(f"\n  All steps succeeded.")


if __name__ == "__main__":
    main()
