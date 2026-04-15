"""
Quick explorer: scan lightning_logs/ to understand the directory structure,
identify which model each version belongs to, and check what metrics are
available. Run this FIRST to see what we're working with before plotting.

Usage:
    python analysis/explore_lightning_logs.py
"""

import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

# The user said logs are under the repo root /lightning_logs
REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "lightning_logs"


def main():
    if not LOG_ROOT.exists():
        print(f"lightning_logs not found at {LOG_ROOT}")
        print("Check path and try again.")
        return

    versions = sorted(LOG_ROOT.glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
    print(f"Found {len(versions)} version directories in {LOG_ROOT}\n")

    summary_rows = []
    for v_dir in versions:
        v_num = int(v_dir.name.split("_")[1])
        info = {"version": v_num, "dir": str(v_dir)}

        # Read hparams.yaml to identify model
        hparams_path = v_dir / "hparams.yaml"
        if hparams_path.exists():
            try:
                with open(hparams_path, "r") as f:
                    hparams = yaml.safe_load(f)
                if hparams and isinstance(hparams, dict):
                    # Common keys that identify the model
                    info["h"] = hparams.get("h")
                    info["input_size"] = hparams.get("input_size")
                    info["max_steps"] = hparams.get("max_steps")
                    info["batch_size"] = hparams.get("batch_size")
                    info["learning_rate"] = hparams.get("learning_rate")
                    info["random_seed"] = hparams.get("random_seed")

                    # Model-specific keys that help identify which model this is
                    if "patch_len" in hparams:
                        info["model_type"] = "PatchTST"
                        info["patch_len"] = hparams.get("patch_len")
                    elif "stack_types" in hparams:
                        info["model_type"] = "NBEATS"
                    elif "decoder_output_dim" in hparams:
                        info["model_type"] = "TiDE"
                    elif "lstm_hidden_size" in hparams:
                        info["model_type"] = "DeepAR"
                        info["lstm_hidden_size"] = hparams.get("lstm_hidden_size")
                    elif "top_k" in hparams:
                        info["model_type"] = "TimesNet"
                    elif "encoder_layers" in hparams and "patch_len" not in hparams and "top_k" not in hparams:
                        # DLinear doesn't have encoder_layers in newer versions
                        # but might in older ones
                        info["model_type"] = "Unknown"
                    else:
                        # DLinear is the simplest — no unique keys
                        # Check if it's a very simple model
                        keys = set(hparams.keys())
                        complex_keys = {"patch_len", "stack_types", "decoder_output_dim",
                                       "lstm_hidden_size", "top_k", "num_kernels"}
                        if not keys.intersection(complex_keys):
                            info["model_type"] = "DLinear (probable)"
                        else:
                            info["model_type"] = "Unknown"
            except Exception as e:
                info["model_type"] = f"Error: {e}"
        else:
            info["model_type"] = "No hparams.yaml"

        # Check metrics.csv
        metrics_path = v_dir / "metrics.csv"
        if metrics_path.exists():
            try:
                metrics = pd.read_csv(metrics_path)
                info["metric_rows"] = len(metrics)
                info["metric_cols"] = ", ".join(metrics.columns.tolist())
                # Get last step number if available
                step_col = next((c for c in metrics.columns if "step" in c.lower()), None)
                if step_col:
                    info["last_step"] = int(metrics[step_col].max())
            except Exception as e:
                info["metric_rows"] = f"Error: {e}"
        else:
            info["metric_rows"] = 0
            info["metric_cols"] = "No metrics.csv"

        summary_rows.append(info)

    df = pd.DataFrame(summary_rows)

    # Print summary grouped by model type
    print("="*80)
    print("SUMMARY BY MODEL TYPE")
    print("="*80)
    if "model_type" in df.columns:
        for model_type, group in df.groupby("model_type"):
            print(f"\n  {model_type}: {len(group)} versions")
            if "h" in group.columns:
                h_vals = group["h"].dropna().unique()
                print(f"    Horizons (h): {sorted(h_vals)}")
            if "input_size" in group.columns:
                is_vals = group["input_size"].dropna().unique()
                print(f"    Input sizes: {sorted(is_vals)}")
            if "random_seed" in group.columns:
                seed_vals = group["random_seed"].dropna().unique()
                print(f"    Seeds: {sorted(seed_vals)}")
            if "metric_rows" in group.columns:
                print(f"    Metric rows range: {group['metric_rows'].min()} - {group['metric_rows'].max()}")

    # Print first 3 and last 3 versions in detail
    print("\n" + "="*80)
    print("FIRST 5 VERSIONS (detail)")
    print("="*80)
    for _, row in df.head(5).iterrows():
        print(f"\n  version_{row['version']}:")
        print(f"    Model: {row.get('model_type', '?')}")
        print(f"    h={row.get('h', '?')}, input_size={row.get('input_size', '?')}, "
              f"seed={row.get('random_seed', '?')}, max_steps={row.get('max_steps', '?')}")
        print(f"    Metrics: {row.get('metric_rows', 0)} rows")
        print(f"    Columns: {row.get('metric_cols', '?')}")

    print("\n" + "="*80)
    print("LAST 5 VERSIONS (detail)")
    print("="*80)
    for _, row in df.tail(5).iterrows():
        print(f"\n  version_{row['version']}:")
        print(f"    Model: {row.get('model_type', '?')}")
        print(f"    h={row.get('h', '?')}, input_size={row.get('input_size', '?')}, "
              f"seed={row.get('random_seed', '?')}, max_steps={row.get('max_steps', '?')}")
        print(f"    Metrics: {row.get('metric_rows', 0)} rows")
        print(f"    Columns: {row.get('metric_cols', '?')}")

    # Print one sample metrics.csv content
    print("\n" + "="*80)
    print("SAMPLE METRICS.CSV CONTENT (first version with data)")
    print("="*80)
    for v_dir in versions:
        metrics_path = v_dir / "metrics.csv"
        if metrics_path.exists():
            try:
                metrics = pd.read_csv(metrics_path)
                if len(metrics) > 0:
                    print(f"\n  From {v_dir.name}:")
                    print(f"  Columns: {metrics.columns.tolist()}")
                    print(f"  Shape: {metrics.shape}")
                    print(f"\n  First 10 rows:")
                    print(metrics.head(10).to_string(index=False))
                    print(f"\n  Last 5 rows:")
                    print(metrics.tail(5).to_string(index=False))
                    break
            except Exception:
                pass

    # Save the full summary for reference
    out_path = RESULTS_DIR / "learning_curves"
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "lightning_logs_inventory.csv", index=False)
    print(f"\n  Full inventory saved to {out_path / 'lightning_logs_inventory.csv'}")


if __name__ == "__main__":
    main()
