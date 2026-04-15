"""
Generate training/validation learning curves for all iteratively-trained models.

Re-runs one representative fit per (model, dataset) with loss capture:
  - 6 neural models: uses a custom PyTorch Lightning Callback injected via
    configure_callbacks monkey-patch (logger property is read-only)
  - LightGBM: direct LGBMRegressor fit with eval_set + record_evaluation

Outputs:
  - results/learning_curves/<Model>_<Dataset>.csv
  - results/plots/learning_curve_<Model>_<Dataset>.png
  - results/plots/learning_curves_grid.png  (7x3 combined grid)

Runtime: ~50-60 min total.
"""

import sys
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

_THIS = Path(__file__).resolve()
if (_THIS.parent / "config.py").exists():
    PROJECT_ROOT = _THIS.parent
elif (_THIS.parent.parent / "config.py").exists():
    PROJECT_ROOT = _THIS.parent.parent
else:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from config import (
    M4_CONFIG, M5_CONFIG, TRAFFIC_CONFIG,
    SEEDS, MAX_STEPS, BATCH_SIZE, EARLY_STOP_PATIENCE, VAL_CHECK_STEPS,
    LR_TRANSFORMER, LR_MLP, LR_RNN,
    PATCHTST_PARAMS, NBEATS_PARAMS, TIDE_PARAMS, DEEPAR_PARAMS,
    TIMESNET_PARAMS, LGBM_PARAMS, RESULTS_DIR,
)
from data_prep.m4_prep import load_m4_monthly
from data_prep.m5_prep import load_m5
from data_prep.traffic_prep import load_traffic

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LC_DIR = RESULTS_DIR / "learning_curves"
PLOTS_DIR = RESULTS_DIR / "plots"
LC_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("M4",      M4_CONFIG,      load_m4_monthly),
    ("M5",      M5_CONFIG,      load_m5),
    ("Traffic", TRAFFIC_CONFIG, load_traffic),
]

SEED = SEEDS[0]

_PATCH_CONFIGS = {
    "M4": {"patch_len": 6, "stride": 3},
    "M5": {"patch_len": 7, "stride": 7},
    "Traffic": {"patch_len": 24, "stride": 12},
}
_DEEPAR_HIDDEN = {"M4": 128, "M5": 128, "Traffic": 256}
_TIMESNET_LAYERS = {"M4": 2, "M5": 2, "Traffic": 3}

# =====================================================================
#  Custom Lightning Callback for loss capture
# =====================================================================

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl


class LossCollector(pl.Callback):
    """Records training and validation loss at each step/check."""

    def __init__(self):
        super().__init__()
        self.records = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        train_loss = None
        for key in ["train_loss", "train_loss_step", "loss"]:
            if key in trainer.callback_metrics:
                train_loss = float(trainer.callback_metrics[key])
                break
        if train_loss is None and outputs is not None:
            if isinstance(outputs, dict) and "loss" in outputs:
                train_loss = float(outputs["loss"])
            elif hasattr(outputs, "item"):
                train_loss = float(outputs)
        if train_loss is not None:
            self.records.append({"step": step, "train_loss": train_loss, "val_loss": np.nan})

    def on_validation_end(self, trainer, pl_module):
        step = trainer.global_step
        val_loss = None
        for key in ["valid_loss", "val_loss", "ptl/val_loss"]:
            if key in trainer.callback_metrics:
                val_loss = float(trainer.callback_metrics[key])
                break
        if val_loss is not None:
            self.records.append({"step": step, "train_loss": np.nan, "val_loss": val_loss})

    def to_dataframe(self):
        if not self.records:
            return None
        df = pd.DataFrame(self.records).sort_values("step").reset_index(drop=True)
        return df


# =====================================================================
#  Neural: inject callback via configure_callbacks monkey-patch
# =====================================================================

def _inject_callback_and_fit(nf, df_train, horizon, collector):
    for m in nf.models:
        original_fn = m.configure_callbacks

        def _patched(orig=original_fn, cb=collector):
            existing = orig()
            if not isinstance(existing, list):
                existing = list(existing) if existing else []
            existing.append(cb)
            return existing

        m.configure_callbacks = _patched

    nf.fit(df=df_train, val_size=horizon)
    return collector.to_dataframe()


def _build_nf(model_name, ds_name, cfg):
    from neuralforecast import NeuralForecast
    h, inp, freq = cfg["horizon"], cfg["input_size"], cfg["freq"]

    if model_name == "PatchTST":
        from neuralforecast.models import PatchTST
        pc = _PATCH_CONFIGS[ds_name]
        m = PatchTST(h=h, input_size=inp, patch_len=pc["patch_len"], stride=pc["stride"],
                     n_heads=PATCHTST_PARAMS["n_heads"], hidden_size=PATCHTST_PARAMS["hidden_size"],
                     encoder_layers=PATCHTST_PARAMS["encoder_layers"],
                     max_steps=MAX_STEPS, batch_size=BATCH_SIZE, learning_rate=LR_TRANSFORMER,
                     random_seed=SEED, early_stop_patience_steps=EARLY_STOP_PATIENCE,
                     val_check_steps=VAL_CHECK_STEPS, scaler_type="standard")
    elif model_name == "NBEATS":
        from neuralforecast.models import NBEATS
        m = NBEATS(h=h, input_size=inp, stack_types=NBEATS_PARAMS["stack_types"],
                   n_blocks=NBEATS_PARAMS["n_blocks"], mlp_units=NBEATS_PARAMS["mlp_units"],
                   max_steps=MAX_STEPS, batch_size=BATCH_SIZE, learning_rate=LR_MLP,
                   random_seed=SEED, early_stop_patience_steps=EARLY_STOP_PATIENCE,
                   val_check_steps=VAL_CHECK_STEPS, scaler_type="standard")
    elif model_name == "TiDE":
        from neuralforecast.models import TiDE
        m = TiDE(h=h, input_size=inp, hidden_size=TIDE_PARAMS["hidden_size"],
                 decoder_output_dim=TIDE_PARAMS["decoder_output_dim"],
                 num_encoder_layers=TIDE_PARAMS["num_encoder_layers"],
                 num_decoder_layers=TIDE_PARAMS["num_decoder_layers"],
                 max_steps=MAX_STEPS, batch_size=BATCH_SIZE, learning_rate=LR_TRANSFORMER,
                 random_seed=SEED, early_stop_patience_steps=EARLY_STOP_PATIENCE,
                 val_check_steps=VAL_CHECK_STEPS, scaler_type="standard")
    elif model_name == "DeepAR":
        from neuralforecast.models import DeepAR
        m = DeepAR(h=h, input_size=inp, lstm_hidden_size=_DEEPAR_HIDDEN[ds_name],
                   lstm_n_layers=DEEPAR_PARAMS["n_layers"],
                   max_steps=MAX_STEPS, batch_size=BATCH_SIZE, learning_rate=LR_RNN,
                   random_seed=SEED, early_stop_patience_steps=EARLY_STOP_PATIENCE,
                   val_check_steps=VAL_CHECK_STEPS, scaler_type="standard")
    elif model_name == "DLinear":
        from neuralforecast.models import DLinear
        m = DLinear(h=h, input_size=inp,
                    max_steps=MAX_STEPS, batch_size=BATCH_SIZE, learning_rate=LR_MLP,
                    random_seed=SEED, early_stop_patience_steps=EARLY_STOP_PATIENCE,
                    val_check_steps=VAL_CHECK_STEPS, scaler_type="standard")
    elif model_name == "TimesNet":
        from neuralforecast.models import TimesNet
        m = TimesNet(h=h, input_size=inp, top_k=TIMESNET_PARAMS["top_k"],
                     num_kernels=TIMESNET_PARAMS["num_kernels"],
                     hidden_size=TIMESNET_PARAMS["d_model"], conv_hidden_size=TIMESNET_PARAMS["d_ff"],
                     encoder_layers=_TIMESNET_LAYERS[ds_name],
                     max_steps=MAX_STEPS, batch_size=BATCH_SIZE, learning_rate=LR_TRANSFORMER,
                     random_seed=SEED, early_stop_patience_steps=EARLY_STOP_PATIENCE,
                     val_check_steps=VAL_CHECK_STEPS, scaler_type="standard")
    else:
        raise ValueError(model_name)
    return NeuralForecast(models=[m], freq=freq)


# =====================================================================
#  LightGBM with eval_set logging
# =====================================================================

def _fit_lgbm(ds_name, cfg, df_train):
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean, RollingStd
    from lightgbm import LGBMRegressor, log_evaluation, record_evaluation

    sl = cfg["season_length"]; freq = cfg["freq"]; horizon = cfg["horizon"]
    lags = list(range(1, sl + 1))
    if 2 * sl not in lags: lags.append(2 * sl)
    lag_transforms = {sl: [RollingMean(window_size=sl), RollingStd(window_size=sl)]}
    date_features = []
    if freq in ("D","d"): date_features = ["dayofweek","month"]
    elif freq in ("h","H"): date_features = ["hour","dayofweek"]
    elif freq in ("ME","MS","M"): date_features = ["month"]

    lgbm = LGBMRegressor(n_estimators=LGBM_PARAMS["n_estimators"],
                         learning_rate=LGBM_PARAMS["learning_rate"],
                         num_leaves=LGBM_PARAMS["num_leaves"],
                         random_state=SEED, verbosity=-1)
    mlf = MLForecast(models={"LightGBM": lgbm}, freq=freq, lags=lags,
                     lag_transforms=lag_transforms,
                     date_features=date_features if date_features else None)
    try:
        prep = mlf.preprocess(df_train)
    except Exception as e:
        print(f"    [WARN] preprocess failed: {e}"); return None
    if prep is None or len(prep) == 0: return None

    tcol, icol, dcol = "y", "unique_id", "ds"
    fcols = [c for c in prep.columns if c not in {icol, dcol, tcol}]
    prep = prep.dropna(subset=fcols)
    if len(prep) == 0: return None

    mx = prep.groupby(icol)[dcol].max().reset_index()
    if freq in ("ME","MS","M"): mx["cut"] = mx[dcol] - pd.DateOffset(months=horizon)
    elif freq in ("h","H"): mx["cut"] = mx[dcol] - pd.Timedelta(hours=horizon)
    else: mx["cut"] = mx[dcol] - pd.Timedelta(days=horizon)
    prep = prep.merge(mx[[icol,"cut"]], on=icol)

    Xt = prep.loc[prep[dcol] <= prep["cut"], fcols].values
    yt = prep.loc[prep[dcol] <= prep["cut"], tcol].values
    Xv = prep.loc[prep[dcol] > prep["cut"], fcols].values
    yv = prep.loc[prep[dcol] > prep["cut"], tcol].values
    if len(Xv)==0 or len(Xt)==0: return None
    print(f"    LightGBM: {len(Xt)} train, {len(Xv)} val, {len(fcols)} features")

    er = {}
    mdl = LGBMRegressor(n_estimators=LGBM_PARAMS["n_estimators"],
                        learning_rate=LGBM_PARAMS["learning_rate"],
                        num_leaves=LGBM_PARAMS["num_leaves"],
                        random_state=SEED, verbosity=-1)
    mdl.fit(Xt, yt, eval_set=[(Xt,yt),(Xv,yv)], eval_names=["train","val"],
            eval_metric="l1", callbacks=[record_evaluation(er), log_evaluation(period=0)])

    if "train" in er and "l1" in er["train"]:
        df = pd.DataFrame({"step": range(1, len(er["train"]["l1"])+1),
                           "train_loss": er["train"]["l1"], "val_loss": er["val"]["l1"]})
        print(f"    [OK] {len(df)} rounds"); return df
    return None


# =====================================================================
#  Plotting
# =====================================================================

def _plot_single(df, mn, dn, is_lgbm=False):
    if df is None or len(df) == 0: return False
    fig, ax = plt.subplots(figsize=(6, 4))

    ht = "train_loss" in df.columns and df["train_loss"].notna().any()
    hv = "val_loss" in df.columns and df["val_loss"].notna().any()

    if ht:
        tr = df[df["train_loss"].notna()]
        tv, ts = tr["train_loss"].values, tr["step"].values
        if not is_lgbm and len(tv) > 20:
            w = max(5, len(tv)//40)
            sm = pd.Series(tv).rolling(w, min_periods=1, center=True).mean().values
            ax.plot(ts, tv, alpha=0.12, color="tab:blue", linewidth=0.4)
            ax.plot(ts, sm, color="tab:blue", linewidth=2, label="Train (smoothed)")
        else:
            ax.plot(ts, tv, color="tab:blue", linewidth=1.5, label="Train Loss")

    if hv:
        vr = df[df["val_loss"].notna()]
        if len(vr) > 0:
            ax.plot(vr["step"].values, vr["val_loss"].values, color="tab:orange",
                    linewidth=2, marker="o", markersize=4, label="Val Loss")

    if hv and not is_lgbm:
        last = int(df["step"].max())
        if last < MAX_STEPS * 0.9:
            ax.axvline(last, color="red", linestyle="--", alpha=0.5)
            ax.annotate("Early stop", xy=(last, ax.get_ylim()[1]*0.9),
                        fontsize=8, color="red", alpha=0.7, ha="right")

    ax.set_xlabel("Boosting Round" if is_lgbm else "Training Step", fontsize=11)
    ax.set_ylabel("Loss (MAE)", fontsize=11)
    ax.set_title(f"{mn} — {dn}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); plt.tight_layout()
    out = PLOTS_DIR / f"learning_curve_{mn}_{dn}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"    Saved: {out}"); return True


def _plot_grid(results):
    morder = ["DLinear","LightGBM","TiDE","NBEATS","PatchTST","DeepAR","TimesNet"]
    dorder = ["M4","M5","Traffic"]
    fig, axes = plt.subplots(len(morder), len(dorder), figsize=(15,24), squeeze=False)
    fig.suptitle("Learning Curves: Training vs Validation Loss",
                 fontsize=16, fontweight="bold", y=0.995)

    for i, mn in enumerate(morder):
        for j, dn in enumerate(dorder):
            ax = axes[i][j]; df = results.get((mn,dn))
            if df is not None and len(df) > 0:
                il = mn == "LightGBM"
                ht = "train_loss" in df.columns and df["train_loss"].notna().any()
                hv = "val_loss" in df.columns and df["val_loss"].notna().any()
                if ht:
                    tr = df[df["train_loss"].notna()]
                    tv, ts = tr["train_loss"].values, tr["step"].values
                    if not il and len(tv)>20:
                        w = max(3,len(tv)//40)
                        sm = pd.Series(tv).rolling(w,min_periods=1,center=True).mean().values
                        ax.plot(ts,tv,alpha=0.08,color="tab:blue",linewidth=0.3)
                        ax.plot(ts,sm,color="tab:blue",linewidth=1.5,label="Train")
                    else: ax.plot(ts,tv,color="tab:blue",linewidth=1,label="Train")
                if hv:
                    vr = df[df["val_loss"].notna()]
                    if len(vr)>0:
                        ax.plot(vr["step"].values,vr["val_loss"].values,
                                color="tab:orange",linewidth=1.5,marker="o",markersize=2,label="Val")
                if hv and not il:
                    last = int(df["step"].max())
                    if last < MAX_STEPS*0.9:
                        ax.axvline(last,color="red",linestyle="--",alpha=0.4,linewidth=0.8)
            else:
                ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center",va="center",
                        fontsize=10,color="gray")
            if i==0: ax.set_title(dn,fontsize=12,fontweight="bold")
            if j==0: ax.set_ylabel(mn,fontsize=11,fontweight="bold")
            if i==len(morder)-1: ax.set_xlabel("Round" if mn=="LightGBM" else "Step",fontsize=9)
            if i==0 and j==len(dorder)-1: ax.legend(fontsize=7,loc="upper right")
            ax.grid(True,alpha=0.2); ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0,0,1,0.99])
    out = PLOTS_DIR / "learning_curves_grid.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nSaved grid: {out}")


# =====================================================================
#  Main
# =====================================================================

NEURAL_MODELS = ["PatchTST","NBEATS","TiDE","DeepAR","DLinear","TimesNet"]

def main():
    print("="*60)
    print("  Learning Curves — 7 models x 3 datasets")
    print("  Method: Callback injection (configure_callbacks patch)")
    print("="*60)

    results = {}
    t_total = time.time()

    for ds_name, cfg, loader in DATASETS:
        ns = cfg.get("n_series_sample")
        print(f"\n{'─'*60}\n  Dataset: {ds_name} (n={ns})\n{'─'*60}")

        t0 = time.time()
        df_tr, df_te = loader(n_series=ns)
        df_all = pd.concat([df_tr, df_te], ignore_index=True)

        mx = df_all["ds"].max()
        off = pd.tseries.frequencies.to_offset(cfg["freq"])
        ts = mx - (cfg["horizon"]-1)*off
        tc = ts - off
        df_w1 = df_all[df_all["ds"] <= tc].copy()

        minr = cfg["input_size"] + cfg["horizon"] + 1
        lens = df_w1.groupby("unique_id").size()
        valid = lens[lens >= minr].index
        df_w1 = df_w1[df_w1["unique_id"].isin(valid)]
        print(f"  Loaded in {time.time()-t0:.1f}s — {df_w1['unique_id'].nunique()} series")

        for mn in NEURAL_MODELS:
            print(f"\n  [{mn}] on {ds_name}...")
            t1 = time.time()
            try:
                nf = _build_nf(mn, ds_name, cfg)
                coll = LossCollector()
                df_m = _inject_callback_and_fit(nf, df_w1, cfg["horizon"], coll)

                if df_m is not None and len(df_m) > 0:
                    nt = df_m["train_loss"].notna().sum()
                    nv = df_m["val_loss"].notna().sum()
                    print(f"    [OK] {nt} train steps, {nv} val checks")
                    df_m.to_csv(LC_DIR / f"{mn}_{ds_name}.csv", index=False)
                    _plot_single(df_m, mn, ds_name)
                    results[(mn, ds_name)] = df_m
                else:
                    print(f"    [WARN] 0 records captured")
                    print(f"    [DEBUG] collector had {len(coll.records)} raw records")
                    results[(mn, ds_name)] = None
            except Exception as e:
                print(f"    [ERROR] {mn} on {ds_name}: {e}")
                import traceback; traceback.print_exc()
                results[(mn, ds_name)] = None
            print(f"    Time: {time.time()-t1:.1f}s")

        print(f"\n  [LightGBM] on {ds_name}...")
        t1 = time.time()
        try:
            df_m = _fit_lgbm(ds_name, cfg, df_w1)
            if df_m is not None:
                df_m.to_csv(LC_DIR / f"LightGBM_{ds_name}.csv", index=False)
                _plot_single(df_m, "LightGBM", ds_name, is_lgbm=True)
                results[("LightGBM", ds_name)] = df_m
            else:
                results[("LightGBM", ds_name)] = None
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[("LightGBM", ds_name)] = None
        print(f"    Time: {time.time()-t1:.1f}s")

    print(f"\n{'='*60}\n  Generating 7x3 grid...")
    _plot_grid(results)

    elapsed = time.time() - t_total
    ok = sum(1 for v in results.values() if v is not None)
    print(f"\n{'='*60}")
    print(f"  Done: {ok}/{len(results)} successful, {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    if ok < len(results):
        failed = [k for k,v in results.items() if v is None]
        print(f"\n  Failed cells: {failed}")
        print("  If ALL neural models failed, paste the [DEBUG] lines above.")

if __name__ == "__main__":
    main()
