"""
FloodTriage — Colab Runner
==========================
Single-file entry-point for Google Colab.

HOW TO USE
----------
1. Open this file in Colab:
       File → Open notebook → GitHub tab → paste the repo URL → open colab_runner.py
   OR click the "Open in Colab" badge in README.md.

2. Before running, upload your three CSV files to Google Drive at:
       MyDrive/Data_Save_1965_2011/
           Precip_mm_Save_1965_2011.csv
           PET_mm_Save_1965_2011.csv
           Discharge_cms_Save_1965_2011.csv

3. Click  Runtime → Run all  (Ctrl+F9).
   All outputs are saved to  MyDrive/Triage/ft_outputs/.

NOTES
-----
- A free T4 GPU runtime is sufficient (and recommended for speed).
- Total runtime: ~45–90 min (LSTM training dominates).
- Re-running is safe: all pip installs are idempotent.
"""

# ── 0. Drive mount & matplotlib backend (must be before any pyplot import) ───
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; savefig works, show() is no-op

import matplotlib.pyplot as plt
from pathlib import Path
import subprocess, sys, os

# ── 1. Clone the repository (skip if already present) ────────────────────────
REPO_URL  = "https://github.com/IngGOHOUEDE/FloodTriage.git"    
REPO_DIR  = Path("/content/FloodTriage")

if not REPO_DIR.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    print(f"Repository cloned to {REPO_DIR}")
else:
    print(f"Repository already present at {REPO_DIR} — pulling latest changes.")
    subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])

# ── 2. Define paths ───────────────────────────────────────────────────────────
# Drive-backed output directory — all figures, CSVs and JSONs land here.
DRIVE_OUTPUT = Path("/content/drive/MyDrive/Triage/ft_outputs")
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Data: auto-downloaded from public Google Drive folder ────────────────────
DRIVE_DATA = Path("/content/Data_Save_1965_2011")
DRIVE_DATA.mkdir(exist_ok=True)

# This downloads the folder if not already present (idempotent)
if not any(DRIVE_DATA.glob("*.csv")):
    subprocess.check_call([
        sys.executable, "-m", "gdown", "--folder",
        "1oODB0F7BslwTi9rMEx1dzh7CqZRL0y-k",   
        "-O", str(DRIVE_DATA),
    ])

os.environ["FT_OUTPUT_DIR"] = str(DRIVE_OUTPUT)
print(f"Figures and artefacts will be saved to: {DRIVE_OUTPUT}")
print(f"Raw data expected at:                   {DRIVE_DATA}")

# ── 3. Install dependencies (idempotent) ─────────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "gdown", "neuralhydrology", "xgboost", "shap", "statsmodels",
])

# ── 4. Make src/ block scripts importable ────────────────────────────────────
SCRIPTS_DIR = str(REPO_DIR / "src")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ── 4. Monkey-patch plt.show() to call plt.close('all') instead ──────────────
# This guarantees figure memory is released after each savefig().
_real_show = plt.show
def _safe_show(*args, **kwargs):
    plt.close("all")   # release figure; savefig() was already called before show()
plt.show = _safe_show

# ── 5. Run Block 1 — Data Preparation & Utilities ────────────────────────────
print("\n" + "="*60)
print("BLOCK 1 — Data Preparation & Utilities")
print("="*60)

import importlib, types

def _run_block(module_name: str):
    """Import (or reload) a block module, with OUTPUT_DIR patched to Drive."""
    if module_name in sys.modules:
        mod = importlib.reload(sys.modules[module_name])
    else:
        mod = importlib.import_module(module_name)
    if hasattr(mod, "OUTPUT_DIR"):
        mod.OUTPUT_DIR = DRIVE_OUTPUT
    return mod

b1 = _run_block("block1_data_prep_utilities")
b1.OUTPUT_DIR = DRIVE_OUTPUT
b1.DRIVE = DRIVE_DATA   # point data loader at the Drive CSV folder

# Run data preparation
df       = b1.load_raw_data(DRIVE_DATA)
df, Q_stats = b1.engineer_features(df)
b1.save_json(Q_stats,  DRIVE_OUTPUT / "quantiles.json")
b1.save_json(b1.SPLITS, DRIVE_OUTPUT / "splits.json")
df.to_csv(DRIVE_OUTPUT / "full.csv")
print("Block 1 complete.")

# ── 6. Run Block 2 — Multi-Horizon Conformal Inference ───────────────────────
print("\n" + "="*60)
print("BLOCK 2 — LSTM Ensemble & Multi-Horizon ACI")
print("="*60)

b2 = _run_block("block2_multiHorizon_conformal")
b2.OUTPUT_DIR = DRIVE_OUTPUT

b2.write_netcdf(df, b2.NH_DATA_DIR, b2.BASIN_ID)
ensemble_preds, run_dirs = b2.train_ensemble(
    df, b2.NH_DATA_DIR, b2.NH_RUN_DIR, DRIVE_OUTPUT, b2.SEEDS
)

# Build per-horizon mu/sigma matrices (shape: N_full × H_MAX)
import numpy as np
def _build_matrix(field):
    cols = []
    for h in range(1, b2.H_MAX + 1):
        uq_h = b2.build_uq(ensemble_preds, df, Q_stats, h=h)
        cols.append(uq_h[field].values)
    return np.column_stack(cols)

mu_matrix    = _build_matrix("Q_mean")
sigma_matrix = _build_matrix("Q_std")

# Day-ahead UQ (h=1) for gateway
uq = b2.build_uq(ensemble_preds, df, Q_stats, h=1)
df["Q_lstm"] = uq["Q_mean"].values
uq.to_csv(DRIVE_OUTPUT / "uq.csv")
df.to_csv(DRIVE_OUTPUT / "full.csv")

scaler = b2.load_scaler(run_dirs[42], target_var="Q")
b1.save_json(
    {"mu_Q": scaler["mean"], "std_Q": scaler["std"],
     "sigma_clim": float(Q_stats["Q_std"])},
    DRIVE_OUTPUT / "lstm_scaler.json",
)

# ACI calibration warm-start
aci = b2.MultiHorizonACI(alpha0=0.10, gamma=0.005, h_max=b2.H_MAX)
import pandas as pd
cal_mask  = (df.index >= b1.SPLITS["cal"][0]) & (df.index <= b1.SPLITS["cal"][1])
cal       = df[cal_mask].copy()
uq_cal    = uq[cal_mask]
cal_obs   = cal["Q"].values
cal_mu    = mu_matrix[cal_mask]
cal_sig   = sigma_matrix[cal_mask]
for h in range(1, b2.H_MAX + 1):
    aci.calibrate_on_window(cal_obs, cal_mu[:, h-1], cal_sig[:, h-1], h=h)

# ACI online test run
test_mask  = (df.index >= b1.SPLITS["test"][0]) & (df.index <= b1.SPLITS["test"][1])
test       = df[test_mask].copy()
uq_test    = uq[test_mask]
test_dates = test.index
test_obs   = test["Q"].values
test_mu    = mu_matrix[test_mask]
test_sig   = sigma_matrix[test_mask]

aci_results = aci.run_online(test_dates, test_obs, test_mu, test_sig)
aci_results.to_csv(DRIVE_OUTPUT / "aci_intervals.csv", index=False)

h1_rows   = aci_results[aci_results["h"] == 1].set_index("date")
Q_lo_test = h1_rows["Q_lo"].reindex(test_dates).values
Q_hi_test = h1_rows["Q_hi"].reindex(test_dates).values
alpha_seq  = h1_rows["alpha_t"].reindex(test_dates).values

print("Block 2 complete.")

# ── 7. Run Block 3 — Gateway, XGBoost & Evaluation ───────────────────────────
print("\n" + "="*60)
print("BLOCK 3 — XGBoost Surrogate, Gateway & Evaluation")
print("="*60)

b3 = _run_block("block3_gateway_evaluation")
b3.OUTPUT_DIR = DRIVE_OUTPUT

# Inject Block 1 utilities into Block 3
for attr in dir(b1):
    if not attr.startswith("__"):
        setattr(b3, attr, getattr(b1, attr))

import xgboost as xgb

# Thresholds
tr_mask = (df.index >= b1.SPLITS["train"][0]) & (df.index <= b1.SPLITS["train"][1])
p30d_thresholds = b3._compute_tercile_thresholds(df["P30d"], tr_mask, b1.WARMUP_DAYS)
asm_thresholds  = b3._compute_tercile_thresholds(df["ASM"],  tr_mask, b1.WARMUP_DAYS)
P30D_Q33, P30D_Q67 = p30d_thresholds

obs_cal    = cal["Q"].values
Q_lstm_cal = cal["Q_lstm"].values

# Aleatoric table & LOYO
(ALEATORIC_TABLE, TAU_ALE,
 achieved_eta_ale, global_var_cal) = b3.build_aleatoric_table(
    cal, uq_cal, Q_stats, p30d_thresholds
)
loyo_ale_df = b3.loyo_tau_ale(cal, Q_stats, p30d_thresholds)
loyo_ale_df.to_csv(DRIVE_OUTPUT / "loyo_tau_ale.csv", index=False)

b3.compute_reliability_stats(obs_cal, Q_lstm_cal, uq_cal, Q_stats, DRIVE_OUTPUT)

tau_regime, GLOBAL_TAU = b3.calibrate_leg1_thresholds(uq_cal, Q_stats)
loyo_leg1_df = b3.loyo_tau_regime(cal, uq_cal, Q_stats)
loyo_leg1_df.to_csv(DRIVE_OUTPUT / "loyo_tau_regime.csv", index=False)

xgb_model = b3.train_xgb(df, uq, Q_stats, DRIVE_OUTPUT)

X_cal_raw     = b3.build_xgb_features(cal, uq_cal)
Q_xgb_cal_all = np.clip(
    Q_lstm_cal + xgb_model.predict(
        xgb.DMatrix(X_cal_raw, feature_names=b3.XGB_FEATURE_NAMES)
    ), 0, None,
)

reg_cal_pred = b1.assign_regime(Q_lstm_cal, Q_stats)
asm_cal_bins = b3.asm_to_bin(cal["ASM"].values, asm_thresholds)
b3.block_permutation_complementarity(
    cal, obs_cal, Q_lstm_cal, Q_xgb_cal_all,
    reg_cal_pred, asm_cal_bins, Q_stats,
)

# Test period
obs_t    = test["Q"].values
Q_lstm_t = test["Q_lstm"].values.copy()
Qm_t     = uq_test["Q_mean"].values
wt_t     = uq_test["w_t"].values
n_test   = len(test.index)

reg_pred_test = b1.assign_regime(Qm_t, Q_stats)
ale_bin_test  = b3.p30d_to_bin(test["P30d"].values, p30d_thresholds)
_fallback     = float(np.nanmean(list(ALEATORIC_TABLE.values())))
uale_t = np.array([
    ALEATORIC_TABLE.get((reg_pred_test[t], ale_bin_test[t]), _fallback)
    for t in range(n_test)
])

leg2_aleatoric_test = uale_t >= TAU_ALE          # ≥ operator
leg3_hydro          = Qm_t >= Q_stats["Q95"]
taus_test = np.array([tau_regime.get(r, GLOBAL_TAU) for r in reg_pred_test])
leg1_stat = wt_t > taus_test

is_defer = b3.build_gateway_mask(
    n_test, wt_t, taus_test, leg2_aleatoric_test, leg3_hydro, Q_lstm_t
)
Q_system, dq_test = b3.apply_xgb_correction(test, uq_test, is_defer, xgb_model)

b3.run_leg_ablation(
    n_test, obs_t, test, uq_test, wt_t, taus_test,
    leg2_aleatoric_test, leg3_hydro, Q_lstm_t, xgb_model, Q_stats,
)
s1_metrics, s2_metrics, pval_mcnemar = b3.compute_strategy_metrics(
    obs_t, Q_lstm_t, Q_system, is_defer, Q_stats
)

reg_test_obs = b1.assign_regime(np.nan_to_num(obs_t, nan=-1), Q_stats)
cov_by_regime = {}
for r in b1.REGIMES:
    idx = (reg_test_obs == r) & ~np.isnan(obs_t)
    if not idx.any():
        cov_by_regime[r] = float("nan")
        continue
    cov_by_regime[r] = float(np.nanmean(
        (Q_lo_test[idx] <= obs_t[idx]) & (obs_t[idx] <= Q_hi_test[idx])
    ))

ope_result = b3.run_ope_dm(obs_t, Q_lstm_t, Q_system, is_defer, Q_stats)
s1_fnr_series, s2_fnr_series, rc_errors = b3.rating_curve_sensitivity(
    obs_t, Q_lstm_t, Q_system, Q_stats, DRIVE_OUTPUT
)
rc_df = b3.sweep_rc_curves(
    n_test, obs_t, Q_lstm_t, wt_t, taus_test,
    leg2_aleatoric_test, leg3_hydro, dq_test, Q_stats, DRIVE_OUTPUT,
)
b3.failure_attribution(obs_t, Q_lstm_t, Q_system, is_defer, Q_stats)
b3.confident_but_wrong(
    test, obs_t, Q_lstm_t, wt_t, uale_t, is_defer,
    Q_stats, GLOBAL_TAU, P30D_Q67, DRIVE_OUTPUT,
)
b3.flood_classification_overlap(obs_t, Q_lstm_t, Q_system, Q_stats, DRIVE_OUTPUT)
b3.profile_single_day_latency(test, uq_test, aci, xgb_model, is_defer)

print("Block 3 complete.")

# ── 8. Run Block 4 — Plotting (all savefig writes go to Drive) ───────────────
print("\n" + "="*60)
print("BLOCK 4 — Plotting & SHAP")
print("="*60)

b4 = _run_block("block4_plotting_orchestration")
b4.OUTPUT_DIR = DRIVE_OUTPUT

# Inject Block 1 & 3 utilities cumulatively into Block 4
for mod in (b1, b3):
    for attr in dir(mod):
        if not attr.startswith("__"):
            setattr(b4, attr, getattr(mod, attr))

b4.temporal_audit(
    n_test, obs_t, Q_lstm_t, Q_system, is_defer,
    leg1_stat, leg2_aleatoric_test, leg3_hydro, Q_stats, test,
)

# SHAP
shap_vals, X_defer, explainer, peak_dq_idx, defer_dates_test = b4.compute_shap(
    test, uq_test, is_defer, xgb_model, DRIVE_OUTPUT
)

# Each plot function: savefig → Drive, then plt.close('all') via patched show()
b4.plot_shap(
    shap_vals, X_defer, explainer, peak_dq_idx,
    defer_dates_test, dq_test, is_defer, DRIVE_OUTPUT,
    xgb_model=xgb_model,
)
b4.plot_coverage_legs(
    cov_by_regime, leg1_stat, leg2_aleatoric_test, leg3_hydro,
    is_defer, alpha0=0.10, output_dir=DRIVE_OUTPUT,
)
b4.plot_rc_and_metrics(rc_df, s1_metrics, s2_metrics, DRIVE_OUTPUT)
b4.plot_rating_curve_sensitivity(rc_errors, s1_fnr_series, s2_fnr_series, DRIVE_OUTPUT)
b4.plot_confident_wrong(
    test, obs_t, Q_lstm_t, wt_t, uale_t, is_defer,
    GLOBAL_TAU, P30D_Q67, Q_stats, DRIVE_OUTPUT,
)
b4.plot_full_hydrograph(
    df, test, uq_test, Q_system, is_defer,
    leg2_aleatoric_test, leg3_hydro, Q_stats, b1.SPLITS, DRIVE_OUTPUT,
)
b4.plot_test_hydrograph(
    test, uq_test, Q_system, is_defer,
    leg2_aleatoric_test, leg3_hydro, Q_stats, DRIVE_OUTPUT,
)
b4.print_operator_dashboard(
    test, obs_t, Q_lstm_t, Q_system, is_defer, wt_t, taus_test,
    uale_t, leg2_aleatoric_test, leg3_hydro, uq_test,
    shap_vals, X_defer, defer_dates_test, xgb_model,
    peak_dq_idx, Q_stats, TAU_ALE,
)
b4.serialise_strategy_metrics(s1_metrics, s2_metrics, DRIVE_OUTPUT)

print("Block 4 complete.")

# ── 9. Run Block 5 — Reporting & Exports ─────────────────────────────────────
print("\n" + "="*60)
print("BLOCK 5 — Reporting & Exports")
print("="*60)

b5 = _run_block("block5_reporting_exports")
b5.OUTPUT_DIR = DRIVE_OUTPUT

# Inject Block 1, 3 & 4 utilities cumulatively into Block 5
for mod in (b1, b3, b4):
    for attr in dir(mod):
        if not attr.startswith("__"):
            setattr(b5, attr, getattr(mod, attr))

b5.print_ensemble_performance(df, uq, Q_stats, b1.SPLITS, b1.WARMUP_DAYS)
b5.print_loyo_summary(loyo_leg1_df, loyo_ale_df)
b5.save_aleatoric_table(
    ALEATORIC_TABLE, TAU_ALE, achieved_eta_ale, global_var_cal,
    P30D_Q33, P30D_Q67, DRIVE_OUTPUT,
)
b5.print_complementarity_heatmap(
    cal, obs_cal, Q_lstm_cal, Q_xgb_cal_all,
    reg_cal_pred, asm_cal_bins,
)
b5.print_kge_bootstrap_report(obs_t, Q_lstm_t, Q_system)
b5.print_mcnemar_summary(obs_t, Q_lstm_t, Q_system, Q_stats)
b5.print_scaler_stats(scaler, Q_stats)

# Recompute Leg 2 and Leg 3 on the calibration period.
_cal_reg   = b1.assign_regime(cal["Q_lstm"].values, Q_stats)
_cal_p30d  = b3.p30d_to_bin(cal["P30d"].values, p30d_thresholds)
_fallback  = float(np.nanmean(list(ALEATORIC_TABLE.values())))
_uale_cal  = np.array([
    ALEATORIC_TABLE.get((_cal_reg[t], _cal_p30d[t]), _fallback)
    for t in range(len(cal))
])
_leg2_cal  = _uale_cal >= TAU_ALE
_leg3_cal  = cal["Q_lstm"].values >= Q_stats["Q95"]

q_hat_nd = b5.compute_nondeferral_quantile(
    cal, uq_cal, _leg2_cal, _leg3_cal, Q_stats,
)

b5.export_aci_params(
    tau_regime, GLOBAL_TAU, TAU_ALE, achieved_eta_ale,
    q_hat_nd, cov_by_regime, is_defer, leg1_stat,
    leg2_aleatoric_test, leg3_hydro, aci.alpha_state, DRIVE_OUTPUT,
)

policy_df = b5.build_policy_df(
    test, Q_lstm_t, dq_test, Q_system, is_defer,
    leg1_stat, leg2_aleatoric_test, leg3_hydro,
    uale_t, wt_t, Q_lo_test, Q_hi_test, alpha_seq, DRIVE_OUTPUT,
)

b1.save_json(
    {
        "tau_per_regime":         tau_regime,
        "GLOBAL_TAU":             GLOBAL_TAU,
        "TAU_ALE":                TAU_ALE,
        "achieved_eta_ale":       achieved_eta_ale,
        "coverage_by_regime_h1":  cov_by_regime,
        "COST_RATIO":             b1.COST_RATIO,
        "NORMALIZED_ESC_COST":    b1.NORMALIZED_ESC_COST,
        "deferral_rate":          float(is_defer.mean()),
        "ope_dm":                 ope_result,
        "mcnemar_p":              pval_mcnemar,
    },
    DRIVE_OUTPUT / "pipeline_summary.json",
)
print("Block 5 complete.")

# ── 10. Verify all figures were written to Drive ──────────────────────────────
print("\n" + "="*60)
print("FIGURE VERIFICATION")
print("="*60)
figs = sorted(DRIVE_OUTPUT.glob("fig_*.png"))
print(f"Found {len(figs)} figure(s) in {DRIVE_OUTPUT}:")
for f in figs:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name}  ({size_kb} KB)")

print("\nFloodTriage pipeline complete. All artefacts saved to Drive.")