"""
FloodTriage — Block 3: XGBoost Surrogate, Tri-Fold Routing Gateway & Evaluation
=================================================================================
Ouémé at Savè, Benin | 1965–2011

Key Components
--------------
1. XGBoost surrogate correction expert trained on calibration residuals.
2. Tri-Fold Routing Gateway:
     Leg 1 — Epistemic safeguard   (w_t > τ_regime, regime-specific thresholds)
     Leg 2 — Aleatoric safeguard   (uale_t ≥ TAU_ALE, uses ≥ for consistency)
     Leg 3 — Threshold Safeguard   (Q_lstm ≥ Q95, hard physical threshold)
   with event-persistence logic on the rising limb.
3. Leg-wise ablation study.
4. Direct Method (DM) Off-Policy Evaluation via deterministic counterfactual
   simulation (LogisticRegression propensity model and DR estimator removed;
   positivity assumption is violated by the deterministic gateway).
5. SHAP interpretability on deferred test days.
6. Flood-day classification overlap table with WMO-compliant verification tags.
7. Failure attribution and confident-but-wrong diagnostic.

OPE Design Note
---------------
The Tri-Fold Gateway is fully deterministic: every timestep maps to a binary
{defer, autonomous} decision without stochasticity. This collapses propensity
scores to 0 or 1, violating the positivity assumption required by Inverse
Propensity Weighting (IPW) and Doubly Robust (DR) estimators. Following the
standard OPE remedy for deterministic evaluation policies (Precup et al. 2000;
Dudík et al. 2011), we adopt the Direct Method (DM) exclusively: rewards are
estimated by swapping each day's action between {S1: autonomous, S2: deferred}
and evaluating the reward model directly. Bootstrap confidence intervals are
computed via a 120-day seasonal block bootstrap.
"""

# ── Imports (Block 1 & 2 must be executed first in notebook sessions) ──────────
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.patches import Patch
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint

try:
    from block1_data_prep_utilities import (
        ASM_LABELS, BASIN_ID, COST_RATIO, H_MAX, NORMALIZED_ESC_COST,
        OUTPUT_DIR, P30D_LABELS, REGIMES, SEEDS, SPLITS, WARMUP_DAYS,
        Q_stats, assign_regime, df, fnr_wilson, kge, mcnemar_fnr_test,
        moving_block_bootstrap, nse, pbias, rmse, save_json,
    )
    from block2_multiHorizon_conformal import (
        MultiHorizonACI, build_uq, uq, ensemble_preds, aci_results,
        Q_lo_test, Q_hi_test, alpha_seq, test_mask, test_dates, mu_matrix,
        sigma_matrix,
    )
except ImportError:
    pass  # Notebook mode — globals already present.

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# §1  Regime and P30d Bin Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_tercile_thresholds(
    series: pd.Series,
    train_mask: pd.Series,
    warmup: int,
) -> tuple[float, float]:
    """Compute training-period Q33 and Q67 tercile thresholds."""
    vals = series[train_mask].values[warmup:]
    return (
        float(np.nanpercentile(vals, 33.33)),
        float(np.nanpercentile(vals, 66.67)),
    )


def asm_to_bin(asm_vals: np.ndarray, thresholds: tuple) -> np.ndarray:
    """Map ASM values to 'dry' / 'moderate' / 'wet' tercile bins."""
    q33, q67 = thresholds
    bins = np.full(len(np.asarray(asm_vals)), "moderate", dtype=object)
    bins[np.asarray(asm_vals) < q33] = "dry"
    bins[np.asarray(asm_vals) >= q67] = "wet"
    return bins


def p30d_to_bin(p30d_vals: np.ndarray, thresholds: tuple) -> np.ndarray:
    """Map P30d values to 'dry' / 'moderate' / 'wet' tercile bins."""
    q33, q67 = thresholds
    p30d_arr = np.asarray(p30d_vals)
    bins = np.full(len(p30d_arr), "moderate", dtype=object)
    bins[p30d_arr < q33] = "dry"
    bins[p30d_arr >= q67] = "wet"
    return bins


# ══════════════════════════════════════════════════════════════════════════════
# §2  Aleatoric Uncertainty Table & TAU_ALE Calibration
# ══════════════════════════════════════════════════════════════════════════════

def build_aleatoric_table(
    cal: pd.DataFrame,
    uq_cal: pd.DataFrame,
    Q_stats: dict,
    p30d_thresholds: tuple,
    min_cell_size: int = 5,
) -> tuple[dict, float, float, float]:
    """Build a regime × P30d aleatoric variance lookup table.

    The table maps (regime, p30d_bin) → conditional residual variance,
    estimated on the calibration period. TAU_ALE is chosen as the unique
    table value whose implied deferral rate is closest to NORMALIZED_ESC_COST.

    Parameters
    ----------
    cal : pd.DataFrame
        Calibration subset of the full dataframe.
    uq_cal : pd.DataFrame
        UQ dataframe (Q_mean, Q_std, w_t) for the calibration period.
    Q_stats : dict
        Training-period quantiles.
    p30d_thresholds : tuple (Q33, Q67)
        P30d tercile thresholds computed from the training period.
    min_cell_size : int
        Minimum cell population; cells below this use the global residual
        variance as a regularisation fallback.

    Returns
    -------
    ALEATORIC_TABLE : dict  {(regime, p30d_bin): variance}
    TAU_ALE : float  Threshold on aleatoric variance.
    achieved_eta_ale : float  Deferral rate implied by TAU_ALE.
    global_var_cal : float  Global calibration residual variance.
    """
    obs_cal    = cal["Q"].values
    Q_lstm_cal = cal["Q_lstm"].values

    reg_cal_pred = assign_regime(Q_lstm_cal, Q_stats)
    ale_bin_cal  = p30d_to_bin(cal["P30d"].values, p30d_thresholds)
    residual_cal = obs_cal - Q_lstm_cal

    global_var_cal = float(np.nanvar(residual_cal))

    ALEATORIC_TABLE: dict = {}
    for reg in REGIMES:
        for p30d_bin in P30D_LABELS:
            mask = (
                (reg_cal_pred == reg)
                & (ale_bin_cal == p30d_bin)
                & ~np.isnan(residual_cal)
            )
            v = (
                float(np.var(residual_cal[mask]))
                if mask.sum() >= min_cell_size
                else global_var_cal
            )
            ALEATORIC_TABLE[(reg, p30d_bin)] = v

    uale_cal = np.array([
        ALEATORIC_TABLE[(reg_cal_pred[t], ale_bin_cal[t])]
        for t in range(len(obs_cal))
    ])

    unique_uale = sorted(set(ALEATORIC_TABLE.values()))
    diffs       = [
        abs(np.mean(uale_cal > t) - NORMALIZED_ESC_COST) for t in unique_uale
    ]
    TAU_ALE          = float(unique_uale[np.argmin(diffs)])
    achieved_eta_ale = float(np.mean(uale_cal > TAU_ALE))

    print(
        f"TAU_ALE = {TAU_ALE:.1f} m³/s²  "
        f"(target η={NORMALIZED_ESC_COST:.3f}, achieved={achieved_eta_ale:.3f})"
    )
    return ALEATORIC_TABLE, TAU_ALE, achieved_eta_ale, global_var_cal


def loyo_tau_ale(
    cal: pd.DataFrame,
    Q_stats: dict,
    p30d_thresholds: tuple,
    min_cell_size: int = 5,
) -> pd.DataFrame:
    """Leave-one-year-out stability analysis for TAU_ALE.

    Parameters
    ----------
    cal : pd.DataFrame  Calibration period dataframe.
    Q_stats : dict  Training-period quantiles.
    p30d_thresholds : tuple  P30d tercile thresholds.
    min_cell_size : int  Minimum cell population for table entries.

    Returns
    -------
    pd.DataFrame  Columns: ['held_out', 'TAU_ALE'].
    """
    cal_years = cal.index.year.unique()
    records = []

    for held_out_year in cal_years:
        tr_mask = cal.index.year != held_out_year
        residual_cv = (
            cal.loc[tr_mask, "Q"].values
            - cal.loc[tr_mask, "Q_lstm"].values
        )
        reg_cv   = assign_regime(cal.loc[tr_mask, "Q_lstm"].values, Q_stats)
        p30d_cv  = p30d_to_bin(cal.loc[tr_mask, "P30d"].values, p30d_thresholds)
        gvar_cv  = float(np.nanvar(residual_cv))

        ale_tbl_cv: dict = {}
        for reg in REGIMES:
            for p30d_bin in P30D_LABELS:
                m = (
                    (reg_cv == reg)
                    & (p30d_cv == p30d_bin)
                    & ~np.isnan(residual_cv)
                )
                ale_tbl_cv[(reg, p30d_bin)] = (
                    float(np.var(residual_cv[m]))
                    if m.sum() >= min_cell_size
                    else gvar_cv
                )

        uale_cv  = np.array([ale_tbl_cv[(reg_cv[t], p30d_cv[t])]
                              for t in range(len(residual_cv))])
        unique_cv = sorted(set(ale_tbl_cv.values()))
        diffs_cv  = [
            abs(np.mean(uale_cv > t) - NORMALIZED_ESC_COST)
            for t in unique_cv
        ]
        records.append({
            "held_out": int(held_out_year),
            "TAU_ALE":  float(unique_cv[np.argmin(diffs_cv)]),
        })

    loyo_df = pd.DataFrame(records)
    print(
        f"TAU_ALE LOYO: mean={loyo_df.TAU_ALE.mean():.1f}  "
        f"std={loyo_df.TAU_ALE.std():.1f}"
    )
    return loyo_df


# ══════════════════════════════════════════════════════════════════════════════
# §3  Leg 1 Threshold Calibration (Regime-Specific τ)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_leg1_thresholds(
    uq_cal: pd.DataFrame,
    Q_stats: dict,
) -> tuple[dict, float]:
    """Calibrate regime-specific epistemic uncertainty thresholds for Leg 1.

    For each regime r, sweep τ over the calibration w_t distribution and
    select the value whose implied deferral rate ≈ NORMALIZED_ESC_COST.

    Parameters
    ----------
    uq_cal : pd.DataFrame  UQ for calibration period.
    Q_stats : dict         Training-period quantiles.

    Returns
    -------
    tau_regime : dict[str, float]  Per-regime τ thresholds.
    GLOBAL_TAU : float             Median of regime-specific τ values.
    """
    Q_lstm_cal = uq_cal["Q_mean"].values
    wt_c       = uq_cal["w_t"].values
    reg_c      = assign_regime(Q_lstm_cal, Q_stats)
    tau_regime: dict = {}

    for r in REGIMES:
        wt_r = wt_c[(reg_c == r) & ~np.isnan(wt_c)]
        if len(wt_r) == 0:
            tau_regime[r] = float(np.nanmedian(wt_c))
            continue
        thresholds = np.linspace(0, np.nanpercentile(wt_r, 99.5), 300)
        diffs      = [
            abs(np.mean(wt_r > t) - NORMALIZED_ESC_COST) for t in thresholds
        ]
        tau_regime[r] = float(thresholds[np.argmin(diffs)])

    GLOBAL_TAU = float(np.nanmedian(list(tau_regime.values())))

    print(f"\nLeg 1 τ thresholds (target η={NORMALIZED_ESC_COST:.3f}):")
    for r, v in tau_regime.items():
        mask_r = (reg_c == r) & ~np.isnan(wt_c)
        eta_r  = float(np.mean(wt_c[mask_r] > v)) if mask_r.any() else np.nan
        print(f"  {r:8s}: τ={v:.3f}  implied η={eta_r:.3f}")
    print(f"  GLOBAL_TAU (median): {GLOBAL_TAU:.3f}")
    return tau_regime, GLOBAL_TAU


def loyo_tau_regime(
    cal: pd.DataFrame,
    uq_cal: pd.DataFrame,
    Q_stats: dict,
) -> pd.DataFrame:
    """Leave-one-year-out stability for Leg 1 τ thresholds."""
    cal_years = cal.index.year.unique()
    records   = []

    for held_out_year in cal_years:
        cv_mask = cal.index.year != held_out_year
        Qm_cv   = uq_cal["Q_mean"].values[cv_mask]
        wt_cv   = uq_cal["w_t"].values[cv_mask]
        reg_cv  = assign_regime(Qm_cv, Q_stats)

        tau_cv: dict = {}
        for r in REGIMES:
            wt_r = wt_cv[(reg_cv == r) & ~np.isnan(wt_cv)]
            if len(wt_r) == 0:
                tau_cv[r] = float(np.nanmedian(wt_cv))
                continue
            thresholds = np.linspace(0, np.nanpercentile(wt_r, 99.5), 300)
            diffs      = [
                abs(np.mean(wt_r > t) - NORMALIZED_ESC_COST) for t in thresholds
            ]
            tau_cv[r] = float(thresholds[np.argmin(diffs)])

        row = {"held_out_year": int(held_out_year)}
        for r in REGIMES:
            row[f"tau_{r}"] = tau_cv[r]
        records.append(row)

    loyo_df = pd.DataFrame(records)
    print("\n--- Leg 1 Threshold Stability (LOYO-CV) ---")
    print(loyo_df.to_string(index=False))
    return loyo_df


# ══════════════════════════════════════════════════════════════════════════════
# §4  XGBoost Surrogate Expert
# ══════════════════════════════════════════════════════════════════════════════

XGB_FEATURE_NAMES = [
    "Q_lstm", "w_t", "P", "PET", "P7d", "P30d", "P90d",
    "ASM", "BF_proxy", "delta_Q", "sin_month", "cos_month",
]


def build_xgb_features(
    sub_df: pd.DataFrame,
    sub_uq: pd.DataFrame,
) -> np.ndarray:
    """Construct the feature matrix for XGBoost.

    Features
    --------
    Q_lstm      : Day-ahead ensemble mean streamflow forecast.
    w_t         : Normalised epistemic uncertainty spread.
    P, PET      : Daily precipitation and potential evapotranspiration.
    P7d, P30d, P90d : Rolling precipitation sums (7 / 30 / 90 day).
    ASM         : EWMA antecedent soil moisture proxy.
    BF_proxy    : 7-day rolling minimum Q_lstm (baseflow proxy).
    delta_Q     : Finite-difference hydrograph slope (rising / falling limb).
    sin_month, cos_month : Monthly cyclicity encoding.

    Parameters
    ----------
    sub_df : pd.DataFrame  Sub-period rows of the full dataframe.
    sub_uq : pd.DataFrame  Corresponding UQ rows.

    Returns
    -------
    np.ndarray of shape (n, 12), dtype float32.
    """
    Q_lstm_arr = sub_df["Q_lstm"].values
    # np.gradient requires ≥2 elements; use np.diff padded with 0 for the
    # first entry so the function is safe for single-row inference calls.
    if len(Q_lstm_arr) >= 2:
        delta_Q = np.gradient(Q_lstm_arr)
    else:
        delta_Q = np.zeros_like(Q_lstm_arr)
    bf_proxy   = pd.Series(Q_lstm_arr).rolling(7, min_periods=1).min().values
    month      = sub_df.index.month.values
    sin_month  = np.sin(2 * np.pi * month / 12)
    cos_month  = np.cos(2 * np.pi * month / 12)

    X = np.column_stack([
        Q_lstm_arr,
        sub_uq["w_t"].values,
        sub_df["P"].values,
        sub_df["PET"].values,
        sub_df["P7d"].values,
        sub_df["P30d"].values,
        sub_df["P90d"].values,
        sub_df["ASM"].values,
        bf_proxy,
        delta_Q,
        sin_month,
        cos_month,
    ])
    return X.astype(np.float32)


def train_xgb(
    df: pd.DataFrame,
    uq: pd.DataFrame,
    Q_stats: dict,
    output_dir: Path,
) -> xgb.Booster:
    """Train the XGBoost surrogate on training-period residuals.

    Cost-sensitive sample weights (COST_RATIO × for Q ≥ Q95) are applied
    to prioritise correct correction on high-flow events. A temporal
    validation split at 1998-01-01 is used for early stopping, avoiding
    any use of the calibration or test periods.

    Hyperparameters (printed below for reproducibility)
    ---------------------------------------------------
    learning_rate   : 0.05
    max_depth       : 5
    subsample       : 0.8
    colsample_bytree: 0.8
    n_estimators    : 800 (with early stopping, rounds=40)
    """
    XGB_PARAMS = {
        "objective":        "reg:squarederror",
        "learning_rate":    0.05,
        "max_depth":        5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma":            1.0,
        "lambda":           2.0,
        "seed":             42,
        "eval_metric":      "rmse",
    }

    # ── Print hyperparameters  ────────────────────────────────────
    print("\nXGBoost Hyperparameters:")
    for k, v in {
        "n_estimators":    800,
        "learning_rate":   XGB_PARAMS["learning_rate"],
        "max_depth":       XGB_PARAMS["max_depth"],
        "subsample":       XGB_PARAMS["subsample"],
        "colsample_bytree": XGB_PARAMS["colsample_bytree"],
    }.items():
        print(f"  {k:20s}: {v}")

    tr_mask = (df.index >= SPLITS["train"][0]) & (df.index <= SPLITS["train"][1])
    train_skip = df[tr_mask].iloc[WARMUP_DAYS:]
    uq_skip    = uq[tr_mask].iloc[WARMUP_DAYS:]

    X_raw = build_xgb_features(train_skip, uq_skip)
    y_raw = train_skip["Q"].values - train_skip["Q_lstm"].values

    mask_tr = ~np.isnan(y_raw) & ~np.any(np.isnan(X_raw), axis=1)
    X_train = X_raw[mask_tr]
    y_train = y_raw[mask_tr]

    obs_train = train_skip["Q"].values[mask_tr]
    weights   = np.where(
        obs_train >= Q_stats["Q95"], float(COST_RATIO), 1.0
    )

    val_split_date  = "1998-01-01"
    train_dates_msk = train_skip.index[mask_tr]
    val_idx         = train_dates_msk >= pd.Timestamp(val_split_date)

    X_fit, y_fit, w_fit = X_train[~val_idx], y_train[~val_idx], weights[~val_idx]
    X_val, y_val, w_val = X_train[val_idx],  y_train[val_idx],  weights[val_idx]

    dtrain = xgb.DMatrix(
        X_fit, label=y_fit, weight=w_fit, feature_names=XGB_FEATURE_NAMES
    )
    dval = xgb.DMatrix(
        X_val, label=y_val, weight=w_val, feature_names=XGB_FEATURE_NAMES
    )

    xgb_model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=800,
        evals=[(dval, "val")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )

    xgb_model.save_model(str(output_dir / "xgb_surrogate.json"))
    save_json(
        {
            **XGB_PARAMS,
            "num_boost_round":      800,
            "early_stopping_rounds": 40,
            "best_iteration":        xgb_model.best_iteration,
            "best_val_rmse":         float(xgb_model.best_score),
            "val_split_date":        val_split_date,
            "val_metric":            "rmse",
            "n_features":            len(XGB_FEATURE_NAMES),
            "feature_names":         XGB_FEATURE_NAMES,
        },
        output_dir / "xgb_hyperparams.json",
    )
    print(f"XGBoost trained.  Best iteration: {xgb_model.best_iteration}")
    return xgb_model


def apply_xgb_correction(
    sub_df: pd.DataFrame,
    sub_uq: pd.DataFrame,
    defer_mask: np.ndarray,
    xgb_model: xgb.Booster,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the XGBoost correction to deferred timesteps.

    Parameters
    ----------
    sub_df : pd.DataFrame  Sub-period rows of the full dataframe.
    sub_uq : pd.DataFrame  Corresponding UQ rows.
    defer_mask : (N,) bool  True where deferral is active.
    xgb_model : xgb.Booster  Trained surrogate model.

    Returns
    -------
    Q_corrected : (N,) array  Streamflow after correction (≥ 0).
    dq          : (N,) array  Raw XGBoost residual corrections.
    """
    Q_lstm_arr  = sub_df["Q_lstm"].values.copy()
    X           = build_xgb_features(sub_df, sub_uq)
    dq          = xgb_model.predict(xgb.DMatrix(X, feature_names=XGB_FEATURE_NAMES))
    Q_corrected = Q_lstm_arr.copy()
    Q_corrected[defer_mask] = np.clip(
        Q_lstm_arr[defer_mask] + dq[defer_mask], 0, None
    )
    return Q_corrected, dq


# ══════════════════════════════════════════════════════════════════════════════
# §5  Reliability Diagram (Raw Spearman on unbinned pairs)
# ══════════════════════════════════════════════════════════════════════════════

def compute_reliability_stats(
    obs_cal: np.ndarray,
    Q_lstm_cal: np.ndarray,
    uq_cal: pd.DataFrame,
    Q_stats: dict,
    output_dir: Path,
) -> dict:
    """Compute Spearman ρ on raw observation-residual pairs (no binning).

    Parameters
    ----------
    obs_cal : (N,) array  Observed streamflow (calibration period).
    Q_lstm_cal : (N,) array  LSTM ensemble mean (calibration period).
    uq_cal : pd.DataFrame  UQ for calibration period.
    Q_stats : dict  Training-period quantiles.
    output_dir : Path  Artefact directory.

    Returns
    -------
    dict  {'spearman_rho_raw', 'spearman_p_raw', 'monotone'}.
    """
    wt_cal_vals = uq_cal["w_t"].values
    valid       = ~np.isnan(obs_cal)
    abs_resid   = np.abs(obs_cal[valid] - Q_lstm_cal[valid])

    # Raw (unbinned) Spearman correlation
    rho_raw, p_raw = spearmanr(wt_cal_vals[valid], abs_resid)
    is_monotone    = float(rho_raw) > 0

    print(f"Raw Spearman (unbinned): ρ={rho_raw:.3f}  p={p_raw:.4f}  "
          f"n={valid.sum()}")

    stats = {
        "spearman_rho_raw": float(rho_raw),
        "spearman_p_raw":   float(p_raw),
        "monotone":         bool(is_monotone),
    }
    save_json(stats, output_dir / "reliability_diag.json")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# §6  Tri-Fold Routing Gateway
# ══════════════════════════════════════════════════════════════════════════════

def build_gateway_mask(
    n_test: int,
    wt_t: np.ndarray,
    taus_test: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    Q_lstm_t: np.ndarray,
) -> np.ndarray:
    """Apply the Tri-Fold Gateway with event-persistence logic.

    Leg 1 — Epistemic safeguard    : w_t > τ_regime
    Leg 2 — Aleatoric safeguard    : uale_t ≥ TAU_ALE  (≥ matches test logic)
    Leg 3 — Threshold Safeguard    : Q_lstm ≥ Q95

    Event persistence: once deferred, the system remains deferred while the
    predicted hydrograph is on the rising limb (dQ_pred > 0). This prevents
    chattering at event boundaries and reduces gateway switch frequency (churn).

    Parameters
    ----------
    n_test : int  Length of the test period.
    wt_t : (n_test,) array  Normalised epistemic uncertainty.
    taus_test : (n_test,) array  Per-day regime-matched threshold.
    leg2_aleatoric_test : (n_test,) bool  Leg 2 activation flags.
    leg3_hydro : (n_test,) bool  Leg 3 activation flags.
    Q_lstm_t : (n_test,) array  Day-ahead ensemble mean streamflow.

    Returns
    -------
    is_defer : (n_test,) bool  True where deferral is active.
    """
    # Pad with Q_lstm_t[0] to prevent a spurious spike on the first day
    dQ_pred   = np.diff(np.concatenate([[Q_lstm_t[0]], Q_lstm_t]))
    leg1_stat = wt_t > taus_test

    is_defer = np.zeros(n_test, dtype=bool)
    prev_def = False

    for t in range(n_test):
        defer = leg1_stat[t] or leg2_aleatoric_test[t] or leg3_hydro[t]
        # Event-persistence: stay deferred on rising limb after a prior deferral
        if not defer and prev_def and dQ_pred[t] > 0:
            defer = True
        is_defer[t] = defer
        prev_def = defer

    return is_defer


# ══════════════════════════════════════════════════════════════════════════════
# §7  Leg-Wise Ablation Study
# ══════════════════════════════════════════════════════════════════════════════

def run_leg_ablation(
    n_test: int,
    obs_t: np.ndarray,
    test: pd.DataFrame,
    uq_test: pd.DataFrame,
    wt_t: np.ndarray,
    taus_test: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    Q_lstm_t: np.ndarray,
    xgb_model: xgb.Booster,
    Q_stats: dict,
) -> pd.DataFrame:
    """Evaluate each gateway leg in isolation and in combination.

    Parameters
    ----------
    All parameters follow the naming conventions from the gateway section.

    Returns
    -------
    pd.DataFrame  Columns: ['name', 'deferral_rate', 'FNR'].
    """
    q95 = Q_stats["Q95"]

    def _eval_mask(name: str, mask: np.ndarray) -> dict:
        q_abl, _ = apply_xgb_correction(test, uq_test, mask, xgb_model)
        f_val, _ = fnr_wilson(obs_t, q_abl, q95)
        return {
            "name":          name,
            "deferral_rate": float(mask.mean() * 100),
            "FNR":           float(f_val) if not np.isnan(f_val) else np.nan,
        }

    ablation_strategies = {
        "Base Autonomous (S1)":        np.zeros(n_test, dtype=bool),
        "Leg 1 Active Only":           wt_t > taus_test,
        "Leg 2 Active Only":           leg2_aleatoric_test.copy(),     
        "Leg 3 (Threshold Safeguard) Only":      leg3_hydro.copy(),
        "Full Tri-Fold Policy (S2)":   build_gateway_mask(
            n_test, wt_t, taus_test, leg2_aleatoric_test, leg3_hydro, Q_lstm_t
        ),
    }

    rows = []
    print("\n--- Gateway Leg Ablation Analysis ---")
    for name, mask in ablation_strategies.items():
        row = _eval_mask(name, mask)
        rows.append(row)
        print(
            f"  {name:<35} "
            f"Deferral: {row['deferral_rate']:>5.1f}%  "
            f"FNR: {row['FNR']:.3f}"
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# §8  Off-Policy Evaluation — Direct Method (DM)
# ══════════════════════════════════════════════════════════════════════════════
#
# The LogisticRegression propensity model, IPW weights, and Doubly Robust
# estimator from the original code are removed entirely.  The gateway is
# deterministic, so propensity scores collapse to {0, 1}, violating positivity.
# We use the Direct Method exclusively (Dudík et al. 2011; Precup et al. 2000).
#
# DM reward model:
#   R = -1  if a flood day was missed (Q_pred < Q95 while Q_obs ≥ Q95)
#   R =  0  otherwise
#
# Counterfactual simulation:
#   V(S1) = mean reward if ALL days use the autonomous LSTM (never defer)
#   V(S2) = mean reward if ALL days use the LSTM+XGB+Human system (defer per policy)
# ─────────────────────────────────────────────────────────────────────────────

def run_ope_dm(
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    Q_stats: dict,
    n_boot: int = 1000,
    block_size: int = 120,
    seed: int = 42,
) -> dict:
    """Direct Method OPE with seasonal block bootstrap confidence intervals.

    Parameters
    ----------
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  Autonomous LSTM predictions (S1).
    Q_system : (N,) array  Full system predictions (S2).
    is_defer : (N,) bool  Gateway deferral mask.
    Q_stats : dict  Must contain 'Q95'.
    n_boot : int  Bootstrap resamples.
    block_size : int  Seasonal block length in days.
    seed : int  Random seed.

    Returns
    -------
    dict  Keys: 'V_S1', 'V_S2', 'CI_S2_lo', 'CI_S2_hi', 'S2_better'.
    """
    q95 = Q_stats["Q95"]

    def _reward(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Flood-miss indicator: -1 if flood missed, 0 otherwise."""
        return np.where(
            (~np.isnan(obs)) & (obs >= q95) & (pred < q95),
            -1.0,
            0.0,
        )

    r_s1 = _reward(obs_t, Q_lstm_t)
    r_s2 = _reward(obs_t, Q_system)

    V_s1 = float(r_s1.mean())
    V_s2 = float(r_s2.mean())

    # Seasonal block bootstrap for V_S2
    rng      = np.random.default_rng(seed)
    n        = len(obs_t)
    boot_v2  = []

    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size, size=n // block_size + 1)
        idx    = np.concatenate([
            np.arange(s, min(s + block_size, n)) for s in starts
        ])[:n]
        boot_v2.append(float(_reward(obs_t[idx], Q_system[idx]).mean()))

    CI_lo = float(np.percentile(boot_v2, 2.5))
    CI_hi = float(np.percentile(boot_v2, 97.5))

    print(f"\nOPE — Direct Method (DM) with {block_size}-day seasonal bootstrap:")
    print(f"  V_DM(S1) = {V_s1:.4f}")
    print(
        f"  V_DM(S2) = {V_s2:.4f}  "
        f"95% CI = [{CI_lo:.4f}, {CI_hi:.4f}]  "
        f"{'S2 better ✓' if V_s2 > V_s1 else 'S1 better ✗'}"
    )

    return {
        "V_S1":      V_s1,
        "V_S2":      V_s2,
        "CI_S2_lo":  CI_lo,
        "CI_S2_hi":  CI_hi,
        "S2_better": bool(V_s2 > V_s1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §9  Strategy Metrics & Comparison
# ══════════════════════════════════════════════════════════════════════════════

def compute_strategy_metrics(
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    Q_stats: dict,
) -> tuple[dict, dict, float]:
    """Compute full metric suite for S1 (LSTM) and S2 (system) strategies.

    Parameters
    ----------
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  LSTM-only predictions.
    Q_system : (N,) array  Full system predictions.
    is_defer : (N,) bool  Gateway deferral mask.
    Q_stats : dict  Training-period quantiles.

    Returns
    -------
    s1_metrics, s2_metrics : dict  Full metric dictionaries.
    pval_mcnemar : float  McNemar FNR comparison p-value.
    """
    valid_t = ~np.isnan(obs_t)
    q95     = Q_stats["Q95"]

    fnr_s1, fnr_s1_ci = fnr_wilson(obs_t, Q_lstm_t, q95)
    fnr_s2, fnr_s2_ci = fnr_wilson(obs_t, Q_system,  q95)

    s1_boot = moving_block_bootstrap(obs_t, Q_lstm_t)
    s2_boot = moving_block_bootstrap(obs_t, Q_system)

    s1_metrics = {
        "strategy": "S1-LSTM",
        "NSE":      nse(obs_t[valid_t], Q_lstm_t[valid_t]),
        "KGE":      kge(obs_t[valid_t], Q_lstm_t[valid_t]),
        "RMSE":     rmse(obs_t[valid_t], Q_lstm_t[valid_t]),
        "FNR":      fnr_s1,
        "PBIAS":    pbias(obs_t[valid_t], Q_lstm_t[valid_t]),
        "eta":      float("nan"),
        "FNR_CI":   fnr_s1_ci,
        "CIs":      s1_boot,
    }

    s2_metrics = {
        "strategy": "S2-LSTM+XGB+Human",
        "NSE":      nse(obs_t[valid_t], Q_system[valid_t]),
        "KGE":      kge(obs_t[valid_t], Q_system[valid_t]),
        "RMSE":     rmse(obs_t[valid_t], Q_system[valid_t]),
        "FNR":      fnr_s2,
        "PBIAS":    pbias(obs_t[valid_t], Q_system[valid_t]),
        "eta":      float(is_defer.mean()),
        "FNR_CI":   fnr_s2_ci,
        "CIs":      s2_boot,
    }

    pval_mcnemar = mcnemar_fnr_test(obs_t, Q_lstm_t, Q_system, q95)
    delta_h      = fnr_s1 - fnr_s2

    print("\n" + "=" * 62)
    print("Strategy Comparison — Test 2008–2011")
    print("=" * 62)
    for m in [s1_metrics, s2_metrics]:
        print(f"  {m['strategy']}")
        print(f"    NSE  = {m['NSE']:.3f} "
              f"[{m['CIs'].get('NSE', (0, 0))[0]:.3f}, "
              f"{m['CIs'].get('NSE', (0, 0))[1]:.3f}]")
        print(f"    KGE  = {m['KGE']:.3f} "
              f"[{m['CIs'].get('KGE', (0, 0))[0]:.3f}, "
              f"{m['CIs'].get('KGE', (0, 0))[1]:.3f}]")
        print(f"    FNR  = {m['FNR']:.3f} "
              f"[{m['FNR_CI'][0]:.3f}, {m['FNR_CI'][1]:.3f}]")
    print(f"\n  δ_h = {delta_h:.3f}  η = {s2_metrics['eta']:.3f}")
    print(f"  McNemar p-value (S1 vs S2 FNR): {pval_mcnemar:.4e}")

    return s1_metrics, s2_metrics, pval_mcnemar


# ══════════════════════════════════════════════════════════════════════════════
# §10  Rating Curve Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════════════

def rating_curve_sensitivity(
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    Q_stats: dict,
    output_dir: Path,
) -> tuple[list, list]:
    """Evaluate FNR under systematic rating curve errors ±20 %.

    Parameters
    ----------
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  LSTM-only predictions.
    Q_system : (N,) array  Full system predictions.
    Q_stats : dict  Training-period quantiles.
    output_dir : Path  Artefact directory.

    Returns
    -------
    s1_fnr_series, s2_fnr_series : list[float]  FNR per RC error level.
    """
    rc_errors = [-0.20, -0.10, 0.00, 0.10, 0.20]
    s1_fnr_series, s2_fnr_series = [], []

    print("\nRating-Curve Uncertainty Sensitivity Analysis:")
    for rc_err in rc_errors:
        obs_adj  = obs_t * (1.0 + rc_err)
        q95_adj  = Q_stats["Q95"] * (1.0 + rc_err)   
        f1, _    = fnr_wilson(obs_adj, Q_lstm_t, q95_adj)
        f2, _    = fnr_wilson(obs_adj, Q_system,  q95_adj)
        s1_fnr_series.append(f1)
        s2_fnr_series.append(f2)
        print(f"  RC {rc_err:+.0%}: S1_FNR={f1:.3f}  S2_FNR={f2:.3f}  Δ={f1-f2:.3f}")

    return s1_fnr_series, s2_fnr_series, rc_errors


# ══════════════════════════════════════════════════════════════════════════════
# §11  Risk-Coverage Sweep (RC Curves with event-persistence parity)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_rc_curves(
    n_test: int,
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    wt_t: np.ndarray,
    taus_test: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    dq_test: np.ndarray,
    Q_stats: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """Sweep Leg 1 threshold τ to generate Risk-Coverage curves.

    All three strategies (S1-LSTM, S2-Oracle, S2-XGB) use the EXACT same
    routing logic (including event-persistence) for a fair comparison.

    Parameters
    ----------
    n_test : int  Length of the test period.
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  Day-ahead ensemble mean.
    wt_t : (N,) array  Normalised epistemic uncertainty.
    taus_test : (N,) array  Per-day regime-matched threshold.
    leg2_aleatoric_test : (N,) bool  Leg 2 activation flags.
    leg3_hydro : (N,) bool  Leg 3 (Threshold Safeguard) activation flags.
    dq_test : (N,) array  XGBoost residual corrections.
    Q_stats : dict  Training-period quantiles.
    output_dir : Path  Artefact directory.

    Returns
    -------
    pd.DataFrame  RC curve records with columns: tau, coverage, risk, strategy.
    """
    q95       = Q_stats["Q95"]
    thresholds = np.quantile(wt_t, np.linspace(0, 0.999, 100))
    dq_oracle  = np.where(~np.isnan(obs_t), obs_t - Q_lstm_t, 0.0)

    # S1-LSTM baseline FNR is constant (never defers)
    s1_baseline_fnr, _ = fnr_wilson(obs_t, Q_lstm_t, q95)

    # Leg 2 and Leg 3 are disabled during the sweep so the curve isolates
    # Leg 1's (epistemic threshold's) contribution to coverage vs. risk.
    _leg2_off = np.zeros(n_test, dtype=bool)
    _leg3_off = np.zeros(n_test, dtype=bool)

    rc_rows = []
    for tau in thresholds:
        defer_rc = build_gateway_mask(
            n_test, wt_t,
            np.full(n_test, tau),   # uniform Leg 1 τ sweep
            _leg2_off, _leg3_off, Q_lstm_t,
        )
        coverage = float((~defer_rc).mean())

        # S1-LSTM: constant horizontal baseline  
        rc_rows.append({
            "tau": tau, "coverage": coverage,
            "risk": s1_baseline_fnr,
            "strategy": "S1-LSTM",
        })

        # S2-Oracle
        Q_oracle = np.where(
            defer_rc, np.clip(Q_lstm_t + dq_oracle, 0, None), Q_lstm_t
        )
        rc_rows.append({
            "tau": tau, "coverage": coverage,
            "risk": fnr_wilson(obs_t, Q_oracle, q95)[0],
            "strategy": "S2-Oracle",
        })

        # S2-XGB
        Q_xgb_rc = np.where(
            defer_rc, np.clip(Q_lstm_t + dq_test, 0, None), Q_lstm_t
        )
        rc_rows.append({
            "tau": tau, "coverage": coverage,
            "risk": fnr_wilson(obs_t, Q_xgb_rc, q95)[0],
            "strategy": "S2-XGB",
        })

    rc_df = pd.DataFrame(rc_rows)
    rc_df.to_csv(output_dir / "rc_curves.csv", index=False)
    print(
        f"RC curves generated across {len(thresholds)} threshold steps.  "
        f"S1-LSTM baseline FNR = {s1_baseline_fnr:.3f} (constant)."
    )
    return rc_df


# ══════════════════════════════════════════════════════════════════════════════
# §12  Failure Attribution & Confident-But-Wrong Diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def failure_attribution(
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    Q_stats: dict,
) -> pd.DataFrame:
    """Decompose missed floods into gateway misses vs. XGB correction failures.

    Parameters
    ----------
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  LSTM-only predictions.
    Q_system : (N,) array  Full system predictions.
    is_defer : (N,) bool  Gateway deferral mask.
    Q_stats : dict  Training-period quantiles.

    Returns
    -------
    pd.DataFrame  Failure type counts.
    """
    q95 = Q_stats["Q95"]

    n_s1_miss   = int(((obs_t >= q95) & (Q_lstm_t < q95)  & ~np.isnan(obs_t)).sum())
    n_s2_miss   = int(((obs_t >= q95) & (Q_system  < q95)  & ~np.isnan(obs_t)).sum())
    n_gate_miss = int(((obs_t >= q95) & ~is_defer  & (Q_lstm_t < q95) & ~np.isnan(obs_t)).sum())
    n_xgb_miss  = int(((obs_t >= q95) &  is_defer  & (Q_system  < q95) & ~np.isnan(obs_t)).sum())

    table = pd.DataFrame([
        {
            "Type":      "Gateway missed event (LSTM failure, not deferred)",
            "Count":     n_gate_miss,
            "Mitigable": "Tighten TAU_ALE / Q95 threshold or add more legs",
        },
        {
            "Type":      "XGB correction insufficient on deferred days",
            "Count":     n_xgb_miss,
            "Mitigable": "More calibration data / heteroscedastic XGB head",
        },
        {
            "Type":      "Total missed floods (S2 system)",
            "Count":     n_s2_miss,
            "Mitigable": "—",
        },
        {
            "Type":      "Total missed floods (S1 LSTM only)",
            "Count":     n_s1_miss,
            "Mitigable": "—",
        },
    ])
    print("\nFailure attribution (test period):")
    print(table.to_string(index=False))
    return table


def confident_but_wrong(
    test: pd.DataFrame,
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    wt_t: np.ndarray,
    uale_t: np.ndarray,
    is_defer: np.ndarray,
    Q_stats: dict,
    GLOBAL_TAU: float,
    P30D_Q67: float,
    output_dir: Path,
) -> dict:
    """Identify and analyse flood days where the gateway fired no leg.

    A 'confident-but-wrong' day is one where:
    - The observed flow exceeded Q95 (flood), AND
    - The gateway deferred to no leg (w_t ≤ τ, uale < TAU_ALE, Q_lstm < Q95), AND
    - The LSTM prediction missed the flood.

    Parameters
    ----------
    All arrays aligned to the test period.

    Returns
    -------
    dict  Summary statistics for confident-but-wrong days.
    """
    q95 = Q_stats["Q95"]
    confident_wrong_mask = (
        (obs_t >= q95)
        & ~is_defer
        & (Q_lstm_t < q95)
        & ~np.isnan(obs_t)
    )
    n_cw = int(confident_wrong_mask.sum())

    print(f"\n{'=' * 65}")
    print(f"Confident-But-Wrong Diagnostic  (n={n_cw} days)")
    print(f"{'=' * 65}")
    print("  Flood days where all three gateway legs were inactive.")

    if n_cw > 0:
        cw_idx   = np.where(confident_wrong_mask)[0]
        cw_dates = test.index[cw_idx]

        cw_wt    = wt_t[cw_idx]
        cw_uale  = uale_t[cw_idx]
        cw_p7    = test["P7d"].values[cw_idx]
        cw_p30   = test["P30d"].values[cw_idx]
        cw_qobs  = obs_t[cw_idx]
        cw_qlstm = Q_lstm_t[cw_idx]

        print(f"\n  Feature summary ({n_cw} days):")
        print(f"    {'Metric':20s}  {'Mean':>8s}  {'Median':>8s}  {'Max':>8s}")
        print(f"    {'-' * 50}")
        for label, arr in [
            ("Q_obs (m³/s)",    cw_qobs),
            ("Q_lstm (m³/s)",   cw_qlstm),
            ("w_t (epistemic)", cw_wt),
            ("uale (m³/s²)",    cw_uale),
            ("P7d (mm)",        cw_p7),
            ("P30d (mm)",       cw_p30),
        ]:
            print(f"    {label:20s}  {np.mean(arr):8.1f}  "
                  f"{np.median(arr):8.1f}  {np.max(arr):8.1f}")

        # Group into events (gap ≤ 2 days)
        ev_start, ev_last = cw_dates[0], cw_dates[0]
        cw_event_dates = []
        for d in cw_dates[1:]:
            if (d - ev_last).days <= 2:
                ev_last = d
            else:
                cw_event_dates.append((ev_start, ev_last))
                ev_start = ev_last = d
        cw_event_dates.append((ev_start, ev_last))

        q95_80pct = 0.80 * q95
        cw_caught_lower_q95 = int((cw_qlstm >= q95_80pct).sum())

        summary = {
            "n_confident_wrong":    n_cw,
            "n_events":             len(cw_event_dates),
            "mean_Q_obs":           float(np.mean(cw_qobs)),
            "mean_Q_lstm":          float(np.mean(cw_qlstm)),
            "mean_wt":              float(np.mean(cw_wt)),
            "mean_uale":            float(np.mean(cw_uale)),
            "mean_P30d":            float(np.mean(cw_p30)),
            "caught_by_lower_q95":  cw_caught_lower_q95,
            "event_dates": [
                (str(s.date()), str(e.date())) for s, e in cw_event_dates
            ],
        }
    else:
        print("  None — all flood days deferred or correctly forecast.")
        summary = {"n_confident_wrong": 0}

    save_json(summary, output_dir / "confident_wrong.json")
    print("=" * 65)
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# §13  Flood-Day Classification Overlap Table
# ══════════════════════════════════════════════════════════════════════════════

def flood_classification_overlap(
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    Q_stats: dict,
    output_dir: Path,
) -> dict:
    """Compute a WMO-compliant flood-day verification contingency table.

    False Alarm Ratio (FAR) and False Alarm Rate (FPR) are computed
    separately with clear naming to avoid WMO terminology ambiguity.

    Parameters
    ----------
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  LSTM-only predictions.
    Q_system : (N,) array  Full system predictions.
    Q_stats : dict  Training-period quantiles.
    output_dir : Path  Artefact directory.

    Returns
    -------
    dict  Contingency table and verification statistics.
    """
    q95           = Q_stats["Q95"]
    valid         = ~np.isnan(obs_t)
    obs_flood     = obs_t[valid] >= q95
    lstm_flood    = Q_lstm_t[valid] >= q95
    sys_flood     = Q_system[valid] >= q95

    n_valid       = int(valid.sum())
    n_flood_days  = int(obs_flood.sum())
    n_nflood_days = int((~obs_flood).sum())

    fday  = obs_flood
    nfday = ~obs_flood

    both_catch_f    = int(( lstm_flood[fday] &  sys_flood[fday]).sum())
    both_miss_f     = int((~lstm_flood[fday] & ~sys_flood[fday]).sum())
    only_s1_f       = int(( lstm_flood[fday] & ~sys_flood[fday]).sum())
    only_s2_f       = int((~lstm_flood[fday] &  sys_flood[fday]).sum())

    s1_fa           = int(lstm_flood[nfday].sum())
    s2_fa           = int(sys_flood[nfday].sum())
    s1_pos          = int(lstm_flood.sum())
    s2_pos          = int(sys_flood.sum())

    far_s1 = float(s1_fa / max(s1_pos, 1))
    far_s2 = float(s2_fa / max(s2_pos, 1))
    fpr_s1 = float(s1_fa / max(n_nflood_days, 1))
    fpr_s2 = float(s2_fa / max(n_nflood_days, 1))

    print("\n" + "=" * 65)
    print("Flood Day Classification Overlap — Test 2008–2011")
    print("=" * 65)
    print(f"  Flood days (n={n_flood_days}):")
    for label, count in [
        ("Both caught", both_catch_f),
        ("Both missed", both_miss_f),
        ("Only S1 caught", only_s1_f),
        ("Only S2 caught", only_s2_f),
    ]:
        print(f"    {label:<20s}: {count:4d}  ({100*count/max(n_flood_days,1):.1f}%)")
    print(f"\n  Non-flood days (n={n_nflood_days}):")
    print(f"    S1 FAR (ratio): {far_s1:.3f}  FPR (rate): {fpr_s1:.3f}")
    print(f"    S2 FAR (ratio): {far_s2:.3f}  FPR (rate): {fpr_s2:.3f}")

    result = {
        "n_valid": n_valid, "n_flood_days": n_flood_days,
        "n_nonflood_days": n_nflood_days,
        "FAR_Ratio_S1": far_s1, "FAR_Ratio_S2": far_s2,
        "FPR_Rate_S1": fpr_s1,  "FPR_Rate_S2": fpr_s2,
        "flood_days": {
            "both_caught": both_catch_f, "both_missed": both_miss_f,
            "only_s1": only_s1_f, "only_s2": only_s2_f,
        },
    }
    save_json(result, output_dir / "error_overlap.json")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# §14  Block-Permutation Complementarity Test
# ══════════════════════════════════════════════════════════════════════════════

def block_permutation_complementarity(
    cal: pd.DataFrame,
    obs_cal: np.ndarray,
    Q_lstm_cal: np.ndarray,
    Q_xgb_cal_all: np.ndarray,
    reg_cal_pred: np.ndarray,
    asm_cal_bins: np.ndarray,
    Q_stats: dict,
    n_perm: int = 1000,
) -> float:
    """Block-permutation test for spatial variance of oracle complementarity.

    Parameters
    ----------
    cal : pd.DataFrame  Calibration dataframe.
    obs_cal, Q_lstm_cal, Q_xgb_cal_all : (N,) array
        Observed, LSTM, and XGBoost calibration predictions.
    reg_cal_pred, asm_cal_bins : (N,) object array  Regime / ASM labels.
    Q_stats : dict  Training-period quantiles.
    n_perm : int  Number of permutation samples.

    Returns
    -------
    float  Block-permutation p-value.
    """
    def _rmse_arr(o, s):
        m = ~np.isnan(o) & ~np.isnan(s)
        return float(np.sqrt(np.mean((o[m] - s[m]) ** 2))) if m.sum() > 0 else np.nan

    heatmap = pd.DataFrame(index=ASM_LABELS, columns=REGIMES, dtype=float)
    for reg in REGIMES:
        for asm in ASM_LABELS:
            cell  = (reg_cal_pred == reg) & (asm_cal_bins == asm)
            valid = cell & ~np.isnan(obs_cal)
            if valid.sum() < 5:
                heatmap.loc[asm, reg] = np.nan
                continue
            o  = obs_cal[valid]
            lp = Q_lstm_cal[valid]
            hp = Q_xgb_cal_all[valid]
            oracle = np.where(np.abs(o - lp) <= np.abs(o - hp), lp, hp)
            heatmap.loc[asm, reg] = round(
                float(min(_rmse_arr(o, lp), _rmse_arr(o, hp)) - _rmse_arr(o, oracle)), 4
            )

    vals     = heatmap.values.flatten().astype(float)
    obs_var  = float(np.var(vals[~np.isnan(vals)]))
    rng_comp = np.random.default_rng(42)
    cal_idx  = cal.index
    months   = cal.index.to_period("M").unique()
    month_indices = [np.where(cal_idx.to_period("M") == m)[0] for m in months]
    null_vars = []

    for _ in range(n_perm):
        perm_order = rng_comp.permutation(len(month_indices)) 
        new_reg = np.full_like(reg_cal_pred, fill_value="low")
        new_asm = np.full_like(asm_cal_bins, fill_value="moderate")

        for src, dst in zip(perm_order, range(len(month_indices))):
            src_idx = month_indices[src]
            dst_idx = month_indices[dst]
            min_len = min(len(src_idx), len(dst_idx))
            new_reg[dst_idx[:min_len]] = reg_cal_pred[src_idx[:min_len]]
            new_asm[dst_idx[:min_len]] = asm_cal_bins[src_idx[:min_len]]

        null_v = []
        for reg in REGIMES:
            for asm in ASM_LABELS:
                m_    = (new_reg == reg) & (new_asm == asm)
                valid = m_ & ~np.isnan(obs_cal)
                if valid.sum() < 5:
                    continue
                o  = obs_cal[valid]
                lp = Q_lstm_cal[valid]
                hp = Q_xgb_cal_all[valid]
                oracle = np.where(np.abs(o - lp) <= np.abs(o - hp), lp, hp)
                null_v.append(
                    min(_rmse_arr(o, lp), _rmse_arr(o, hp)) - _rmse_arr(o, oracle)
                )
        if null_v:
            null_vars.append(float(np.var(null_v)))

    p_block = float(np.mean(np.array(null_vars) >= obs_var))
    print(f"Block-permutation p={p_block:.3f}  (Variance={obs_var:.3f})")
    return p_block


# ══════════════════════════════════════════════════════════════════════════════
# §15  Computational Latency Profiling
# ══════════════════════════════════════════════════════════════════════════════

def profile_single_day_latency(
    test: pd.DataFrame,
    uq_test: pd.DataFrame,
    aci: "MultiHorizonACI",
    xgb_model: xgb.Booster,
    is_defer: np.ndarray,
    n_reps: int = 100,
) -> float:
    """Profile mean single-day forward inference latency in milliseconds.

    Parameters
    ----------
    test : pd.DataFrame  Test period dataframe (one row used).
    uq_test : pd.DataFrame  Corresponding UQ row.
    aci : MultiHorizonACI  Calibrated ACI engine.
    xgb_model : xgb.Booster  Trained surrogate.
    is_defer : (N,) bool  Gateway mask (for context).
    n_reps : int  Number of repetitions for stable timing.

    Returns
    -------
    float  Mean latency in milliseconds.
    """
    # Pass 2 rows so build_xgb_features has ≥2 elements for np.gradient,
    # then take only the first row's prediction for the latency measurement.
    row_df  = test.iloc[:2]
    row_uq  = uq_test.iloc[:2]
    mu_val  = float(uq_test["Q_mean"].values[0])
    sig_val = float(uq_test["Q_std"].values[0])

    latencies = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _  = aci.build_interval(mu_val, sig_val, h=1)
        _  = xgb_model.predict(
            xgb.DMatrix(
                build_xgb_features(row_df, row_uq),
                feature_names=XGB_FEATURE_NAMES,
            )
        )
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(latencies))
    print(f"Single-day inference latency ({n_reps} reps): {mean_ms:.3f} ms")
    return mean_ms
