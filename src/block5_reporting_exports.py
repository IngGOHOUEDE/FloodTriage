"""
FloodTriage — Block 5: Ensemble Performance Reporting, Diagnostics & Exports
=============================================================================
Ouémé at Savè, Benin | 1965–2011

Covers the remaining logic from the original notebook not captured in
Blocks 1–4:

1. Ensemble LSTM performance table (Train / Cal / Test) with FNR and PBIAS.
2. w_t distribution summary (max / 99th / 95th percentiles, NaN count).
3. LOYO stability summary printer (Leg 1 + Leg 2 combined report).
4. Aleatoric table save / reload helpers.
5. ACI parameter JSON export (unified schema matching pipeline_summary.json).
6. Policy DataFrame construction and export (policy.csv).
7. Complementarity heatmap printer.
8. Ensemble-period KGE confidence interval report (block-bootstrap).
9. McNemar summary printer with decision statement.
10. Scaler statistics printer.
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Assume Block 1 globals are in scope (notebook) or import below ─────────────
try:
    from block1_data_prep_utilities import (
        ASM_LABELS, BASIN_ID, COST_RATIO, H_MAX, NORMALIZED_ESC_COST,
        OUTPUT_DIR, P30D_LABELS, REGIMES, SEEDS, SPLITS, WARMUP_DAYS,
        Q_stats, assign_regime, df, fnr_wilson, kge, mcnemar_fnr_test,
        moving_block_bootstrap, nse, pbias, rmse, save_json,
    )
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §1  Ensemble LSTM Performance Table
# ══════════════════════════════════════════════════════════════════════════════

def print_ensemble_performance(
    df: pd.DataFrame,
    uq: pd.DataFrame,
    Q_stats: dict,
    splits: dict,
    warmup_days: int,
) -> None:
    """Print NSE / KGE / RMSE / FNR / PBIAS for Train, Cal, and Test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with 'Q' (observed) and 'Q_lstm' (ensemble mean).
    uq : pd.DataFrame
        Uncertainty quantification dataframe with 'w_t'.
    Q_stats : dict
        Training-period quantiles; must contain 'Q95', 'Q_std'.
    splits : dict
        Date-range dict with keys 'train', 'cal', 'test'.
    warmup_days : int
        Number of leading training days to skip (LSTM warm-up).
    """
    q95 = Q_stats["Q95"]

    print("=" * 62)
    print("Ensemble LSTM Performance")
    print("=" * 62)

    for label, (s, e), skip in [
        ("Train", splits["train"], warmup_days),
        ("Cal",   splits["cal"],   0),
        ("Test",  splits["test"],  0),
    ]:
        sub = df.loc[s:e]
        obs = sub["Q"].values[skip:]
        sim = sub["Q_lstm"].values[skip:]
        fnr_pt, _ = fnr_wilson(obs, sim, q95)
        print(
            f"{label:5s}  "
            f"NSE={nse(obs, sim):.3f}  "
            f"KGE={kge(obs, sim):.3f}  "
            f"RMSE={rmse(obs, sim):.1f}  "
            f"FNR={fnr_pt:.3f}  "
            f"PBIAS={pbias(obs, sim):+.1f}%"
        )

    wt = uq["w_t"].values
    print(
        f"\nw_t  max={np.nanmax(wt):.3f}  "
        f"99th={np.nanpercentile(wt, 99):.3f}  "
        f"95th={np.nanpercentile(wt, 95):.3f}  "
        f"NaN days={int(np.isnan(wt).sum())}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# §2  LOYO Stability Summary Printer
# ══════════════════════════════════════════════════════════════════════════════

def print_loyo_summary(
    loyo_leg1_df: pd.DataFrame,
    loyo_ale_df: pd.DataFrame,
) -> None:
    """Print a combined LOYO stability summary for Leg 1 and Leg 2 thresholds.

    Parameters
    ----------
    loyo_leg1_df : pd.DataFrame
        Output of :func:`block3_gateway_evaluation.loyo_tau_regime`.
        Columns: 'held_out_year', 'tau_low', 'tau_normal', 'tau_flood'.
    loyo_ale_df : pd.DataFrame
        Output of :func:`block3_gateway_evaluation.loyo_tau_ale`.
        Columns: 'held_out', 'TAU_ALE'.
    """
    print("\n" + "=" * 62)
    print("LOYO Threshold Stability Summary")
    print("=" * 62)

    print("\nLeg 1 — Regime-Specific τ (Epistemic):")
    for r in REGIMES:
        col = f"tau_{r}"
        if col in loyo_leg1_df.columns:
            vals = loyo_leg1_df[col]
            print(f"  τ_{r:8s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"range=[{vals.min():.3f}, {vals.max():.3f}]")

    print("\nLeg 2 — TAU_ALE (Aleatoric):")
    vals = loyo_ale_df["TAU_ALE"]
    print(f"  TAU_ALE:    mean={vals.mean():.1f}  std={vals.std():.1f}  "
          f"range=[{vals.min():.1f}, {vals.max():.1f}]")


# ══════════════════════════════════════════════════════════════════════════════
# §3  Aleatoric Table Save / Reload
# ══════════════════════════════════════════════════════════════════════════════

def save_aleatoric_table(
    ALEATORIC_TABLE: dict,
    TAU_ALE: float,
    achieved_eta_ale: float,
    global_var_cal: float,
    P30D_Q33: float,
    P30D_Q67: float,
    output_dir: Path,
) -> None:
    """Persist the aleatoric table and associated calibration parameters.

    Parameters
    ----------
    ALEATORIC_TABLE : dict  {(regime, p30d_bin): variance}
    TAU_ALE : float  Aleatoric deferral threshold.
    achieved_eta_ale : float  Implied deferral rate.
    global_var_cal : float  Global calibration residual variance (fallback).
    P30D_Q33, P30D_Q67 : float  P30d tercile thresholds.
    output_dir : Path  Artefact directory.
    """
    json_safe = {f"{k[0]}|{k[1]}": v for k, v in ALEATORIC_TABLE.items()}
    with open(output_dir / "aleatoric_table.json", "w") as fh:
        json.dump(json_safe, fh, indent=2)

    params = {
        "TAU_ALE":           TAU_ALE,
        "achieved_eta_ale":  achieved_eta_ale,
        "global_var_cal":    global_var_cal,
        "P30D_Q33_mm":       P30D_Q33,
        "P30D_Q67_mm":       P30D_Q67,
    }
    with open(output_dir / "aleatoric_params.json", "w") as fh:
        json.dump(params, fh, indent=2)

    print("Aleatoric table and params saved.")


def load_aleatoric_table(output_dir: Path) -> tuple[dict, dict]:
    """Reload the aleatoric table and params from JSON.

    Parameters
    ----------
    output_dir : Path  Artefact directory.

    Returns
    -------
    ALEATORIC_TABLE : dict  {(regime, p30d_bin): variance}
    params : dict  TAU_ALE and threshold values.
    """
    with open(output_dir / "aleatoric_table.json") as fh:
        raw = json.load(fh)
    ALEATORIC_TABLE = {}
    for k, v in raw.items():
        regime, p30d_bin = k.split("|")
        ALEATORIC_TABLE[(regime, p30d_bin)] = float(v)

    with open(output_dir / "aleatoric_params.json") as fh:
        params = json.load(fh)

    return ALEATORIC_TABLE, params


# ══════════════════════════════════════════════════════════════════════════════
# §4  Policy DataFrame Construction & Export
# ══════════════════════════════════════════════════════════════════════════════

def build_policy_df(
    test: pd.DataFrame,
    Q_lstm_t: np.ndarray,
    dq_test: np.ndarray,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    leg1_stat: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    uale_t: np.ndarray,
    wt_t: np.ndarray,
    Q_lo_test: np.ndarray,
    Q_hi_test: np.ndarray,
    alpha_seq: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """Construct and save the per-day policy record DataFrame (policy.csv).

    Leg 3 column is named 'leg3_threshold_safeguard' for naming consistency.

    Parameters
    ----------
    All arrays are aligned to the test period index.
    output_dir : Path  Artefact directory.

    Returns
    -------
    pd.DataFrame  One row per test day.
    """
    policy_df = pd.DataFrame(
        {
            "action":             np.where(is_defer, "defer", "model"),
            "Q_lstm":             Q_lstm_t,
            "dQ_xgb":             dq_test,
            "Q_system":           Q_system,
            "leg1_epistemic":     leg1_stat,
            "leg2_aleatoric":     leg2_aleatoric_test,
            "leg3_threshold_safeguard":     leg3_hydro,      # renamed from leg3_hydro 
            "uale_t":             uale_t,
            "w_t":                wt_t,
            "Q_lo":               Q_lo_test,
            "Q_hi":               Q_hi_test,
            "alpha_t":            alpha_seq,
        },
        index=test.index,
    )
    policy_df.to_csv(output_dir / "policy.csv")
    print(f"Policy DataFrame saved ({len(policy_df)} rows).")
    return policy_df


# ══════════════════════════════════════════════════════════════════════════════
# §5  Complementarity Heatmap Printer
# ══════════════════════════════════════════════════════════════════════════════

def print_complementarity_heatmap(
    cal: pd.DataFrame,
    obs_cal: np.ndarray,
    Q_lstm_cal: np.ndarray,
    Q_xgb_cal_all: np.ndarray,
    reg_cal_pred: np.ndarray,
    asm_cal_bins: np.ndarray,
) -> None:
    """Print the oracle RMSE complementarity heatmap (regime × ASM).

    This summarises how much the oracle selector (choosing the better of LSTM
    vs XGBoost per timestep) improves over the best single model in each cell.

    Parameters
    ----------
    cal : pd.DataFrame  Calibration dataframe.
    obs_cal, Q_lstm_cal, Q_xgb_cal_all : (N,) array  Calibration predictions.
    reg_cal_pred, asm_cal_bins : (N,) object array  Regime / ASM bin labels.
    """

    def _rmse_arr(o: np.ndarray, s: np.ndarray) -> float:
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
            o      = obs_cal[valid]
            lp     = Q_lstm_cal[valid]
            hp     = Q_xgb_cal_all[valid]
            oracle = np.where(np.abs(o - lp) <= np.abs(o - hp), lp, hp)
            heatmap.loc[asm, reg] = round(
                float(
                    min(_rmse_arr(o, lp), _rmse_arr(o, hp)) - _rmse_arr(o, oracle)
                ),
                4,
            )

    print("\nComplementarity Heatmap — Oracle RMSE gain  (m³/s)")
    print("  (positive = oracle selector beats best single model in that cell)")
    print(heatmap.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# §6  Ensemble KGE Bootstrap Report
# ══════════════════════════════════════════════════════════════════════════════

def print_kge_bootstrap_report(
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
) -> None:
    """Print KGE point estimate and 95 % block-bootstrap CI for S1 and S2.

    Parameters
    ----------
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  LSTM-only predictions.
    Q_system : (N,) array  Full system predictions.
    """
    from block1_data_prep_utilities import block_bootstrap_hydrology

    s1_ci = block_bootstrap_hydrology(obs_t, Q_lstm_t)
    s2_ci = block_bootstrap_hydrology(obs_t, Q_system)

    print("\nKGE Block-Bootstrap Report (7-day blocks, 1 000 resamples):")
    print(f"  S1-LSTM   KGE={kge(obs_t, Q_lstm_t):.3f} "
          f"[{s1_ci[0]:.3f}, {s1_ci[1]:.3f}]")
    print(f"  S2-System KGE={kge(obs_t, Q_system):.3f}  "
          f"[{s2_ci[0]:.3f}, {s2_ci[1]:.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# §7  McNemar Summary Printer
# ══════════════════════════════════════════════════════════════════════════════

def print_mcnemar_summary(
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    Q_stats: dict,
    alpha: float = 0.05,
) -> None:
    """Print the McNemar test result with a plain-language decision statement.

    Parameters
    ----------
    obs_t : (N,) array  Observed streamflow.
    Q_lstm_t : (N,) array  LSTM-only predictions.
    Q_system : (N,) array  Full system predictions.
    Q_stats : dict  Must contain 'Q95'.
    alpha : float  Significance threshold (default 0.05).
    """
    p = mcnemar_fnr_test(obs_t, Q_lstm_t, Q_system, Q_stats["Q95"])
    fnr_s1, _ = fnr_wilson(obs_t, Q_lstm_t, Q_stats["Q95"])
    fnr_s2, _ = fnr_wilson(obs_t, Q_system,  Q_stats["Q95"])
    delta      = fnr_s1 - fnr_s2

    print("\nMcNemar FNR Test (S1 vs S2):")
    print(f"  δ_FNR = {delta:+.3f}  (S1={fnr_s1:.3f}, S2={fnr_s2:.3f})")
    print(f"  p-value = {p:.4e}")

    if p < alpha:
        direction = "reduction" if delta > 0 else "increase"
        print(
            f"  ✓ Statistically significant FNR {direction} at α={alpha} "
            f"(p={p:.4e} < {alpha})."
        )
    else:
        print(
            f"  ✗ No statistically significant FNR difference at α={alpha} "
            f"(p={p:.4e} ≥ {alpha})."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §8  Scaler Statistics Printer
# ══════════════════════════════════════════════════════════════════════════════

def print_scaler_stats(scaler: dict, Q_stats: dict) -> None:
    """Print loaded scaler statistics and climatological baseline.

    Parameters
    ----------
    scaler : dict  Keys 'mean', 'std' from :func:`block2.load_scaler`.
    Q_stats : dict  Training-period quantiles.
    """
    print("\nNeuralHydrology Q Scaler:")
    print(f"  μ_Q   = {scaler['mean']:.3f} m³/s")
    print(f"  σ_Q   = {scaler['std']:.3f} m³/s")
    print(f"  σ_clim (training std) = {Q_stats['Q_std']:.3f} m³/s")
    print(
        "  Note: z-score normalisation is applied during training; "
        "clip_targets_to_zero ensures Q_pred ≥ 0 in all outputs."
    )


# ══════════════════════════════════════════════════════════════════════════════
# §9  ACI Parameter Export (unified schema)
# ══════════════════════════════════════════════════════════════════════════════

def export_aci_params(
    tau_regime: dict,
    GLOBAL_TAU: float,
    TAU_ALE: float,
    achieved_eta_ale: float,
    q_hat_nondeferral: float,
    cov_by_regime: dict,
    is_defer: np.ndarray,
    leg1_stat: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    aci_alpha_state: np.ndarray,
    output_dir: Path,
) -> None:
    """Export the unified ACI parameter JSON (aci_params.json).

    Replaces the fragmented aci_params.json from the original notebook with a
    schema that includes multi-horizon alpha state and Leg 3 (Threshold Safeguard) label.

    Parameters
    ----------
    tau_regime : dict  Per-regime Leg 1 thresholds.
    GLOBAL_TAU : float  Median of regime τ values.
    TAU_ALE : float  Leg 2 aleatoric threshold.
    achieved_eta_ale : float  Implied Leg 2 deferral rate.
    q_hat_nondeferral : float  Non-deferred split conformal quantile (h=1).
    cov_by_regime : dict  ACI h=1 empirical coverage per regime.
    is_defer, leg1_stat, leg2_aleatoric_test, leg3_hydro : bool arrays.
    aci_alpha_state : (H_MAX,) array  Per-horizon α after test run.
    output_dir : Path  Artefact directory.
    """
    payload = {
        "tau_per_regime":              tau_regime,
        "GLOBAL_TAU":                  GLOBAL_TAU,
        "TAU_ALE":                     TAU_ALE,
        "achieved_eta_ale":            achieved_eta_ale,
        "q_hat_nondeferral_h1":        q_hat_nondeferral,
        "coverage_by_regime_h1":       cov_by_regime,
        "COST_RATIO":                  COST_RATIO,
        "NORMALIZED_ESC_COST":         NORMALIZED_ESC_COST,
        "deferral_rate":               float(is_defer.mean()),
        "leg1_epistemic_rate":         float(leg1_stat.mean()),
        "leg2_aleatoric_rate":         float(leg2_aleatoric_test.mean()),
        "leg3_threshold_safeguard_rate":         float(leg3_hydro.mean()),
        "aci_alpha_state_after_test":  aci_alpha_state.tolist(),
    }
    with open(output_dir / "aci_params.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print("aci_params.json saved.")


# ══════════════════════════════════════════════════════════════════════════════
# §10  Non-Deferred Conformal Quantile (h=1, diagnostic)
# ══════════════════════════════════════════════════════════════════════════════

def compute_nondeferral_quantile(
    cal: pd.DataFrame,
    uq_cal: pd.DataFrame,
    leg2_aleatoric_cal: np.ndarray,
    leg3_cal: np.ndarray,
    Q_stats: dict,
    alpha0: float = 0.10,
) -> float:
    """Compute the h=1 split-conformal score quantile on non-deferred cal days.

    This is a sensitivity diagnostic: it quantifies how different the conformal
    quantile is when estimated only on low-risk (non-deferred) calibration days,
    compared to the full calibration pool.

    Parameters
    ----------
    cal : pd.DataFrame  Calibration dataframe.
    uq_cal : pd.DataFrame  Calibration UQ (Q_mean, Q_std, w_t).
    leg2_aleatoric_cal : (N,) bool  Leg 2 activations on calibration set.
    leg3_cal : (N,) bool  Leg 3 activations on calibration set.
    Q_stats : dict  Training-period quantiles.
    alpha0 : float  Nominal miscoverage rate.

    Returns
    -------
    float  Non-deferred split-conformal quantile q̂_nd.
    """
    obs_c  = cal["Q"].values
    Qm_c   = uq_cal["Q_mean"].values
    Qs_c   = uq_cal["Q_std"].values + 1e-6
    valid_c = ~np.isnan(obs_c)

    scores_all = np.abs(obs_c[valid_c] - Qm_c[valid_c]) / Qs_c[valid_c]
    q_hat_all  = float(np.quantile(scores_all, 1 - alpha0))

    nondefer_mask = (~leg2_aleatoric_cal) & (~leg3_cal) & valid_c
    if nondefer_mask.sum() < 20:
        q_hat_nd = q_hat_all
    else:
        scores_nd = np.abs(
            obs_c[nondefer_mask] - Qm_c[nondefer_mask]
        ) / Qs_c[nondefer_mask]
        q_hat_nd = float(np.quantile(scores_nd, 1 - alpha0))

    delta_q = q_hat_nd - q_hat_all
    print(
        f"\nSplit-conformal q̂ (all cal):          {q_hat_all:.3f}"
    )
    print(
        f"Split-conformal q̂ (non-deferred cal): {q_hat_nd:.3f}  "
        f"(n={nondefer_mask.sum()})   Δq̂={delta_q:+.3f}"
    )
    return q_hat_nd
