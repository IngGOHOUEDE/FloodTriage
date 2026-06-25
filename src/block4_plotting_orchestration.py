"""
FloodTriage — Block 4: Plotting, SHAP Interpretability & Orchestration
=======================================================================
Ouémé at Savè, Benin | 1965–2011

Key Components
--------------
1. SHAP TreeExplainer on deferred test days (feature importance, beeswarm,
   waterfall).
2. Coverage / leg-activation bar charts.
3. Rating Curve (RC) sweep plots with S1 as horizontal baseline.
4. Metric comparison bar chart with 95 % confidence intervals.
5. Rating-curve sensitivity plot with fixed axis tick alignment.
6. Temporal reliability / churn audit.
7. Full-record hydrograph with per-leg deferral band colouring.
8. Orchestration entrypoint calling Blocks 1–3 in sequence.

"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json
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

# Assume Blocks 1–3 are in scope (notebook) or imported below
try:
    from block1_data_prep_utilities import (
        ASM_LABELS, BASIN_ID, COST_RATIO, H_MAX, NORMALIZED_ESC_COST,
        OUTPUT_DIR, P30D_LABELS, REGIMES, SEEDS, SPLITS, WARMUP_DAYS,
        Q_stats, assign_regime, df, fnr_wilson, kge, mcnemar_fnr_test,
        moving_block_bootstrap, nse, pbias, rmse, save_json,
    )
    from block3_gateway_evaluation import (
        XGB_FEATURE_NAMES, apply_xgb_correction, build_gateway_mask,
        build_xgb_features, confident_but_wrong, failure_attribution,
        flood_classification_overlap, profile_single_day_latency,
        run_leg_ablation, run_ope_dm, sweep_rc_curves,
        compute_strategy_metrics, rating_curve_sensitivity,
    )
except ImportError:
    pass  # Notebook mode — globals already present.

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# §1  SHAP Interpretability
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(
    test: pd.DataFrame,
    uq_test: pd.DataFrame,
    is_defer: np.ndarray,
    xgb_model: xgb.Booster,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, shap.TreeExplainer, int]:
    """Compute SHAP values for all deferred test days.

    Parameters
    ----------
    test : pd.DataFrame  Test period dataframe.
    uq_test : pd.DataFrame  UQ for test period.
    is_defer : (N,) bool  Gateway deferral mask.
    xgb_model : xgb.Booster  Trained surrogate.
    output_dir : Path  Artefact directory.

    Returns
    -------
    shap_vals : (n_deferred, n_features) array
    X_defer : (n_deferred, n_features) array
    explainer : shap.TreeExplainer
    peak_dq_idx : int  Index of peak XGBoost correction within deferred set.
    """
    defer_idx_test   = np.where(is_defer)[0]
    defer_dates_test = test.index[defer_idx_test]

    X_defer    = build_xgb_features(test.iloc[defer_idx_test], uq_test.iloc[defer_idx_test])
    X_defer_df = pd.DataFrame(X_defer, columns=XGB_FEATURE_NAMES)

    explainer  = shap.TreeExplainer(xgb_model)
    shap_vals  = explainer.shap_values(X_defer_df)

    shap_df = pd.DataFrame(shap_vals, index=defer_dates_test, columns=XGB_FEATURE_NAMES)
    shap_df.to_csv(output_dir / "shap_values.csv")
    print(f"SHAP computed for {len(defer_dates_test)} deferred test days.")

    # Peak index: target 2010-09-15 (highest discharge in test record).
    # Falls back to nearest deferred day if that exact date was not deferred.
    target_date = pd.Timestamp("2010-09-15")
    if target_date in defer_dates_test:
        peak_dq_idx = int(np.where(defer_dates_test == target_date)[0][0])
    else:
        deltas = np.abs((defer_dates_test - target_date).days)
        peak_dq_idx = int(np.argmin(deltas))

    return shap_vals, X_defer, explainer, peak_dq_idx, defer_dates_test


# ══════════════════════════════════════════════════════════════════════════════
# §2  Plotting Functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap(
    shap_vals: np.ndarray,
    X_defer: np.ndarray,
    explainer: shap.TreeExplainer,
    peak_dq_idx: int,
    defer_dates_test: pd.DatetimeIndex,
    dq_test: np.ndarray,
    is_defer: np.ndarray,
    output_dir: Path,
    xgb_model: "xgb.Booster" = None,
) -> None:
    """Plot SHAP feature importance, beeswarm, and waterfall."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Plot 1: Mean |SHAP| bar chart
    ax = axes[0]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)
    ax.barh([XGB_FEATURE_NAMES[i] for i in order], mean_abs[order],
            color="steelblue", alpha=0.85)
    ax.set_xlabel("Mean |SHAP| on ΔQ prediction  (m³/s)")
    ax.set_title("XGBoost SHAP importance\n(all deferred test days)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Beeswarm (top 8 features)
    ax2   = axes[1]
    top_k = 8
    top_feats = np.argsort(mean_abs)[::-1][:top_k]
    rng_bee = np.random.default_rng(42)

    for j, fi in enumerate(top_feats[::-1]):
        feat_col = shap_vals[:, fi]
        feat_raw = X_defer[:, fi]
        val_range = feat_raw.max() - feat_raw.min()
        normed = (feat_raw - feat_raw.min()) / (val_range if val_range > 0 else 1e-8)
        jitter = rng_bee.uniform(-0.25, 0.25, len(feat_col))
        sc = ax2.scatter(feat_col, np.full_like(feat_col, j) + jitter,
                         c=normed, cmap="coolwarm", s=8, alpha=0.5,
                         vmin=0, vmax=1)

    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels([XGB_FEATURE_NAMES[fi] for fi in top_feats[::-1]], fontsize=8)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("SHAP value  (m³/s)")
    ax2.set_title("SHAP beeswarm (top 8)\nBlue=low value, Red=high value")
    plt.colorbar(sc, ax=ax2, label="Feature value (normed)", shrink=0.7)

    # Plot 3: Waterfall for peak event
    ax3       = axes[2]
    peak_sv   = shap_vals[peak_dq_idx]
    peak_date = str(defer_dates_test[peak_dq_idx].date())
    peak_base = float(np.asarray(explainer.expected_value).item())
    ord3      = np.argsort(np.abs(peak_sv))[::-1][:8]
    vals3     = peak_sv[ord3]
    colors3   = ["crimson" if v > 0 else "steelblue" for v in vals3]

    ax3.barh([XGB_FEATURE_NAMES[i] for i in ord3[::-1]], vals3[::-1],
             color=colors3[::-1], alpha=0.85)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_xlabel("SHAP contribution to ΔQ  (m³/s)")
    ax3.set_title(f"Waterfall — peak deferred event\n{peak_date}")
    ax3.grid(True, alpha=0.3)

    dq_deferred = xgb_model.predict(
        xgb.DMatrix(
            pd.DataFrame(X_defer, columns=XGB_FEATURE_NAMES),
            feature_names=XGB_FEATURE_NAMES,
        )
    ) if xgb_model is not None else np.zeros(len(X_defer))
    ax3.text(
        0.98, 0.02,
        f"Base ΔQ = {peak_base:.1f} m³/s\n"
        f"SHAP sum = {peak_sv.sum():.1f} m³/s\n"
        f"XGB ΔQ = {dq_deferred[peak_dq_idx]:.1f} m³/s",
        transform=ax3.transAxes, fontsize=8,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig_shap_xgb.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_coverage_legs(
    cov_by_regime: dict,
    leg1_stat: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    is_defer: np.ndarray,
    alpha0: float,
    output_dir: Path,
) -> None:
    """Plot ACI coverage by regime and per-leg deferral activation rates.

    Leg 3 is labelled "Leg 3 (Threshold Safeguard)" throughout.

    Parameters
    ----------
    cov_by_regime : dict  {regime: empirical coverage fraction}.
    leg1/2/3 : arrays  Per-leg activation flags.
    is_defer : bool array  Total gateway deferral mask.
    alpha0 : float  Nominal miscoverage rate (for target coverage line).
    output_dir : Path  Artefact directory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    colors = ["steelblue", "orange", "red"]
    for i, r in enumerate(REGIMES):
        v = cov_by_regime.get(r, np.nan)
        ax.bar(i, 0.0 if np.isnan(v) else v, color=colors[i], alpha=0.85, label=r)
    ax.axhline(1 - alpha0, color="black", ls="--", lw=1.5,
               label=f"Target {1 - alpha0:.0%}")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Normal", "Flood"])
    ax.set_ylabel("Empirical ACI coverage (h=1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Horizon ACI Coverage by Regime (test, h=1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    legs   = ["Leg 1\n(Epistemic)", "Leg 2\n(Aleatoric)", "Leg 3\n(Threshold Safeguard)"]
    rates  = [leg1_stat.mean(), leg2_aleatoric_test.mean(), leg3_hydro.mean()]
    colors2 = ["steelblue", "forestgreen", "crimson"]
    ax2.bar(legs, rates, color=colors2, alpha=0.85)
    ax2.axhline(is_defer.mean(), color="black", ls="--", lw=1.5,
                label=f"Total deferred {is_defer.mean():.2f}")
    ax2.set_ylabel("Deferral rate")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Tri-Fold Gateway — Leg Activation (test)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_coverage_legs.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_rc_and_metrics(
    rc_df: pd.DataFrame,
    s1_metrics: dict,
    s2_metrics: dict,
    output_dir: Path,
) -> None:
    """Plot Risk-Coverage curves and metric comparison grids.

    Parameters
    ----------
    rc_df : pd.DataFrame  RC curve records from Block 3.
    s1_metrics, s2_metrics : dict  Strategy metric dictionaries.
    output_dir : Path  Artefact directory.
    """
    # ── Figure 1: Risk-Coverage curve (Single plot layout) ────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    style = {
        "S2-Oracle": ("-",  "green",     2.0),
        "S2-XGB":    ("--", "red",       2.0),
        "S1-LSTM":   (":",  "steelblue", 1.5),
    }

    for strat, (ls, col, lw) in style.items():
        sub = rc_df[rc_df["strategy"] == strat].sort_values("coverage")
        if strat == "S1-LSTM":
            # Plot as horizontal baseline (constant FNR)  
            ax.axhline(sub["risk"].mean(), ls=ls, color=col, lw=lw,
                       label=f"{strat} (baseline, never defers)")
        else:
            ax.plot(sub["coverage"], sub["risk"], ls=ls, color=col, lw=lw,
                    label=strat)

    ax.scatter(
        [1 - s2_metrics["eta"]], [s2_metrics["FNR"]],
        color="red", s=100, edgecolors="black", zorder=5,
        label="S2 Operating Point",
    )
    
    # Increased font sizes for readability in publications
    ax.set_xlabel("Coverage (Autonomy rate)", fontsize=13)
    ax.set_ylabel("Flood Miss Rate (FNR)", fontsize=13)
    ax.set_title("Selective Prediction Performance (RC Curve)", fontsize=15)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_rc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Figure 2: 2x2 Metric Subplots (Horizontal Lollipop Style) ─────────────
    # Initialize the 2x2 grid without sharex, as metrics have different scales
    fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(11, 7))
    
    # Flatten the 2D array of axes to easily iterate over them
    axes2_flat = axes2.flatten()

    # Helper to get CI bounds for error bars
    def _ci(m_dict, key):
        if key == "FNR":
            return m_dict["FNR_CI"]
        return m_dict["CIs"].get(key, (m_dict[key], m_dict[key]))

    metric_cfg = [
        ("FNR",   "FNR (Lower is Better)",   False),
        ("KGE",   "KGE (Higher is Better)",  False),
        ("NSE",   "NSE (Higher is Better)",  False),
        ("PBIAS", "PBIAS % (Closer to 0)",   True),
    ]

    models = [
        ("S1", "steelblue", "S1-LSTM"),
        ("S2", "red",       "S2-System"),
    ]

    for ax_i, (key, xlabel, add_zero) in zip(axes2_flat, metric_cfg):
        # Reverse the models list so S1 sits above S2 visually on the Y-axis
        for y_idx, (strat, color, label) in enumerate(reversed(models)):
            m_dict = s1_metrics if strat == "S1" else s2_metrics
            val    = m_dict[key]
            lo, hi = _ci(m_dict, key)
            
            # Draw the horizontal whisker (Confidence Interval)
            ax_i.hlines(y_idx, lo, hi, color=color, lw=2.0)
            
            # Draw the dot (Operating Point)
            ax_i.plot(val, y_idx, marker="o", markersize=8, color=color, 
                      label=f"{label} = {val:.3f}")
            
        if add_zero:
            ax_i.axvline(0, color="black", lw=0.8, ls="--", label="Zero bias")
            
        # Configure Y-axis with categorical model names with increased fonts
        ax_i.set_yticks([0, 1])
        ax_i.set_yticklabels([m[2] for m in reversed(models)], fontsize=13)
        ax_i.set_xlabel(xlabel, fontsize=13)
        ax_i.tick_params(axis='x', labelsize=11)
        
        # Placed legend perfectly at the center of the subplot
        ax_i.legend(fontsize=10, loc="center")
        ax_i.grid(True, axis='x', alpha=0.3)

    # Use suptitle to span the title centrally over both subplots
    fig2.suptitle(
        "Metric Comparison w/ 95% CIs (Test 2008–2011)\n"
        "Error bars show bootstrap / Wilson confidence intervals",
        fontsize=15,
        ha="center"
    )
    
    # Adjust tight_layout rect to ensure the suptitle isn't cut off or overlapping
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(output_dir / "fig_rc_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_rating_curve_sensitivity(
    rc_errors: list,
    s1_fnr_series: list,
    s2_fnr_series: list,
    output_dir: Path,
) -> None:
    """Plot FNR vs. rating curve systematic error.

    Parameters
    ----------
    rc_errors : list[float]  Rating curve error factors (e.g. [-0.2, -0.1, …]).
    s1_fnr_series, s2_fnr_series : list[float]  FNR at each error level.
    output_dir : Path  Artefact directory.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rc_errors, s1_fnr_series, marker="o", ls="--",
            color="steelblue", label="S1: Autonomous LSTM")
    ax.plot(rc_errors, s2_fnr_series, marker="s", ls="-",
            color="crimson", label="S2: Tri-Fold System")
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Rating Curve Systematic Error")
    ax.set_ylabel("False Negative Rate (FNR)")
    ax.set_title("Systemic Resilience vs. Rating Curve Shifts") 
    ax.set_xticks(rc_errors)
    ax.set_xticklabels([f"{x:+.0%}" for x in rc_errors])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_rating_curve_sensitivity.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_confident_wrong(
    test: pd.DataFrame,
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    wt_t: np.ndarray,
    uale_t: np.ndarray,
    is_defer: np.ndarray,
    GLOBAL_TAU: float,
    P30D_Q67: float,
    Q_stats: dict,
    output_dir: Path,
) -> None:
    """Plot epistemic signal and antecedent precipitation on confident-wrong days."""
    q95 = Q_stats["Q95"]
    mask = (
        (obs_t >= q95)
        & ~is_defer
        & (Q_lstm_t < q95)
        & ~np.isnan(obs_t)
    )
    n_cw = int(mask.sum())
    if n_cw == 0:
        print("No confident-wrong days — plot skipped.")
        return

    cw_idx  = np.where(mask)[0]
    cw_wt   = wt_t[cw_idx]
    cw_p30  = test["P30d"].values[cw_idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1 = axes[0]
    ax1.hist(cw_wt, bins=10, color="steelblue", alpha=0.8, edgecolor="white")
    ax1.axvline(GLOBAL_TAU, color="red", ls="--", lw=1.5,
                label=f"GLOBAL_TAU={GLOBAL_TAU:.3f}")
    ax1.set_xlabel("w_t (epistemic uncertainty) on missed flood days")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Epistemic signal on {n_cw} confident-wrong days")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(cw_p30, bins=10, color="forestgreen", alpha=0.8, edgecolor="white")
    ax2.axvline(P30D_Q67, color="red", ls="--", lw=1.5,
                label=f"P30D_Q67={P30D_Q67:.0f} mm (wet threshold)")
    ax2.set_xlabel("P30d (mm) on missed flood days")
    ax2.set_ylabel("Count")
    ax2.set_title("Antecedent precipitation on confident-wrong days")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_confident_wrong.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_full_hydrograph(
    df: pd.DataFrame,
    test: pd.DataFrame,
    uq_test: pd.DataFrame,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    Q_stats: dict,
    SPLITS: dict,
    output_dir: Path,
) -> None:
    """Plot the full 1965–2011 hydrograph with per-leg deferral shading.

    Parameters
    ----------
    All parameters follow notebook global naming conventions.
    output_dir : Path  Artefact directory.
    """
    n_test = len(test.index)
    Qm_t   = uq_test["Q_mean"].values
    Qs_t   = uq_test["Q_std"].values + 1e-6

    LABEL_SIZE = 12
    TICK_SIZE  = 11
    TITLE_SIZE = 13

    fig, ax1 = plt.subplots(figsize=(13, 5))

    prec = df["P"].values
    ax1.bar(df.index, prec, color="black", alpha=0.8, width=1,
            label="Precipitation (P)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")
    ax1.set_ylabel("P (mm/d)", color="black",
                   fontsize=LABEL_SIZE, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="black", labelsize=TICK_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_SIZE)
    ax1.set_ylim(0, max(100, float(np.nanmax(prec))) * 2.0)
    ax1.invert_yaxis()
    ax1.set_title(
        "Ouémé at Savè — Full record 1965–2011 | "
        "LSTM Deep Ensemble × XGBoost Surrogate × Human",
        fontsize=TITLE_SIZE, fontweight="bold",
    )

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["Q"], color="red", lw=1.5, label="Observed Q", zorder=3)
    ax2.plot(df.index, df["Q_lstm"], color="blue", lw=1.2, ls="--",
             alpha=0.6, label="LSTM ensemble mean")
    ax2.fill_between(
        test.index,
        np.clip(Qm_t - 1.96 * Qs_t, 0, None),
        Qm_t + 1.96 * Qs_t,
        color="blue", alpha=0.12, label="95% ensemble band (test)",
    )
    ax2.plot(test.index, Q_system, color="darkgreen", lw=1.8,
             label="S2 system (LSTM+XGB+Human)", zorder=4)

    # Per-leg deferral bands — Leg 3 labelled "Threshold Safeguard"  
    for t_i in range(n_test):
        if not is_defer[t_i]:
            continue
        if leg3_hydro[t_i]:
            col = "red"          # Leg 3 (Threshold Safeguard)
        elif leg2_aleatoric_test[t_i]:
            col = "forestgreen"  # Leg 2 (Aleatoric)
        else:
            col = "orange"       # Leg 1 (Epistemic)

        ax2.axvspan(
            test.index[t_i] - pd.Timedelta("0.5D"),
            test.index[t_i] + pd.Timedelta("0.5D"),
            alpha=0.15, color=col, lw=0,
        )

    ax2.axhline(Q_stats["Q95"], color="orange", ls=":", lw=1.5,
                label=f"Q95={Q_stats['Q95']:.0f} m³/s")
    ax2.axvspan(pd.Timestamp(SPLITS["cal"][0]),
                pd.Timestamp(SPLITS["cal"][1]),
                alpha=0.10, color="gray", label="Calibration Split")
    ax2.axvspan(pd.Timestamp(SPLITS["test"][0]),
                pd.Timestamp(SPLITS["test"][1]),
                alpha=0.08, color="salmon", label="Test Split")

    ax2.set_ylabel("Q (m³/s)", color="blue",
                   fontsize=LABEL_SIZE, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="blue", labelsize=TICK_SIZE)

    ax2.set_ylim(0, max(2500, float(np.nanmax(df["Q"].values))) * 1.3)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    custom_patches = [
        Patch(color="red",         alpha=0.3, label="Deferral (Leg 3 (Threshold Safeguard))"),
        Patch(color="forestgreen", alpha=0.3, label="Deferral (Leg 2 (Aleatoric))"),
        Patch(color="orange",      alpha=0.3, label="Deferral (Leg 1 (Epistemic))"),
    ]
    all_handles = lines1 + lines2 + custom_patches
    all_labels  = labs1 + labs2 + [p.get_label() for p in custom_patches]
    ax2.legend(all_handles, all_labels, loc="upper left", fontsize=9,
               framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_hydrograph_v5.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def plot_test_hydrograph(
    test: pd.DataFrame,
    uq_test: pd.DataFrame,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    Q_stats: dict,
    output_dir: Path,
) -> None:
    """Plot hydrograph for the test period only (2008-2011).

    Calibration Split and Test Split shading bands are omitted because the
    entire visible window is the test period. Deferral band colouring and
    legend entries follow the same conventions as the full hydrograph.

    Parameters
    ----------
    test : pd.DataFrame  Test period dataframe (2008-01-01 to 2011-12-31).
    uq_test : pd.DataFrame  Corresponding UQ rows.
    Q_system : (N,) array  Full system (S2) predictions.
    is_defer : (N,) bool  Gateway deferral mask.
    leg2_aleatoric_test : (N,) bool  Leg 2 activation flags.
    leg3_hydro : (N,) bool  Leg 3 (Threshold Safeguard) activation flags.
    Q_stats : dict  Training-period quantiles.
    output_dir : Path  Artefact directory.
    """
    n_test     = len(test.index)
    Qm_t       = uq_test["Q_mean"].values
    Qs_t       = uq_test["Q_std"].values + 1e-6
    Q_lstm_t   = test["Q_lstm"].values
    obs_t      = test["Q"].values

    LABEL_SIZE = 12
    TICK_SIZE  = 11
    TITLE_SIZE = 13

    fig, ax1 = plt.subplots(figsize=(13, 5))

    prec = test["P"].values
    ax1.bar(test.index, prec, color="black", alpha=0.8, width=1,
            label="Precipitation (P)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax1.set_xlabel("Date", fontsize=LABEL_SIZE, fontweight="bold")
    ax1.set_ylabel("P (mm/d)", color="black",
                   fontsize=LABEL_SIZE, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="black", labelsize=TICK_SIZE)
    ax1.tick_params(axis="x", labelsize=TICK_SIZE)
    ax1.set_ylim(0, max(100, float(np.nanmax(prec))) * 2.0)
    ax1.invert_yaxis()
    ax1.set_title(
        "Ouémé at Savè — Test Period 2008–2011 | "
        "LSTM Deep Ensemble × XGBoost Surrogate × Human",
        fontsize=TITLE_SIZE, fontweight="bold",
    )

    ax2 = ax1.twinx()
    ax2.plot(test.index, obs_t, color="red", lw=1.5, label="Observed Q", zorder=3)
    ax2.plot(test.index, Q_lstm_t, color="blue", lw=1.2, ls="--",
             alpha=0.6, label="LSTM ensemble mean")
    ax2.fill_between(
        test.index,
        np.clip(Qm_t - 1.96 * Qs_t, 0, None),
        Qm_t + 1.96 * Qs_t,
        color="blue", alpha=0.12, label="95% ensemble band",
    )
    ax2.plot(test.index, Q_system, color="darkgreen", lw=1.8,
             label="S2 system (LSTM+XGB+Human)", zorder=4)

    # Per-leg deferral shading — no Calibration/Test split bands
    for t_i in range(n_test):
        if not is_defer[t_i]:
            continue
        if leg3_hydro[t_i]:
            col = "red"
        elif leg2_aleatoric_test[t_i]:
            col = "forestgreen"
        else:
            col = "orange"
        ax2.axvspan(
            test.index[t_i] - pd.Timedelta("0.5D"),
            test.index[t_i] + pd.Timedelta("0.5D"),
            alpha=0.15, color=col, lw=0,
        )

    ax2.axhline(Q_stats["Q95"], color="orange", ls=":", lw=1.5,
                label=f"Q95={Q_stats['Q95']:.0f} m³/s")

    ax2.set_ylabel("Q (m³/s)", color="blue",
                   fontsize=LABEL_SIZE, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="blue", labelsize=TICK_SIZE)
    ax2.set_ylim(0, max(2500, float(np.nanmax(obs_t[~np.isnan(obs_t)]))) * 1.3)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    custom_patches = [
        Patch(color="red",         alpha=0.3, label="Deferral (Leg 3 (Threshold Safeguard))"),
        Patch(color="forestgreen", alpha=0.3, label="Deferral (Leg 2 (Aleatoric))"),
        Patch(color="orange",      alpha=0.3, label="Deferral (Leg 1 (Epistemic))"),
    ]
    all_handles = lines1 + lines2 + custom_patches
    all_labels  = labs1 + labs2 + [p.get_label() for p in custom_patches]
    ax2.legend(all_handles, all_labels, loc="upper left", fontsize=9,
               framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_hydrograph_test_only.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# §3  Temporal Reliability & Churn Audit
# ══════════════════════════════════════════════════════════════════════════════

def temporal_audit(
    n_test: int,
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    leg1_stat: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    Q_stats: dict,
    test: pd.DataFrame,
) -> None:
    """Compute regime-stratified FNR, churn reduction, and event persistence.

    Parameters
    ----------
    n_test : int  Length of the test period.
    All other parameters follow notebook global conventions.
    """
    q95          = Q_stats["Q95"]
    reg_test_obs = assign_regime(np.nan_to_num(obs_t, nan=-1), Q_stats)

    # Regime-stratified FNR (S2)
    fnr_by_regime: dict = {}
    for r in REGIMES:
        idx   = (reg_test_obs == r) & ~np.isnan(obs_t)
        flood = obs_t[idx] >= q95
        if flood.sum() == 0:
            fnr_by_regime[r] = np.nan
            continue
        missed            = flood & (Q_system[idx] < q95)
        fnr_by_regime[r]  = float(missed.sum() / flood.sum())
    print(f"\nFNR by regime (S2, test): {fnr_by_regime}")

    # Gateway churn reduction
    policy_arr  = is_defer.astype(int)
    raw_any_leg = (leg1_stat | leg2_aleatoric_test | leg3_hydro).astype(int)
    kappa_p     = int(np.sum(np.diff(policy_arr) != 0))
    kappa_i     = int(np.sum(np.diff(raw_any_leg) != 0))
    churn_red   = 100.0 * (kappa_i - kappa_p) / max(kappa_i, 1)

    # Count isolated event-persistence activations
    dQ_lstm = np.append([0], np.diff(Q_lstm_t))
    event_persist_fires = sum(
        1 for t in range(1, n_test)
        if not (leg1_stat[t] or leg2_aleatoric_test[t] or leg3_hydro[t])
        and is_defer[t - 1]
        and dQ_lstm[t] > 0
        and is_defer[t]
    )

    print(f"  κ_smoothed={kappa_p}  κ_raw={kappa_i}  "
          f"churn_reduction={churn_red:.1f}%")
    print(f"  Event-persistence triggers: {event_persist_fires}")


# ══════════════════════════════════════════════════════════════════════════════
# §4  Operator Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def print_operator_dashboard(
    test: pd.DataFrame,
    obs_t: np.ndarray,
    Q_lstm_t: np.ndarray,
    Q_system: np.ndarray,
    is_defer: np.ndarray,
    wt_t: np.ndarray,
    taus_test: np.ndarray,
    uale_t: np.ndarray,
    leg2_aleatoric_test: np.ndarray,
    leg3_hydro: np.ndarray,
    uq_test: pd.DataFrame,
    shap_vals: np.ndarray,
    X_defer: np.ndarray,
    defer_dates_test: pd.DatetimeIndex,
    xgb_model: xgb.Booster,
    peak_dq_idx: int,
    Q_stats: dict,
    TAU_ALE: float,
) -> None:
    """Print a structured human-operator briefing for the peak deferred event.

    Leg 3 is labelled "Threshold Safeguard".
    """
    defer_idx_test = np.where(is_defer)[0]
    peak_t         = defer_idx_test[peak_dq_idx]
    peak_date      = str(defer_dates_test[peak_dq_idx].date())

    Qs_t      = uq_test["Q_std"].values + 1e-6
    ord3      = np.argsort(
        np.abs(shap_vals[peak_dq_idx])
    )[::-1][:5]

    dq_deferred = xgb_model.predict(
        xgb.DMatrix(
            pd.DataFrame(X_defer, columns=XGB_FEATURE_NAMES),
            feature_names=XGB_FEATURE_NAMES,
        )
    )

    q95 = Q_stats["Q95"]
    print("\n" + "=" * 65)
    print("HUMAN OPERATOR DASHBOARD — Peak Deferred Event")
    print("=" * 65)
    print(f"  Date:               {peak_date}")
    print(f"  Routing trigger:")
    print(f"    Leg 1 (Epistemic):    {bool(wt_t[peak_t] > taus_test[peak_t])}  "
          f"w_t={wt_t[peak_t]:.3f} vs τ={taus_test[peak_t]:.3f}")
    print(f"    Leg 2 (Aleatoric):    {bool(leg2_aleatoric_test[peak_t])}  "
          f"uale={uale_t[peak_t]:.1f} m³/s² vs TAU_ALE={TAU_ALE:.1f}")
    print(f"    Leg 3 (Threshold Safeguard):    {bool(leg3_hydro[peak_t])}  "
          f"Q_lstm={Q_lstm_t[peak_t]:.1f} vs Q95={q95:.1f} m³/s")
    print(f"  LSTM forecast:        {Q_lstm_t[peak_t]:.1f} m³/s")
    print(f"  Ensemble spread:      {Qs_t[peak_t]:.1f} m³/s  "
          f"(w_t={wt_t[peak_t]:.2f})")
    print(f"  XGB correction:       {dq_deferred[peak_dq_idx]:+.1f} m³/s")
    print(f"  System forecast:      {Q_system[peak_t]:.1f} m³/s")
    print(f"  Q95 threshold:        {q95:.1f} m³/s")
    print(f"  {'⚠ FLOOD WARNING' if Q_system[peak_t] >= q95 else '  Normal flow'}")
    print(f"\n  XGBoost rationale (SHAP, top 5 drivers):")
    for i in ord3:
        fname = XGB_FEATURE_NAMES[i]
        fval  = float(X_defer[peak_dq_idx, i])
        sval  = float(shap_vals[peak_dq_idx, i])
        arrow = "↑" if sval > 0 else "↓"
        print(f"    {fname:15s} = {fval:8.2f}   {arrow} {abs(sval):.2f} m³/s")
    print(f"\n  [HUMAN ACTION: Accept / Modify / Reject]")
    print(f"  (Simulation: deterministically ACCEPT)")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# §5  Metric Serialisation
# ══════════════════════════════════════════════════════════════════════════════

def serialise_strategy_metrics(
    s1_metrics: dict,
    s2_metrics: dict,
    output_dir: Path,
) -> None:
    """Flatten and save strategy metric dictionaries to CSV."""
    export_rows = []
    for m in [s1_metrics, s2_metrics]:
        flat = {k: v for k, v in m.items() if k not in ["CIs", "FNR_CI"]}
        flat["FNR_low"], flat["FNR_high"] = m["FNR_CI"]
        for metric in ["NSE", "KGE", "RMSE", "PBIAS"]:
            flat[f"{metric}_low"], flat[f"{metric}_high"] = m["CIs"].get(
                metric, (np.nan, np.nan)
            )
        export_rows.append(flat)
    pd.DataFrame(export_rows).to_csv(output_dir / "strategy_metrics.csv", index=False)
    print("Strategy metrics saved.")


# ══════════════════════════════════════════════════════════════════════════════
# §6  Full Orchestration Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline() -> None:
    """Execute the complete FloodTriage pipeline (Blocks 1 → 4).

    Call this function at the bottom of a Google Colab notebook or from a
    terminal after ensuring all dependencies are installed.

    The single authoritative length variable for the test period is ``n_test``.
    """
    import subprocess
    import sys

    # ── 0. Install dependencies ───────────────────────────────────────────────
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "gdown", "neuralhydrology", "shap", "xgboost",
        "scikit-learn", "statsmodels",
    ])

    # ── 1. Data preparation ───────────────────────────────────────────────────
    from block1_data_prep_utilities import (
        ASM_LABELS, BASIN_ID, COST_RATIO, DEVICE, DRIVE, H_MAX,
        NH_DATA_DIR, NH_RUN_DIR, NORMALIZED_ESC_COST, OUTPUT_DIR,
        P30D_LABELS, REGIMES, SEEDS, SPLITS, WARMUP_DAYS,
        assign_regime, fnr_wilson, kge, load_json, mcnemar_fnr_test,
        moving_block_bootstrap, nse, pbias, rmse, save_json,
    )
    from block1_data_prep_utilities import load_raw_data, engineer_features

    df = load_raw_data(DRIVE)
    df, Q_stats = engineer_features(df)
    save_json(Q_stats, OUTPUT_DIR / "quantiles.json")
    save_json(SPLITS, OUTPUT_DIR / "splits.json")
    df.to_csv(OUTPUT_DIR / "full.csv")

    # ── 2. Ensemble training & multi-horizon ACI ──────────────────────────────
    from block2_multiHorizon_conformal import (
        MultiHorizonACI, build_uq, load_scaler,
        train_ensemble, write_netcdf,
    )

    write_netcdf(df, NH_DATA_DIR, BASIN_ID)
    ensemble_preds, run_dirs = train_ensemble(
        df, NH_DATA_DIR, NH_RUN_DIR, OUTPUT_DIR, SEEDS
    )

    # Build per-horizon mu / sigma matrices
    def _build_matrix(field: str) -> np.ndarray:
        cols = []
        for h in range(1, H_MAX + 1):
            uq_h = build_uq(ensemble_preds, df, Q_stats, h=h)
            cols.append(uq_h[field].values)
        return np.column_stack(cols)

    mu_matrix    = _build_matrix("Q_mean")
    sigma_matrix = _build_matrix("Q_std")

    # Day-ahead UQ for gateway
    uq = build_uq(ensemble_preds, df, Q_stats, h=1)
    df["Q_lstm"] = uq["Q_mean"].values
    uq.to_csv(OUTPUT_DIR / "uq.csv")
    df.to_csv(OUTPUT_DIR / "full.csv")

    scaler = load_scaler(run_dirs[42], target_var="Q")
    save_json(
        {"mu_Q": scaler["mean"], "std_Q": scaler["std"],
         "sigma_clim": float(Q_stats["Q_std"])},
        OUTPUT_DIR / "lstm_scaler.json",
    )

    # ACI calibration
    aci = MultiHorizonACI(alpha0=0.10, gamma=0.005, h_max=H_MAX)
    cal_mask = (
        (df.index >= SPLITS["cal"][0]) & (df.index <= SPLITS["cal"][1])
    )
    cal = df[cal_mask].copy()
    uq_cal = uq[cal_mask]
    cal_obs = cal["Q"].values
    cal_mu  = mu_matrix[cal_mask]
    cal_sig = sigma_matrix[cal_mask]
    for h in range(1, H_MAX + 1):
        aci.calibrate_on_window(cal_obs, cal_mu[:, h - 1], cal_sig[:, h - 1], h=h)

    # ACI test period
    test_mask = (
        (df.index >= SPLITS["test"][0]) & (df.index <= SPLITS["test"][1])
    )
    test       = df[test_mask].copy()
    uq_test    = uq[test_mask]
    test_obs   = test["Q"].values
    test_mu    = mu_matrix[test_mask]
    test_sig   = sigma_matrix[test_mask]
    test_dates = test.index

    aci_results = aci.run_online(test_dates, test_obs, test_mu, test_sig)
    aci_results.to_csv(OUTPUT_DIR / "aci_intervals.csv", index=False)

    h1_rows   = aci_results[aci_results["h"] == 1].set_index("date")
    Q_lo_test = h1_rows["Q_lo"].reindex(test_dates).values
    Q_hi_test = h1_rows["Q_hi"].reindex(test_dates).values
    alpha_seq  = h1_rows["alpha_t"].reindex(test_dates).values

    # ── 3. XGBoost, gateway, evaluation ──────────────────────────────────────
    from block3_gateway_evaluation import (
        XGB_FEATURE_NAMES, apply_xgb_correction, block_permutation_complementarity,
        build_aleatoric_table, build_gateway_mask, build_xgb_features,
        calibrate_leg1_thresholds, compute_reliability_stats,
        compute_strategy_metrics, confident_but_wrong, failure_attribution,
        flood_classification_overlap, loyo_tau_ale as _loyo_ale,
        loyo_tau_regime as _loyo_leg1, p30d_to_bin, asm_to_bin,
        profile_single_day_latency, rating_curve_sensitivity,
        run_leg_ablation, run_ope_dm, sweep_rc_curves, train_xgb,
        _compute_tercile_thresholds,
    )

    # Compute thresholds
    tr_mask = (df.index >= SPLITS["train"][0]) & (df.index <= SPLITS["train"][1])
    p30d_thresholds = _compute_tercile_thresholds(df["P30d"], tr_mask, WARMUP_DAYS)
    asm_thresholds  = _compute_tercile_thresholds(df["ASM"],  tr_mask, WARMUP_DAYS)
    P30D_Q33, P30D_Q67 = p30d_thresholds
    ASM_Q33,  ASM_Q67  = asm_thresholds

    obs_cal    = cal["Q"].values
    Q_lstm_cal = cal["Q_lstm"].values

    # Aleatoric table
    ALEATORIC_TABLE, TAU_ALE, achieved_eta_ale, global_var_cal = build_aleatoric_table(
        cal, uq_cal, Q_stats, p30d_thresholds
    )
    loyo_ale_df = _loyo_ale(cal, Q_stats, p30d_thresholds)
    loyo_ale_df.to_csv(OUTPUT_DIR / "loyo_tau_ale.csv", index=False)

    # Reliability stats (raw Spearman)
    compute_reliability_stats(obs_cal, Q_lstm_cal, uq_cal, Q_stats, OUTPUT_DIR)

    # Leg 1 thresholds
    tau_regime, GLOBAL_TAU = calibrate_leg1_thresholds(uq_cal, Q_stats)
    loyo_leg1_df = _loyo_leg1(cal, uq_cal, Q_stats)
    loyo_leg1_df.to_csv(OUTPUT_DIR / "loyo_tau_regime.csv", index=False)

    # XGBoost surrogate
    xgb_model = train_xgb(df, uq, Q_stats, OUTPUT_DIR)

    # Calibration XGBoost predictions
    X_cal_raw     = build_xgb_features(cal, uq_cal)
    Q_xgb_cal_all = np.clip(
        Q_lstm_cal + xgb_model.predict(
            xgb.DMatrix(X_cal_raw, feature_names=XGB_FEATURE_NAMES)
        ),
        0, None,
    )

    # Block permutation test
    reg_cal_pred = assign_regime(Q_lstm_cal, Q_stats)
    asm_cal_bins = asm_to_bin(cal["ASM"].values, asm_thresholds)
    block_permutation_complementarity(
        cal, obs_cal, Q_lstm_cal, Q_xgb_cal_all,
        reg_cal_pred, asm_cal_bins, Q_stats,
    )

    # Test period gateway inputs
    obs_t      = test["Q"].values
    Q_lstm_t   = test["Q_lstm"].values.copy()
    Qm_t       = uq_test["Q_mean"].values
    wt_t       = uq_test["w_t"].values
    n_test     = len(test.index)   # single authoritative length variable  

    reg_pred_test = assign_regime(Qm_t, Q_stats)
    ale_bin_test  = p30d_to_bin(test["P30d"].values, p30d_thresholds)
    _fallback     = float(np.nanmean(list(ALEATORIC_TABLE.values())))
    uale_t = np.array([
        ALEATORIC_TABLE.get((reg_pred_test[t], ale_bin_test[t]), _fallback)
        for t in range(n_test)
    ])

    # Leg activations — Leg 2 uses ≥ 
    leg2_aleatoric_test = uale_t >= TAU_ALE
    leg3_hydro          = Qm_t >= Q_stats["Q95"]

    taus_test = np.array([tau_regime.get(r, GLOBAL_TAU) for r in reg_pred_test])

    # Gateway
    is_defer = build_gateway_mask(
        n_test, wt_t, taus_test, leg2_aleatoric_test, leg3_hydro, Q_lstm_t
    )
    leg1_stat = wt_t > taus_test

    # XGBoost correction
    Q_system, dq_test = apply_xgb_correction(test, uq_test, is_defer, xgb_model)

    # Ablation
    run_leg_ablation(
        n_test, obs_t, test, uq_test, wt_t, taus_test,
        leg2_aleatoric_test, leg3_hydro, Q_lstm_t, xgb_model, Q_stats,
    )

    # Strategy metrics
    s1_metrics, s2_metrics, pval_mcnemar = compute_strategy_metrics(
        obs_t, Q_lstm_t, Q_system, is_defer, Q_stats
    )

    # ACI coverage by regime for h=1
    reg_test_obs = assign_regime(np.nan_to_num(obs_t, nan=-1), Q_stats)
    cov_by_regime: dict = {}
    for r in REGIMES:
        idx = (reg_test_obs == r) & ~np.isnan(obs_t)
        if not idx.any():
            cov_by_regime[r] = np.nan
            continue
        cov_by_regime[r] = float(
            np.nanmean(
                (Q_lo_test[idx] <= obs_t[idx]) & (obs_t[idx] <= Q_hi_test[idx])
            )
        )

    # OPE (Direct Method)
    ope_result = run_ope_dm(obs_t, Q_lstm_t, Q_system, is_defer, Q_stats)

    # Rating curve sensitivity
    s1_fnr_series, s2_fnr_series, rc_errors = rating_curve_sensitivity(
        obs_t, Q_lstm_t, Q_system, Q_stats, OUTPUT_DIR
    )

    # RC sweep
    rc_df = sweep_rc_curves(
        n_test, obs_t, Q_lstm_t, wt_t, taus_test,
        leg2_aleatoric_test, leg3_hydro, dq_test, Q_stats, OUTPUT_DIR,
    )

    # Temporal audit
    temporal_audit(
        n_test, obs_t, Q_lstm_t, Q_system, is_defer,
        leg1_stat, leg2_aleatoric_test, leg3_hydro, Q_stats, test,
    )

    # Failure attribution
    failure_attribution(obs_t, Q_lstm_t, Q_system, is_defer, Q_stats)

    # Confident-but-wrong
    confident_but_wrong(
        test, obs_t, Q_lstm_t, wt_t, uale_t, is_defer,
        Q_stats, GLOBAL_TAU, P30D_Q67, OUTPUT_DIR,
    )

    # Flood classification
    flood_classification_overlap(obs_t, Q_lstm_t, Q_system, Q_stats, OUTPUT_DIR)

    # Latency profile  
    profile_single_day_latency(test, uq_test, aci, xgb_model, is_defer)

    # ── 4. Plotting ────────────────────────────────────────────────────────────
    shap_vals, X_defer, explainer, peak_dq_idx, defer_dates_test = compute_shap(
        test, uq_test, is_defer, xgb_model, OUTPUT_DIR
    )

    plot_shap(
        shap_vals, X_defer, explainer, peak_dq_idx,
        defer_dates_test, dq_test, is_defer, OUTPUT_DIR,
    )
    plot_coverage_legs(
        cov_by_regime, leg1_stat, leg2_aleatoric_test, leg3_hydro, is_defer,
        alpha0=0.10, output_dir=OUTPUT_DIR,
    )
    plot_rc_and_metrics(rc_df, s1_metrics, s2_metrics, OUTPUT_DIR)
    plot_rating_curve_sensitivity(rc_errors, s1_fnr_series, s2_fnr_series, OUTPUT_DIR)
    plot_confident_wrong(
        test, obs_t, Q_lstm_t, wt_t, uale_t, is_defer,
        GLOBAL_TAU, P30D_Q67, Q_stats, OUTPUT_DIR,
    )
    plot_full_hydrograph(
        df, test, uq_test, Q_system, is_defer,
        leg2_aleatoric_test, leg3_hydro, Q_stats, SPLITS, OUTPUT_DIR,
    )

    # Operator dashboard
    print_operator_dashboard(
        test, obs_t, Q_lstm_t, Q_system, is_defer, wt_t, taus_test,
        uale_t, leg2_aleatoric_test, leg3_hydro, uq_test,
        shap_vals, X_defer, defer_dates_test, xgb_model,
        peak_dq_idx, Q_stats, TAU_ALE,
    )

    # Serialise
    serialise_strategy_metrics(s1_metrics, s2_metrics, OUTPUT_DIR)
    save_json(
        {
            "tau_per_regime":          tau_regime,
            "GLOBAL_TAU":              GLOBAL_TAU,
            "TAU_ALE":                 TAU_ALE,
            "achieved_eta_ale":        achieved_eta_ale,
            "coverage_by_regime_h1":   cov_by_regime,
            "COST_RATIO":              COST_RATIO,
            "NORMALIZED_ESC_COST":     NORMALIZED_ESC_COST,
            "deferral_rate":           float(is_defer.mean()),
            "leg1_epistemic_rate":     float(leg1_stat.mean()),
            "leg2_aleatoric_rate":     float(leg2_aleatoric_test.mean()),
            "leg3_threshold_safeguard_rate":     float(leg3_hydro.mean()),
            "ope_dm":                  ope_result,
            "mcnemar_p":               pval_mcnemar,
        },
        OUTPUT_DIR / "pipeline_summary.json",
    )
    print("\nFloodTriage pipeline complete. All artefacts saved to:", OUTPUT_DIR)


# ── Script entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_full_pipeline()
