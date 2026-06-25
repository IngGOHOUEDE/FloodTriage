"""
FloodTriage — Block 1: Data Preparation, Utilities & Metrics
=============================================================
Ouémé at Savè, Benin | 1965–2011

Architecture
------------
- 5-member LSTM deep ensemble  (NeuralHydrology back-end)
- XGBoost surrogate correction expert
- Tri-Fold Routing Gateway     (epistemic | aleatoric | Threshold Safeguard)
- Multi-track horizon-specific Adaptive Conformal Inference (ACI)
  per Zaffran et al. (2022) and Xu & Xie (2021)

References
----------
Xu, C., & Xie, Y. (2021). Conformal Prediction for Time Series (EnbPI). ICML.
Zaffran, M., Dieuleveut, A., et al. (2022). Adaptive Conformal Predictions for
    Time Series. ICML.
Sun, Y., et al. (2022). Copula Conformal Prediction for Multi-step Time Series
    Forecasting. ICLR.
Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI.
Cortes, C., et al. (2016). Learning with rejection. ALT.

PEP 8 compliant. Dead code removed. Duplicate blocks eliminated.
"""

# ── Standard Library ───────────────────────────────────────────────────────────
import io
import json
import math
import os
import pickle
import sys
import textwrap
import time
import warnings
from pathlib import Path

# ── Scientific Stack ───────────────────────────────────────────────────────────
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xarray as xr
import xgboost as xgb
import yaml
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.stats.proportion import proportion_confint

# ── NeuralHydrology ────────────────────────────────────────────────────────────
from neuralhydrology.modelzoo import get_model
from neuralhydrology.nh_run import eval_run, start_run
from neuralhydrology.utils.config import Config

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# §0  Global Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = Path("/content")
OUTPUT_DIR = DATA_ROOT / "ft_data"
NH_DATA_DIR = DATA_ROOT / "nh_data"
NH_RUN_DIR = DATA_ROOT / "nh_runs"
DRIVE = DATA_ROOT / "Data_Save"

for _p in [OUTPUT_DIR, NH_DATA_DIR, NH_RUN_DIR]:
    _p.mkdir(exist_ok=True)

BASIN_ID = "save"

# ── Cost-Sensitive Deferral Policy (Elkan 2001; Cortes et al. 2016) ───────────
# C_err / C_esc = 10 means one missed flood costs 10× a false escalation.
COST_RATIO = 10
NORMALIZED_ESC_COST = 1.0 / COST_RATIO   # 0.10
# ANALYTICAL_TAU_STAR (0.90) has been removed — it was unused downstream.

# ── Multi-Horizon Forecast Settings ──────────────────────────────────────────
H_MAX = 15         # Maximum forecast lead time in days (operational ECMWF cycle)
SEEDS = [42, 43, 44, 45, 46]
WARMUP_DAYS = 365
WARMUP_END = "1966-12-31"

# Splits
# NOTE: The two-year gap 2000–2001 between training end (1999-12-31) and
# calibration start (2002-01-01) is intentional. It prevents any residual
# LSTM hidden-state memory or feature rolling windows (P7d, P30d, P90d, ASM)
# from leaking training-period information into the calibration set. The
# first 90 days of calibration are additionally masked (see §2).
SPLITS = {
    "train": ("1965-01-01", "1999-12-31"),
    "cal":   ("2002-01-01", "2007-12-31"),
    "test":  ("2008-01-01", "2011-12-31"),
}

# Regimes and labels used throughout
REGIMES = ["low", "normal", "flood"]
ASM_LABELS = ["dry", "moderate", "wet"]
P30D_LABELS = ["dry", "moderate", "wet"]


# ══════════════════════════════════════════════════════════════════════════════
# §1  Data Loading & Pre-processing
# ══════════════════════════════════════════════════════════════════════════════

def _load(path: Path, name: str) -> pd.Series:
    """Load a single-column daily hydro-met CSV into a named daily Series.

    Parameters
    ----------
    path : Path
        Full file path.
    name : str
        Output series name.

    Returns
    -------
    pd.Series
        Daily time series with DatetimeIndex, frequency 'D'.
    """
    s = pd.read_csv(
        path,
        header=0,
        parse_dates=[0],
        index_col=0,
        na_values=["", " "],
    ).iloc[:, 0].rename(name)
    s.index.name = "date"
    return s.asfreq("D")


def load_raw_data(drive: Path) -> pd.DataFrame:
    """Download (via gdown) and assemble the raw P / PET / Q dataframe.

    Handles arbitrarily long PET gaps (months to years) by forward- then
    back-filling. The fill is validated with an assertion so that downstream
    calculations never propagate NaN silently.

    Parameters
    ----------
    drive : Path
        Local mount point of the Google-Drive folder.

    Returns
    -------
    pd.DataFrame
        Columns: ['P', 'PET', 'Q']  — daily, 1965-01-01 to 2011-12-31.
    """
    os.system(f"gdown --folder 1yKfjGlt8Bzlm8hJu6EdlIM9d6QmAbVy2 -O {drive}")

    files = {
        "P":   "Precip_mm_Save_1965_2011.csv",
        "PET": "PET_mm_Save_1965_2011.csv",
        "Q":   "Discharge_cms_Save_1965_2011.csv",
    }
    raw_data = {k: _load(drive / f, k) for k, f in files.items()}
    df = pd.DataFrame(raw_data).loc["1965-01-01":"2011-12-31"]

    # ── PET gap-fill (supports multi-month / multi-year gaps) ─────────────────
    pet_missing = df["PET"].isna()
    n_missing = int(pet_missing.sum())
    if n_missing > 0:
        gap_lengths = pet_missing.groupby((~pet_missing).cumsum()).sum()
        max_gap = int(gap_lengths.max())
        print(f"PET missing values: {n_missing}  |  Max consecutive gap: {max_gap} days")
        print("  Strategy: forward-fill then back-fill (handles multi-month gaps).")
    else:
        print("PET: no missing values.")

    df["PET"] = df["PET"].ffill().bfill()
    assert not df["PET"].isna().any(), "PET still contains NaN after fill."

    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add hydrological state features and compute training-period quantiles.

    Features added in-place
    -----------------------
    P7d, P30d, P90d : Rolling precipitation sums.
    ASM             : Exponentially Weighted Moving Average antecedent soil
                      moisture proxy, normalised to [0, 1] over the training
                      period.  alpha = 0.05 → ~20-day memory time constant,
                      representative of a 0–100 cm soil column.
    regime          : Categorical flow regime label ('low', 'normal', 'flood').

    Data-Leakage Boundary Treatment
    --------------------------------
    The 2000–2001 gap between training and calibration windows means that any
    feature computed with a lookback window (rolling sum or EWMA) reaching
    back before 2002-01-01 carries ghost memory from the training period.
    Following the treatment for P7d / P30d / P90d, ASM is also masked to NaN
    for the first 90 days of calibration (2002-01-01 – 2002-03-31). XGBoost
    handles NaN natively; LSTM sees these as missing inputs and clips them.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with columns ['P', 'PET', 'Q'].

    Returns
    -------
    df : pd.DataFrame
        Augmented dataframe.
    Q_stats : dict
        Training-period streamflow quantiles and moments.
    """
    tr_mask = (df.index >= SPLITS["train"][0]) & (df.index <= SPLITS["train"][1])
    tr_q = df.loc[tr_mask, "Q"].dropna()

    # Rolling precipitation sums
    df["P7d"] = df["P"].rolling(7, min_periods=1).sum()
    df["P30d"] = df["P"].rolling(30, min_periods=1).sum()
    df["P90d"] = df["P"].rolling(90, min_periods=1).sum()

    # Mask the first 90 days of calibration to prevent cross-gap leakage
    boundary = slice("2002-01-01", "2002-03-31")
    df.loc[boundary, ["P7d", "P30d", "P90d"]] = np.nan

    # Antecedent Soil Moisture proxy via EWMA on net forcing (P − PET)
    alpha_decay = 0.05
    wetness_raw = (df["P"] - df["PET"]).ewm(alpha=alpha_decay, adjust=False).mean()
    w_min = wetness_raw[tr_mask].min()
    w_max = wetness_raw[tr_mask].max()
    df["ASM"] = ((wetness_raw - w_min) / (w_max - w_min + 1e-8)).clip(0, 1)

    # Mirror the calibration boundary mask for ASM  
    df.loc[boundary, "ASM"] = np.nan

    print(f"P30d @ 2002-01-01 (after mask): "
          f"{df.loc['2002-01-01', 'P30d']}")    # should be NaN
    print(f"ASM  @ 2002-01-01 (after mask): "
          f"{df.loc['2002-01-01', 'ASM']}")      # should be NaN

    # Training-period quantiles
    Q_stats = {
        "Q50":    float(tr_q.quantile(0.50)),
        "Q95":    float(tr_q.quantile(0.95)),
        "Q99":    float(tr_q.quantile(0.99)),
        "Q_mean": float(tr_q.mean()),
        "Q_std":  float(tr_q.std()),
    }

    df["regime"] = assign_regime(df["Q"].fillna(-1), Q_stats)

    return df, Q_stats


def assign_regime(
    q_vals: np.ndarray,
    Q_stats: dict,
) -> np.ndarray:
    """Categorise streamflow values into operational risk regime labels.

    Boundaries are set from training-period quantiles so the function is
    leakage-free when called on any period.

    Parameters
    ----------
    q_vals : array-like
        Streamflow values (m³/s).
    Q_stats : dict
        Must contain keys 'Q50' and 'Q95'.

    Returns
    -------
    np.ndarray of dtype object
        Values in {'low', 'normal', 'flood'}.
    """
    q_arr = np.asarray(q_vals, float)
    reg = np.full(q_arr.shape, "low", dtype=object)
    reg[q_arr >= Q_stats["Q50"]] = "normal"
    reg[q_arr >= Q_stats["Q95"]] = "flood"
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# §2  Metric Suite
# ══════════════════════════════════════════════════════════════════════════════

def _mask(o: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN pairs from observed / simulated arrays."""
    m = ~np.isnan(o) & ~np.isnan(s)
    return o[m], s[m]


def nse(o: np.ndarray, s: np.ndarray) -> float:
    """Nash–Sutcliffe Efficiency."""
    o, s = _mask(o, s)
    d = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((o - s) ** 2) / d) if d > 0 else np.nan


def kge(o: np.ndarray, s: np.ndarray) -> float:
    """Kling–Gupta Efficiency."""
    o, s = _mask(o, s)
    if len(o) < 2:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return float(
        1 - math.sqrt(
            (r - 1) ** 2
            + (s.std() / o.std() - 1) ** 2
            + (s.mean() / o.mean() - 1) ** 2
        )
    )


def rmse(o: np.ndarray, s: np.ndarray) -> float:
    """Root Mean Square Error."""
    o, s = _mask(o, s)
    return float(np.sqrt(np.mean((o - s) ** 2))) if len(o) else np.nan


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent Bias."""
    o, s = _mask(np.asarray(obs, float), np.asarray(sim, float))
    return float(100 * (s - o).sum() / (o.sum() + 1e-8))


def fnr_wilson(
    obs: np.ndarray,
    pred: np.ndarray,
    q95: float,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float]]:
    """False Negative Rate with Wilson score confidence interval.

    Uses ``statsmodels.stats.proportion.proportion_confint`` (method='wilson')
    as a rigorous replacement for the hand-rolled binomial interval.

    Parameters
    ----------
    obs, pred : array-like
        Observed and predicted streamflow (m³/s).
    q95 : float
        Flood threshold (Q95 of training period).
    alpha : float
        Two-sided significance level (default 0.05 → 95 % CI).

    Returns
    -------
    p_hat : float
        Point estimate FNR.
    ci : (float, float)
        Wilson 95 % confidence interval (lower, upper).
    """
    flood = np.asarray(obs) >= q95
    valid = flood & ~np.isnan(obs) & ~np.isnan(pred)
    n = int(valid.sum())

    if n == 0:
        return np.nan, (np.nan, np.nan)

    k = int((valid & (np.asarray(pred) < q95)).sum())
    p_hat = k / n

    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(p_hat), (float(lo), float(hi))


def mcnemar_fnr_test(
    obs: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    q95: float,
) -> float:
    """McNemar chi-squared p-value comparing flood-miss rates of two models.

    Parameters
    ----------
    obs : array-like  Observed streamflow.
    pred1, pred2 : array-like  Model 1 and Model 2 predictions.
    q95 : float  Flood threshold.

    Returns
    -------
    float  p-value (two-sided, continuity-corrected).
    """
    from scipy.stats import chi2

    flood = np.asarray(obs) >= q95
    valid = flood & ~np.isnan(obs) & ~np.isnan(pred1) & ~np.isnan(pred2)

    if valid.sum() == 0:
        return np.nan

    missed_1 = np.asarray(pred1)[valid] < q95
    missed_2 = np.asarray(pred2)[valid] < q95

    b = int((missed_1 & ~missed_2).sum())   # model 1 misses, model 2 catches
    c = int((~missed_1 & missed_2).sum())   # model 1 catches, model 2 misses

    if b + c == 0:
        return 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2.sf(chi2_stat, df=1))


def moving_block_bootstrap(
    obs: np.ndarray,
    sim: np.ndarray,
    block_size: int = 60,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """95 % CIs for NSE / KGE / RMSE / PBIAS via moving-block bootstrap.

    Parameters
    ----------
    obs, sim : array-like  Observed and simulated streamflow.
    block_size : int  Bootstrap block length in days.
    n_boot : int  Number of resamples.
    seed : int  Random seed for reproducibility.

    Returns
    -------
    dict  Keys: 'NSE', 'KGE', 'RMSE', 'PBIAS' → (lo_2.5, hi_97.5).
    """
    rng = np.random.default_rng(seed)
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[valid], sim[valid]
    n = len(o)

    if n < block_size:
        return {}

    n_blocks = int(np.ceil(n / block_size))
    boot_res = {"NSE": [], "KGE": [], "RMSE": [], "PBIAS": []}

    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate(
            [np.arange(st, st + block_size) for st in starts]
        )[:n]
        boot_res["NSE"].append(nse(o[idx], s[idx]))
        boot_res["KGE"].append(kge(o[idx], s[idx]))
        boot_res["RMSE"].append(rmse(o[idx], s[idx]))
        boot_res["PBIAS"].append(pbias(o[idx], s[idx]))

    return {
        k: (
            float(np.nanpercentile(v, 2.5)),
            float(np.nanpercentile(v, 97.5)),
        )
        for k, v in boot_res.items()
    }


def block_bootstrap_hydrology(
    obs: np.ndarray,
    sim: np.ndarray,
    block_size: int = 7,
    n_boot: int = 1000,
) -> tuple[float, float]:
    """Compact KGE block bootstrap returning 95 % CI tuple only."""
    rng = np.random.default_rng(42)
    n = len(obs)
    n_blocks = n // block_size
    kge_b = []

    for _ in range(n_boot):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        b_obs, b_sim = [], []
        for b in sampled:
            start = b * block_size
            end = start + block_size
            b_obs.extend(obs[start:end])
            b_sim.extend(sim[start:end])
        kge_b.append(kge(np.array(b_obs), np.array(b_sim)))

    return (
        float(np.percentile(kge_b, 2.5)),
        float(np.percentile(kge_b, 97.5)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3  Persistence: save / load artefacts
# ══════════════════════════════════════════════════════════════════════════════

def save_json(obj: dict, path: Path) -> None:
    """Serialise a dict to a JSON file."""
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: Path) -> dict:
    """Load a JSON file as a dict."""
    with open(path) as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# §4  Entrypoint — run data preparation
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = load_raw_data(DRIVE)
    df, Q_stats = engineer_features(df)

    # Persist
    save_json(Q_stats, OUTPUT_DIR / "quantiles.json")
    save_json(SPLITS, OUTPUT_DIR / "splits.json")
    df.to_csv(OUTPUT_DIR / "full.csv")

    print("\nData Preparation Complete.")
    print({k: f"{v:.3f}" for k, v in Q_stats.items()})
    print(f"COST_RATIO         : {COST_RATIO}")
    print(f"NORMALIZED_ESC_COST: {NORMALIZED_ESC_COST:.3f}")
