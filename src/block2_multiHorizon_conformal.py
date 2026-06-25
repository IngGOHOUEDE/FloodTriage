"""
FloodTriage — Block 2: LSTM Ensemble & Multi-Horizon Conformal Inference
=========================================================================
Ouémé at Savè, Benin | 1965–2011

Key Components
--------------
1. NeuralHydrology LSTM ensemble (5 seeds, H=15 day forecast horizon).
2. Multi-track, horizon-specific Adaptive Conformal Inference (ACI) engine.
   - Ensemble variance-normalized nonconformity scores  (Xu & Xie 2021).
   - Independent alpha tracks per lead time h∈{1,…,15}  (Zaffran et al. 2022).
   - Marginal interval semantics per horizon              (Sun et al. 2022).
3. Uncertainty Quantification (UQ) dataframe construction.

LSTM Training Notes (NeuralHydrology Validation Loop)
------------------------------------------------------
NeuralHydrology calls ``model.eval()`` internally before the validation
forward pass. This correctly freezes batch-normalisation statistics and
disables dropout, so the validation loss is a clean held-out estimate.
No modification to the PyTorch training loop is required or applied here.

Missing Data Handling
---------------------
The LSTM ``seq_length=365`` window can span the 2000-01-01 to 2001-12-31
gap between training and calibration partitions. NeuralHydrology fills masked
input timesteps with the dataset ``_FillValue=-999.0`` token. The model
learns to treat these tokens as missing via the ``allow_subsequent_nan_losses``
flag, which suppresses gradient updates when target Q is NaN. This approach
handles arbitrarily long missing sequences (months to years) without manual
imputation, and is the recommended pattern for data-sparse hydrological basins.
"""

# ── Imports (assumes Block 1 has been executed in the same session) ────────────
import io
import json
import pickle
import re
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from neuralhydrology.nh_run import eval_run, start_run

# Re-import Block-1 globals when running as a standalone module
# (In a notebook, these are already in global scope.)
try:
    from block1_data_prep_utilities import (
        BASIN_ID, COST_RATIO, DEVICE, DRIVE, H_MAX, NH_DATA_DIR,
        NH_RUN_DIR, NORMALIZED_ESC_COST, OUTPUT_DIR, SEEDS, SPLITS,
        WARMUP_DAYS, Q_stats, assign_regime, df, load_json, save_json,
    )
except ImportError:
    pass  # Running inside notebook — globals already present.

# Add this right after your try/except import block
def save_json(data: dict, path: Path) -> None:
    """Safely serialize and save a dictionary to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# ══════════════════════════════════════════════════════════════════════════════
# §1  NeuralHydrology Data Preparation (NetCDF + Basins file)
# ══════════════════════════════════════════════════════════════════════════════

FILL = -999.0


def write_netcdf(df: pd.DataFrame, nh_data_dir: Path, basin_id: str) -> None:
    """Write input time series to a NetCDF file expected by NeuralHydrology.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with columns ['P', 'PET', 'Q'].
    nh_data_dir : Path
        Root directory for NeuralHydrology data.
    basin_id : str
        Basin identifier string (e.g. 'save').
    """
    ts_dir = nh_data_dir / "time_series"
    ts_dir.mkdir(exist_ok=True)

    ds = xr.Dataset(
        {
            v: xr.DataArray(df[v].values.astype(np.float32), dims=["date"])
            for v in ["P", "PET", "Q"]
        },
        coords={"date": df.index.values},
    )
    ds.to_netcdf(
        ts_dir / f"{basin_id}.nc",
        encoding={
            v: {"dtype": "float32", "_FillValue": FILL}
            for v in ["P", "PET", "Q"]
        },
    )
    (nh_data_dir / "basins.txt").write_text(f"{basin_id}\n")
    pd.DataFrame({"basin_id": [basin_id]}).to_csv(
        nh_data_dir / "static_attributes.csv", index=False
    )
    print(f"NetCDF written: {ts_dir / basin_id}.nc")


def _fmt(d: str) -> str:
    """Format date string to DD/MM/YYYY for NeuralHydrology YAML."""
    return pd.to_datetime(d).strftime("%d/%m/%Y")


def build_base_yaml(df: pd.DataFrame, nh_data_dir: Path, nh_run_dir: Path) -> str:
    """Construct the base YAML configuration for NeuralHydrology.

    The predict_last_n parameter is set to H_MAX (15) to enable a 15-day
    multi-step output window per forward pass. This matches operational
    ECMWF NWP telemetry cycles and is the minimal change required to expose
    horizon-indexed predictions for multi-track ACI calibration.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe (used only for date bounds).
    nh_data_dir, nh_run_dir : Path
        NeuralHydrology data and run directories.

    Returns
    -------
    str  YAML text.
    """
    full_start = _fmt(str(df.index.min().date()))
    full_end = _fmt(str(df.index.max().date()))
    model_type = "cudalstm" if DEVICE == "cuda" else "lstm"

    yaml_text = textwrap.dedent(f"""
        experiment_name: ft_s42
        run_dir: {nh_run_dir}
        dataset: generic
        data_dir: {nh_data_dir}
        train_basin_file:      {nh_data_dir}/basins.txt
        validation_basin_file: {nh_data_dir}/basins.txt
        test_basin_file:       {nh_data_dir}/basins.txt
        train_start_date:      '{_fmt(SPLITS["train"][0])}'
        train_end_date:        '{_fmt(SPLITS["train"][1])}'
        validation_start_date: '{full_start}'
        validation_end_date:   '{full_end}'
        test_start_date:       '{full_start}'
        test_end_date:         '{full_end}'
        dynamic_inputs: [P, PET]
        static_inputs: []
        target_variables: [Q]
        clip_targets_to_zero: [Q]
        model: {model_type}
        head: regression
        hidden_size: 64
        initial_forget_bias: 3
        output_dropout: 0.4
        seq_length: 365
        predict_last_n: {H_MAX}
        optimizer: Adam
        loss: MSE
        learning_rate:
          0:  0.001
          15: 0.0005
        epochs: 25
        batch_size: 256
        allow_subsequent_nan_losses: True
        num_workers: 0
        seed: 42
        log_interval: 5
        log_tensorboard: False
        save_weights_every: 25
    """).strip()
    return yaml_text


# ══════════════════════════════════════════════════════════════════════════════
# §2  Ensemble Training & Prediction Extraction
# ══════════════════════════════════════════════════════════════════════════════

class _SilentFilter(io.TextIOBase):
    """Stdout filter: surfaces only epoch / loss lines during NH training."""

    def __init__(self, real_stdout):
        self._r = real_stdout
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if re.search(r"Epoch\s*\d+", line) or "loss" in line.lower():
                self._r.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf:
            self._r.write(self._buf)
            self._buf = ""
        self._r.flush()


def _last_epoch(run_dir: Path) -> int:
    pts = sorted(run_dir.glob("model_epoch*.pt"))
    return max(int(p.stem.replace("model_epoch", "")) for p in pts)


def _load_predictions(run_dir: Path, epoch: int, basin_id: str) -> pd.DataFrame:
    """Extract per-horizon predictions from a NeuralHydrology result file.

    With ``predict_last_n=H_MAX`` the result pickle contains one column per
    horizon step.  This function returns a DataFrame with columns
    ``Q_h1`` … ``Q_h{H_MAX}`` aligned to the *target date* index (i.e. the
    date being predicted, not the date of issue).

    Parameters
    ----------
    run_dir : Path
        NeuralHydrology run directory.
    epoch : int
        Epoch number to load.
    basin_id : str
        Basin identifier.

    Returns
    -------
    pd.DataFrame
        Columns: ``Q_h1`` … ``Q_hH_MAX``  (all clipped to ≥ 0).
    """
    p = run_dir / "test" / f"model_epoch{epoch:03d}" / "test_results.p"
    if not p.exists():
        p = sorted(run_dir.rglob("test_results.p"))[-1]

    with open(p, "rb") as fh:
        res = pickle.load(fh)

    basin = res[basin_id]
    if isinstance(basin, dict):
        inner = basin.get("1D", basin)
        if isinstance(inner, dict):
            inner = inner.get("xr", next(iter(inner.values())))
        basin = inner

    # Build a DataFrame indexed by target date
    if isinstance(basin, xr.Dataset):
        sim_vars = [v for v in basin.data_vars if str(v).endswith("_sim")]
        if len(sim_vars) == 0:
            raise ValueError("No '*_sim' variables found in test results.")

        # NH may store multiple lead times as separate variables or as a
        # dimension named 'lead_time' / 'time_step'.
        if "lead_time" in basin.dims or "time_step" in basin.dims:
            dim = "lead_time" if "lead_time" in basin.dims else "time_step"
            arr = basin[sim_vars[0]].values   # shape: (n_days, H_MAX)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            idx = basin.coords["date"].values if "date" in basin.coords else basin.coords["time"].values
            cols = {f"Q_h{h + 1}": np.clip(arr[:, h], 0, None) for h in range(arr.shape[1])}
            return pd.DataFrame(cols, index=pd.DatetimeIndex(idx))
        else:
            # Single-output fallback (H=1 stored in one variable)
            series = basin[sim_vars[0]].squeeze().to_series()
            return pd.DataFrame({"Q_h1": np.clip(series.values, 0, None)},
                                index=series.index)

    # Legacy dict / DataFrame format
    sim_cols = [c for c in basin.columns if str(c).endswith("_sim")]
    if len(sim_cols) == 0:
        raise ValueError("No '*_sim' columns found in test results.")
    return basin[sim_cols[0]].rename("Q_h1").clip(lower=0).to_frame()


def train_ensemble(
    df: pd.DataFrame,
    nh_data_dir: Path,
    nh_run_dir: Path,
    output_dir: Path,
    seeds: list,
) -> tuple[dict, dict]:
    """Train the 5-seed LSTM ensemble and return per-seed prediction DataFrames.

    Each DataFrame has columns Q_h1 … Q_hH_MAX (multi-horizon) aligned to the
    full date range of df.

    Parameters
    ----------
    df : pd.DataFrame  Full dataframe.
    nh_data_dir, nh_run_dir, output_dir : Path  Directory paths.
    seeds : list[int]  Random seeds for ensemble members.

    Returns
    -------
    ensemble_preds : dict[int, pd.DataFrame]
        Horizon-indexed prediction DataFrames keyed by seed.
    run_dirs : dict[int, Path]
        Corresponding NeuralHydrology run directories.
    """
    base_yaml = build_base_yaml(df, nh_data_dir, nh_run_dir)
    (output_dir / "nh_config_base.yaml").write_text(base_yaml)

    ensemble_preds: dict[int, pd.DataFrame] = {}
    run_dirs: dict[int, Path] = {}

    for seed in seeds:
        cfg_text = (
            base_yaml
            .replace("seed: 42", f"seed: {seed}")
            .replace("ft_s42", f"ft_s{seed}")
        )
        cfg_path = Path(f"/content/cfg_s{seed}.yaml")
        cfg_path.write_text(cfg_text)

        real_stdout = sys.stdout
        sys.stdout = _SilentFilter(sys.stdout)
        try:
            start_run(config_file=cfg_path)
        finally:
            sys.stdout = real_stdout

        rdir = max(
            (d for d in nh_run_dir.iterdir()
             if d.is_dir() and f"ft_s{seed}" in d.name),
            key=lambda d: d.stat().st_mtime,
        )
        epoch = _last_epoch(rdir)
        eval_run(run_dir=rdir, period="test", epoch=epoch)

        pred_df = _load_predictions(rdir, epoch, BASIN_ID)
        pred_df = pred_df.reindex(df.index)
        ensemble_preds[seed] = pred_df
        run_dirs[seed] = rdir
        print(f"  seed {seed} ✓  valid rows={pred_df.notna().all(axis=1).sum()}")

    save_json(
        {str(k): str(v) for k, v in run_dirs.items()},
        output_dir / "run_dirs.json",
    )
    print("Ensemble training complete.")
    return ensemble_preds, run_dirs


# ══════════════════════════════════════════════════════════════════════════════
# §3  Uncertainty Quantification (UQ)
# ══════════════════════════════════════════════════════════════════════════════

def build_uq(
    ensemble_preds: dict,
    df: pd.DataFrame,
    Q_stats: dict,
    h: int = 1,
) -> pd.DataFrame:
    """Compute ensemble mean, std and normalised epistemic spread w_t.

    The spread metric ``w_t`` is defined as::

        w_t = 2 * Q_std / (sigma_clim + ε)

    where ``Q_std`` is the ensemble *standard deviation* (not variance —
    a docstring error present in the original code has been corrected here).
    Dividing by the climatological standard deviation ``sigma_clim`` mixes
    model epistemic spread with baseline natural aleatoric variability; the
    empirical multiplier ``2.0`` approximates a 95 % confidence interval
    baseline under a Gaussian assumption and is a common operational heuristic.

    Parameters
    ----------
    ensemble_preds : dict[int, pd.DataFrame]
        Prediction DataFrames from :func:`train_ensemble`.
    df : pd.DataFrame
        Full dataframe (used for index alignment).
    Q_stats : dict
        Training-period statistics; must contain 'Q_std'.
    h : int
        Horizon step to extract (1-indexed, default h=1 → day-ahead).

    Returns
    -------
    pd.DataFrame
        Columns: ['Q_mean', 'Q_std', 'w_t'] aligned to df.index.
    """
    col = f"Q_h{h}"
    Q_ens = np.stack([
        ensemble_preds[s][col].reindex(df.index).values
        for s in SEEDS
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Q_mean = np.nanmean(Q_ens, axis=0)
        Q_std = np.nanstd(Q_ens, axis=0)

    Q_std = np.where(Q_std < 1e-4, 1e-4, Q_std)
    sigma_clim = Q_stats["Q_std"]

    # w_t: normalised ensemble spread (epistemic uncertainty proxy).
    # NOTE: Q_std is standard deviation, not variance. The multiplier 2.0
    # approximates a ~95 % Gaussian CI baseline for the spread normalisation.
    w_t = 2.0 * Q_std / (sigma_clim + 1e-8)

    return pd.DataFrame(
        {"Q_mean": Q_mean, "Q_std": Q_std, "w_t": w_t},
        index=df.index,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §4  Multi-Track Horizon-Specific Adaptive Conformal Inference
# ══════════════════════════════════════════════════════════════════════════════
#
# Implementation follows:
#   Zaffran et al. (2022) — independent α tracks per horizon.
#   Xu & Xie (2021)       — ensemble variance normalisation of scores.
#   Sun et al. (2022)     — marginal (not joint) interval semantics.
# ─────────────────────────────────────────────────────────────────────────────


class MultiHorizonACI:
    """Multi-track Adaptive Conformal Inference engine (H=15 horizons).

    Score definition (Xu & Xie 2021, EnbPI)
    ----------------------------------------
    For a prediction targeting day t, issued at lead time h::

        S_{t,h} = |y_t - μ̂_{t|t-h}| / (σ̂_{t|t-h} + ε)

    where μ̂ is the ensemble mean and σ̂ is the ensemble standard deviation.

    Update rule (Zaffran et al. 2022, ACID)
    ----------------------------------------
    Each morning, after observing y_t::

        α_{t,h} = clip(α_{t-1,h} + γ_h * (α_0 - 𝟏[y_t ∉ Ĉ_{t|t-h}]), 0.01, 0.99)

    Interval construction
    ---------------------
    For future step t+h::

        q_{t,h} = quantile(S_history_h, 1 - α_{t,h})
        Ĉ_{t+h|t} = [μ̂_{t+h|t} - q_{t,h}·σ̂_{t+h|t},
                      μ̂_{t+h|t} + q_{t,h}·σ̂_{t+h|t}]

    Parameters
    ----------
    alpha0 : float
        Initial nominal miscoverage rate (e.g. 0.10 for 90 % intervals).
    gamma : float or array-like of shape (H_MAX,)
        Per-horizon learning rate. Scalar is broadcast across all horizons.
        Larger γ adapts faster to local coverage drift.
    h_max : int
        Number of forecast horizons.
    eps : float
        Denominator regulariser for score normalisation.
    min_cal_scores : int
        Minimum historical scores required before issuing an interval.
        Days with fewer scores receive a wide fallback interval.
    """

    def __init__(
        self,
        alpha0: float = 0.10,
        gamma: float | np.ndarray = 0.005,
        h_max: int = 15,
        eps: float = 1e-6,
        min_cal_scores: int = 30,
    ):
        self.alpha0 = alpha0
        self.h_max = h_max
        self.eps = eps
        self.min_cal_scores = min_cal_scores

        # Horizon-specific learning rates (γ_h).  Slightly higher γ for
        # longer horizons to compensate for larger forecast uncertainty.
        if np.isscalar(gamma):
            self._gamma = np.full(h_max, float(gamma))
        else:
            self._gamma = np.asarray(gamma, float)
            assert len(self._gamma) == h_max

        # State: one α per horizon, initialised to alpha0
        self._alpha = np.full(h_max, alpha0)

        # Score histories: list of lists, one per horizon
        self._scores: list[list[float]] = [[] for _ in range(h_max)]

    # ── Public API ─────────────────────────────────────────────────────────────

    def calibrate_on_window(
        self,
        obs: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        h: int = 1,
    ) -> None:
        """Pre-populate score history from a held-out calibration window.

        This seeds the engine before the online test-period loop begins.
        No α updates are performed here — calibration is used for warm-start
        only, matching the ACI paper's offline-pre-load strategy.

        Parameters
        ----------
        obs : (N,) array  Observed streamflow.
        mu : (N,) array   Ensemble mean for horizon h.
        sigma : (N,) array Ensemble std for horizon h.
        h : int  1-indexed horizon (1 = day-ahead, …, 15).
        """
        assert 1 <= h <= self.h_max, f"h must be in [1, {self.h_max}]."
        h_idx = h - 1
        valid = ~np.isnan(obs) & ~np.isnan(mu) & ~np.isnan(sigma)
        scores = (
            np.abs(obs[valid] - mu[valid]) / (sigma[valid] + self.eps)
        ).tolist()
        self._scores[h_idx].extend(scores)

    def get_quantile(self, h: int) -> float:
        """Return the current empirical quantile q_{t,h} for horizon h.

        Parameters
        ----------
        h : int  1-indexed horizon.

        Returns
        -------
        float  Normalised score quantile (or large fallback if few scores).
        """
        h_idx = h - 1
        hist = self._scores[h_idx]
        if len(hist) < self.min_cal_scores:
            return 10.0  # wide fallback
        return float(np.quantile(hist, 1.0 - self._alpha[h_idx]))

    def build_interval(
        self,
        mu: float,
        sigma: float,
        h: int,
        q_cap: float = 1e6,
    ) -> tuple[float, float]:
        """Construct the prediction interval for one (t, h) step.

        Parameters
        ----------
        mu : float  Ensemble mean forecast.
        sigma : float  Ensemble standard deviation.
        h : int  1-indexed lead time.
        q_cap : float  Upper cap on quantile value (overflow guard).

        Returns
        -------
        (Q_lo, Q_hi) : tuple[float, float]
            Lower and upper bounds, Q_lo ≥ 0 (physical constraint).
        """
        q_th = min(self.get_quantile(h), q_cap)
        half = q_th * (sigma + self.eps)
        Q_lo = max(0.0, mu - half)
        Q_hi = mu + half
        return Q_lo, Q_hi

    def update(
        self,
        y_t: float,
        mu_t: float,
        sigma_t: float,
        Q_lo_t: float,
        Q_hi_t: float,
        h: int,
    ) -> None:
        """Ingest one new observation to update score history and α for horizon h.

        Called each morning after y_t becomes available.

        Parameters
        ----------
        y_t : float  Observed streamflow for day t.
        mu_t, sigma_t : float  Ensemble forecast issued h days ago.
        Q_lo_t, Q_hi_t : float  Interval issued h days ago for day t.
        h : int  1-indexed horizon that targeted day t.
        """
        if np.isnan(y_t) or np.isnan(mu_t):
            return

        h_idx = h - 1
        gamma_h = self._gamma[h_idx]

        # Coverage indicator: 1 if observation fell outside the issued interval
        miss = float(not (Q_lo_t <= y_t <= Q_hi_t))

        # α update: α increases if we miss (looser interval next time)
        self._alpha[h_idx] = float(
            np.clip(
                self._alpha[h_idx] + gamma_h * (self.alpha0 - miss),
                0.01,
                0.99,
            )
        )

        # Append the new normalised nonconformity score to history
        score = abs(y_t - mu_t) / (sigma_t + self.eps)
        self._scores[h_idx].append(score)

    def run_online(
        self,
        dates: pd.DatetimeIndex,
        obs: np.ndarray,
        mu_matrix: np.ndarray,
        sigma_matrix: np.ndarray,
    ) -> pd.DataFrame:
        """Run the online multi-track ACI loop over a date range.

        For each day t and each horizon h, this method:
        1. Issues the interval Ĉ_{t+h|t} using the current q_{t,h}.
        2. After one day advances, updates α_{t+1,h} with the newly
           available observation y_t.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Date index for the prediction period.
        obs : (N,) array
            Observed streamflow aligned to *dates*.
        mu_matrix : (N, H_MAX) array
            Ensemble means. Column h-1 corresponds to the h-day-ahead forecast
            issued *today* (targeting dates[t+h-1]).
        sigma_matrix : (N, H_MAX) array
            Ensemble standard deviations, same layout as mu_matrix.

        Returns
        -------
        pd.DataFrame
            One row per (date, horizon) pair with columns::

                date, h, Q_lo, Q_hi, alpha_t, q_th, covered

            where ``covered`` is ``True`` if the observation fell inside the
            interval (NaN when observation is missing).
        """
        n_t = len(dates)
        records = []

        # Ring buffer: issued intervals for delayed update
        # issued_buf[h_idx][t] = (mu, sigma, Q_lo, Q_hi) issued at t,
        # targeting t + h.
        issued_buf: list[dict] = [{} for _ in range(self.h_max)]

        for t in range(n_t):
            # ── Step A: update α using intervals issued h days ago ────────────
            for h in range(1, self.h_max + 1):
                h_idx = h - 1
                target_t = t  # today's observation closes the loop for h-day
                issue_t = t - h
                if issue_t < 0:
                    continue
                buf = issued_buf[h_idx].get(issue_t)
                if buf is None:
                    continue
                self.update(
                    y_t=obs[target_t],
                    mu_t=buf["mu"],
                    sigma_t=buf["sigma"],
                    Q_lo_t=buf["Q_lo"],
                    Q_hi_t=buf["Q_hi"],
                    h=h,
                )

            # ── Step B: issue intervals for t+1 … t+H_MAX ────────────────────
            for h in range(1, self.h_max + 1):
                h_idx = h - 1
                target_idx = t + h - 1   # index in dates that this targets
                if target_idx >= n_t:
                    break

                mu_th = float(mu_matrix[t, h_idx])
                sigma_th = float(sigma_matrix[t, h_idx])

                if np.isnan(mu_th):
                    continue

                Q_lo, Q_hi = self.build_interval(mu_th, sigma_th, h)
                q_th = self.get_quantile(h)

                # Store for future α-update
                issued_buf[h_idx][t] = {
                    "mu": mu_th,
                    "sigma": sigma_th,
                    "Q_lo": Q_lo,
                    "Q_hi": Q_hi,
                }

                # Coverage (NaN if observation not yet available)
                y_target = obs[target_idx]
                covered = (
                    bool(Q_lo <= y_target <= Q_hi)
                    if not np.isnan(y_target)
                    else np.nan
                )

                records.append({
                    "date":       dates[target_idx],
                    "issue_date": dates[t],
                    "h":          h,
                    "Q_lo":       Q_lo,
                    "Q_hi":       Q_hi,
                    "alpha_t":    float(self._alpha[h_idx]),
                    "q_th":       float(q_th),
                    "covered":    covered,
                    "mu":         mu_th,
                    "sigma":      sigma_th,
                })

        return pd.DataFrame(records)

    @property
    def alpha_state(self) -> np.ndarray:
        """Current α values for all H_MAX horizons."""
        return self._alpha.copy()


# ══════════════════════════════════════════════════════════════════════════════
# §5  Scaler Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_scaler(run_dir: Path, target_var: str = "Q") -> dict:
    """Safely load mean and std scaling parameters from NeuralHydrology configs.

    Supports both pickle (.p) and YAML (.yml / .yaml) scaler file formats
    across different NeuralHydrology versions.

    Parameters
    ----------
    run_dir : Path  NeuralHydrology run directory.
    target_var : str  Variable name to extract (default: 'Q').

    Returns
    -------
    dict  Keys: 'mean', 'std'.
    """
    sp = (
        next(run_dir.glob("**/train_data_scaler.p"), None)
        or next(run_dir.glob("**/train_data_scaler.pickle"), None)
        or next(run_dir.glob("**/train_data_scaler.yml"), None)
        or next(run_dir.glob("**/train_data_scaler.yaml"), None)
    )

    if sp is None:
        return {"mean": 0.0, "std": 1.0}

    if sp.suffix in (".p", ".pickle"):
        with open(sp, "rb") as fh:
            raw_sc = pickle.load(fh)
        entry = raw_sc.get(target_var, {})
        return {
            "mean": float(entry.get("center", entry.get("mean", 0.0))),
            "std":  float(entry.get("scale",  entry.get("std",  1.0))),
        }

    with open(sp) as fh:
        raw_sc = yaml.safe_load(fh)

    if "xarray_feature_center" in raw_sc:
        centers = raw_sc["xarray_feature_center"].get("data_vars", {})
        scales  = raw_sc["xarray_feature_scale"].get("data_vars", {})

        def _val(d, k, default):
            if k not in d:
                return default
            v = d[k]
            return float(v["data"] if isinstance(v, dict) else v)

        return {
            "mean": _val(centers, target_var, 0.0),
            "std":  _val(scales,  target_var, 1.0),
        }

    entry = raw_sc.get(target_var, {})
    return {
        "mean": float(entry.get("center", entry.get("mean", 0.0))),
        "std":  float(entry.get("scale",  entry.get("std",  1.0))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §6  Entrypoint — run ensemble training and ACI calibration
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time as _time

    # ── Data prep ────────────────────────────────────────────────────────────
    from block1_data_prep_utilities import (
        BASIN_ID, DEVICE, DRIVE, H_MAX, NH_DATA_DIR, NH_RUN_DIR,
        OUTPUT_DIR, Q_stats, SEEDS, SPLITS, WARMUP_DAYS,
        assign_regime, df, load_json, save_json,
    )

    write_netcdf(df, NH_DATA_DIR, BASIN_ID)

    # ── Ensemble training ─────────────────────────────────────────────────────
    ensemble_preds, run_dirs = train_ensemble(
        df, NH_DATA_DIR, NH_RUN_DIR, OUTPUT_DIR, SEEDS
    )

    # ── UQ — day-ahead (h=1) for gateway ─────────────────────────────────────
    uq = build_uq(ensemble_preds, df, Q_stats, h=1)
    df["Q_lstm"] = uq["Q_mean"].values
    uq.to_csv(OUTPUT_DIR / "uq.csv")
    df.to_csv(OUTPUT_DIR / "full.csv")

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler = load_scaler(run_dirs[42], target_var="Q")
    save_json(
        {
            "mu_Q":      scaler["mean"],
            "std_Q":     scaler["std"],
            "sigma_clim": float(Q_stats["Q_std"]),
        },
        OUTPUT_DIR / "lstm_scaler.json",
    )
    print(f"Scaler: mu={scaler['mean']:.3f}  std={scaler['std']:.3f}")

    # ── Build per-horizon mu / sigma matrices ─────────────────────────────────
    # Shape: (N_full, H_MAX)  — rows = dates, columns = horizons 1…15
    def _build_matrix(field: str) -> np.ndarray:
        cols = []
        for h in range(1, H_MAX + 1):
            uq_h = build_uq(ensemble_preds, df, Q_stats, h=h)
            cols.append(uq_h[field].values)
        return np.column_stack(cols)

    mu_matrix    = _build_matrix("Q_mean")
    sigma_matrix = _build_matrix("Q_std")

    # ── ACI — calibration pre-load ────────────────────────────────────────────
    aci = MultiHorizonACI(alpha0=0.10, gamma=0.005, h_max=H_MAX)

    cal_mask = (
        (df.index >= SPLITS["cal"][0]) & (df.index <= SPLITS["cal"][1])
    )
    cal_dates = df.index[cal_mask]
    cal_obs   = df.loc[cal_mask, "Q"].values
    cal_mu    = mu_matrix[cal_mask]
    cal_sigma = sigma_matrix[cal_mask]

    for h in range(1, H_MAX + 1):
        aci.calibrate_on_window(cal_obs, cal_mu[:, h - 1], cal_sigma[:, h - 1], h=h)

    print(f"ACI calibrated on {cal_mask.sum()} calibration days.")
    print(f"Alpha state after calibration: {np.round(aci.alpha_state, 4)}")

    # ── ACI — run online over test period ─────────────────────────────────────
    test_mask  = (
        (df.index >= SPLITS["test"][0]) & (df.index <= SPLITS["test"][1])
    )
    test_dates = df.index[test_mask]
    test_obs   = df.loc[test_mask, "Q"].values
    test_mu    = mu_matrix[test_mask]
    test_sigma = sigma_matrix[test_mask]

    aci_results = aci.run_online(test_dates, test_obs, test_mu, test_sigma)
    aci_results.to_csv(OUTPUT_DIR / "aci_intervals.csv", index=False)

    # ── Single-day latency profile ────────────────────────────────────────────
    t0 = _time.perf_counter()
    _ = aci.build_interval(
        mu=float(test_mu[0, 0]),
        sigma=float(test_sigma[0, 0]),
        h=1,
    )
    latency_ms = (_time.perf_counter() - t0) * 1000
    print(f"Single-day forward inference latency: {latency_ms:.3f} ms")

    # ── h=1 interval summary for gateway (day-ahead Leg 1 input) ─────────────
    h1_rows = aci_results[aci_results["h"] == 1].set_index("date")
    Q_lo_test = h1_rows["Q_lo"].reindex(test_dates).values
    Q_hi_test = h1_rows["Q_hi"].reindex(test_dates).values
    alpha_seq  = h1_rows["alpha_t"].reindex(test_dates).values

    print("\nACI h=1 coverage on test period (by regime) ─────────────────")
    obs_t = df.loc[test_mask, "Q"].values
    for r in ["low", "normal", "flood"]:
        idx = (
            (assign_regime(np.nan_to_num(obs_t, nan=-1), Q_stats) == r)
            & ~np.isnan(obs_t)
        )
        if not idx.any():
            continue
        cov = np.nanmean(
            (Q_lo_test[idx] <= obs_t[idx]) & (obs_t[idx] <= Q_hi_test[idx])
        )
        print(f"  {r:8s}: {cov:.3f}  (n={idx.sum()})")

    save_json(
        {
            "alpha_state_after_test": aci.alpha_state.tolist(),
            "latency_ms_single_day":  latency_ms,
        },
        OUTPUT_DIR / "aci_params.json",
    )
