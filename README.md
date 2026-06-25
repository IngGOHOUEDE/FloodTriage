# FloodTriage 🌊

**Probabilistic flood-forecasting with intelligent human-escalation routing**  
*Ouémé at Savè, Benin | Daily discharge 1965–2011*

---

## What this does

FloodTriage is a research pipeline that predicts daily river discharge and decides—automatically—whether the forecast is reliable enough to act on, or whether it should be escalated to a human operator.

The system chains three technologies:

1. **LSTM Deep Ensemble** — 5 NeuralHydrology models trained with different seeds produce a probabilistic streamflow forecast at up to 15-day lead times.
2. **Multi-Horizon Adaptive Conformal Inference (ACI)** — calibrates prediction intervals per lead time so that coverage guarantees remain valid even as the data distribution drifts.
3. **Tri-Fold Routing Gateway** — three sequential safeguards decide whether to trust the forecast or defer to a human:
   - *Leg 1 (Epistemic)* — ensemble disagreement too large (`w_t > τ_regime`)
   - *Leg 2 (Aleatoric)* — inherent data uncertainty too high (`uale_t ≥ τ_ale`)
   - *Leg 3 (Threshold)* — raw LSTM output already near the Q95 flood threshold
   
   On deferred days an **XGBoost surrogate** corrects the LSTM bias before the alert is issued.

The cost asymmetry is explicit: one missed flood costs 10× a false escalation (`COST_RATIO = 10`).

---

## Repository layout

```
FloodTriage/
├── README.md
├── colab_runner.py          ← Single-click Colab entry-point  ✅ start here
├── requirements.txt         ← Pip dependencies
├── .gitignore
│
├── src/                     ← Modular pipeline blocks
│   ├── block1_data_prep_utilities.py     # Data loading, feature engineering, metrics
│   ├── block2_multiHorizon_conformal.py  # LSTM ensemble + ACI
│   ├── block3_gateway_evaluation.py      # Gateway, XGBoost, evaluation
│   ├── block4_plotting_orchestration.py  # All figures + SHAP plots
│   └── block5_reporting_exports.py       # Reports, CSVs, JSON exports
│
├── data/
│   └── Data_Save_1965_2011/             ← Raw hydro-met CSVs  (add yours here)
│       ├── Precip_mm_Save_1965_2011.csv
│       ├── PET_mm_Save_1965_2011.csv
│       └── Discharge_cms_Save_1965_2011.csv
│
└── outputs/                 ← Auto-created at runtime (figures, CSVs, JSONs)
```

---

## Quick start — Google Colab (recommended for beginners)

**Step 1 — Upload your data to Google Drive**

Create a folder called `Data_Save_1965_2011` inside `MyDrive/` and upload the three CSV files:
- `Precip_mm_Save_1965_2011.csv`
- `PET_mm_Save_1965_2011.csv`
- `Discharge_cms_Save_1965_2011.csv`

**Step 2 — Open the notebook in Colab**

Click the badge below (or copy the URL manually):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IngGOHOUEDE/FloodTriage/blob/main/colab_runner.py)

> Replace `IngGOHOUEDE` with your GitHub username after pushing the repo.

**Step 3 — Run all cells**

`Runtime → Run all` — the pipeline will:
- Mount Google Drive
- Install all dependencies
- Run Blocks 1 → 5 sequentially
- Save all figures and artefacts to `MyDrive/Triage/ft_outputs/`

**Estimated runtime:** ~45–90 min on a free T4 GPU runtime (LSTM training dominates).

---

## Local / manual setup

```bash
# 1. Clone
git clone https://github.com/IngGOHOUEDE/FloodTriage.git
cd FloodTriage

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Put raw CSVs in data/Data_Save_1965_2011/

# 5. Run block by block (or use colab_runner.py as a script template)
python src/block1_data_prep_utilities.py
```

> **Note:** LSTM training requires a CUDA-capable GPU for reasonable speed. CPU fallback works but is ~10× slower.

---

## Key outputs

| File | Description |
|---|---|
| `outputs/uq.csv` | Day-ahead uncertainty quantification table |
| `outputs/aci_intervals.csv` | Per-horizon ACI prediction intervals |
| `outputs/policy.csv` | Full test-period decision log (defer/autonomous + reason) |
| `outputs/pipeline_summary.json` | All thresholds, coverage metrics, McNemar p-value |
| `outputs/fig_*.png` | ~15 publication-ready figures |
| `outputs/quantiles.json` | Training-period Q50 / Q95 / Q99 thresholds |

---

## Dependencies

Core stack: `PyTorch`, `NeuralHydrology`, `XGBoost`, `SHAP`, `statsmodels`, `xarray`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

See `requirements.txt` for pinned versions.

---

## Data splits

| Split | Period | Purpose |
|---|---|---|
| Train | 1965-01-01 – 1999-12-31 | LSTM + XGBoost fitting |
| *(gap)* | 2000-01-01 – 2001-12-31 | Prevents hidden-state leakage |
| Calibration | 2002-01-01 – 2007-12-31 | ACI calibration, threshold tuning |
| Test | 2008-01-01 – 2011-12-31 | Held-out evaluation |

---

## References

- Xu & Xie (2021). *Conformal Prediction for Time Series (EnbPI).* ICML.
- Zaffran et al. (2022). *Adaptive Conformal Predictions for Time Series.* ICML.
- Sun et al. (2022). *Copula Conformal Prediction for Multi-step Time Series.* ICLR.
- Elkan (2001). *The foundations of cost-sensitive learning.* IJCAI.
- Cortes et al. (2016). *Learning with rejection.* ALT.

---

## License

MIT — see `LICENSE`.
