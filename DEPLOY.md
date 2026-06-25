# FloodTriage — GitHub Deployment Guide
## Step-by-step for beginners

---

## Prerequisites

- A **GitHub account** (free) → https://github.com/signup
- **Git** installed on your machine → https://git-scm.com/downloads  
  *(skip if you will use the GitHub web interface)*
- A **Google account** for Colab + Drive (free)

---

## PART 1 — Create the GitHub repository

### Option A — GitHub web interface (easiest)

1. Go to https://github.com/new
2. Fill in:
   - **Repository name:** `FloodTriage`
   - **Description:** `Probabilistic flood-forecasting with intelligent human-escalation routing`
   - **Visibility:** Public *(or Private — your choice)*
   - ✅ Check **Add a README file**
3. Click **Create repository**

### Option B — GitHub CLI (faster if you have it)

```bash
gh repo create FloodTriage --public --description "Probabilistic flood-forecasting with intelligent human-escalation routing"
```

---

## PART 2 — Upload the files

### Option A — GitHub web interface (drag and drop)

1. Open your new repository page on GitHub
2. Click **Add file → Upload files**
3. Drag and drop **all files and folders** from this package:
   ```
   README.md
   LICENSE
   requirements.txt
   .gitignore
   colab_runner.py
   src/                     ← drag the entire folder
   data/                    ← drag the entire folder (with your CSV files inside)
   outputs/
   .github/
   ```
4. Scroll down, type a commit message (e.g. `Initial commit — FloodTriage pipeline`), click **Commit changes**

> **Note:** GitHub's web uploader does not support hidden files like `.gitignore` or `.github/`. Use the Git command-line method below to include them.

### Option B — Git command line (recommended)

```bash
# 1. Clone the empty repo you just created
git clone https://github.com/YOUR_USERNAME/FloodTriage.git
cd FloodTriage

# 2. Copy all the FloodTriage files into this folder
#    (replace /path/to/FloodTriage_package with wherever you extracted the files)
cp -r /path/to/FloodTriage_package/. .

# 3. Add your data CSVs
mkdir -p data/Data_Save_1965_2011
cp /path/to/your/csvs/*.csv data/Data_Save_1965_2011/

# 4. Stage, commit and push
git add .
git commit -m "Initial commit — FloodTriage pipeline"
git push origin main
```

---

## PART 3 — Update the Colab URL in colab_runner.py

1. Open `colab_runner.py` on GitHub (click the file, then the pencil ✏️ icon)
2. Find this line near the top:
   ```python
   REPO_URL  = "https://github.com/YOUR_USERNAME/FloodTriage.git"
   ```
3. Replace `YOUR_USERNAME` with your actual GitHub username
4. Click **Commit changes**

Also update the badge URL in `README.md` the same way:
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/FloodTriage/blob/main/colab_runner.py)
```

---

## PART 4 — Add your data to Google Drive

1. Open https://drive.google.com
2. Create a folder: `MyDrive/Data_Save_1965_2011/`
3. Upload the three CSV files into that folder:
   - `Precip_mm_Save_1965_2011.csv`
   - `PET_mm_Save_1965_2011.csv`
   - `Discharge_cms_Save_1965_2011.csv`

---

## PART 5 — Run the pipeline in Colab

1. Go to your repository on GitHub
2. Click the **"Open in Colab"** badge in `README.md`  
   *(or go to https://colab.research.google.com → File → Open notebook → GitHub tab → paste your repo URL → open `colab_runner.py`)*
3. In Colab: **Runtime → Change runtime type → T4 GPU** (free, recommended)
4. Click **Runtime → Run all** (Ctrl+F9)
5. When prompted, click **Connect to Google Drive** and grant access
6. Wait ~45–90 minutes — progress is printed for each of the 5 blocks
7. All outputs appear in `MyDrive/Triage/ft_outputs/`

---

## PART 6 — Verify your deployment

After running, confirm these files exist in `MyDrive/Triage/ft_outputs/`:

```
quantiles.json          ← Q50, Q95, Q99 thresholds
splits.json             ← Train / cal / test date ranges
uq.csv                  ← Day-ahead uncertainty table
aci_intervals.csv       ← Multi-horizon prediction intervals
policy.csv              ← Test-period decision log
pipeline_summary.json   ← All metrics and thresholds
fig_*.png               ← ~15 publication-ready figures
```

If you see at least 10 `fig_*.png` files and `pipeline_summary.json` with non-zero values, the deployment is successful ✅

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: neuralhydrology` | Re-run cell 3 (pip install). Sometimes Colab needs a runtime restart after install. |
| `FileNotFoundError: Precip_mm_Save_1965_2011.csv` | Check the folder name on Drive matches `Data_Save_1965_2011` exactly (case-sensitive). |
| LSTM training takes >3 hours | Make sure you selected **T4 GPU** (not CPU). Runtime → Change runtime type. |
| `git pull` fails in Colab | Delete the `/content/FloodTriage` folder in the Colab file browser and re-run. |
| Drive not mounted | Run only cell 0 first, complete the Drive authentication, then run all. |

---

## Keeping the repo up to date

Whenever you modify a block script locally:

```bash
git add src/block3_gateway_evaluation.py   # or whichever file changed
git commit -m "Fix XGBoost feature names"
git push origin main
```

The next time you run `colab_runner.py`, it will automatically `git pull` the latest version.

---

## Repository structure reference

```
FloodTriage/
├── README.md                              ← Project overview
├── DEPLOY.md                              ← This file
├── LICENSE                                ← MIT
├── requirements.txt                       ← pip dependencies
├── .gitignore                             ← Excludes outputs, checkpoints, venvs
├── colab_runner.py                        ← Single entry-point (run this in Colab)
│
├── src/
│   ├── block1_data_prep_utilities.py      ← Data loading, features, metrics
│   ├── block2_multiHorizon_conformal.py   ← LSTM ensemble + ACI
│   ← block3_gateway_evaluation.py        ← Gateway, XGBoost, evaluation
│   ├── block4_plotting_orchestration.py   ← Figures + SHAP
│   └── block5_reporting_exports.py        ← Reports, CSVs, JSON exports
│
├── data/
│   ├── README.md                          ← Data format instructions
│   └── Data_Save_1965_2011/               ← Put your CSVs here (gitignored)
│
├── outputs/                               ← Auto-generated at runtime (gitignored)
│
└── .github/
    └── workflows/
        └── ci.yml                         ← Auto-runs lint on every push
```
