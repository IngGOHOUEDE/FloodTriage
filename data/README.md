# Data — Data_Save_1965_2011

Place the three raw hydro-met CSV files here:

```
data/Data_Save_1965_2011/
├── Precip_mm_Save_1965_2011.csv       # Daily precipitation (mm)
├── PET_mm_Save_1965_2011.csv          # Daily potential evapotranspiration (mm)
└── Discharge_cms_Save_1965_2011.csv   # Daily discharge at Savè gauge (m³/s)
```

## CSV format expected

Each file must have:
- **Column 1:** Date in any pandas-parseable format (e.g. `YYYY-MM-DD`)
- **Column 2:** The numeric value

Example (`Discharge_cms_Save_1965_2011.csv`):

```
date,Q
1965-01-01,12.4
1965-01-02,11.9
...
```

Missing values should be encoded as empty strings or spaces — the loader handles them via `na_values=["", " "]`.

## Google Drive alternative (Colab)

When running via `colab_runner.py`, the data folder is downloaded automatically from Google Drive using `gdown`. See the `DRIVE` path constant in `block1_data_prep_utilities.py`.
