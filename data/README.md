# Data

The raw input tables are not published in this repository. To re-run the pipeline, place these three files in this folder:

| File | What it is | Rows | Used by |
|---|---|---|---|
| `Firm_CEO_Compensation.csv` | executive-year compensation table: pay components (`salary`, `bonus`, `stock_awards`, `option_awards`, …), role flag (`ceoann`) and `age` for each executive, firm (`gvkey`) and year | 6,039 | `build_panel.py` |
| `Firm_Information.csv` | firm-year financial table: `ASSETS`, `SALES`, `MKTVAL`, `SEQ`, `ROA`, `SALECHG`, `ASSETCHG`, `TRS1YR`, `TRS3YR`, … for each firm (`GVKEY`) and year | 11,070 | `build_panel.py` |
| `DataItems.xlsx` | data dictionary for the items in the two tables (`ItemNbr`, `ItemName`, `ItemDescription`) | — | reference only |

Both tables cover fiscal years 2011–2013 and are joined on company ID (`gvkey` / `GVKEY`) and year. With the files in place:

```bash
python build_panel.py   # -> results/panel.csv, results/missingness.csv
python run_models.py    # -> results/*.csv
```

Outputs from one full run are committed under `results/`, so the results can be inspected without the raw data. The three files are ignored by git (see `.gitignore`), so a local copy will not be published by accident.
