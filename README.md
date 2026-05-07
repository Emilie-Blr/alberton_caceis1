# CACEIS Human Capital Intelligence Cockpit — Updated Streamlit

This package updates the previous golden-source Streamlit prototype with the following additions:

- Risk-adjusted HCV Score distribution;
- Strategic Value vs Organizational Risk snapshot in Executive Overview;
- detailed Human Capital Portfolio matrix;
- department filter with fallback logic: business group, service, entity, role family;
- average HCV score by department with a minimum threshold of 20 analytical records;
- radar chart comparing selected segment against portfolio average;
- dedicated Recommendations page;
- population share displayed in key tables instead of raw population counts;
- updated private/local enterprise AI copilot wording.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data

Place the CACEIS Excel files in the same folder as the app or enter their folder path in the sidebar.

The app uses cached CSV files from `streamlit_outputs/` when available. If you want the new department filter and score distribution to work from cache, force a rebuild once from Excel so that `valuation_view.csv` is created.

## Recommended demo settings

- Keep `FAST_MODE` enabled for initial demos.
- Use `Force rebuild from Excel` once after this update.
- Then uncheck it and use cached CSVs for faster demos.

## Governance

The app is designed for aggregated decision support. It does not display anonymous employee IDs and should not be used for individual ranking or disciplinary decisions.
