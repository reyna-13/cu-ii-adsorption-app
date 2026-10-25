# Cu(II) Adsorption Random Forest Demo

A small Streamlit prototype for the PBL project on predicting Cu(II) adsorption capacity from selected experimental conditions.

## Files
- `app.py` — Streamlit user interface and model pipeline
- `cu_ii_training_data.csv` — candidate training rows from the current project workbook
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Important scientific limitation
This is an exploratory demo, not a validated predictor. The candidate dataset is small, strongly imbalanced by paper/source, and still requires source-level verification. The earlier leave-one-paper-out validation gave MAE around 61 mg/g and R² around -0.01 for the shallow Random Forest. Do not use the output for engineering design or as a substitute for experimental measurements.

## Inputs
The current prototype uses:
- pH
- initial Cu(II) concentration (mg/L)
- contact time (min)
- adsorbent dose (g/L)
- adsorbent type
- synthesis method

Temperature, BET surface area, and O/C ratio are not included as inputs in this first app version.
