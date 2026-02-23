# Mobile Money Fraud Detection

Hybrid unsupervised fraud detection prototype for mobile money transactions.

This project combines:
- Isolation Forest
- Autoencoder (PyTorch)
- DBSCAN

The app provides:
- Real-time transaction simulation and blocking
- Batch anomaly analysis
- Investigation workflow with manual review
- Objective tracking dashboard aligned to final-year project objectives

## Project Structure

```text
MMF-Detection-RUN2/
|-- app.py
|-- models.py
|-- preprocessing.py
|-- requirements.txt
|-- scripts/
|   `-- generate_sample_data.py
|-- tests/
|   `-- test_preprocessing_fix.py
|-- docs/
|   |-- PROJECT_PROCESS.md
|   |-- GETTING_STARTED.md
|   |-- QUICK_START.md
|   `-- DASHBOARD_UPDATES.md
|-- data/
|   |-- train_data.csv
|   |-- test_data.csv
|   `-- demo_data.csv
`-- trained_models/
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Regenerate synthetic data:

```bash
python scripts/generate_sample_data.py
```

3. Run Streamlit app:

```bash
streamlit run app.py
```

4. Login:
- Username: `admin`
- Password: `admin123`

## Data Requirements

Required columns:
- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`

Optional columns:
- `isFraud`
- `isFlaggedFraud`

## Testing

Run preprocessing smoke test:

```bash
python tests/test_preprocessing_fix.py
```

## Notes

- `trained_models/` stores saved models and scaler.
- `data/` contains local synthetic datasets for demo/testing.
- Main technical/process documentation is in `docs/PROJECT_PROCESS.md`.
