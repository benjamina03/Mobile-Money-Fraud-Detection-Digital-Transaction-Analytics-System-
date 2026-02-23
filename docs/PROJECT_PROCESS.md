# Project Process Documentation

## 1. Problem Statement

The goal is to detect suspicious mobile money transactions without relying on labeled training data. The system must support:
- Real-time fraud scoring and blocking
- Batch fraud analysis
- Manual validation workflow
- Action recommendations after review

## 2. Objectives and System Mapping

### Objective 1: Upload a mobile money dataset for analysis
Implemented through:
- Real-Time Monitor file upload
- Batch Analysis file upload

### Objective 2: Analyze transactions against normal behavior
Implemented through:
- Unsupervised hybrid model
- Feature engineering + scaling
- Hybrid anomaly score generation

### Objective 3: Flag fraudulent activities
Implemented through:
- Threshold-based blocking logic (`hybrid_score > threshold`)
- Alerts and flagged transaction queue

### Objective 4: Validate flagged transactions through manual inspection
Implemented through:
- Investigation page review queue
- Manual labels: Confirmed Fraud / False Positive / Pending

### Objective 5: Recommend actions after manual analysis
Implemented through:
- Risk-aware recommendation rules
- Recommendations shown on Dashboard and Investigation pages

## 3. End-to-End Architecture

### 3.1 Data Layer
- Input: CSV transaction data (PaySim-like schema)
- Optional synthetic generator: `scripts/generate_sample_data.py`
- Demo data stored in `data/`

### 3.2 Preprocessing Layer (`preprocessing.py`)
Pipeline steps:
1. Feature engineering
   - Transaction velocity
   - Origin/destination balance error
   - Type one-hot encoding
   - Amount and balance ratio features
2. Feature selection with fixed model feature order
3. Scaling (`StandardScaler` by default)

### 3.3 Modeling Layer (`models.py`)
Hybrid ensemble components:
- Isolation Forest
- Autoencoder (PyTorch)
- DBSCAN

Hybrid score:
`0.4 * isolation + 0.4 * autoencoder + 0.2 * dbscan`

Calibration improvements:
- Training-time normalization bounds are saved
- Data-driven default threshold computed from training score percentile
- Models and scaler persisted in `trained_models/`

### 3.4 Application Layer (`app.py`)
Main pages:
- Dashboard
- Real-Time Monitor
- Batch Analysis
- Investigation and Reports

Visualization:
- Plotly line, scatter, histogram, pie, area, and bar charts
- White/blue design system and objective progress cards

## 4. Runtime Workflow

### Real-Time Mode
1. User uploads test CSV
2. App loads or trains models
3. Transactions are simulated one-by-one
4. Each transaction gets hybrid score + decision
5. Flagged transactions enter investigation queue

### Batch Mode
1. User uploads CSV
2. App preprocesses all records
3. Predicts anomaly decisions in bulk
4. Displays charts and downloadable anomaly file
5. Pushes anomalies into investigation queue

### Investigation Mode
1. Analyst reviews flagged records
2. Assigns manual review status
3. System updates recommendations and metrics
4. Full report can be downloaded

## 5. Project Restructure Performed

- Consolidated documentation into `docs/`
- Moved generated datasets into `data/`
- Moved utility script to `scripts/generate_sample_data.py`
- Moved test script to `tests/test_preprocessing_fix.py`
- Updated app paths to support `data/test_data.csv`

## 6. Validation and Test Checklist

### Technical checks
- `python -m py_compile app.py models.py preprocessing.py`
- `python tests/test_preprocessing_fix.py`

### Functional checks
- Login works with demo credentials
- Real-Time Monitor blocks transactions with calibrated threshold
- Batch Analysis identifies anomalies and shows Plotly outputs
- Investigation page supports manual review decisions
- Dashboard objective table reflects activity and evidence

## 7. Known Constraints

- Credentials are hardcoded for prototype purposes
- Model behavior depends on dataset quality and threshold setting
- Torch load warning may appear due upstream default `weights_only=False`

## 8. Suggested Next Steps

1. Replace hardcoded auth with secure user management.
2. Add unit and integration tests under `tests/`.
3. Add automated model evaluation report (precision/recall when labels exist).
4. Introduce model monitoring and threshold tuning UI.
5. Add database-backed audit logging for production deployment.
