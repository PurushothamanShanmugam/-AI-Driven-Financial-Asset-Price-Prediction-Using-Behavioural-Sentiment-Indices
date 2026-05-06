<div align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.56+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Active-2EA44F?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-1E3A5F?style=for-the-badge"/>

<br/>

# 📈 SentiPrice

### AI-Driven Assessment of Financial Assets to Predict Market Value

*Predicting next-day and 10-day High & Low prices using Machine Learning augmented with Behavioural Sentiment Indices*

<br/>

**MSc Artificial Intelligence for Business Intelligence**  
**University of Leicester · 2025**

**Student:** Purushothaman Shanmugam &nbsp;|&nbsp; **Supervisor:** Andrey Morozov

<br/>

[🚀 Run Dashboard](#-run-the-dashboard) · [📖 Documentation](#-project-overview) · [🧪 Tests](#-testing) · [📊 Results](#-results)

---

</div>

## 📖 Project Overview

**SentiPrice** answers one research question:

> *"How effectively do machine learning models augmented with investor sentiment data predict next-day price ranges of financial assets, in comparison to price-only baseline models?"*

Most price prediction models stop at OHLCV (Open, High, Low, Close, Volume). This project goes further — by combining classical price features with **behavioural sentiment indices** (VIX, VXN, GVZ, MOVE, Fear & Greed), training **seven ML models** side-by-side, and validating on completely unseen **2025 data**.

The result is a working AI system with an interactive Streamlit dashboard that gives a clear **green / red investment signal** for each asset.

---

## 🎯 Assets & Sentiment Indices

| Asset | Type | Sentiment Index | Volatility Level |
|-------|------|-----------------|-----------------|
| **SPY** | Broad Equity | VIX | High |
| **QQQ** | Tech Equity | VXN | Extreme |
| **GLD** | Safe-Haven Gold | GVZ | Moderate |
| **TLT** | Long-Term Bonds | MOVE | Moderate |
| **BTC_USD** | Cryptocurrency | Fear & Greed (0–100) | Extreme |

> **Note on TLT:** TLT failed on the 2025 test set due to a regime shift — the 2022 rate hike cycle pushed bond prices into territory the model was never trained on (R² ≈ −1.83). This is treated as a genuine finding, not a bug, and TLT is excluded from the live Investment Advisor.

---

## 🏗️ Project Structure

```
finsentinel/
│
├── app/
│   └── dashboard.py              # Streamlit dashboard (6 tabs)
│
├── data/
│   ├── raw/                      # Original downloaded CSVs
│   └── processed/                # Feature-engineered prepared files
│       └── {ASSET}_prepared.csv
│
├── models/                       # Trained .joblib model files
│   └── {ASSET}_{MODEL}.joblib
│
├── outputs/
│   ├── metrics/
│   │   ├── all_model_summary.csv # R², MAE, RMSE per asset/model/split
│   │   └── r2_audit.csv          # Validation audit on 2025 data
│   ├── predictions/
│   │   └── {ASSET}_all_predictions.csv
│   └── reports/
│       └── project_report.md
│
├── tests/                        # pytest test suite (150 tests)
│   ├── conftest.py
│   ├── test_data_loading.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_predictions.py
│   ├── test_pipeline.py
│   ├── test_behavioural_finance.py
│   └── test_validation.py
│
├── main.py                       # Full pipeline runner
├── generate_outputs.py           # Prediction + metrics generation
├── environment.yml               # Conda environment spec
├── requirements.txt              # pip requirements
└── README.md
```

---

## ⚙️ Setup

### Option 1 — Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/finsentinel.git
cd finsentinel

# Create and activate the environment
conda env create -f environment.yml
conda activate finance_behavior_project
```

### Option 2 — pip

```bash
pip install -r requirements.txt
```

### Requirements snapshot

```
streamlit>=1.56
scikit-learn>=1.4
pandas>=2.1
numpy>=1.26
plotly>=5.20
joblib>=1.3
matplotlib>=3.8
yfinance>=0.2
```

---

## 🚀 Run the Pipeline

### 1. Full pipeline (data → features → models → outputs)

```bash
python main.py
```

This runs all 26 steps: data loading, feature engineering, cleaning, normalisation, model training, comparison, hyperparameter tuning, and output generation.

### 2. Generate predictions only (after training)

```bash
python generate_outputs.py
```

### 3. Run the dashboard

```bash
streamlit run app/dashboard.py
```

| URL | Access |
|-----|--------|
| `http://localhost:8501` | Same machine only |
| `http://172.20.10.9:8501` | Any device on the same network |

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🏠 **Overview** | Project KPIs, top model per asset, R² summary |
| 📊 **Model Comparison** | Bar charts comparing all models across all assets |
| 🎯 **Predictions** | Actual vs predicted line charts with High–Low bands |
| 🌐 **3D Historical** | 3D scatter of historical OHLCV data |
| ⬆️ **Upload & Infer** | Upload your own CSV and get predictions instantly |
| 💡 **Investment Advisor** | Enter today's price → get a green / red trade signal |
| 📄 **Report** | Generate and download the full HTML project report |

---

## 🧠 Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Dataset loading and asset mapping |
| 2 | Behavioural finance feature integration (VIX, VXN, GVZ, MOVE, F&G) |
| 3 | Preprocessing and exploratory data analysis |
| 4 | Data cleaning and outlier handling |
| 5 | Missing-value handling (ffill, interpolation) |
| 6 | Summary statistics |
| 7 | Normalisation, z-scoring, and smoothing |
| 8 | First model results |
| 9 | Hyperparameter tuning (GridSearchCV + TimeSeriesSplit) |
| 10 | Algorithm comparison across 7 models |
| 11 | First cross-model comparison |
| 12 | Model improvement via regularisation |
| 13 | Reorganised outputs for one-glance comparison |
| 14–15 | Essential charts and platform revision |
| 16–20 | Dashboard build and completion |
| 21–22 | Chart alignment, interactivity, upload + hover |
| 23–24 | Improved models and best-model validation |
| 25–26 | Comparison with previous indicators + T+1/T+10 side-by-side |

---

## 🤖 Models

| Model | Notes |
|-------|-------|
| **Linear Regression** | Baseline reference |
| **Ridge (L2)** | Shrinks coefficients; smoother fit |
| **Lasso (L1)** | Automatic feature selection; top performer on SPY, QQQ, GLD |
| **ElasticNet (L1+L2)** | Best performer on BTC_USD |
| **Random Forest** | Non-linear; prone to overfit on crypto |
| **XGBoost** | Gradient boosting with regularisation |
| **LSTM** | Deep sequence model with 20-day lookback |

All models are trained with **TimeSeriesSplit cross-validation** and a `shift(1)` guard on every feature column to prevent forward-looking data leakage.

---

## 📈 Results

### Best model per asset (2025 unseen test data)

| Asset | Best Model | R² (Test) |
|-------|-----------|-----------|
| SPY | Lasso | **0.96** |
| QQQ | Lasso | **0.95** |
| GLD | Lasso | **0.95** |
| BTC_USD | ElasticNet | **0.74** (direction: ~77%) |
| TLT | — | **−1.83** (regime shift failure) |

### Key findings

- ✅ Sentiment-augmented models **significantly outperform** price-only baselines on equities and gold
- ✅ BTC direction predicted correctly in **~77%** of cases
- ✅ T+1 predictions are substantially stronger than T+10 across all assets
- ⚠️ TLT failed — bonds entered price territory outside the training range after 2022 rate hikes
- ⚠️ BTC magnitude remains difficult even with direction accuracy

### R² — Important note

> The project targets **R² between 0.80 and 0.95** to avoid overfitting. The code is designed to minimise leakage and regularise aggressively — but R² cannot be guaranteed before training. SPY, QQQ, and GLD consistently exceed 0.95 on out-of-sample 2025 data.

---

## 🧪 Testing

The test suite has **150 tests** across 8 test files, all of which pass with no errors.

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

| Test File | Coverage Area | Tests |
|-----------|--------------|-------|
| `test_data_loading.py` | OHLCV structure, date parsing, merge | 13 |
| `test_features.py` | Lags, sentiment, volatility, leakage | 17 |
| `test_preprocessing.py` | Cleaning, normalisation, smoothing | 15 |
| `test_models.py` | Training, metrics, regularisation | 18 |
| `test_predictions.py` | Output schema, residuals, signal logic | 14 |
| `test_pipeline.py` | End-to-end, temporal split, metrics | 14 |
| `test_behavioural_finance.py` | Fear & Greed zones, VIX regimes | 15 |
| `test_validation.py` | Unseen set, sentiment vs baseline, TLT | 12 |

---

## 🧬 Feature Engineering

Each row in the prepared dataset contains **56 features** built from:

| Category | Features |
|----------|----------|
| **Price lags** | Close, Open, High, Low lagged 1, 2, 3, 5, 10 days |
| **Return lags** | Daily % returns lagged 1, 2, 3, 5, 10 days |
| **Volatility** | Rolling 30-day σ, annualised, z-scored |
| **Sentiment — raw** | VIX, VXN, GVZ, MOVE, Fear & Greed daily close |
| **Sentiment — lagged** | 1, 2, 3, 5-day lags of each sentiment index |
| **Sentiment — smoothed** | 5-day SMA of each index |
| **Sentiment — z-scored** | 60-day rolling z-score and 252-day percentile rank |

---

## 🛡️ Data Leakage Prevention

This project takes leakage seriously. Every step is protected:

```python
# All features are lagged by at least 1 day before training
df["feature"] = df["raw_value"].shift(1)

# TimeSeriesSplit — no shuffling, strict temporal ordering
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    ...  # val_idx always > max(train_idx)
```

---

## 💡 Investment Advisor Logic

The dashboard's Investment Advisor uses the following signal logic:

```
entry_price = (predicted_high + predicted_low) / 2  ±  up to $10

if entry_price < predicted_low  → ✅ LOW RISK   (price expected to rise)
if entry_price > predicted_high → ⚠️ HIGH RISK  (price expected to fall)
else                            → 〰️ MODERATE RISK (entry inside predicted band)
```

> TLT is permanently excluded from the Advisor due to its regime-shift failure.

---

## 📚 References

- Fama, E. F. (1970). Efficient Capital Markets. *Journal of Finance*
- Shiller, R. J. (2000). *Irrational Exuberance*. Princeton University Press
- Baker, M., & Wurgler, J. (2007). Investor Sentiment in the Stock Market. *Journal of Economic Perspectives*
- CBOE VIX White Paper — <https://www.cboe.com/vix>
- CNN Fear & Greed Index — <https://money.cnn.com/data/fear-and-greed/>
- Pedregosa et al. (2011). Scikit-learn. *Journal of Machine Learning Research*

---

## 📄 Licence

University Dessertaion Project

## Author 
Purushothaman Shanmugam 
Student Id : 249057446
University of Leicester 
Leicester
United Kingdom 

---

<div align="center">

**SentiPrice** · Built as part of MSc AI for Business Intelligence  
University of Leicester · 2025

*"The model assists judgement — it doesn't replace it."*

</div>