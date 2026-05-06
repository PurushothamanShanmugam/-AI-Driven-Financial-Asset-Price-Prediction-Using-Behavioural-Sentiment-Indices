# Beginner Project Guide

## Dissertation title suggestion
**AI-Driven Assessment of Financial Assets to Predict Market Value Using Behavioural Analysis**

## Problem statement
The project predicts the **next-day** and **10th-day** **High** and **Low** prices for financial assets by combining market price history with behavioural and volatility signals.

## Behavioural finance methods covered
- Investor sentiment proxies
- Fear & Greed signals
- Volatility-as-sentiment proxies (VIX, VXN, GVZ, MOVE)
- Trend-following technical indicators
- Momentum indicators
- Mean reversion signals
- Rolling standard deviation as uncertainty proxy

## Machine learning methods covered
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor

## Your 26-step mapping
1. Data sources already integrated from uploaded files.
2. Behavioural + ML research incorporated into feature set and README.
3. Preprocessing and EDA in `preprocess.py` and `analysis.py`.
4. Cleaning in `clean_dataframe()`.
5. Missing-value handling with interpolation, median, forward fill, backward fill.
6. Summary statistics exported to `outputs/reports/`.
7. Normalisation and smoothing in `features.py`.
8. First results in `outputs/all_model_metrics.csv`.
9. Hyperparameter tuning in `GridSearchCV`.
10. Another algorithm comparison built in.
11. First comparison in `outputs/model_summary.csv`.
12. Improvement via regularisation and normalisation.
13. Reorganised outputs into single summary files.
14. Essential charts only are saved in `outputs/figures/`.
15. Platform selection is Streamlit dashboard.
16. Dashboard code in `app/dashboard.py`.
17. Dashboard runs through Streamlit.
18. Model results illustrated in the app and CSV outputs.
19. Platform build completed in code structure.
20. Dashboard completion ready for local run.
21. Charts aligned with R², MAE, and RMSE.
22. Interactivity includes file upload and hover tooltips.
23. Model improvements included via tuning.
24. Cross-asset validation is implemented.
25. Previous vs new indicators can be compared in output files.
26. Side-by-side target prediction for T+1 and T+10 high/low is included.

## Data leakage controls
- No random split.
- No future information in rolling features.
- All rolling and lagged features are shifted.
- Targets are shifted forward.
- TimeSeriesSplit is used.
