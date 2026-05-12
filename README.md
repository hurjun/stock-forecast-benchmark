# Stock Forecast Benchmark

> Comparing **7 time-series forecasting models** on historical stock data.
> **Train:** 1962–1992 &nbsp;|&nbsp; **Test:** 1993–2000 &nbsp;|&nbsp; **Tickers:** ^GSPC, KO, IBM

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/your-notebook-link-here)

---

## Results

![Forecast Comparison](results/forecast_comparison.png)

<!-- LEADERBOARD_START -->
| Rank | Model | MAE | RMSE | MAPE | DA |
| --- | --- | --- | --- | --- | --- |
| 1 | Prophet | 117.45 | 160.27 | 40.13% | 49.46% |
| 2 | LSTM | 136.79 | 186.14 | 40.81% | 17.95% |
| 3 | GRU | 139.18 | 187.71 | 60.32% | 10.63% |
| 4 | ARIMA | 148.01 | 196.91 | 48.12% | 3.73% |
| 5 | XGBoost | 148.34 | 197.18 | 49.54% | 3.75% |
| 6 | LightGBM | 148.56 | 197.36 | 48.64% | 8.57% |
| 7 | Transformer | 149.05 | 197.74 | 60.00% | 20.72% |
<!-- LEADERBOARD_END -->

---

## Models

| Model | Category | Description |
|---|---|---|
| ARIMA | Statistical | Autoregressive Integrated Moving Average — classic univariate baseline |
| Prophet | Statistical | Meta's decomposable model with trend and seasonality components |
| XGBoost | Gradient Boosting | Tree ensemble trained on lag and rolling-window features |
| LightGBM | Gradient Boosting | Fast histogram-based gradient boosting with the same feature set |
| LSTM | Deep Learning | Long Short-Term Memory recurrent network on normalized price sequences |
| GRU | Deep Learning | Gated Recurrent Unit — lighter alternative to LSTM |
| Transformer | Deep Learning | Self-attention encoder for sequential price data |

---

## Project Structure

```
stock-forecast-benchmark/
├── config.yaml             # All hyperparameters and date ranges
├── requirements.txt
├── main.py                 # Entry point
├── data/loader.py          # yfinance download + feature engineering
├── models/                 # One file per model
├── evaluation/             # Metrics + leaderboard generation
├── visualization/          # Matplotlib plotting functions
├── results/                # Generated outputs (figures + leaderboard)
└── notebooks/              # Self-contained Kaggle notebook
```

---

## Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full benchmark (downloads data, trains all models, saves results)
python main.py
```

Results are written to `results/` and the leaderboard table above is updated automatically.

---

## Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average absolute dollar deviation |
| **RMSE** | Root Mean Squared Error — penalises large errors more heavily |
| **MAPE** | Mean Absolute Percentage Error (%) — scale-independent error |
| **DA** | Directional Accuracy (%) — fraction of days with correct up/down prediction |
