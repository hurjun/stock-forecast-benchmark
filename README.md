# Stock Forecast Benchmark

> Comparing **7 time-series forecasting models** on historical stock data.
> **Train:** 1962–1992 &nbsp;|&nbsp; **Test:** 1993–2000 &nbsp;|&nbsp; **Tickers:** ^GSPC, KO, IBM

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/your-notebook-link-here)

---

## Results

![Forecast Comparison](results/forecast_comparison.png)

<!-- LEADERBOARD_START -->
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
