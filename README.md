# Stock Forecast Benchmark

> Comparing **7 time-series forecasting models** on historical stock data.
> **Train:** 1962–1992 &nbsp;|&nbsp; **Test:** 1993–2000 &nbsp;|&nbsp; **Tickers:** ^GSPC, KO, IBM

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/your-notebook-link-here)

---

## Results

All models were trained on 30 years of daily closing prices (1962–1992) and evaluated on a held-out 8-year test period (1993–2000). Metrics below are **averaged across all three tickers**; lower MAE/RMSE/MAPE and higher DA are better.

![Forecast Comparison](results/forecast_comparison.png)

<!-- LEADERBOARD_START -->
| Rank | Model | MAE | RMSE | MAPE | DA | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Prophet** | **117.45** | **160.27** | **40.13%** | **49.46%** | Best overall |
| 2 | LSTM | 136.79 | 186.14 | 40.81% | 17.95% | Best deep learning |
| 3 | GRU | 139.18 | 187.71 | 60.32% | 10.63% | — |
| 4 | ARIMA | 148.01 | 196.91 | 48.12% | 3.73% | — |
| 5 | XGBoost | 148.34 | 197.18 | 49.54% | 3.75% | — |
| 6 | LightGBM | 148.56 | 197.36 | 48.64% | 8.57% | — |
| 7 | *Transformer* | *149.05* | *197.74* | *60.00%* | *20.72%* | Worst overall |
<!-- LEADERBOARD_END -->

> **Bold** = best value in that column &nbsp;|&nbsp; *Italic* = worst value &nbsp;|&nbsp; Metrics averaged across ^GSPC, KO, IBM

![Metrics Bar Chart](results/metrics_bar.png)

---

## Per-Ticker Results

<details>
<summary>^GSPC (S&P 500 Index)</summary>

| Model | MAE | RMSE | MAPE | DA |
|---|---|---|---|---|
| Prophet | 329.84 | 449.37 | 29.67% | 53.7% |
| LSTM | 389.80 | 528.81 | 35.17% | 4.7% |
| GRU | 386.14 | 523.70 | 34.93% | 8.7% |
| ARIMA | 418.74 | 556.35 | 38.57% | 1.4% |
| XGBoost | 419.35 | 556.82 | 38.66% | 0.7% |
| LightGBM | 420.32 | 557.55 | 38.79% | 0.4% |
| Transformer | 416.93 | 554.96 | 38.32% | 13.5% |

</details>

<details>
<summary>KO (Coca-Cola)</summary>

| Model | MAE | RMSE | MAPE | DA |
|---|---|---|---|---|
| Prophet | 4.36 | 5.55 | 32.16% | 48.0% |
| LSTM | 6.19 | 7.67 | 45.99% | 6.8% |
| GRU | 6.38 | 7.87 | 47.66% | 9.8% |
| ARIMA | 6.34 | 7.83 | 47.22% | 5.3% |
| XGBoost | 6.73 | 8.17 | 51.44% | 6.8% |
| LightGBM | 6.63 | 8.15 | 49.69% | 21.7% |
| Transformer | 6.51 | 7.98 | 49.10% | 24.4% |

</details>

<details>
<summary>IBM</summary>

| Model | MAE | RMSE | MAPE | DA |
|---|---|---|---|---|
| LSTM | 14.37 | 21.95 | 41.27% | 42.3% |
| Prophet | 18.15 | 25.89 | 58.57% | 46.7% |
| XGBoost | 18.94 | 26.55 | 58.53% | 3.7% |
| ARIMA | 18.94 | 26.55 | 58.55% | 4.5% |
| LightGBM | 18.74 | 26.38 | 57.44% | 3.6% |
| Transformer | 23.71 | 30.26 | 92.59% | 24.3% |
| GRU | 25.02 | 31.57 | 98.38% | 13.4% |

</details>

---

## Key Findings

**Prophet ranks first overall.** Its decomposable trend component naturally follows the sustained bull market of 1993–2000, which explains both its low RMSE and its near-random DA (~50%): the model tracks the long-run level well but does not capture daily direction.

**Tree-based models (XGBoost, LightGBM) show near-zero DA on ^GSPC (~0.4–0.7%).** Gradient-boosted trees learned mean-reversion patterns from the choppy 1962–1992 training data and consequently predict a flat or declining price trajectory during a period of strong sustained growth. This is a well-known limitation of ML models on non-stationary financial series.

**LSTM outperforms GRU on IBM, but GRU/Transformer MAPE spikes above 90%.** IBM had an unusual price trajectory in the 1990s (declining through mid-decade, then recovering sharply). Recursive prediction errors compound over 2,000 steps, causing deep-learning models to diverge on this ticker.

**Deep learning is expensive relative to its gains.** LSTM and GRU each take ~18–20 minutes per ticker on CPU vs. seconds for statistical and tree-based models, yet they rank only 2nd and 3rd — only modestly ahead of far cheaper baselines.

---

## Models & Training Time

Training times measured on Google Colab CPU (no GPU) per ticker. Deep-learning models run 50 epochs with batch size 32 on sequences of length 60.

| Model | Category | Train Time / Ticker | Description |
|---|---|---|---|
| ARIMA | Statistical | ~1 s | Autoregressive Integrated Moving Average — classic univariate baseline |
| Prophet | Statistical | ~7 s | Meta's decomposable model with trend and seasonality components |
| XGBoost | Gradient Boosting | ~5 s | Tree ensemble trained on lag and rolling-window features |
| LightGBM | Gradient Boosting | ~3 s | Fast histogram-based gradient boosting with the same feature set |
| LSTM | Deep Learning | ~18 min | Long Short-Term Memory recurrent network on normalized price sequences |
| GRU | Deep Learning | ~19 min | Gated Recurrent Unit — lighter alternative to LSTM |
| Transformer | Deep Learning | ~21 min | Self-attention encoder for sequential price data |

Total wall-clock time for the full benchmark (3 tickers × 7 models): **~3 hours on CPU**.

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
