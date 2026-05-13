| Rank | Model | MAE | RMSE | MAPE | DA | Train Time / Ticker |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Prophet | 117.45 | 160.27 | 40.13% | 49.41% | ~7 s |
| 2 | LSTM | 136.79 | 186.14 | 40.81% | 18.03% | ~16 min |
| 3 | GRU | 139.18 | 187.71 | 60.32% | 12.66% | ~18 min |
| 4 | ETS | 147.98 | 196.88 | 47.96% | 20.04% | < 1 s |
| 5 | ARIMA | 148.01 | 196.91 | 48.12% | 3.71% | ~2 s |
| 6 | XGBoost | 148.34 | 197.18 | 49.55% | 3.85% | ~5 s |
| 7 | LightGBM | 148.49 | 197.27 | 48.15% | 9.11% | ~3 s |
| 8 | Transformer | 149.01 | 197.73 | 59.42% | 22.62% | ~21 min |
| 9 | TCN | 212.09 | 251.54 | 64.72% | 15.62% | ~7 min |

Measured on Google Colab CPU (no GPU). Total benchmark: ~3.5 hours (3 tickers × 9 models).
