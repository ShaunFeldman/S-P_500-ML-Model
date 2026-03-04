# S&P 500 Cross-Sectional Factor ML Portfolio

A full systematic equity strategy built from scratch: raw price data → feature engineering → LightGBM cross-sectional return prediction → walk-forward backtest → HMM risk overlay → Ledoit-Wolf portfolio optimization → personal portfolio analyzer.

This is not a toy backtest. Every design decision mirrors how quantitative equity funds actually operate — from point-in-time universe construction to turnover-adjusted transaction costs to Newey-West HAC standard errors on the alpha regression. It also generates **live factor signals** you can act on today, and analyzes your own holdings.

---

## Results (2017–2025, 8-year backtest)

![Performance Tearsheet](portfolio_performance.png)

### Performance Summary

| Metric | ML Portfolio (Gross) | ML Portfolio (Net of TC) | S&P 500 (SPY) |
|---|---|---|---|
| CAGR | **37.2%** | **35.3%** | 15.8% |
| Sharpe Ratio | **1.49** | **1.42** | 0.71 |
| Sortino Ratio | **2.68** | **2.52** | 1.09 |
| Max Drawdown | **-16.4%** | **-16.7%** | -23.9% |
| Calmar Ratio | **2.27** | **2.12** | 0.66 |
| Info Ratio vs SPY | **0.67** | **0.62** | — |
| Monthly Hit Rate | 51% | 51% | — |

### Factor Alpha (vs Known Risk Premia)

| Model | Alpha (Ann.) | t-stat | Significance |
|---|---|---|---|
| CAPM (1-factor) | **+29.9%** | 3.56 | ✓ Significant at 1% |
| Fama-French 5-factor | **+31.7%** | 4.05 | ✓ Significant at 1% |

FF5 alpha of +31.7% (t=4.05) confirms the strategy generates returns that **cannot be explained by market, size, value, profitability, or investment risk premia** — a stringent academic benchmark.

### Tail Risk

| Metric | ML Portfolio | S&P 500 (SPY) |
|---|---|---|
| Monthly VaR (95%) | -7.1% | -8.2% |
| Monthly CVaR (95%) | -8.9% | -10.2% |
| Monthly VaR (99%) | -10.3% | — |

### Quintile CAGR (Monotonicity Check)

| Quintile | CAGR | Interpretation |
|---|---|---|
| Q1 (bottom 20%) | 10.4% | ← model correctly identifies weakest stocks |
| Q2 | 18.3% | |
| Q3 | 16.2% | |
| Q4 | 26.7% | |
| Q5 (top 20%) | **47.1%** | ← model correctly identifies strongest stocks |
| S&P 500 (SPY) | 15.8% | benchmark |

Clear Q1 → Q5 spread (10.4% → 47.1%) confirms genuine predictive signal, not noise.

---

## Strategy Overview

The core idea is **cross-sectional return prediction**: instead of predicting whether the market goes up or down (nearly impossible), predict which stocks will outperform *relative to each other* each month. This is a tractable, well-studied problem in quantitative finance.

Each month:
1. Restrict the universe to the **150 most liquid S&P 500 stocks** by dollar volume
2. Compute **15 engineered features** per stock (momentum, volatility, mean-reversion signals, market beta)
3. **Rank-normalize** all features cross-sectionally to remove outliers and non-stationarity
4. Use a **LightGBM model** (trained on all prior months only) to predict each stock's return rank
5. Go **long the top quintile** (~30 stocks), rebalance monthly
6. Apply an **HMM regime filter** — move to cash during bear market states
7. In the live module: **Ledoit-Wolf + max-Sharpe optimizer** computes optimal position sizing
8. **Personal portfolio analyzer**: score your current holdings, get VaR/CVaR/beta, flag weak positions

---

## Pipeline

```
Wikipedia S&P 500 list + historical change log
        │
        ▼
yfinance 8yr daily OHLCV (502 tickers)
        │
        ▼
Daily Indicators: Garman-Klass Vol · RSI · Bollinger Bands · ATR · MACD · Dollar Volume
        │
        ▼
Monthly Aggregation (resample to month-end)
        │
        ▼
Point-in-Time Universe Filter (no survivorship bias)
        │
        ▼
Liquidity Filter (top 150 by avg daily dollar volume)
        │
        ▼
Return Features: monthly returns · 1/2/3/6/9/12m momentum
        │
        ▼
Rolling Market Beta (vectorized Cov/Var, 12m window)
        │
        ▼
Cross-Sectional Rank Normalization ([-0.5, 0.5] each month)
        │
        ▼
Walk-Forward LightGBM (expanding window, 24-month warmup, monthly retrain)
        │
        ▼
Quintile Sort → Long Q5 (top 20%) → HMM Regime Filter
        │
        ▼
Transaction Cost Deduction (weight-based turnover × 10bps × 2)
        │                                   │
        ▼                                   ▼
Backtest Tearsheet                  Live Signals + Portfolio Optimizer
CAGR · Sharpe · Sortino · Max DD    Ledoit-Wolf Cov + Max-Sharpe (SLSQP)
Calmar · IR · CAPM α · FF5 α        Personal Portfolio Analysis (VaR/CVaR/Beta)
```

---

## Features (15 total)

| Feature | Category | Description |
|---|---|---|
| `garman_klass_vol` | Volatility | O/H/L/C volatility — more efficient than close-to-close |
| `atr` | Volatility | Average True Range (14-day), z-score normalized |
| `rsi` | Mean-Reversion | Relative Strength Index (20-day) |
| `bb_low` / `bb_mid` / `bb_high` | Mean-Reversion | Bollinger Band levels (20-day, log price) |
| `macd` | Trend | MACD signal (20-day), z-score normalized |
| `ret_1m` | Momentum | 1-month trailing return (short-term reversal) |
| `ret_2m` / `ret_3m` | Momentum | 2–3 month trailing returns |
| `ret_6m` / `ret_9m` / `ret_12m` | Momentum | Intermediate/long-term momentum |
| `dollar_volume` | Liquidity | Avg daily dollar volume (millions) |
| `beta` | Market Sensitivity | Rolling 12-month CAPM beta vs. SPY |

---

## Key Methodological Details

### No Lookahead Bias
The target (next month's return) is constructed by shifting each ticker's return series back by one period within the ticker group. The walk-forward loop enforces this at the model level: the model trained for month `t+1` has never seen any data from `t+1` or later.

### Point-in-Time Universe (Survivorship Bias Elimination)
A naive backtest using today's S&P 500 list only includes stocks that *survived* to the present — excluding every company that was delisted, merged, or went bankrupt. This inflates historical returns. The project reconstructs the monthly constituent list by walking Wikipedia's historical change log backwards.

### Cross-Sectional Rank Normalization
Each feature is rank-transformed cross-sectionally within each month and centered at zero. This eliminates outliers, removes non-stationarity across time, and is standard practice at systematic equity funds.

### Walk-Forward Validation
Expanding training window: to predict month `t`, train on months `1..t-1`. This exactly replicates live trading. A random 80/20 split would introduce look-ahead bias and massively overstate performance.

### HMM Regime Filter
A 3-state Gaussian Hidden Markov Model is fit to the expanding history of SPY monthly returns. When the current state is classified as a bear regime (lowest-mean state), the strategy moves to cash. Re-estimated monthly using only past data.

### Ledoit-Wolf Portfolio Optimization
For live portfolio recommendations, uses Oracle Approximating Shrinkage (Ledoit-Wolf) to estimate the covariance matrix — significantly more stable than the sample covariance with 30 assets and 36 months of data. Max-Sharpe weights are then found via SLSQP constrained optimization (position limits: 0–10% per stock).

### Fama-French 5-Factor Alpha
CAPM alpha only controls for market beta. FF5 additionally controls for size (SMB), value (HML), profitability (RMW), and investment (CMA). Alpha that survives FF5 is genuinely uncorrelated with known systematic risk premia. Standard errors are Newey-West HAC-corrected (3 lags).

### Personal Portfolio Analyzer
Add your own holdings to `MY_PORTFOLIO` at the top of the file. The system scores each position against the current factor model signals, computes historical VaR/CVaR/beta, identifies positions with weak model signals, and suggests additions from the top factor signals not currently held.

---

## Installation

```bash
# Clone
git clone https://github.com/ShaunFeldman/S-P_500-ML-Model.git
cd S-P_500-ML-Model

# Install dependencies
pip install yfinance pandas pandas_ta lightgbm statsmodels pandas_datareader \
            matplotlib hmmlearn scikit-learn scipy

# macOS only: LightGBM requires OpenMP
brew install libomp
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical computation |
| `yfinance` | ≥ 0.2 | Market data download |
| `pandas_ta` | ≥ 0.3 | Technical indicators |
| `lightgbm` | ≥ 4.0 | Gradient boosting model |
| `statsmodels` | ≥ 0.14 | OLS regression (CAPM/FF5) |
| `pandas_datareader` | ≥ 0.10 | Fama-French factor data |
| `hmmlearn` | ≥ 0.3 | HMM regime classification |
| `scikit-learn` | ≥ 1.3 | Ledoit-Wolf covariance shrinkage |
| `scipy` | ≥ 1.11 | Max-Sharpe SLSQP optimization |
| `matplotlib` | ≥ 3.7 | Visualization |

---

## Usage

```bash
python S\&P500_ML_Model.py
```

Runtime: ~20–40 minutes (dominated by walk-forward retraining). Set `RETRAIN_FREQ = 3` in config for faster quarterly retraining.

### Configuration (top of file)

```python
END_DATE           = '2025-09-15'  # backtest end date
N_LIQUID           = 150           # liquid universe size per month
TRAIN_WARMUP       = 24            # months of warmup before first prediction
RETRAIN_FREQ       = 1             # retrain every N months (1 = monthly)
TC_ONE_WAY         = 0.0010        # 10bps one-way transaction cost
RF_ANNUAL          = 0.04          # risk-free rate
USE_PIT_UNIVERSE   = True          # point-in-time S&P 500 membership
USE_HMM_FILTER     = True          # regime-based risk overlay
OPTIMIZE_PORTFOLIO = True          # Ledoit-Wolf + max-Sharpe live portfolio
MAX_POSITION_SIZE  = 0.10          # max 10% single-stock weight

# Enter your current holdings (weights or dollar amounts):
MY_PORTFOLIO = {
    "AAPL": 0.20,
    "MSFT": 0.15,
    # ...
}
```

---

## Output

Running the script produces:

**Console:**
- Full backtest performance table (CAGR, Sharpe, Sortino, Max DD, Calmar, IR, Hit Rate)
- CAPM alpha with t-statistic
- Fama-French 5-factor regression with all loadings (HAC standard errors)
- Tail risk table (VaR/CVaR at 95% and 99%)
- Quintile CAGR monotonicity check
- **Live factor signals**: top 30 stocks ranked by model score as of backtest end
- **Optimized portfolio**: Ledoit-Wolf + max-Sharpe weights with risk stats
- **Your portfolio analysis** (if `MY_PORTFOLIO` filled in): score, quintile, signal, VaR/CVaR/beta, suggested trades

**File:**
- `portfolio_performance.png` — multi-panel tearsheet: cumulative returns (with HMM bear-regime shading), rolling 12-month Sharpe, LightGBM feature importance, personal portfolio comparison (if applicable)

---

## Project Structure

```
S-P_500-ML-Model/
├── S&P500_ML_Model.py        # Full pipeline (17 sections, ~1,000 lines)
├── portfolio_performance.png # Generated tearsheet
└── README.md
```

---

## References

- Fama & French (2015). "A five-factor asset pricing model." *Journal of Financial Economics.*
- Gu, Kelly & Xiu (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies.*
- Garman & Klass (1980). "On the Estimation of Security Price Volatilities from Historical Data." *Journal of Business.*
- Ledoit & Wolf (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis.*
- Ang & Bekaert (2002). "Regime Switches in Interest Rates." *Journal of Business & Economic Statistics.*
