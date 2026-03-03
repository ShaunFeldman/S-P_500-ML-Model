import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

END_DATE      = '2025-09-15'
START_DATE    = pd.to_datetime(END_DATE) - pd.DateOffset(365 * 8)

N_LIQUID      = 150     # top N stocks by dollar volume to trade each month
TRAIN_WARMUP  = 24      # months of history before first prediction
RETRAIN_FREQ  = 3       # retrain every N months (1 = monthly, 3 = quarterly)
TC_ONE_WAY    = 0.0010  # 10bps one-way transaction cost (conservative for large caps)
RF_ANNUAL     = 0.04    # risk-free rate for Sharpe / Sortino

USE_PIT_UNIVERSE = True  # build approximate point-in-time membership from Wiki change log
USE_HMM_FILTER   = True  # risk regime filter on/off
HMM_STATES       = 3
HMM_MIN_OBS      = 36

LGB_PARAMS = {
    'objective':         'regression',
    'metric':            'rmse',
    'n_estimators':      300,
    'learning_rate':     0.05,
    'num_leaves':        31,
    'min_child_samples': 20,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      1,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'random_state':      42,
    'n_jobs':            -1,
    'verbose':           -1,
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def norm_ticker(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().upper().replace('.', '-')

def get_wiki_tables():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})

def find_col(df, candidates):
    lower_map = {}
    for c in df.columns:
        if isinstance(c, tuple):
            key = " ".join([str(part).strip().lower() for part in c if str(part) != 'nan'])
        else:
            key = str(c).strip().lower()
        lower_map[key] = c

    for c in candidates:
        c = c.strip().lower()
        for k, v in lower_map.items():
            if c == k or c in k:
                return v
    return None

def build_constituent_map(tables, monthly_dates):
    # Table 0 is the current constituents. We use table 1 (historical changes)
    # to approximately reconstruct monthly point-in-time members.
    current = tables[0].copy()
    current['Symbol'] = current['Symbol'].map(norm_ticker)
    curr_set = set(current['Symbol'].dropna())

    if len(tables) < 2:
        return {d: curr_set.copy() for d in monthly_dates}

    changes = tables[1].copy()
    date_col = find_col(changes, ['date'])
    add_col = find_col(changes, ['added ticker', 'added'])
    rem_col = find_col(changes, ['removed ticker', 'removed'])
    if date_col is None or (add_col is None and rem_col is None):
        return {d: curr_set.copy() for d in monthly_dates}

    # Wikipedia tables occasionally arrive with MultiIndex columns, so we avoid
    # depending on a newly-added label existing in `changes.columns`.
    change_date = pd.to_datetime(changes[date_col], errors='coerce')
    if add_col is not None:
        added_ticker = changes[add_col].map(norm_ticker)
    else:
        added_ticker = pd.Series(np.nan, index=changes.index)
    if rem_col is not None:
        removed_ticker = changes[rem_col].map(norm_ticker)
    else:
        removed_ticker = pd.Series(np.nan, index=changes.index)

    valid = change_date.notna()
    changes = pd.DataFrame({
        'change_date': change_date[valid],
        'added_ticker': added_ticker[valid],
        'removed_ticker': removed_ticker[valid],
    }).sort_values('change_date', ascending=False)

    by_month = {}
    idx = 0
    for d in sorted(monthly_dates, reverse=True):
        while idx < len(changes) and changes.iloc[idx]['change_date'] > d:
            row = changes.iloc[idx]
            add_t = row['added_ticker']
            rem_t = row['removed_ticker']
            # Reverse current-day change to roll universe backward in time.
            if isinstance(add_t, str) and add_t and add_t in curr_set:
                curr_set.remove(add_t)
            if isinstance(rem_t, str) and rem_t:
                curr_set.add(rem_t)
            idx += 1
        by_month[d] = curr_set.copy()
    return by_month

print("Fetching S&P 500 constituents...")
tables = get_wiki_tables()
sp500  = tables[0]
sp500['Symbol'] = sp500['Symbol'].map(norm_ticker)
symbols_list    = sp500['Symbol'].unique().tolist()

print(f"Downloading {len(symbols_list)} tickers from {START_DATE.date()} to {END_DATE}...")
df = yf.download(
    tickers=symbols_list,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
).stack()
df.index.names = ['date', 'ticker']
df.columns     = df.columns.str.lower()
if 'close' not in df.columns:
    raise RuntimeError("Missing close prices from yfinance download.")
print(f"Downloaded {len(df):,} daily observations across {df.index.get_level_values('ticker').nunique()} tickers.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DAILY TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("Computing daily technical indicators...")

# Garman-Klass volatility: more efficient than close-to-close vol because it
# incorporates the full O/H/L/C price path. A staple in systematic strategies.
df['garman_klass_vol'] = (
    (np.log(df['high']) - np.log(df['low']))**2 / 2
    - (2 * np.log(2) - 1) * (np.log(df['close']) - np.log(df['open']))**2
)

df['rsi'] = df.groupby(level=1)['close'].transform(
    lambda x: pandas_ta.rsi(close=x, length=20)
)

df['bb_low'] = df.groupby(level=1)['close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0]
)
df['bb_mid'] = df.groupby(level=1)['close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1]
)
df['bb_high'] = df.groupby(level=1)['close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2]
)

def compute_atr(stock_data):
    atr = pandas_ta.atr(
        high=stock_data['high'], low=stock_data['low'],
        close=stock_data['close'], length=14
    )
    return atr.sub(atr.mean()).div(atr.std())

def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())

df['atr']          = df.groupby(level=1, group_keys=False).apply(compute_atr)
df['macd']         = df.groupby(level=1, group_keys=False)['close'].apply(compute_macd)
df['dollar_volume'] = (df['close'] * df['volume']) / 1e6

# ══════════════════════════════════════════════════════════════════════════════
# 3. MONTHLY AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

print("Aggregating to monthly frequency...")

# Take the last available trading day of each calendar month for indicator
# values. Use mean dollar volume — a better liquidity estimate than point-in-time.
indicator_cols = [
    'close', 'garman_klass_vol', 'rsi',
    'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd'
]

def safe_stack(frame, level='ticker', dropna=True):
    """pandas-version-safe stack that handles the future_stack parameter."""
    try:
        return frame.stack(level=level, future_stack=True)
    except TypeError:
        return frame.stack(level=level, dropna=dropna)

monthly = safe_stack(
    df[indicator_cols].unstack(level='ticker').resample('ME').last()
)
monthly['dollar_volume'] = safe_stack(
    df[['dollar_volume']].unstack(level='ticker').resample('ME').mean()
).squeeze()

monthly.index.names = ['date', 'ticker']

if USE_PIT_UNIVERSE:
    monthly_dates = monthly.index.get_level_values('date').unique().tolist()
    pit_members = build_constituent_map(tables, monthly_dates)
    membership = pd.Series(
        [
            idx_ticker in pit_members.get(idx_date, set())
            for idx_date, idx_ticker in monthly.index
        ],
        index=monthly.index
    )
    monthly = monthly[membership]
    print("Applied approximate point-in-time S&P 500 membership from Wiki change log.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LIQUIDITY FILTER
# ══════════════════════════════════════════════════════════════════════════════

# Each month, restrict the investable universe to the top N stocks by average
# daily dollar volume. Illiquid stocks have prohibitive trading costs and
# are generally excluded from institutional portfolios.
monthly = (
    monthly
    .groupby(level='date', group_keys=False)
    .apply(lambda g: g.nlargest(N_LIQUID, 'dollar_volume'))
)

print(f"Liquid universe: ~{N_LIQUID} stocks/month | "
      f"{monthly.index.get_level_values('date').nunique()} months of data.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MOMENTUM & RETURN FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# Monthly simple returns from adjusted close.
monthly['monthly_return'] = (
    monthly.groupby(level='ticker')['close'].pct_change()
)

# Multi-horizon momentum: trailing returns capture both short-term momentum
# and long-term mean reversion dynamics.
for n in [1, 2, 3, 6, 9, 12]:
    monthly[f'ret_{n}m'] = (
        monthly.groupby(level='ticker')['close'].pct_change(n)
    )

# ══════════════════════════════════════════════════════════════════════════════
# 6. ROLLING MARKET BETA  (vectorized — no per-ticker looping)
# ══════════════════════════════════════════════════════════════════════════════

# Beta = Cov(stock, market) / Var(market) over a 12-month rolling window.
# Using SPY as the market proxy. Vectorized computation via pandas rolling
# is far faster than calling RollingOLS per ticker.

print("Computing rolling market betas...")
spy_raw     = yf.download('SPY', start=START_DATE, end=END_DATE, auto_adjust=True)
spy_monthly = (
    spy_raw['Close'].squeeze().resample('ME').last()
    .pct_change().rename('sp500_ret')
)

monthly = monthly.join(spy_monthly, on='date')

stock_wide   = monthly['monthly_return'].unstack(level='ticker')
mkt_ret      = spy_monthly.reindex(stock_wide.index)

rolling_cov  = stock_wide.rolling(12).cov(mkt_ret)
rolling_var  = mkt_ret.rolling(12).var()
beta_wide    = rolling_cov.div(rolling_var, axis=0)

beta_long             = safe_stack(beta_wide)
beta_long.index.names = ['date', 'ticker']
monthly['beta']       = beta_long

# ══════════════════════════════════════════════════════════════════════════════
# 7. TARGET VARIABLE  (strict no-lookahead construction)
# ══════════════════════════════════════════════════════════════════════════════

# Shift returns BACK by 1 month per ticker to get the NEXT month's return.
# At any point in time t, we only see features from t; the label is t+1.
# This is the single most important step to prevent data leakage.
monthly['forward_return'] = (
    monthly.groupby(level='ticker')['monthly_return'].shift(-1)
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE MATRIX & CROSS-SECTIONAL NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = [
    # Volatility / risk
    'garman_klass_vol', 'atr',
    # Mean reversion
    'rsi', 'bb_low', 'bb_mid', 'bb_high',
    # Trend / momentum
    'macd', 'ret_1m', 'ret_2m', 'ret_3m', 'ret_6m', 'ret_9m', 'ret_12m',
    # Liquidity & market sensitivity
    'dollar_volume', 'beta',
]

data = monthly[FEATURES + ['forward_return', 'monthly_return', 'sp500_ret']].copy()
data = data.dropna(subset=['forward_return'])

# Cross-sectional rank normalization: within each month, rank each feature
# from 0→1 and center at 0.  This eliminates outliers, makes features
# stationary across time, and is standard practice at systematic quant shops.
# Using transform (not apply) to guarantee the MultiIndex is fully preserved —
# apply strips the groupby level from the index in pandas 2.x.
print("Rank-normalizing features cross-sectionally...")
for col in FEATURES:
    data[col] = data.groupby(level='date')[col].transform(
        lambda x: x.rank(pct=True) - 0.5
    )

# Predict relative rank of forward returns, not raw returns.
# Predicting rank is more robust: outlier returns don't dominate the loss.
data['target'] = data.groupby(level='date')['forward_return'].transform(
    lambda x: x.rank(pct=True)
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. WALK-FORWARD ML TRAINING
# ══════════════════════════════════════════════════════════════════════════════
#
# Expanding-window walk-forward validation:
#   Iteration t: train on all months 0..t-1, predict month t, advance t.
#
# This exactly replicates live trading: the model only ever sees historical
# data. No information from the future bleeds into training. This is the
# gold standard for evaluating time-series ML models — NOT a random
# train/test split, which would massively overstate performance.

all_dates  = data.index.get_level_values('date').unique().sort_values()
if len(all_dates) == 0:
    raise RuntimeError(
        "No model-ready months after preprocessing. "
        "Check feature construction and NaN handling."
    )

effective_warmup = min(TRAIN_WARMUP, max(len(all_dates) - 1, 0))
if effective_warmup < TRAIN_WARMUP:
    print(
        f"Warning: TRAIN_WARMUP={TRAIN_WARMUP} exceeds available model months "
        f"({len(all_dates)}). Using {effective_warmup} month(s) instead."
    )

pred_dates = all_dates[effective_warmup:]
if len(pred_dates) == 0:
    raise RuntimeError(
        f"No prediction months available (model months={len(all_dates)}, "
        f"warmup={effective_warmup})."
    )

predictions = []
model = None  # will hold the last trained model for feature importance

print(f"\nWalk-forward backtest:")
print(f"  Training warmup : {effective_warmup} months")
print(f"  Retrain every   : {RETRAIN_FREQ} month(s)")
print(f"  Prediction span : {pred_dates[0].strftime('%Y-%m')} → {pred_dates[-1].strftime('%Y-%m')}")
print(f"  Total periods   : {len(pred_dates)}\n")

for i, pred_date in enumerate(pred_dates):

    train = data.loc[data.index.get_level_values('date') < pred_date]
    test  = data.loc[data.index.get_level_values('date') == pred_date]

    if test.empty:
        continue

    # Retrain on schedule; reuse last model otherwise
    if i % RETRAIN_FREQ == 0:
        X_tr = train[FEATURES]
        y_tr = train['target']

        valid   = y_tr.notna()          # LightGBM handles NaN features natively
        X_tr    = X_tr[valid]
        y_tr    = y_tr[valid]

        if len(X_tr) < 200:
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_tr, y_tr)

    if model is None:
        continue

    result                   = test.copy()
    result['predicted_rank'] = model.predict(test[FEATURES])
    predictions.append(result)

    if i % 12 == 0 or i == len(pred_dates) - 1:
        print(f"  [{i+1:>3}/{len(pred_dates)}]  {pred_date.strftime('%Y-%m')}  "
              f"| training obs: {len(X_tr):>5,}")

if not predictions:
    raise RuntimeError(
        "No predictions were generated. Check data availability and training configuration."
    )

pred_df = pd.concat(predictions)
print("\nWalk-forward complete.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 10. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def hmm_exposure_filter(benchmark, n_states=3, min_obs=36):
    if (not USE_HMM_FILTER) or (not HMM_AVAILABLE):
        if USE_HMM_FILTER and not HMM_AVAILABLE:
            print("hmmlearn not installed; skipping HMM regime filter.")
        return pd.Series(1.0, index=benchmark.index, name='hmm_exposure')

    exposures = []
    for i in range(len(benchmark)):
        if i < min_obs:
            exposures.append(1.0)
            continue

        hist = benchmark.iloc[:i + 1].dropna()
        if len(hist) < min_obs:
            exposures.append(1.0)
            continue

        obs = hist.values.reshape(-1, 1)
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=300,
                random_state=42
            )
            model.fit(obs)
            states = model.predict(obs)
            state_means = pd.Series(obs.ravel()).groupby(states).mean()
            current_state = states[-1]
            exposures.append(float(state_means.loc[current_state] > state_means.median()))
        except Exception:
            exposures.append(1.0)

    return pd.Series(exposures, index=benchmark.index, name='hmm_exposure')

# Sort stocks into quintiles by predicted return rank each month.
# Long the top quintile (Q5 ≈ top 20% = ~30 stocks).
# Equal weight within the quintile — simple and hard to beat in practice.

pred_df['quintile'] = pred_df.groupby(level='date')['predicted_rank'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
)

strategy_gross   = (
    pred_df[pred_df['quintile'] == 4]
    .groupby(level='date')['forward_return']
    .mean()
)

# Align benchmark to strategy dates (drop any gaps)
benchmark_returns = spy_monthly.reindex(strategy_gross.index).dropna()
strategy_gross    = strategy_gross.reindex(benchmark_returns.index)
hmm_exposure      = hmm_exposure_filter(
    benchmark_returns, n_states=HMM_STATES, min_obs=HMM_MIN_OBS
).reindex(strategy_gross.index).fillna(1.0)
strategy_gross    = strategy_gross * hmm_exposure + (1 - hmm_exposure) * (RF_ANNUAL / 12)

# ══════════════════════════════════════════════════════════════════════════════
# 11. TRANSACTION COST ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

# Estimate monthly portfolio turnover: what fraction of holdings change each month?
# Apply 10bps one-way (bid-ask) cost — conservative for S&P 500 large caps.

top_q = pred_df[pred_df['quintile'] == 4].copy()
top_q['w_raw'] = top_q.groupby(level='date')['forward_return'].transform(lambda x: 1 / len(x))
weights = top_q['w_raw'].unstack(level='ticker').reindex(strategy_gross.index).fillna(0.0)
weights = weights.mul(hmm_exposure, axis=0)

turnover = weights.diff().abs().sum(axis=1).fillna(0.0) / 2.0
tc_series = turnover * 2 * TC_ONE_WAY
strategy_net = strategy_gross - tc_series

print(f"Avg monthly turnover : {turnover.mean():.0%}")
print(f"Avg TC drag / month  : {tc_series.mean() * 10_000:.1f} bps\n")

# ══════════════════════════════════════════════════════════════════════════════
# 12. PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

RF_MONTHLY = RF_ANNUAL / 12

def annualized_return(rets, freq=12):
    r = rets.dropna()
    return (1 + r).prod() ** (freq / len(r)) - 1

def sharpe(rets, freq=12):
    r = rets.dropna()
    excess = r - RF_MONTHLY
    return excess.mean() / excess.std() * np.sqrt(freq)

def sortino(rets, freq=12):
    r      = rets.dropna()
    excess = r - RF_MONTHLY
    down   = excess[excess < 0].std()
    return excess.mean() / down * np.sqrt(freq) if down > 0 else np.inf

def max_drawdown(rets):
    cum  = (1 + rets.dropna()).cumprod()
    peak = cum.expanding().max()
    return ((cum - peak) / peak).min()

def calmar(rets, freq=12):
    mdd = abs(max_drawdown(rets))
    return annualized_return(rets, freq) / mdd if mdd > 0 else np.inf

def info_ratio(port, bench):
    excess = port.dropna() - bench.reindex(port.dropna().index)
    return excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else np.nan

def hit_rate(port, bench):
    excess = port.dropna() - bench.reindex(port.dropna().index)
    return (excess > 0).mean()

# CAPM alpha regression: does the strategy generate returns beyond market exposure?
def capm_alpha(port, bench):
    common   = pd.concat([port, bench], axis=1).dropna()
    common.columns = ['port', 'bench']
    excess_p = common['port']  - RF_MONTHLY
    excess_m = common['bench'] - RF_MONTHLY
    result   = sm.OLS(excess_p, sm.add_constant(excess_m)).fit()
    return result.params['const'] * 12, result.tvalues['const'], result.params['bench']

alpha, t_alpha, capm_beta = capm_alpha(strategy_gross, benchmark_returns)

rows = []
for label, rets in [('Gross', strategy_gross),
                     ('Net of TC', strategy_net),
                     ('S&P 500 (SPY)', benchmark_returns)]:
    rows.append({
        'CAGR':      f"{annualized_return(rets):.1%}",
        'Sharpe':    f"{sharpe(rets):.2f}",
        'Sortino':   f"{sortino(rets):.2f}",
        'Max DD':    f"{max_drawdown(rets):.1%}",
        'Calmar':    f"{calmar(rets):.2f}",
        'IR vs SPY': f"{info_ratio(rets, benchmark_returns):.2f}" if 'SPY' not in label else '—',
        'Hit Rate':  f"{hit_rate(rets, benchmark_returns):.0%}" if 'SPY' not in label else '—',
    })

perf = pd.DataFrame(rows, index=['Gross', 'Net of TC', 'S&P 500 (SPY)'])
print("══════════════════ PERFORMANCE SUMMARY ══════════════════")
print(perf.to_string())
print(f"\nCAPM Alpha (annualized): {alpha:.1%}  (t = {t_alpha:.2f})")
print(f"CAPM Beta:               {capm_beta:.2f}")
print("═════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 12b. FAMA-FRENCH 5-FACTOR ALPHA
# ══════════════════════════════════════════════════════════════════════════════
# CAPM alpha only controls for market exposure. The Fama-French 5-factor model
# additionally controls for size (SMB), value (HML), profitability (RMW), and
# investment (CMA) — the standard academic benchmark for systematic strategies.
# Alpha that survives FF5 is genuinely uncorrelated with known risk premia.

try:
    import pandas_datareader.data as web

    ff5_raw = web.DataReader(
        'F-F_Research_Data_5_Factors_2x3', 'famafrench',
        start=strategy_gross.index[0], end=strategy_gross.index[-1]
    )[0] / 100  # convert from percent to decimal

    ff5_raw.index = ff5_raw.index.to_timestamp('M')
    ff5_aligned   = ff5_raw.reindex(strategy_gross.index).dropna()
    port_excess   = strategy_gross.reindex(ff5_aligned.index) - ff5_aligned['RF']

    X_ff5      = sm.add_constant(ff5_aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
    ff5_result = sm.OLS(port_excess, X_ff5).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

    ff5_alpha_ann = ff5_result.params['const'] * 12
    ff5_alpha_t   = ff5_result.tvalues['const']
    ff5_r2        = ff5_result.rsquared

    print("══════════ FAMA-FRENCH 5-FACTOR REGRESSION ══════════════")
    print(f"  Alpha (annualized) : {ff5_alpha_ann:.2%}  (t = {ff5_alpha_t:.2f})")
    print(f"  R²                 : {ff5_r2:.3f}")
    print(f"  Factor loadings:")
    for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
        print(f"    {f:8s}  β = {ff5_result.params[f]:+.3f}  "
              f"(t = {ff5_result.tvalues[f]:+.2f})")
    print("═════════════════════════════════════════════════════════\n")

except Exception as e:
    print(f"FF5 alpha unavailable: {e}\n")
    ff5_alpha_ann = np.nan
    ff5_alpha_t   = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 13. QUINTILE MONOTONICITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
# A well-specified factor model should show monotonically increasing returns
# from Q1 (bottom) to Q5 (top). This confirms the model has genuine
# predictive power, not just noise.

print("══════════ QUINTILE CAGR  (monotonicity check) ══════════")
for q in range(5):
    q_rets = (
        pred_df[pred_df['quintile'] == q]
        .groupby(level='date')['forward_return']
        .mean()
    )
    n      = len(q_rets.dropna())
    label  = '← bottom' if q == 0 else ('← top   ' if q == 4 else '        ')
    print(f"  Q{q+1}  {annualized_return(q_rets):>8.1%}   n={n}  {label}")
print(f"  SPY {annualized_return(benchmark_returns):>8.1%}  (benchmark)")
print("═════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 14. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(14, 18))
fig.suptitle(
    'S&P 500 Factor ML Portfolio — Performance Tearsheet',
    fontsize=15, fontweight='bold', y=0.99
)

# ── Panel 1: Cumulative returns ───────────────────────────────────────────────
ax = axes[0]

cum_gross = (1 + strategy_gross).cumprod()
cum_net   = (1 + strategy_net).cumprod()
cum_bench = (1 + benchmark_returns).cumprod()

ax.plot(cum_gross, label='ML Portfolio (Gross)',     color='steelblue',  lw=2.5)
ax.plot(cum_net,   label='ML Portfolio (Net of TC)', color='navy',        lw=2.0, linestyle='--')
ax.plot(cum_bench, label='S&P 500 (SPY)',            color='darkorange',  lw=2.0, linestyle='-.')

out_mask = cum_gross.values >= cum_bench.reindex(cum_gross.index).values
ax.fill_between(
    cum_gross.index, cum_gross, cum_bench.reindex(cum_gross.index),
    where=out_mask,  alpha=0.12, color='green', label='Outperformance'
)
ax.fill_between(
    cum_gross.index, cum_gross, cum_bench.reindex(cum_gross.index),
    where=~out_mask, alpha=0.12, color='red',   label='Underperformance'
)

stats_text = (
    f"Gross CAGR: {annualized_return(strategy_gross):.1%}  |  "
    f"Net CAGR: {annualized_return(strategy_net):.1%}  |  "
    f"SPY CAGR: {annualized_return(benchmark_returns):.1%}  |  "
    f"Sharpe (gross): {sharpe(strategy_gross):.2f}  |  "
    f"CAPM α: {alpha:.1%} (t={t_alpha:.1f})"
)
ax.set_title(f'Cumulative Returns ($1 Initial)\n{stats_text}', fontsize=10)
ax.set_ylabel('Portfolio Value ($)')
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.1f'))
ax.grid(alpha=0.3)

# ── Panel 2: Rolling 12-month Sharpe ratio ────────────────────────────────────
ax = axes[1]

rs_port  = (strategy_gross.rolling(12).mean()
            / strategy_gross.rolling(12).std() * np.sqrt(12))
rs_bench = (benchmark_returns.rolling(12).mean()
            / benchmark_returns.rolling(12).std() * np.sqrt(12))

ax.plot(rs_port,  label='ML Portfolio', color='steelblue',  lw=2)
ax.plot(rs_bench, label='S&P 500 (SPY)', color='darkorange', lw=2, linestyle='--')
ax.axhline(1.0,  color='gray',  lw=0.8, linestyle=':', label='Sharpe = 1')
ax.axhline(0.0,  color='black', lw=0.8)

ax.fill_between(
    rs_port.index, rs_port.fillna(0), 0,
    where=rs_port.fillna(0) >= 0, alpha=0.15, color='green'
)
ax.fill_between(
    rs_port.index, rs_port.fillna(0), 0,
    where=rs_port.fillna(0) < 0,  alpha=0.15, color='red'
)

ax.set_title('Rolling 12-Month Sharpe Ratio', fontsize=11)
ax.set_ylabel('Sharpe Ratio')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)

# ── Panel 3: LightGBM feature importance ─────────────────────────────────────
ax = axes[2]

imp        = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
bar_colors = ['steelblue' if v >= imp.median() else 'lightsteelblue' for v in imp.values]
ax.barh(imp.index, imp.values, color=bar_colors, edgecolor='white', height=0.7)
ax.set_title('LightGBM Feature Importance  (last trained model)', fontsize=11)
ax.set_xlabel('Importance Score (split count)')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('portfolio_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → portfolio_performance.png")
