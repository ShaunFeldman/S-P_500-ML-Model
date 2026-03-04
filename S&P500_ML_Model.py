import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import statsmodels.api as sm
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta
import lightgbm as lgb
import warnings
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

END_DATE   = '2026-03-3'
START_DATE = pd.to_datetime(END_DATE) - pd.DateOffset(365 * 8)

N_LIQUID_US  = 120     # top N US stocks by dollar volume each month
N_LIQUID_CA  = 30      # top N Canadian stocks by dollar volume each month
N_LIQUID     = N_LIQUID_US + N_LIQUID_CA   # total universe size
TRAIN_WARMUP = 24      # months of history before first prediction
RETRAIN_FREQ = 1       # retrain every N months (1 = monthly, 3 = quarterly)
TC_ONE_WAY   = 0.0010  # 10bps one-way transaction cost
RF_ANNUAL    = 0.04    # risk-free rate for Sharpe / Sortino
RF_MONTHLY   = RF_ANNUAL / 12

UNIVERSE_MARKETS = ['US', 'CA']   # 'US' = S&P 500,  'CA' = TSX 60

USE_PIT_UNIVERSE = True   # approximate point-in-time S&P 500 membership
USE_HMM_FILTER   = True   # risk-off during bear regime
HMM_STATES       = 3
HMM_MIN_OBS      = 36

# ─── Portfolio optimizer settings ─────────────────────────────────────────────
OPTIMIZE_PORTFOLIO = True   # Ledoit-Wolf + max-Sharpe for live portfolio
MAX_POSITION_SIZE  = 0.10   # max single-stock weight (10%)
MIN_POSITION_SIZE  = 0.00   # long-only

# ─── ETFs: risk metrics only — excluded from model training ───────────────────
PORTFOLIO_ETFS = {'VFV.TO', 'XEF.TO', 'XIC.TO', 'XEQT.TO', 'SPY', 'QQQ', 'VOO'}

# ─── YOUR CURRENT PORTFOLIO ───────────────────────────────────────────────────
# Enter ticker → weight or dollar amount (normalized automatically).
# Canadian stocks: use .TO suffix  (e.g. "RY.TO").
# ETFs: VaR/CVaR/beta computed; model score shown as "N/A (ETF)".
# Set MY_PORTFOLIO = {} to skip personal portfolio analysis.
MY_PORTFOLIO = {
    # US stocks (S&P 500)
    "GOOGL":    1391,
    "MU":        932,
    "MSFT":      348,
    "NVDA":      460,

    # Canadian stocks (TSX) — values already converted to USD at ~0.70
    "RY.TO":    1222,   # Royal Bank of Canada
    "BMO.TO":   1096,   # Bank of Montreal
    "TD.TO":     996,   # TD Bank

    # Canadian ETFs — risk metrics only, no model score
    "XEQT.TO":  3155,   # iShares Core Equity ETF Portfolio (global)
    "VFV.TO":    909,   # Vanguard S&P 500 ETF (CAD)
    "XEF.TO":    557,   # iShares MSCI EAFE (international developed)
    "XIC.TO":    428,   # iShares S&P/TSX Composite (Canadian market)
}

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
    current = tables[0].copy()
    current['Symbol'] = current['Symbol'].map(norm_ticker)
    curr_set = set(current['Symbol'].dropna())

    if len(tables) < 2:
        return {d: curr_set.copy() for d in monthly_dates}

    changes = tables[1].copy()
    date_col = find_col(changes, ['date'])
    add_col  = find_col(changes, ['added ticker', 'added'])
    rem_col  = find_col(changes, ['removed ticker', 'removed'])
    if date_col is None or (add_col is None and rem_col is None):
        return {d: curr_set.copy() for d in monthly_dates}

    change_date    = pd.to_datetime(changes[date_col], errors='coerce')
    added_ticker   = changes[add_col].map(norm_ticker) if add_col is not None else pd.Series(np.nan, index=changes.index)
    removed_ticker = changes[rem_col].map(norm_ticker) if rem_col is not None else pd.Series(np.nan, index=changes.index)

    valid   = change_date.notna()
    changes = pd.DataFrame({
        'change_date':    change_date[valid],
        'added_ticker':   added_ticker[valid],
        'removed_ticker': removed_ticker[valid],
    }).sort_values('change_date', ascending=False)

    by_month = {}
    idx = 0
    for d in sorted(monthly_dates, reverse=True):
        while idx < len(changes) and changes.iloc[idx]['change_date'] > d:
            row   = changes.iloc[idx]
            add_t = row['added_ticker']
            rem_t = row['removed_ticker']
            if isinstance(add_t, str) and add_t and add_t in curr_set:
                curr_set.remove(add_t)
            if isinstance(rem_t, str) and rem_t:
                curr_set.add(rem_t)
            idx += 1
        by_month[d] = curr_set.copy()
    return by_month

def get_tsx60_tickers():
    """
    Return TSX 60 constituent tickers in yfinance format (with .TO suffix).
    Tries Wikipedia first; falls back to a hardcoded list if scraping fails.
    """
    # Hardcoded TSX 60 (yfinance format, accurate as of 2025)
    TSX60_FALLBACK = [
        'AEM.TO', 'ATD.TO', 'BAM.TO', 'BCE.TO', 'BMO.TO', 'BN.TO', 'BNS.TO',
        'CAE.TO', 'CCL-B.TO', 'CM.TO', 'CNQ.TO', 'CNR.TO', 'CP.TO', 'CSU.TO',
        'CVE.TO', 'DOL.TO', 'ENB.TO', 'EQB.TO', 'FM.TO', 'FNV.TO', 'GIB-A.TO',
        'GWO.TO', 'H.TO', 'IFC.TO', 'IMO.TO', 'KEY.TO', 'L.TO', 'MFC.TO',
        'MG.TO', 'MRU.TO', 'NA.TO', 'NTR.TO', 'OVV.TO', 'POW.TO', 'PPL.TO',
        'QBR-B.TO', 'RCI-B.TO', 'RY.TO', 'SAP.TO', 'SLF.TO', 'SU.TO',
        'T.TO', 'TD.TO', 'TIH.TO', 'TRP.TO', 'WCN.TO', 'WN.TO', 'WSP.TO', 'X.TO',
    ]
    try:
        tbls = pd.read_html(
            "https://en.wikipedia.org/wiki/S%26P/TSX_60",
            storage_options={"User-Agent": "Mozilla/5.0"}
        )
        for t in tbls:
            sym_col = find_col(t, ['ticker symbol', 'ticker', 'symbol'])
            if sym_col is not None:
                raw = t[sym_col].dropna().astype(str).str.strip().str.upper()
                tickers = []
                for s in raw:
                    s = s.replace('.', '-')        # AGF.B → AGF-B (yfinance format)
                    if 1 <= len(s) <= 8 and s.replace('-', '').isalpha():
                        tickers.append(s + '.TO')
                if len(tickers) >= 30:
                    print(f"  Fetched {len(tickers)} TSX 60 tickers from Wikipedia.")
                    return tickers
    except Exception as e:
        print(f"  TSX 60 Wikipedia fetch failed ({e}), using hardcoded list.")
    return TSX60_FALLBACK


print("Fetching S&P 500 constituents...")
tables       = get_wiki_tables()
sp500        = tables[0]
sp500['Symbol'] = sp500['Symbol'].map(norm_ticker)
us_symbols   = sp500['Symbol'].unique().tolist()

ca_symbols = []
if 'CA' in UNIVERSE_MARKETS:
    print("Fetching TSX 60 constituents...")
    ca_symbols = get_tsx60_tickers()
    print(f"  {len(ca_symbols)} Canadian tickers added to universe.")

ca_tickers_set = set(ca_symbols)
symbols_list   = us_symbols + ca_symbols

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
print(f"Downloaded {len(df):,} daily observations across "
      f"{df.index.get_level_values('ticker').nunique()} tickers.")

# ── FX: Convert Canadian stocks from CAD → USD ────────────────────────────────
# All prices are converted to USD before any feature computation.
# This means momentum, dollar volume, and returns all reflect USD-equivalent values,
# so Canadian and US stocks are directly comparable in the cross-sectional model.
fx_daily = None
ca_in_df = [t for t in ca_tickers_set if t in df.index.get_level_values('ticker').unique()]
if ca_in_df:
    print("Downloading CAD/USD FX rate for currency normalization...")
    fx_raw   = yf.download('CADUSD=X', start=START_DATE, end=END_DATE,
                            auto_adjust=False, progress=False)
    fx_daily = fx_raw['Close'].squeeze().rename('cadusd').ffill().bfill()
    ca_mask  = df.index.get_level_values('ticker').isin(ca_in_df)
    fx_vals  = fx_daily.reindex(df.index.get_level_values('date')).values
    for col in ['close', 'open', 'high', 'low']:
        if col in df.columns:
            df.loc[ca_mask, col] = df.loc[ca_mask, col].values * fx_vals[ca_mask]
    print(f"  CAD→USD applied to {len(ca_in_df)} Canadian tickers.\n")
else:
    print()

# ══════════════════════════════════════════════════════════════════════════════
# 2. DAILY TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

print("Computing daily technical indicators...")

# Garman-Klass vol: uses O/H/L/C — more efficient than close-to-close.
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

indicator_cols = [
    'close', 'garman_klass_vol', 'rsi',
    'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd'
]

def safe_stack(frame, level='ticker', dropna=True):
    """pandas-version-safe stack (handles future_stack param change in 2.x)."""
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
    pit_members   = build_constituent_map(tables, monthly_dates)
    membership    = pd.Series(
        [
            idx_ticker.endswith('.TO')                           # CA: always included
            or idx_ticker in pit_members.get(idx_date, set())   # US: PIT filter
            for idx_date, idx_ticker in monthly.index
        ],
        index=monthly.index
    )
    monthly = monthly[membership]
    print("Applied PIT membership: S&P 500 history (US) + full TSX 60 (CA).")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LIQUIDITY FILTER
# ══════════════════════════════════════════════════════════════════════════════

def _liquidity_filter(g):
    """Per-market liquidity quota: top N_LIQUID_US (US) + top N_LIQUID_CA (CA)."""
    ca_mask = g.index.get_level_values('ticker').str.endswith('.TO')
    top_us  = g[~ca_mask].nlargest(N_LIQUID_US, 'dollar_volume')
    top_ca  = g[ca_mask].nlargest(N_LIQUID_CA, 'dollar_volume')
    return pd.concat([top_us, top_ca])

monthly = monthly.groupby(level='date', group_keys=False).apply(_liquidity_filter)
print(f"Liquid universe: ~{N_LIQUID} stocks/month (≤{N_LIQUID_US} US + ≤{N_LIQUID_CA} CA) | "
      f"{monthly.index.get_level_values('date').nunique()} months of data.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MOMENTUM & RETURN FEATURES
# ══════════════════════════════════════════════════════════════════════════════

monthly['monthly_return'] = monthly.groupby(level='ticker')['close'].pct_change()

for n in [1, 2, 3, 6, 9, 12]:
    monthly[f'ret_{n}m'] = monthly.groupby(level='ticker')['close'].pct_change(n)

# ══════════════════════════════════════════════════════════════════════════════
# 6. ROLLING MARKET BETA  (vectorized)
# ══════════════════════════════════════════════════════════════════════════════

print("Computing rolling market betas...")
spy_raw     = yf.download('SPY', start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
spy_monthly = (
    spy_raw['Close'].squeeze().resample('ME').last()
    .pct_change().rename('sp500_ret')
)

monthly    = monthly.join(spy_monthly, on='date')
stock_wide = monthly['monthly_return'].unstack(level='ticker')
mkt_ret    = spy_monthly.reindex(stock_wide.index)

rolling_cov        = stock_wide.rolling(12).cov(mkt_ret)
rolling_var        = mkt_ret.rolling(12).var()
beta_wide          = rolling_cov.div(rolling_var, axis=0)
beta_long          = safe_stack(beta_wide)
beta_long.index.names = ['date', 'ticker']
monthly['beta']    = beta_long

# ══════════════════════════════════════════════════════════════════════════════
# 7. TARGET VARIABLE  (strict no-lookahead)
# ══════════════════════════════════════════════════════════════════════════════

monthly['forward_return'] = (
    monthly.groupby(level='ticker')['monthly_return'].shift(-1)
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE MATRIX & CROSS-SECTIONAL NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = [
    'garman_klass_vol', 'atr',
    'rsi', 'bb_low', 'bb_mid', 'bb_high',
    'macd', 'ret_1m', 'ret_2m', 'ret_3m', 'ret_6m', 'ret_9m', 'ret_12m',
    'dollar_volume', 'beta',
]

data = monthly[FEATURES + ['forward_return', 'monthly_return', 'sp500_ret']].copy()
data = data.dropna(subset=['forward_return'])

# Cross-sectional rank normalization: rank within each month → [0,1] → center.
# Eliminates outliers, makes features stationary. Standard at quant shops.
# transform (not apply) preserves full MultiIndex in pandas 2.x.
print("Rank-normalizing features cross-sectionally...")
for col in FEATURES:
    data[col] = data.groupby(level='date')[col].transform(
        lambda x: x.rank(pct=True) - 0.5
    )

data['target'] = data.groupby(level='date')['forward_return'].transform(
    lambda x: x.rank(pct=True)
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. WALK-FORWARD ML TRAINING
# ══════════════════════════════════════════════════════════════════════════════

all_dates        = data.index.get_level_values('date').unique().sort_values()
effective_warmup = min(TRAIN_WARMUP, max(len(all_dates) - 1, 0))
pred_dates       = all_dates[effective_warmup:]

if len(pred_dates) == 0:
    raise RuntimeError("No prediction months available. Check data and config.")

predictions = []
model       = None

print(f"\nWalk-forward backtest:")
print(f"  Warmup   : {effective_warmup} months")
print(f"  Retrain  : every {RETRAIN_FREQ} month(s)")
print(f"  Span     : {pred_dates[0]:%Y-%m} → {pred_dates[-1]:%Y-%m}  ({len(pred_dates)} periods)\n")

for i, pred_date in enumerate(pred_dates):
    train = data.loc[data.index.get_level_values('date') < pred_date]
    test  = data.loc[data.index.get_level_values('date') == pred_date]
    if test.empty:
        continue

    if i % RETRAIN_FREQ == 0:
        X_tr, y_tr = train[FEATURES], train['target']
        valid = y_tr.notna()
        X_tr, y_tr = X_tr[valid], y_tr[valid]
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
        print(f"  [{i+1:>3}/{len(pred_dates)}]  {pred_date:%Y-%m}  | train obs: {len(X_tr):>5,}")

if not predictions:
    raise RuntimeError("No predictions generated. Check data availability.")

pred_df = pd.concat(predictions)
print("\nWalk-forward complete.\n")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (used by sections 10–16)
# ══════════════════════════════════════════════════════════════════════════════

def annualized_return(rets, freq=12):
    r = rets.dropna()
    return (1 + r).prod() ** (freq / len(r)) - 1

def sharpe(rets, freq=12):
    r      = rets.dropna()
    excess = r - RF_MONTHLY
    return excess.mean() / excess.std() * np.sqrt(freq)

def sortino(rets, freq=12):
    r    = rets.dropna()
    exc  = r - RF_MONTHLY
    down = exc[exc < 0].std()
    return exc.mean() / down * np.sqrt(freq) if down > 0 else np.inf

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

def capm_alpha(port, bench):
    common             = pd.concat([port, bench], axis=1).dropna()
    common.columns     = ['port', 'bench']
    excess_p, excess_m = common['port'] - RF_MONTHLY, common['bench'] - RF_MONTHLY
    result             = sm.OLS(excess_p, sm.add_constant(excess_m)).fit()
    return result.params['const'] * 12, result.tvalues['const'], result.params['bench']

def portfolio_var_cvar(rets, conf=0.95):
    """Historical VaR and CVaR (Expected Shortfall) at given confidence."""
    r       = rets.dropna().sort_values()
    cutoff  = int((1 - conf) * len(r))
    var     = -r.iloc[cutoff]
    cvar    = -r.iloc[:cutoff].mean()
    return var, cvar

def max_sharpe_weights(returns_df, rf=RF_MONTHLY, max_w=MAX_POSITION_SIZE, min_w=MIN_POSITION_SIZE):
    """
    Ledoit-Wolf shrinkage covariance + SLSQP max-Sharpe portfolio.
    Returns pd.Series of optimized weights indexed by ticker.
    Fallback: equal weight if optimization fails.
    """
    clean = returns_df.dropna(how='all', axis=1).dropna(how='any', axis=0)
    n     = clean.shape[1]
    if clean.shape[0] < 12 or n < 2:
        return pd.Series(np.ones(n) / n, index=clean.columns)

    lw  = LedoitWolf().fit(clean.values)
    cov = lw.covariance_
    mu  = clean.mean().values

    def neg_sharpe(w):
        port_ret = float(w @ mu) - rf
        port_vol = float(np.sqrt(w @ cov @ w))
        return -port_ret / port_vol if port_vol > 1e-10 else 0.0

    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method='SLSQP',
        bounds=[(min_w, max_w)] * n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
        options={'maxiter': 1000, 'ftol': 1e-9},
    )
    w = result.x if result.success else np.ones(n) / n
    w = np.maximum(w, 0)
    w /= w.sum()
    return pd.Series(w, index=clean.columns)

# ══════════════════════════════════════════════════════════════════════════════
# 10. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def hmm_exposure_filter(benchmark, n_states=3, min_obs=36):
    """
    Fit a Gaussian HMM on cumulative benchmark returns each month.
    Returns 1.0 (fully invested) in bull/neutral regimes, 0.0 in the
    worst-mean regime (bear market — park in cash at RF_MONTHLY).
    Using expanding window so only past data is used at each step.
    """
    if (not USE_HMM_FILTER) or (not HMM_AVAILABLE):
        if USE_HMM_FILTER and not HMM_AVAILABLE:
            print("hmmlearn not installed — skipping HMM regime filter.")
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
            hmm_model = GaussianHMM(
                n_components=n_states, covariance_type='full',
                n_iter=300, random_state=42
            )
            hmm_model.fit(obs)
            states      = hmm_model.predict(obs)
            state_means = pd.Series(obs.ravel()).groupby(states).mean()
            bear_state  = state_means.idxmin()
            exposures.append(0.0 if states[-1] == bear_state else 1.0)
        except Exception:
            exposures.append(1.0)

    return pd.Series(exposures, index=benchmark.index, name='hmm_exposure')

pred_df['quintile'] = pred_df.groupby(level='date')['predicted_rank'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
)

strategy_gross    = pred_df[pred_df['quintile'] == 4].groupby(level='date')['forward_return'].mean()
benchmark_returns = spy_monthly.reindex(strategy_gross.index).dropna()
strategy_gross    = strategy_gross.reindex(benchmark_returns.index)

print("Fitting HMM regime filter (expanding window)...")
hmm_exposure   = hmm_exposure_filter(
    benchmark_returns, n_states=HMM_STATES, min_obs=HMM_MIN_OBS
).reindex(strategy_gross.index).fillna(1.0)

strategy_gross = strategy_gross * hmm_exposure + (1 - hmm_exposure) * RF_MONTHLY

# ══════════════════════════════════════════════════════════════════════════════
# 11. TRANSACTION COST ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

top_q      = pred_df[pred_df['quintile'] == 4].copy()
top_q['w'] = top_q.groupby(level='date')['forward_return'].transform(lambda x: 1 / len(x))
weights    = top_q['w'].unstack(level='ticker').reindex(strategy_gross.index).fillna(0.0)
weights    = weights.mul(hmm_exposure, axis=0)

turnover     = weights.diff().abs().sum(axis=1).fillna(0.0) / 2.0
tc_series    = turnover * 2 * TC_ONE_WAY
strategy_net = strategy_gross - tc_series

print(f"Avg monthly turnover : {turnover.mean():.0%}")
print(f"Avg TC drag / month  : {tc_series.mean() * 10_000:.1f} bps\n")

# ══════════════════════════════════════════════════════════════════════════════
# 11b. PORTFOLIO TAIL RISK (VaR / CVaR)
# ══════════════════════════════════════════════════════════════════════════════

var95,  cvar95  = portfolio_var_cvar(strategy_net, 0.95)
var99,  cvar99  = portfolio_var_cvar(strategy_net, 0.99)
bvar95, bcvar95 = portfolio_var_cvar(benchmark_returns, 0.95)

print("══════════════ TAIL RISK  (Monthly, Historical) ══════════════")
print(f"  {'':30s}  {'ML Portfolio':>14}  {'SPY':>10}")
print(f"  {'VaR  (95%, 1-in-20 month)':30s}  {-var95:>13.1%}  {-bvar95:>9.1%}")
print(f"  {'CVaR (95%, Exp. Shortfall)':30s}  {-cvar95:>13.1%}  {-bcvar95:>9.1%}")
print(f"  {'VaR  (99%, 1-in-100 month)':30s}  {-var99:>13.1%}")
print("══════════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 12. PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

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
print(f"\nCAPM Alpha (ann.) : {alpha:.1%}  (t = {t_alpha:.2f})")
print(f"CAPM Beta         : {capm_beta:.2f}")
print("═════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 12b. FAMA-FRENCH 5-FACTOR ALPHA
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pandas_datareader.data as web

    ff5_raw = web.DataReader(
        'F-F_Research_Data_5_Factors_2x3', 'famafrench',
        start=strategy_gross.index[0], end=strategy_gross.index[-1]
    )[0] / 100

    ff5_raw.index = ff5_raw.index.to_timestamp('M')
    ff5_aligned   = ff5_raw.reindex(strategy_gross.index).dropna()
    port_excess   = strategy_gross.reindex(ff5_aligned.index) - ff5_aligned['RF']

    X_ff5      = sm.add_constant(ff5_aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
    ff5_result = sm.OLS(port_excess, X_ff5).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

    ff5_alpha_ann = ff5_result.params['const'] * 12
    ff5_alpha_t   = ff5_result.tvalues['const']
    ff5_r2        = ff5_result.rsquared

    print("══════════ FAMA-FRENCH 5-FACTOR REGRESSION ══════════════")
    print(f"  Alpha (ann.) : {ff5_alpha_ann:.2%}  (t = {ff5_alpha_t:.2f})")
    print(f"  R²           : {ff5_r2:.3f}")
    for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
        print(f"  {f:8s}  β = {ff5_result.params[f]:+.3f}  (t = {ff5_result.tvalues[f]:+.2f})")
    print("═════════════════════════════════════════════════════════\n")

except Exception as e:
    print(f"FF5 alpha unavailable: {e}\n")
    ff5_alpha_ann = np.nan
    ff5_alpha_t   = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 13. QUINTILE MONOTONICITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

print("══════════ QUINTILE CAGR  (monotonicity check) ══════════")
for q in range(5):
    q_rets = pred_df[pred_df['quintile'] == q].groupby(level='date')['forward_return'].mean()
    label  = '← bottom' if q == 0 else ('← top' if q == 4 else '')
    print(f"  Q{q+1}  {annualized_return(q_rets):>7.1%}   n={len(q_rets.dropna())}  {label}")
print(f"  SPY {annualized_return(benchmark_returns):>7.1%}  (benchmark)")
print("═════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 15. LIVE FACTOR SIGNALS  (as of END_DATE)
# ══════════════════════════════════════════════════════════════════════════════
# Uses the last trained model to score the current investable universe.
# These are the model's actual recommendations as of the backtest end date.

last_date      = pred_df.index.get_level_values('date').max()
last_preds     = pred_df.xs(last_date, level='date').copy()
last_preds_raw = monthly.xs(last_date, level='date').copy()  # un-normalized for display

# Rank scores descending (higher = stronger buy signal)
last_preds = last_preds.sort_values('predicted_rank', ascending=False)
top_n      = 30

print(f"══════════ LIVE FACTOR SIGNALS  ({last_date:%Y-%m-%d}) ══════════")
print(f"  Model's top {top_n} long candidates (by predicted return rank):\n")
print(f"  {'#':>3}  {'Ticker':<8}  {'Score':>7}  {'Quintile':>8}  "
      f"{'1m Ret':>7}  {'12m Ret':>8}  {'Beta':>6}")
print("  " + "─" * 60)

top_signals = last_preds.head(top_n)
for rank_i, (ticker, row) in enumerate(top_signals.iterrows(), 1):
    raw = last_preds_raw.loc[ticker] if ticker in last_preds_raw.index else None
    ret_1m  = f"{raw['monthly_return']:.1%}" if raw is not None and pd.notna(raw.get('monthly_return')) else '  N/A'
    ret_12m = f"{raw.get('ret_12m', np.nan):.1%}" if raw is not None and pd.notna(raw.get('ret_12m')) else '  N/A'
    beta_v  = f"{raw.get('beta', np.nan):.2f}" if raw is not None and pd.notna(raw.get('beta')) else ' N/A'
    q_label = f"Q{int(row['quintile'])+1}" if pd.notna(row['quintile']) else 'Q?'
    print(f"  {rank_i:>3}.  {ticker:<8}  {row['predicted_rank']:>7.4f}  {q_label:>8}  "
          f"{ret_1m:>7}  {ret_12m:>8}  {beta_v:>6}")

current_regime = "BULLISH" if hmm_exposure.iloc[-1] == 1.0 else "BEARISH / RISK-OFF"
print(f"\n  Current HMM Regime   : {current_regime}")
print(f"  Equity Exposure      : {hmm_exposure.iloc[-1]:.0%}")
if hmm_exposure.iloc[-1] < 1.0:
    print(f"  ⚠  Model is RISK-OFF — consider trimming equity exposure")
print("═════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 16. PORTFOLIO OPTIMIZER  (Ledoit-Wolf + Max Sharpe)
# ══════════════════════════════════════════════════════════════════════════════

all_returns = monthly['monthly_return'].unstack('ticker')

if OPTIMIZE_PORTFOLIO:
    print("══════════ OPTIMIZED LIVE PORTFOLIO ══════════════════════")
    print(f"  Method : Ledoit-Wolf shrinkage covariance + Max-Sharpe (SLSQP)")
    print(f"  Universe: top-quintile stocks as of {last_date:%Y-%m}")
    print(f"  Position limits: [{MIN_POSITION_SIZE:.0%}, {MAX_POSITION_SIZE:.0%}]\n")

    # Build return history for top-quintile stocks (last 36 months)
    top_q_tickers = last_preds[last_preds['quintile'] == 4].index.tolist()
    lookback      = 36
    ret_history   = all_returns[
        [t for t in top_q_tickers if t in all_returns.columns]
    ].iloc[-lookback:]

    opt_weights = max_sharpe_weights(ret_history)
    opt_weights = opt_weights[opt_weights > 0.001].sort_values(ascending=False)

    # Compute stats for the optimized portfolio (in-sample, for display only)
    opt_ret_hist = (ret_history[opt_weights.index] * opt_weights).sum(axis=1).dropna()
    opt_var95, opt_cvar95 = portfolio_var_cvar(opt_ret_hist, 0.95)

    print(f"  {'Ticker':<8}  {'Weight':>8}  {'Shrunk Cov Contribution':>24}")
    print("  " + "─" * 45)
    for ticker, w in opt_weights.items():
        bar = "█" * int(w * 100)
        print(f"  {ticker:<8}  {w:>8.2%}  {bar}")

    print(f"\n  Portfolio stats  (past {lookback}m, optimized weights):")
    print(f"  Ann. Return : {annualized_return(opt_ret_hist):.1%}")
    print(f"  Sharpe      : {sharpe(opt_ret_hist):.2f}")
    print(f"  VaR  (95%)  : {-opt_var95:.1%}")
    print(f"  CVaR (95%)  : {-opt_cvar95:.1%}")
    print("═════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 17. YOUR PORTFOLIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if MY_PORTFOLIO:
    # Normalize weights
    total   = sum(MY_PORTFOLIO.values())
    my_w    = {k: v / total for k, v in MY_PORTFOLIO.items()}

    print("══════════════ YOUR PORTFOLIO ANALYSIS ══════════════════")
    print(f"  Date : {last_date:%Y-%m-%d}  |  Holdings : {len(my_w)}")

    # Score each holding
    rows_p = []
    for ticker, w in sorted(my_w.items(), key=lambda x: -x[1]):
        if ticker in PORTFOLIO_ETFS:
            score, q_num, signal = np.nan, None, "N/A (ETF)"
        elif ticker in last_preds.index:
            score    = last_preds.loc[ticker, 'predicted_rank']
            q_num    = int(last_preds.loc[ticker, 'quintile']) + 1
            signal   = "↑ HOLD/ADD" if q_num == 5 else ("↓ TRIM" if q_num <= 2 else "→ NEUTRAL")
        else:
            score, q_num, signal = np.nan, None, "? NOT IN UNIVERSE"
        rows_p.append({
            'Ticker':      ticker,
            'Weight':      f"{w:.1%}",
            'Model Score': f"{score:.4f}" if pd.notna(score) else "N/A",
            'Quintile':    f"Q{q_num}" if q_num else "N/A",
            'Signal':      signal,
        })

    port_df = pd.DataFrame(rows_p).set_index('Ticker')
    print(f"\n  Holdings:\n")
    for _, row in port_df.iterrows():
        print(f"    {row.name:<8}  {row['Weight']:>6}  {row['Model Score']:>10}  "
              f"{row['Quintile']:>5}  {row['Signal']}")

    # Portfolio risk metrics
    # Download any tickers missing from the training universe (ETFs, non-S&P 500/TSX 60)
    missing = [t for t in my_w if t not in all_returns.columns]
    if missing:
        print(f"\n  Fetching data for missing tickers: {', '.join(missing)} ...")
        try:
            extra_raw = yf.download(missing, start=START_DATE, end=END_DATE,
                                    auto_adjust=True, progress=False)
            if isinstance(extra_raw.columns, pd.MultiIndex):
                extra_close = extra_raw['Close']
            else:
                extra_close = pd.DataFrame(extra_raw['Close'])
                extra_close.columns = [missing[0]] if len(missing) == 1 else missing
            extra_monthly = extra_close.resample('ME').last().pct_change()
            # FX: convert any .TO tickers from CAD returns to USD returns
            if fx_daily is not None:
                fx_monthly_ret = fx_daily.resample('ME').last().pct_change()
                for col in extra_monthly.columns:
                    t = col if isinstance(col, str) else col[0]
                    if t.endswith('.TO'):
                        fx_r = fx_monthly_ret.reindex(extra_monthly.index)
                        extra_monthly[col] = (1 + extra_monthly[col]) * (1 + fx_r) - 1
            for col in extra_monthly.columns:
                t = col if isinstance(col, str) else col[0]
                all_returns[t] = extra_monthly[col]
        except Exception as e:
            print(f"  Could not download missing tickers: {e}")

    user_tickers = [t for t in my_w if t in all_returns.columns]
    if user_tickers:
        w_arr     = np.array([my_w[t] for t in user_tickers])
        w_arr    /= w_arr.sum()
        user_rets = (all_returns[user_tickers] * w_arr).sum(axis=1).dropna()

        # Beta
        common = pd.concat([user_rets, spy_monthly], axis=1).dropna()
        if len(common) > 12:
            c        = np.cov(common.iloc[:, 0], common.iloc[:, 1])
            my_beta  = c[0, 1] / c[1, 1]
        else:
            my_beta = np.nan

        my_var95,  my_cvar95  = portfolio_var_cvar(user_rets, 0.95)
        my_var99,  my_cvar99  = portfolio_var_cvar(user_rets, 0.99)

        print(f"\n  Risk Metrics (based on {len(user_rets)} months of history):")
        print(f"    Market Beta (vs SPY)    : {my_beta:.2f}")
        print(f"    Monthly VaR  (95%)      : {-my_var95:.1%}   (1-in-20 month loss)")
        print(f"    Monthly CVaR (95%)      : {-my_cvar95:.1%}   (expected loss, worst 5%)")
        print(f"    Monthly VaR  (99%)      : {-my_var99:.1%}   (1-in-100 month loss)")
        print(f"    Historical CAGR         : {annualized_return(user_rets):.1%}")
        print(f"    Sharpe Ratio            : {sharpe(user_rets):.2f}")
        print(f"    Max Drawdown            : {max_drawdown(user_rets):.1%}")
    else:
        print("\n  (No historical data found for your tickers in the S&P 500 dataset)")

    # Regime warning
    print(f"\n  Current Regime  : {current_regime}")
    print(f"  HMM Exposure    : {hmm_exposure.iloc[-1]:.0%}")
    if hmm_exposure.iloc[-1] < 1.0:
        print(f"  ⚠  RISK-OFF — model recommends reducing equity exposure")

    # Overlap with model's top quintile
    top_q_set  = set(last_preds[last_preds['quintile'] == 4].index.tolist())
    overlap    = set(my_w.keys()) & top_q_set
    overlap_w  = sum(my_w.get(t, 0) for t in overlap)
    print(f"\n  Portfolio ∩ Model Top Quintile : {', '.join(sorted(overlap)) or 'none'}")
    print(f"  Weight in top-quintile stocks  : {overlap_w:.1%}")

    # Suggested actions
    trim_list = [t for t in my_w if t in last_preds.index and last_preds.loc[t, 'quintile'] <= 1]
    add_list  = [t for t in last_preds[last_preds['quintile'] == 4]
                 .sort_values('predicted_rank', ascending=False).index
                 if t not in my_w][:5]

    print(f"\n  Suggested Actions:")
    if trim_list:
        print(f"    REDUCE / EXIT  : {', '.join(trim_list)}  (bottom-quintile signal)")
    if add_list:
        print(f"    ADD / INITIATE : {', '.join(add_list)}  (top model signals not held)")
    if not trim_list and not add_list:
        print(f"    Portfolio is well-aligned with model signals.")
    print("═════════════════════════════════════════════════════════\n")

# ══════════════════════════════════════════════════════════════════════════════
# 14. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

n_panels = 4 if MY_PORTFOLIO and user_tickers else 3
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 6 * n_panels))
fig.suptitle(
    'Multi-Market Factor ML Portfolio (S&P 500 + TSX 60) — Performance Tearsheet',
    fontsize=15, fontweight='bold', y=0.99
)

# ── Panel 1: Cumulative returns + HMM regime shading ─────────────────────────
ax = axes[0]

cum_gross = (1 + strategy_gross).cumprod()
cum_net   = (1 + strategy_net).cumprod()
cum_bench = (1 + benchmark_returns).cumprod()

# Shade bear-regime periods (HMM exposure = 0)
bear_periods = hmm_exposure[hmm_exposure == 0.0].index
if len(bear_periods) > 0:
    in_bear = False
    bear_start = None
    regime_dates = hmm_exposure.index
    for i, d in enumerate(regime_dates):
        exp = hmm_exposure.loc[d]
        if exp == 0.0 and not in_bear:
            bear_start = d
            in_bear    = True
        elif exp > 0.0 and in_bear:
            ax.axvspan(bear_start, d, alpha=0.15, color='red', label='HMM Bear Regime' if bear_start == bear_periods[0] else '')
            in_bear = False
    if in_bear:
        ax.axvspan(bear_start, regime_dates[-1], alpha=0.15, color='red')

ax.plot(cum_gross, label='ML Portfolio (Gross)',     color='steelblue', lw=2.5)
ax.plot(cum_net,   label='ML Portfolio (Net of TC)', color='navy',       lw=2.0, linestyle='--')
ax.plot(cum_bench, label='S&P 500 (SPY)',            color='darkorange', lw=2.0, linestyle='-.')

out_mask = cum_gross.values >= cum_bench.reindex(cum_gross.index).values
ax.fill_between(cum_gross.index, cum_gross, cum_bench.reindex(cum_gross.index),
                where=out_mask,  alpha=0.10, color='green', label='Outperformance')
ax.fill_between(cum_gross.index, cum_gross, cum_bench.reindex(cum_gross.index),
                where=~out_mask, alpha=0.10, color='red',   label='Underperformance')

stats_text = (
    f"Gross CAGR: {annualized_return(strategy_gross):.1%}  |  "
    f"Net CAGR: {annualized_return(strategy_net):.1%}  |  "
    f"SPY: {annualized_return(benchmark_returns):.1%}  |  "
    f"Sharpe (gross): {sharpe(strategy_gross):.2f}  |  "
    f"CAPM α: {alpha:.1%} (t={t_alpha:.1f})"
)
ax.set_title(f'Cumulative Returns ($1 Initial)\n{stats_text}', fontsize=10)
ax.set_ylabel('Portfolio Value ($)')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.1f'))
ax.grid(alpha=0.3)

# ── Panel 2: Rolling 12-month Sharpe ─────────────────────────────────────────
ax = axes[1]

rs_port  = strategy_gross.rolling(12).mean() / strategy_gross.rolling(12).std() * np.sqrt(12)
rs_bench = benchmark_returns.rolling(12).mean() / benchmark_returns.rolling(12).std() * np.sqrt(12)

ax.plot(rs_port,  label='ML Portfolio', color='steelblue',  lw=2)
ax.plot(rs_bench, label='S&P 500 (SPY)', color='darkorange', lw=2, linestyle='--')
ax.axhline(1.0, color='gray',  lw=0.8, linestyle=':', label='Sharpe = 1')
ax.axhline(0.0, color='black', lw=0.8)
ax.fill_between(rs_port.index, rs_port.fillna(0), 0,
                where=rs_port.fillna(0) >= 0, alpha=0.12, color='green')
ax.fill_between(rs_port.index, rs_port.fillna(0), 0,
                where=rs_port.fillna(0) < 0,  alpha=0.12, color='red')
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
ax.set_xlabel('Importance (split count)')
ax.grid(alpha=0.3, axis='x')

# ── Panel 4: Your portfolio vs model vs SPY (if portfolio provided) ───────────
if MY_PORTFOLIO and user_tickers:
    ax = axes[3]

    # Align all series to the same date range
    user_cum   = (1 + user_rets.reindex(cum_net.index).dropna()).cumprod()
    model_cum  = cum_net.reindex(user_cum.index)
    bench_cum2 = cum_bench.reindex(user_cum.index)

    ax.plot(user_cum,   label='Your Portfolio',          color='mediumseagreen', lw=2.5)
    ax.plot(model_cum,  label='ML Model (Net)',           color='navy',           lw=2.0, linestyle='--')
    ax.plot(bench_cum2, label='S&P 500 (SPY)',            color='darkorange',     lw=2.0, linestyle='-.')

    if user_tickers:
        my_ann = annualized_return(user_rets)
        my_sr  = sharpe(user_rets)
        my_mdd = max_drawdown(user_rets)
        ax.set_title(
            f'Your Portfolio vs Model vs SPY\n'
            f'Your CAGR: {my_ann:.1%}  |  Sharpe: {my_sr:.2f}  |  Max DD: {my_mdd:.1%}',
            fontsize=10
        )
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.1f'))
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('portfolio_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → portfolio_performance.png")
