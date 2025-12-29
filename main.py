import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import logging

# --- Configuration matching main.py ---
PLANNER_PARAMS = {
    "S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13, 
    "S2_SMA": 400, "S2_STOP": 0.27, "S2_PROX": 0.05
}

TUMBLER_PARAMS = {
    "SMA1": 32, "SMA2": 114, 
    "STOP": 0.043, "TAKE_PROFIT": 0.126, 
    "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077, 
    "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]
}

GAINER_PARAMS = {
    "GA_WEIGHTS": {"MACD_1H": 0.8, "MACD_1D": 0.4, "SMA_1D": 0.4},
    "MACD_1H": {
        'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 
        'weights': [0.45, 0.43, 0.01]
    },
    "MACD_1D": {
        'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 
        'weights': [0.87, 0.92, 0.73]
    },
    "SMA_1D": {
        'params': [40, 120, 390], 
        'weights': [0.6, 0.8, 0.4]
    }
}

# Normalization
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0
CAP_SPLIT = 0.333

class DataManager:
    @staticmethod
    def fetch_binance_data(symbol="BTC/USDT", timeframe="1h", years=8):
        """Fetches OHLCV data from Binance via CCXT with caching."""
        exchange = ccxt.binance({'enableRateLimit': True})
        
        # Calculate start time
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=365 * years)
        since = int(start_date.timestamp() * 1000)
        
        all_candles = []
        limit = 1000
        
        print(f"Fetching {years} years of {timeframe} data for {symbol}...")
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                if not ohlcv:
                    break
                
                all_candles.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                # Progress indicator
                last_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000, tz=timezone.utc)
                print(f"Fetched up to {last_date.date()}... ({len(all_candles)} candles)")
                
                if last_date >= now:
                    break
                    
                # Small sleep to respect rate limits gently
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)
                continue

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df

    @staticmethod
    def resample_to_1d(df_1h):
        """Resamples 1H data to 1D data."""
        df_1d = df_1h.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df_1d

class BacktestEngine:
    def __init__(self, df_1h):
        self.df_1h = df_1h
        self.df_1d = DataManager.resample_to_1d(df_1h)
        self.results = []
        self.logs = []
        
        # Pre-calculate indicators to speed up the loop
        self._precalc_indicators()

    def _get_sma(self, series, window):
        return series.rolling(window=window).mean()

    def _precalc_indicators(self):
        """Pre-calculates indicators for vectorized lookups."""
        # 1D Indicators for Planner & Tumbler
        d = self.df_1d
        self.ind_1d = pd.DataFrame(index=d.index)
        self.ind_1d['close'] = d['close']
        
        # Planner SMAs
        self.ind_1d['p_sma120'] = self._get_sma(d['close'], PLANNER_PARAMS["S1_SMA"])
        self.ind_1d['p_sma400'] = self._get_sma(d['close'], PLANNER_PARAMS["S2_SMA"])
        
        # Tumbler SMAs
        self.ind_1d['t_sma1'] = self._get_sma(d['close'], TUMBLER_PARAMS["SMA1"])
        self.ind_1d['t_sma2'] = self._get_sma(d['close'], TUMBLER_PARAMS["SMA2"])
        
        # Tumbler III
        w = TUMBLER_PARAMS["III_WIN"]
        log_ret = np.log(d['close'] / d['close'].shift(1))
        # Avoid division by zero
        denom = log_ret.abs().rolling(w).sum()
        self.ind_1d['iii'] = (log_ret.rolling(w).sum().abs() / denom).fillna(0)

        # Gainer Indicators (1D part)
        # SMA Logic
        for p in GAINER_PARAMS["SMA_1D"]['params']:
            self.ind_1d[f'g_sma_{p}'] = self._get_sma(d['close'], p)
            
        # MACD Logic (1D)
        for i, (f, s, sig) in enumerate(GAINER_PARAMS["MACD_1D"]['params']):
            fast = d['close'].ewm(span=f, adjust=False).mean()
            slow = d['close'].ewm(span=s, adjust=False).mean()
            macd = fast - slow
            signal = macd.ewm(span=sig, adjust=False).mean()
            self.ind_1d[f'g_macd_val_{i}'] = macd
            self.ind_1d[f'g_macd_sig_{i}'] = signal

        # 1H Indicators for Gainer
        h = self.df_1h
        self.ind_1h = pd.DataFrame(index=h.index)
        self.ind_1h['close'] = h['close']
        
        for i, (f, s, sig) in enumerate(GAINER_PARAMS["MACD_1H"]['params']):
            fast = h['close'].ewm(span=f, adjust=False).mean()
            slow = h['close'].ewm(span=s, adjust=False).mean()
            macd = fast - slow
            signal = macd.ewm(span=sig, adjust=False).mean()
            self.ind_1h[f'g_macd_val_{i}'] = macd
            self.ind_1h[f'g_macd_sig_{i}'] = signal

    def calculate_decay(self, entry_date, current_date, decay_days):
        if entry_date is None: return 1.0
        days_since = (current_date - entry_date).total_seconds() / 86400
        if days_since >= decay_days: return 0.0
        weight = 1.0 - (days_since / decay_days) ** 2
        return max(0.0, weight)

    def run(self, initial_capital=10000.0):
        # Initial State
        state = {
            "planner": {
                "s1_equity": initial_capital * CAP_SPLIT,
                "s2_equity": initial_capital * CAP_SPLIT,
                "last_price": 0.0,
                "last_lev_s1": 0.0,
                "last_lev_s2": 0.0,
                "s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False, "trend": 0},
                "s2": {"peak_equity": 0.0, "stopped": False, "trend": 0}
            },
            "tumbler": {"flat_regime": False}
        }
        
        portfolio_cash = initial_capital
        # We process hour by hour
        timestamps = self.df_1h.index
        
        # Alignment: We need to know which 1D row corresponds to the current 1H timestamp
        # Forward fill 1D indicators to 1H index to simulate "latest known daily close"
        # In a real scenario, at 14:00 today, we only know YESTERDAY'S daily close for indicators relying on completed days.
        # We shift 1D indicators by 1 day to avoid lookahead bias.
        ind_1d_shifted = self.ind_1d.shift(1).reindex(timestamps, method='ffill')
        
        # Pre-convert to dicts for speed in loop
        ind_1h_dict = self.ind_1h.to_dict('index')
        ind_1d_dict = ind_1d_shifted.to_dict('index')
        
        equity_curve = []
        
        start_time = time.time()
        
        for ts in timestamps:
            current_price = ind_1h_dict[ts]['close']
            daily_data = ind_1d_dict[ts]
            
            # Skip if data is missing (NaN) due to shift
            if pd.isna(daily_data.get('close')):
                equity_curve.append({'timestamp': ts, 'equity': portfolio_cash, 'leverage': 0})
                continue

            # --- 1. PLANNER LOGIC ---
            p_state = state["planner"]
            
            # Init State if first run
            if p_state["last_price"] == 0.0:
                p_state["last_price"] = daily_data['close']
                # Reset equities to portion of current capital if needed, but here we track fixed buckets
                p_state["s1"]["peak_equity"] = p_state["s1_equity"]
                p_state["s2"]["peak_equity"] = p_state["s2_equity"]

            # Update Virtual Equity (Daily Step Logic applied Hourly? No, applying daily changes)
            # To faithfully replicate "run_planner", we check change from last saved price.
            # In live bot, this runs hourly.
            pct_change = (daily_data['close'] - p_state["last_price"]) / p_state["last_price"] if p_state["last_price"] > 0 else 0
            
            # We only update virtual equity if the "daily" price has changed (i.e., new day)
            # OR we simulate it continuously. The script says: "run_planner(df_1d...)"
            # If df_1d updates once a day, pct_change is 0 for 23 hours.
            # Let's check if new day by comparing prices or dates.
            # Ideally, we update virtual equity on every daily candle close.
            
            if pct_change != 0:
                p_state["s1_equity"] *= (1.0 + pct_change * p_state["last_lev_s1"])
                p_state["s2_equity"] *= (1.0 + pct_change * p_state["last_lev_s2"])
                p_state["last_price"] = daily_data['close']
            
            # Strategy 1
            sma120 = daily_data['p_sma120']
            s1_trend = 1 if daily_data['close'] > sma120 else -1
            s1 = p_state["s1"]
            
            if s1["trend"] != s1_trend:
                s1["trend"] = s1_trend
                s1["entry_date"] = ts
                s1["stopped"] = False
                s1["peak_equity"] = p_state["s1_equity"]
            
            if p_state["s1_equity"] > s1["peak_equity"]: s1["peak_equity"] = p_state["s1_equity"]
            dd_s1 = (s1["peak_equity"] - p_state["s1_equity"]) / s1["peak_equity"] if s1["peak_equity"] > 0 else 0
            if dd_s1 > PLANNER_PARAMS["S1_STOP"]: s1["stopped"] = True
            
            s1_lev = 0.0
            if not s1["stopped"]:
                decay = self.calculate_decay(s1["entry_date"], ts, PLANNER_PARAMS["S1_DECAY"])
                s1_lev = float(s1_trend) * decay

            # Strategy 2
            sma400 = daily_data['p_sma400']
            s2_trend = 1 if daily_data['close'] > sma400 else -1
            s2 = p_state["s2"]
            
            if s2["trend"] != s2_trend:
                s2["trend"] = s2_trend
                s2["stopped"] = False
                s2["peak_equity"] = p_state["s2_equity"]
                
            if p_state["s2_equity"] > s2["peak_equity"]: s2["peak_equity"] = p_state["s2_equity"]
            dd_s2 = (s2["peak_equity"] - p_state["s2_equity"]) / s2["peak_equity"] if s2["peak_equity"] > 0 else 0
            if dd_s2 > PLANNER_PARAMS["S2_STOP"]: s2["stopped"] = True
            
            dist_pct = abs(daily_data['close'] - sma400) / sma400
            is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
            tgt_size = 0.5 if is_prox else 1.0
            
            s2_lev = 0.0
            if s2["stopped"]:
                if is_prox:
                    s2["stopped"] = False
                    s2["peak_equity"] = p_state["s2_equity"]
                    s2_lev = float(s2_trend) * tgt_size
            else:
                s2_lev = float(s2_trend) * tgt_size
                
            planner_lev = max(-2.0, min(2.0, s1_lev + s2_lev))
            
            # Save state for next step
            p_state["last_lev_s1"] = s1_lev
            p_state["last_lev_s2"] = s2_lev

            # --- 2. TUMBLER LOGIC ---
            t_state = state["tumbler"]
            iii = daily_data['iii']
            
            lev_t = TUMBLER_PARAMS["LEVS"][2]
            if iii < TUMBLER_PARAMS["III_TH"][0]: lev_t = TUMBLER_PARAMS["LEVS"][0]
            elif iii < TUMBLER_PARAMS["III_TH"][1]: lev_t = TUMBLER_PARAMS["LEVS"][1]
            
            if iii < TUMBLER_PARAMS["FLAT_THRESH"]: t_state["flat_regime"] = True
            
            if t_state["flat_regime"]:
                sma1, sma2 = daily_data['t_sma1'], daily_data['t_sma2']
                curr = daily_data['close']
                b1, b2 = sma1 * TUMBLER_PARAMS["BAND"], sma2 * TUMBLER_PARAMS["BAND"]
                if abs(curr - sma1) <= b1 or abs(curr - sma2) <= b2:
                    t_state["flat_regime"] = False
            
            tumbler_lev = 0.0
            if not t_state["flat_regime"]:
                s1, s2 = daily_data['t_sma1'], daily_data['t_sma2']
                c = daily_data['close']
                if c > s1 and c > s2: tumbler_lev = lev_t
                elif c < s1 and c < s2: tumbler_lev = -lev_t
            
            # Normalize Tumbler
            tumbler_lev = tumbler_lev * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)

            # --- 3. GAINER LOGIC ---
            # MACD 1H
            m1h_score = 0.0
            total_w = sum(GAINER_PARAMS["MACD_1H"]['weights'])
            for i, w in enumerate(GAINER_PARAMS["MACD_1H"]['weights']):
                val = ind_1h_dict[ts][f'g_macd_val_{i}']
                sig = ind_1h_dict[ts][f'g_macd_sig_{i}']
                m1h_score += (1.0 if val > sig else -1.0) * w
            m1h_score /= total_w
            
            # MACD 1D
            m1d_score = 0.0
            total_w = sum(GAINER_PARAMS["MACD_1D"]['weights'])
            for i, w in enumerate(GAINER_PARAMS["MACD_1D"]['weights']):
                val = daily_data[f'g_macd_val_{i}']
                sig = daily_data[f'g_macd_sig_{i}']
                m1d_score += (1.0 if val > sig else -1.0) * w
            m1d_score /= total_w
            
            # SMA 1D
            s1d_score = 0.0
            total_w = sum(GAINER_PARAMS["SMA_1D"]['weights'])
            for i, w in enumerate(GAINER_PARAMS["SMA_1D"]['weights']):
                p = GAINER_PARAMS["SMA_1D"]['params'][i]
                sma = daily_data[f'g_sma_{p}']
                s1d_score += (1.0 if daily_data['close'] > sma else -1.0) * w
            s1d_score /= total_w
            
            g_weights = GAINER_PARAMS["GA_WEIGHTS"]
            gainer_lev = (m1h_score * g_weights["MACD_1H"] + 
                          m1d_score * g_weights["MACD_1D"] + 
                          s1d_score * g_weights["SMA_1D"]) / sum(g_weights.values())
            
            gainer_lev *= TARGET_STRAT_LEV

            # --- 4. PORTFOLIO AGGREGATION ---
            # Total Target Leverage
            # In main.py: target_qty = (n_p + n_t + n_g) * strat_cap / curr_price
            # This implies the strategies are additive on the SAME capital base (CAP_SPLIT of total).
            
            total_lev = planner_lev + tumbler_lev + gainer_lev
            
            # Calculate PnL for this hour
            # We assume we held 'total_lev' from previous hour to this hour.
            # Wait, 'total_lev' is the target for the NEXT period.
            # We need the leverage set at T-1 to calc PnL at T.
            
            # Retrieve prev leverage
            prev_lev = self.results[-1]['leverage'] if self.results else 0.0
            
            # Hourly Return
            # If ts is T, ind_1h_dict[ts] is T.
            # We need return from T-1 to T.
            # Since we iterate sequentially, we can track price change.
            
            prev_price = self.results[-1]['price'] if self.results else current_price
            
            step_ret = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            
            # Cost handling (Funding + Slippage approx)
            # Simple assumption: 0.06% slippage + fees per full turn, roughly distributed
            # Funding is variable. Let's assume a small drag 0.01% per day => ~0.0004% per hour
            drag = 0.000004 
            
            pnl = (step_ret * prev_lev) - (abs(prev_lev) * drag)
            
            portfolio_cash *= (1.0 + pnl)
            
            equity_curve.append({
                'timestamp': ts,
                'price': current_price,
                'equity': portfolio_cash,
                'leverage': total_lev, # This is target for next step
                'planner_lev': planner_lev,
                'tumbler_lev': tumbler_lev,
                'gainer_lev': gainer_lev
            })

        return pd.DataFrame(equity_curve).set_index('timestamp')
