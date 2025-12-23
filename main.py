#!/usr/bin/env python3
"""
master_trader.py - Clean Trading Engine
Features:
- "Fair 2.0" Normalization (All strategies target 2.0x Max).
- Position-Aware Limit Chaser (Cancel-All logic).
- Minimalistic Logging with [HH:MM] timestamps.
- Slow-Log Protection (0.1s pause) for Railway console.
- Atomic State Management for strategy persistence.
- Full Gainer Ensemble (MACD 1H/1D + SMA 1D) matching Analysis.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import kraken_futures as kf
import kraken_ohlc

# --- Global Configuration ---
# Changed to Perpetual Contract as requested
SYMBOL_FUTS = "PF_XBTUSD"
SYMBOL_OHLC = "XBTUSD"
CAP_SPLIT = 0.333
LIMIT_CHASE_DURATION = 720
CHASE_INTERVAL = 60
MIN_TRADE_SIZE = 0.0002
LIMIT_OFFSET_TICKS = 1

# Normalization Constants
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

STATE_FILE = Path("master_state.json")

# --- Custom Slow Logging Handler ---
class SlowStreamHandler(logging.StreamHandler):
    """Pauses for 0.1s after every log message to prevent console flooding."""
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
            time.sleep(0.1) 
        except Exception:
            self.handleError(record)

# Setup Minimalist Logging
log = logging.getLogger("MASTER")
log.setLevel(logging.INFO)
slow_h = SlowStreamHandler(sys.stdout)
# Log format: [HH:MM] Message
slow_h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M"))
log.addHandler(slow_h)

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

# --- Strategy Parameters ---
PLANNER_PARAMS = {"S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13, "S2_SMA": 400}
TUMBLER_PARAMS = {"SMA1": 32, "SMA2": 114, "STOP": 0.043, "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077, "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]}

# Updated GAINER_PARAMS to match binance_analysis.py exactly
# External GA Weights: [0.8, 0, 0.4, 0, 0, 0.4] -> MACD_1H: 0.8, MACD_1D: 0.4, SMA_1D: 0.4
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

# --- State Management ---
def load_state():
    defaults = {
        "planner": {"entry_date": None, "peak_equity": 0.0, "stopped": False},
        "tumbler": {"flat_regime": False}
    }
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                for k, v in defaults.items():
                    if k not in saved: saved[k] = v
                return saved
        except Exception as e: log.error(f"State load error: {e}")
    return defaults

def save_state(state):
    try:
        # Atomic Write
        temp_file = STATE_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception as e: log.error(f"State save error: {e}")

# --- Strategy Helpers ---
def get_sma(prices, window):
    if len(prices) < window: return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def get_market_price(api):
    try:
        resp = api.get_tickers()
        for t in resp.get("tickers", []):
            if t.get("symbol") == SYMBOL_FUTS: return float(t.get("markPrice"))
    except: pass
    return 0.0

def get_net_position(api):
    try:
        resp = api.get_open_positions()
        for p in resp.get("openPositions", []):
            if p.get('symbol') == SYMBOL_FUTS:
                size = float(p.get('size', 0.0))
                return size if p.get('side') == 'long' else -size
    except: pass
    return 0.0

# --- Strategy Logic ---
def run_planner(df_1d, state, capital):
    s = state["planner"]
    if s["peak_equity"] < capital: s["peak_equity"] = capital
    price = df_1d['close'].iloc[-1]
    sma120 = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"])
    sma400 = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"])
    s1_lev = 0.0
    if price > sma120:
        if not s["stopped"]:
            if not s["entry_date"]: s["entry_date"] = datetime.now(timezone.utc).isoformat()
            days = (datetime.now(timezone.utc) - datetime.fromisoformat(s["entry_date"]).replace(tzinfo=timezone.utc)).total_seconds() / 86400
            s1_lev = 1.0 * max(0.0, 1.0 - (days / PLANNER_PARAMS["S1_DECAY"])**2)
    else:
        s["stopped"], s["peak_equity"], s["entry_date"] = False, capital, datetime.now(timezone.utc).isoformat()
        s1_lev = -1.0
    s2_lev = 1.0 if price > sma400 else 0.0
    return max(-2.0, min(2.0, s1_lev + s2_lev))

def run_tumbler(df_1d, state, capital):
    s = state["tumbler"]
    w = TUMBLER_PARAMS["III_WIN"]
    if len(df_1d) < w+1: return 0.0
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0).iloc[-1]
    lev = TUMBLER_PARAMS["LEVS"][2]
    if iii < TUMBLER_PARAMS["III_TH"][0]: lev = TUMBLER_PARAMS["LEVS"][0]
    elif iii < TUMBLER_PARAMS["III_TH"][1]: lev = TUMBLER_PARAMS["LEVS"][1]
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]: s["flat_regime"] = True
    if s["flat_regime"]:
        sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
        if abs(df_1d['close'].iloc[-1] - sma1) <= sma1 * TUMBLER_PARAMS["BAND"]: s["flat_regime"] = False
    if s["flat_regime"]: return 0.0
    sma1, sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"]), get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    curr = df_1d['close'].iloc[-1]
    return lev if (curr > sma1 and curr > sma2) else (-lev if (curr < sma1 and curr < sma2) else 0.0)

def run_gainer(df_1h, df_1d):
    """
    Implements the exact logic from binance_analysis.py with 3-tuple parameter sets
    and specific GA weights: MACD_1H (0.8), MACD_1D (0.4), SMA_1D (0.4).
    """
    def calc_macd_pos(prices, config):
        params = config['params']
        weights = config['weights']
        composite = 0.0
        # Iterate over the 3 parameter sets (e.g., 97/366/47, etc.)
        for (f, s, sig_p), w in zip(params, weights):
            # Using adjust=False to match binance_analysis.py logic
            fast = prices.ewm(span=f, adjust=False).mean()
            slow = prices.ewm(span=s, adjust=False).mean()
            macd = fast - slow
            sig_line = macd.ewm(span=sig_p, adjust=False).mean()
            val = 1.0 if macd.iloc[-1] > sig_line.iloc[-1] else -1.0
            composite += val * w
        
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else composite

    def calc_sma_pos(prices, config):
        params = config['params']
        weights = config['weights']
        composite = 0.0
        current_price = prices.iloc[-1]
        # Iterate over the 3 SMA windows
        for p, w in zip(params, weights):
            sma_val = get_sma(prices, p)
            val = 1.0 if current_price > sma_val else -1.0
            composite += val * w
            
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else composite

    # 1. MACD 1H (GA Weight: 0.8)
    macd_1h_raw = calc_macd_pos(df_1h['close'], GAINER_PARAMS["MACD_1H"])
    s_macd_1h = macd_1h_raw * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1H"]

    # 2. MACD 1D (GA Weight: 0.4)
    macd_1d_raw = calc_macd_pos(df_1d['close'], GAINER_PARAMS["MACD_1D"])
    s_macd_1d = macd_1d_raw * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1D"]

    # 3. SMA 1D (GA Weight: 0.4)
    sma_1d_raw = calc_sma_pos(df_1d['close'], GAINER_PARAMS["SMA_1D"])
    s_sma_1d = sma_1d_raw * GAINER_PARAMS["GA_WEIGHTS"]["SMA_1D"]

    # Normalize by sum of GA weights (0.8 + 0.4 + 0.4 = 1.6)
    total_ga_weight = sum(GAINER_PARAMS["GA_WEIGHTS"].values())
    
    return (s_macd_1h + s_macd_1d + s_sma_1d) / total_ga_weight

# --- Execution ---
def limit_chaser(api, target_qty):
    if dry: return
    log.info(f"Chase Start | Tgt: {target_qty:.4f}")
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass

    for i in range(int(LIMIT_CHASE_DURATION / CHASE_INTERVAL)):
        curr_pos = get_net_position(api)
        delta = target_qty - curr_pos
        if abs(delta) < MIN_TRADE_SIZE:
            log.info(f"Chase Done | Delta: {delta:.5f}")
            break
        
        # Sledgehammer Cleanup
        try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
        except: pass
        time.sleep(0.5)

        side = "buy" if delta > 0 else "sell"
        size = round(abs(delta), 4)
        tk = api.get_tickers()
        limit_px = 0
        for t in tk.get("tickers", []):
            if t["symbol"] == SYMBOL_FUTS:
                limit_px = int(float(t["bid"])) if side == "buy" else int(float(t["ask"]))
                limit_px += (-LIMIT_OFFSET_TICKS if side == "buy" else LIMIT_OFFSET_TICKS)
                break
        
        if limit_px > 0:
            try:
                resp = api.send_order({"orderType": "lmt", "symbol": SYMBOL_FUTS, "side": side, "size": size, "limitPrice": limit_px, "postOnly": True})
                log.info(f"Chase {i+1} | {side.upper()} {size} @ {limit_px} | {resp.get('result')}")
            except: pass
        
        time.sleep(CHASE_INTERVAL)
    
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass

def manage_stops(api, net_size, price):
    if dry or abs(net_size) < MIN_TRADE_SIZE: return
    side = "sell" if net_size > 0 else "buy"
    qty = round(abs(net_size) * 0.33, 4)
    # Planner & Tumbler Stops
    for stop_pct in [PLANNER_PARAMS["S1_STOP"], TUMBLER_PARAMS["STOP"]]:
        px = int(price * (1 - stop_pct)) if side == "sell" else int(price * (1 + stop_pct))
        try:
            r = api.send_order({"orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, "size": qty, "stopPrice": px, "reduceOnly": True})
            log.info(f"Stop Placed | Px: {px} | {r.get('result')}")
        except: pass

def run_cycle(api):
    log.info("--- CYCLE ---")
    try:
        df_1h = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 60)
        df_1d = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 1440)
    except Exception as e:
        log.error(f"OHLC Fail: {e}")
        return

    state = load_state()
    curr_price = get_market_price(api) or df_1h['close'].iloc[-1]
    try:
        accts = api.get_accounts()
        strat_cap = float(accts["accounts"]["flex"]["portfolioValue"]) * CAP_SPLIT
        log.info(f"Cap: ${strat_cap:.0f} | Price: ${curr_price:.1f}")
    except: return
    
    # Fair 2.0 Calculation
    r_p = run_planner(df_1d, state, strat_cap)
    r_t = run_tumbler(df_1d, state, strat_cap)
    r_g = run_gainer(df_1h, df_1d)
    
    n_p, n_t, n_g = r_p, r_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV), r_g * TARGET_STRAT_LEV
    log.info(f"Levs | P:{n_p:.2f} T:{n_t:.2f} G:{n_g:.2f}")

    target_qty = (n_p + n_t + n_g) * strat_cap / curr_price
    limit_chaser(api, target_qty)
    
    final_pos = get_net_position(api)
    manage_stops(api, final_pos, curr_price)
    save_state(state)
    log.info("--- END ---")

def main():
    api_key, api_sec = os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit("No API Keys")
    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    if os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}:
        run_cycle(api)
        
    while True:
        now = datetime.now(timezone.utc)
        wait = ((now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0) - now).total_seconds()
        log.info(f"Sleep {wait/60:.0f}m")
        time.sleep(wait)
        run_cycle(api)

if __name__ == "__main__":
    main()
