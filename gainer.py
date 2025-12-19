#!/usr/bin/env python3
"""
gainer.py - Hourly Ensemble Strategy
Target: FF_XBTUSD_260130
Logic: Weighted Ensemble of MACD 1H (0.8), MACD 1D (0.4), SMA 1D (0.4).
Execution: Delta Trading (Limit -> Market).
"""

import logging
import os
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import kraken_ohlc
import kraken_futures as kf

# --- CONFIG ---
SYMBOL_FUTS = "FF_XBTUSD_260130" # Jan 2030 Fixed
SYMBOL_OHLC = "XBTUSD"
INTERVAL_FAST = 60    # 1H
INTERVAL_SLOW = 1440  # 1D

# Weights from Optimization
# Order: MACD_1H, MACD_4H, MACD_1D, SMA_1H, SMA_4H, SMA_1D
ENSEMBLE_WEIGHTS = [0.8, 0.0, 0.4, 0.0, 0.0, 0.4]

# Sub-Strategy Parameters
STRAT_MACD_1H = {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]}
STRAT_MACD_1D = {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]}
STRAT_SMA_1D = {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}

# Execution Config
LIMIT_OFFSET_PCT = 0.0002
STOP_WAIT_TIME = 300 # 5 mins wait for hourly is safer
MIN_TRADE_SIZE = 0.0001
dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

log = logging.getLogger("gainer")
STATE_FILE = Path("gainer_state.json")

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f: return json.load(f)
        except: pass
    return {"trades": [], "last_run": None}

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)

# --- INDICATOR LOGIC ---
def calculate_macd_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite = 0.0
    
    # We only need the LAST value for live trading
    # But EWM requires some history. We calculate on the full series.
    for (f, s, sig_p), w in zip(params, weights):
        fast = prices.ewm(span=f, adjust=False).mean()
        slow = prices.ewm(span=s, adjust=False).mean()
        macd = fast - slow
        sig_line = macd.ewm(span=sig_p, adjust=False).mean()
        
        # Signal: 1.0 if MACD > Sig else -1.0
        val = 1.0 if macd.iloc[-1] > sig_line.iloc[-1] else -1.0
        composite += val * w
        
    total_w = sum(weights)
    return composite / total_w if total_w > 0 else 0.0

def calculate_sma_pos(prices, strat_config):
    params = strat_config['params']
    weights = strat_config['weights']
    composite = 0.0
    
    current_price = prices.iloc[-1]
    
    for p, w in zip(params, weights):
        if len(prices) < p: continue
        sma = prices.rolling(window=p).mean().iloc[-1]
        val = 1.0 if current_price > sma else -1.0
        composite += val * w
        
    total_w = sum(weights)
    return composite / total_w if total_w > 0 else 0.0

def get_ensemble_signal(df_1h, df_1d):
    # 1. MACD 1H
    s1 = calculate_macd_pos(df_1h['close'], STRAT_MACD_1H) * ENSEMBLE_WEIGHTS[0]
    
    # 2. MACD 1D
    s3 = calculate_macd_pos(df_1d['close'], STRAT_MACD_1D) * ENSEMBLE_WEIGHTS[2]
    
    # 3. SMA 1D
    s6 = calculate_sma_pos(df_1d['close'], STRAT_SMA_1D) * ENSEMBLE_WEIGHTS[5]
    
    total_weight = sum(ENSEMBLE_WEIGHTS)
    if total_weight == 0: return 0.0
    
    # Sum active components
    raw_signal = (s1 + s3 + s6) / total_weight
    
    # Clamp -1 to 1
    return max(-1.0, min(1.0, raw_signal))

# --- EXECUTION ---
def get_current_net_position(api):
    try:
        resp = api.get_open_positions()
        for p in resp.get("openPositions", []):
            if p.get('symbol') == SYMBOL_FUTS:
                size = float(p.get('size', 0.0))
                return size if p.get('side') == 'long' else -size
        return 0.0
    except: return 0.0

def get_market_price(api):
    try:
        resp = api.get_tickers()
        for t in resp.get("tickers", []):
            if t.get("symbol") == SYMBOL_FUTS: return float(t.get("markPrice"))
        raise ValueError("Ticker not found")
    except: return 0.0

def execute_delta(api, current_qty, target_qty, price):
    delta = target_qty - current_qty
    abs_delta = round(abs(delta), 4)
    
    if abs_delta < MIN_TRADE_SIZE:
        return "SKIPPED_SMALL"
    
    side = "buy" if delta > 0 else "sell"
    limit_px = int(round(price * (1.0 - LIMIT_OFFSET_PCT) if side == "buy" else price * (1.0 + LIMIT_OFFSET_PCT)))
    
    log.info(f"EXEC DELTA: {side} {abs_delta} @ {limit_px}")
    if dry: return "DRY_FILLED"
    
    try:
        # 1. Limit
        api.send_order({"orderType": "lmt", "symbol": SYMBOL_FUTS, "side": side, "size": abs_delta, "limitPrice": limit_px})
        time.sleep(STOP_WAIT_TIME)
        api.cancel_all_orders({"symbol": SYMBOL_FUTS})
        
        # 2. Check & Market
        new_curr = get_current_net_position(api)
        rem_delta = round(target_qty - new_curr, 4)
        
        if abs(rem_delta) >= MIN_TRADE_SIZE:
            log.info(f"Fallback MKT: {rem_delta}")
            api.send_order({"orderType": "mkt", "symbol": SYMBOL_FUTS, "side": "buy" if rem_delta > 0 else "sell", "size": abs(rem_delta)})
            time.sleep(5)
            
        return "EXECUTED"
    except Exception as e:
        log.error(f"Exec Error: {e}")
        return "ERROR"

def hourly_trade(api, capital_pct=1.0) -> str:
    log.info("--- Starting Gainer (Hourly) ---")
    try:
        # Data
        df_1h = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 60)
        df_1d = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 1440)
        
        # Signal
        net_signal = get_ensemble_signal(df_1h, df_1d)
        log.info(f"Ensemble Signal: {net_signal:.4f}")
        
        # Capital
        accts = api.get_accounts()
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        collateral = total_pv * capital_pct
        
        # Sizing
        price = get_market_price(api)
        if price == 0: price = df_1h['close'].iloc[-1]
        
        # Target Notional = Signal * Collateral
        # e.g. 0.5 Signal * $1000 = $500 Long
        target_qty = (collateral * net_signal) / price
        current_qty = get_current_net_position(api)
        
        log.info(f"Alloc: ${collateral:.2f} | TgtQty: {target_qty:.4f} | Curr: {current_qty:.4f}")
        
        res = execute_delta(api, current_qty, target_qty, price)
        
        state = load_state()
        state["last_run"] = datetime.now().isoformat()
        if res == "EXECUTED":
            state["trades"].append({"date": datetime.now().isoformat(), "signal": net_signal, "price": price})
        save_state(state)
        
        return f"{res} (Sig: {net_signal:.2f})"
        
    except Exception as e:
        log.exception(f"Gainer Failed: {e}")
        return f"ERROR: {e}"
