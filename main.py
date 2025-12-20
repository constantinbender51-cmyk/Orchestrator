#!/usr/bin/env python3
"""
master_trader.py - Unified Orchestrator for Planner, Tumbler, and Gainer
Target: FF_XBTUSD_260130 (Fixed Jan 2030)
Features:
- Slow Logging: 0.1s pause after every log to protect console.
- Position-Aware Limit Chaser.
- "Fair 2.0" Strategy Normalization.
- Concise Status Logging.
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

# --- Custom Slow Logging Handler ---
class SlowStreamHandler(logging.StreamHandler):
    """Pauses for 0.1s after every log message to prevent console flooding."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
            time.sleep(0.1)  # The requested 0.1s pause
        except Exception:
            self.handleError(record)

# --- Global Configuration ---
SYMBOL_FUTS = "FF_XBTUSD_260130"
SYMBOL_OHLC = "XBTUSD"
CAP_SPLIT = 0.333  # Equal split

# Execution Settings
LIMIT_CHASE_DURATION = 720  # 12 minutes
CHASE_INTERVAL = 60         # Update every minute
MIN_TRADE_SIZE = 0.0002     # Safe minimum to avoid invalidSize errors
LIMIT_OFFSET_TICKS = 1      # Start 1 tick away from spread

# Normalization Constants (Targeting Max Leverage 2.0 per strategy)
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

# Setup Logging
log = logging.getLogger("MASTER")
log.setLevel(logging.INFO)
slow_handler = SlowStreamHandler(sys.stdout)
slow_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
log.addHandler(slow_handler)

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
STATE_FILE = Path("master_state.json")

# --- Strategy Parameters ---
PLANNER_PARAMS = {
    "S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13,
    "S2_SMA": 400, "S2_PROX": 0.05, "S2_STOP": 0.27
}

TUMBLER_PARAMS = {
    "SMA1": 32, "SMA2": 114, "STOP": 0.043, "TP": 0.126,
    "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077,
    "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]
}

GAINER_PARAMS = {
    "WEIGHTS": [0.8, 0.0, 0.4, 0.0, 0.0, 0.4],
    "MACD_1H": {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]},
    "MACD_1D": {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]},
    "SMA_1D":  {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}
}

# --- State Management ---
def load_state():
    defaults = {
        "planner": {"entry_date": None, "peak_equity": 0.0, "stopped": False},
        "tumbler": {"flat_regime": False},
        "gainer": {},
        "virtual_equity": {"planner": 0.0, "tumbler": 0.0, "gainer": 0.0},
        "trades": []
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
        serializable_state = {k: v for k, v in state.items() if k != "df_1d"}
        with open(STATE_FILE, "w") as f: json.dump(serializable_state, f, indent=2)
    except Exception as e: log.error(f"State save error: {e}")

# --- Helper Functions ---
def get_sma(prices, window):
    if len(prices) < window: return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def get_market_price(api):
    try:
        resp = api.get_tickers()
        for t in resp.get("tickers", []):
            if t.get("symbol") == SYMBOL_FUTS: 
                return float(t.get("markPrice"))
    except Exception as e: log.error(f"Ticker fetch failed: {e}")
    return 0.0

def get_net_position(api):
    try:
        resp = api.get_open_positions()
        for p in resp.get("openPositions", []):
            if p.get('symbol') == SYMBOL_FUTS:
                size = float(p.get('size', 0.0))
                return size if p.get('side') == 'long' else -size
    except Exception as e: log.error(f"Pos fetch failed: {e}")
    return 0.0

# --- Strategy Modules ---
def run_planner(df_1d, state, capital):
    s = state["planner"]
    if s["peak_equity"] < capital: s["peak_equity"] = capital
    if s["peak_equity"] > 0:
        dd = (s["peak_equity"] - capital) / s["peak_equity"]
        if dd > PLANNER_PARAMS["S1_STOP"]: 
            if not s["stopped"]: log.info(f"Planner S1 Soft Stop Triggered (DD: {dd*100:.2f}%)")
            s["stopped"] = True
    else: s["peak_equity"] = capital

    price = df_1d['close'].iloc[-1]
    sma120 = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"])
    sma400 = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"])

    s1_lev = 0.0
    if price > sma120:
        if not s["stopped"]:
            if not s["entry_date"]: s["entry_date"] = datetime.now(timezone.utc).isoformat()
            entry_dt = datetime.fromisoformat(s["entry_date"]).replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
            w = max(0.0, 1.0 - (days / PLANNER_PARAMS["S1_DECAY"])**2)
            s1_lev = 1.0 * w
    else:
        s["stopped"] = False
        s["peak_equity"] = capital
        s["entry_date"] = datetime.now(timezone.utc).isoformat()
        s1_lev = -1.0

    s2_lev = 1.0 if price > sma400 else 0.0
    if capital > s["peak_equity"]: s["peak_equity"] = capital
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
    
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]:
        if not s["flat_regime"]: log.info(f"Tumbler: Entering Flat Regime (III={iii:.4f})")
        s["flat_regime"] = True
    
    if s["flat_regime"]:
        sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
        sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
        curr = df_1d['close'].iloc[-1]
        b = TUMBLER_PARAMS["BAND"]
        if abs(curr - sma1) <= sma1*b or abs(curr - sma2) <= sma2*b:
             log.info("Tumbler: Releasing Flat Regime")
             s["flat_regime"] = False
    
    if s["flat_regime"]: return 0.0
    sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
    sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    curr = df_1d['close'].iloc[-1]
    if curr > sma1 and curr > sma2: return lev
    if curr < sma1 and curr < sma2: return -lev
    return 0.0

def run_gainer(df_1h, df_1d):
    def calc_macd(prices, params, weights):
        comp = 0.0
        for (f,s,sig), w in zip(params, weights):
            fast = prices.ewm(span=f, adjust=False).mean()
            slow = prices.ewm(span=s, adjust=False).mean()
            macd = fast - slow
            signal = macd.ewm(span=sig, adjust=False).mean()
            val = 1.0 if macd.iloc[-1] > signal.iloc[-1] else -1.0
            comp += val * w
        return comp
    def calc_sma(prices, params, weights):
        comp = 0.0
        curr = prices.iloc[-1]
        for p, w in zip(params, weights):
            if len(prices) < p: continue
            sma = prices.rolling(window=p).mean().iloc[-1]
            val = 1.0 if curr > sma else -1.0
            comp += val * w
        return comp
    w = GAINER_PARAMS["WEIGHTS"]
    s1 = calc_macd(df_1h['close'], GAINER_PARAMS["MACD_1H"]['params'], GAINER_PARAMS["MACD_1H"]['weights']) * w[0]
    s3 = calc_macd(df_1d['close'], GAINER_PARAMS["MACD_1D"]['params'], GAINER_PARAMS["MACD_1D"]['weights']) * w[2]
    s6 = calc_sma(df_1d['close'], GAINER_PARAMS["SMA_1D"]['params'], GAINER_PARAMS["SMA_1D"]['weights']) * w[5]
    total = sum(w)
    return (s1 + s3 + s6) / total if total > 0 else 0.0

# --- Execution Engine ---

def limit_chaser(api, target_qty):
    if dry: return
    log.info(f"Starting Limit Chaser. Target Net Position: {target_qty:.4f}")
    order_id = None
    
    for i in range(int(LIMIT_CHASE_DURATION / CHASE_INTERVAL)):
        curr_pos = get_net_position(api)
        delta = target_qty - curr_pos
        
        if abs(delta) < MIN_TRADE_SIZE:
            log.info(f"Chaser Complete: Target reached (Pos: {curr_pos:.4f})")
            if order_id:
                try: api.cancel_order({"orderId": order_id, "symbol": SYMBOL_FUTS})
                except: pass
            break
            
        side = "buy" if delta > 0 else "sell"
        size = round(abs(delta), 4)
        
        limit_px = 0
        try:
            tk = api.get_tickers()
            for t in tk["tickers"]:
                if t["symbol"] == SYMBOL_FUTS:
                    limit_px = int(float(t["bid"])) if side == "buy" else int(float(t["ask"]))
                    limit_px += (-LIMIT_OFFSET_TICKS if side == "buy" else LIMIT_OFFSET_TICKS)
                    break
        except Exception as e:
            log.error(f"Ticker fetch failed: {e}")
            time.sleep(5)
            continue
            
        try:
            if order_id:
                api.cancel_order({"orderId": order_id, "symbol": SYMBOL_FUTS})
            
            payload = {"orderType": "lmt", "symbol": SYMBOL_FUTS, "side": side, "size": size, "limitPrice": limit_px, "postOnly": True}
            resp = api.send_order(payload)
            
            result_str = resp.get("result", "unknown")
            status_str = resp.get("sendStatus", {}).get("status", "unknown")
            log.info(f"Order {i+1} [{side} {size} @ {limit_px}]: {result_str} | {status_str}")
            
            if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                order_id = resp["sendStatus"]["order_id"]
        except Exception as e:
            log.error(f"Chaser Execution Error: {e}")
        
        time.sleep(CHASE_INTERVAL)
        
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass

def manage_virtual_stops(api, state, net_size, price, cap_per_strat):
    net_size = round(net_size, 4)
    if dry or abs(net_size) < MIN_TRADE_SIZE: return
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass
    
    side = "sell" if net_size > 0 else "buy"
    qty = round(abs(net_size) * 0.33, 4)
    if qty < MIN_TRADE_SIZE: return

    # 1. Planner Stop
    stop_px = int(price * (1 - PLANNER_PARAMS["S1_STOP"])) if side == "sell" else int(price * (1 + PLANNER_PARAMS["S1_STOP"]))
    try:
        resp = api.send_order({"orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, "size": qty, "stopPrice": stop_px, "reduceOnly": True})
        log.info(f"Planner Stop: {resp.get('result')} | {resp.get('sendStatus', {}).get('status')}")
    except: pass

    # 2. Tumbler Stop
    stop_px = int(price * (1 - TUMBLER_PARAMS["STOP"])) if side == "sell" else int(price * (1 + TUMBLER_PARAMS["STOP"]))
    try:
        resp = api.send_order({"orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, "size": qty, "stopPrice": stop_px, "reduceOnly": True})
        log.info(f"Tumbler Stop: {resp.get('result')} | {resp.get('sendStatus', {}).get('status')}")
    except: pass

def run_cycle(api):
    log.info(">>> CYCLE START <<<")
    try:
        df_1h = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 60)
        df_1d = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 1440)
    except Exception as e:
        log.error(f"Failed to fetch OHLC data: {e}")
        return

    state = load_state()
    state["df_1d"] = df_1d 
    curr_price = get_market_price(api) or df_1h['close'].iloc[-1]
    
    try:
        accts = api.get_accounts()
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        strat_cap = total_pv * CAP_SPLIT
        log.info(f"Total PV: ${total_pv:.2f} | StratAlloc: ${strat_cap:.2f}")
    except: return
    
    raw_planner = run_planner(df_1d, state, strat_cap)
    raw_tumbler = run_tumbler(df_1d, state, strat_cap)
    raw_gainer = run_gainer(df_1h, df_1d)
    
    norm_planner = raw_planner
    norm_tumbler = raw_tumbler * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
    norm_gainer  = raw_gainer  * TARGET_STRAT_LEV
    
    log.info(f"Raw Levs  | Plan: {raw_planner:.2f} | Tumb: {raw_tumbler:.2f} | Gain: {raw_gainer:.2f}")
    log.info(f"Fair Levs | Plan: {norm_planner:.2f} | Tumb: {norm_tumbler:.2f} | Gain: {norm_gainer:.2f}")

    lev_total = norm_planner + norm_tumbler + norm_gainer
    target_qty = lev_total * strat_cap / curr_price
    
    limit_chaser(api, target_qty)
    
    final_pos = get_net_position(api)
    manage_virtual_stops(api, state, final_pos, curr_price, strat_cap)
    save_state(state)
    log.info(">>> CYCLE END <<<")

def wait_until_next_hour():
    now = datetime.now(timezone.utc)
    next_run = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    wait = (next_run - now).total_seconds()
    log.info(f"Sleeping until {next_run.strftime('%H:%M')} ({wait/60:.1f}m)")
    time.sleep(wait)

def main():
    api_key, api_sec = os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit("No API Keys")
    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    if os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}:
        run_cycle(api)
        
    while True:
        wait_until_next_hour()
        run_cycle(api)

if __name__ == "__main__":
    main()
