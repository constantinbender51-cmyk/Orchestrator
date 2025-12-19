#!/usr/bin/env python3
"""
master_trader.py - Unified Orchestrator for Planner, Tumbler, and Gainer
Target: FF_XBTUSD_260130 (Fixed Jan 2030)
Features:
- Single File, Single Pair.
- Net Leverage Aggregation (Internal Netting).
- Execution: Limit Chaser (Post-Only) - NO Market Orders.
- Independent Virtual Stop Losses.
- FIX: Strict size rounding and checks to prevent 'invalidSize' errors.
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
SYMBOL_FUTS = "FF_XBTUSD_260130"
SYMBOL_OHLC = "XBTUSD"
CAP_SPLIT = 0.333  # Equal split

# Execution Settings
LIMIT_CHASE_DURATION = 720  # 12 minutes
CHASE_INTERVAL = 60         # Update every minute
MIN_TRADE_SIZE = 0.0002     # Safe minimum to avoid invalidSize errors
LIMIT_OFFSET_TICKS = 1      # Start 1 tick away from spread

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("MASTER")
dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

STATE_FILE = Path("master_state.json")

# --- Strategy Parameters ---

# Planner
PLANNER_PARAMS = {
    "S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13,
    "S2_SMA": 400, "S2_PROX": 0.05, "S2_STOP": 0.27
}

# Tumbler
TUMBLER_PARAMS = {
    "SMA1": 32, "SMA2": 114, "STOP": 0.043, "TP": 0.126,
    "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077,
    "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]
}

# Gainer
GAINER_PARAMS = {
    "WEIGHTS": [0.8, 0.0, 0.4, 0.0, 0.0, 0.4], # MACD1H, 4H, 1D, SMA1H, 4H, 1D
    "MACD_1H": {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]},
    "MACD_1D": {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]},
    "SMA_1D":  {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}
}

# --- State Management ---
def load_state():
    defaults = {
        "planner": {"entry_date": None, "peak_equity": 0.0, "stopped": False},
        "tumbler": {"flat_regime": False},
        "gainer": {}, # Gainer has no specific state retention needed for signals
        "virtual_equity": {"planner": 0.0, "tumbler": 0.0, "gainer": 0.0},
        "trades": []
    }
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                # Merge defaults
                for k, v in defaults.items():
                    if k not in saved: saved[k] = v
                return saved
        except Exception as e: log.error(f"State load error: {e}")
    return defaults

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)
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
                log.info(f"Ticker Data: Bid={t.get('bid')} Ask={t.get('ask')} Last={t.get('last')}")
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
    """Returns Target Leverage (-2.0 to 2.0)"""
    s = state["planner"]
    # Update Virtual Equity & Trailing Stops
    if s["peak_equity"] < capital: s["peak_equity"] = capital # Reset if cap added
    
    # Check stops based on VIRTUAL equity
    if s["peak_equity"] > 0:
        dd = (s["peak_equity"] - capital) / s["peak_equity"]
        # Soft stop check (actual hard stops are placed on exchange)
        # We just track state here to prevent re-entry
        if dd > PLANNER_PARAMS["S1_STOP"]: 
            if not s["stopped"]: log.info(f"Planner S1 Soft Stop Triggered (DD: {dd*100:.2f}%)")
            s["stopped"] = True
    else: s["peak_equity"] = capital

    price = df_1d['close'].iloc[-1]
    sma120 = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"])
    sma400 = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"])

    # S1
    s1_lev = 0.0
    if price > sma120:
        if not s["stopped"]:
            if not s["entry_date"]: s["entry_date"] = datetime.now(timezone.utc).isoformat()
            # Decay
            entry_dt = datetime.fromisoformat(s["entry_date"]).replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
            w = max(0.0, 1.0 - (days / PLANNER_PARAMS["S1_DECAY"])**2)
            s1_lev = 1.0 * w
    else:
        # Reset
        s["stopped"] = False
        s["peak_equity"] = capital # Reset peak
        s["entry_date"] = datetime.now(timezone.utc).isoformat()
        s1_lev = -1.0 # Short (fresh decay)

    # S2
    s2_lev = 0.0
    if price > sma400:
        s2_lev = 1.0
    else:
        s2_lev = 0.0
    
    # Save back to state
    if capital > s["peak_equity"]: s["peak_equity"] = capital
    
    net = max(-2.0, min(2.0, s1_lev + s2_lev))
    return net

def run_tumbler(df_1d, state, capital):
    """Returns Target Leverage"""
    s = state["tumbler"]
    
    # III Calculation
    w = TUMBLER_PARAMS["III_WIN"]
    if len(df_1d) < w+1: return 0.0
    
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0).iloc[-1]
    
    # Leverage Tier
    lev = TUMBLER_PARAMS["LEVS"][2] # High
    if iii < TUMBLER_PARAMS["III_TH"][0]: lev = TUMBLER_PARAMS["LEVS"][0] # Low
    elif iii < TUMBLER_PARAMS["III_TH"][1]: lev = TUMBLER_PARAMS["LEVS"][1] # Mid
    
    # Flat Regime
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]:
        if not s["flat_regime"]: log.info(f"Tumbler: Entering Flat Regime (III={iii:.4f})")
        s["flat_regime"] = True
    
    # Release Check
    if s["flat_regime"]:
        sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
        sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
        curr = df_1d['close'].iloc[-1]
        
        # Band check
        b = TUMBLER_PARAMS["BAND"]
        if abs(curr - sma1) <= sma1*b or abs(curr - sma2) <= sma2*b:
             log.info("Tumbler: Releasing Flat Regime")
             s["flat_regime"] = False
    
    if s["flat_regime"]: return 0.0
    
    # Signal
    sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
    sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    curr = df_1d['close'].iloc[-1]
    
    if curr > sma1 and curr > sma2: return lev # Long
    if curr < sma1 and curr < sma2: return -lev # Short
    return 0.0

def run_gainer(df_1h, df_1d):
    """Returns Target Leverage (-1.0 to 1.0)"""
    
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

    # Weights: MACD1H(0), 4H(1), 1D(2), SMA1H(3), 4H(4), 1D(5)
    w = GAINER_PARAMS["WEIGHTS"]
    
    s1 = calc_macd(df_1h['close'], GAINER_PARAMS["MACD_1H"]['params'], GAINER_PARAMS["MACD_1H"]['weights']) * w[0]
    s3 = calc_macd(df_1d['close'], GAINER_PARAMS["MACD_1D"]['params'], GAINER_PARAMS["MACD_1D"]['weights']) * w[2]
    s6 = calc_sma(df_1d['close'], GAINER_PARAMS["SMA_1D"]['params'], GAINER_PARAMS["SMA_1D"]['weights']) * w[5]
    
    total = sum(w)
    return (s1 + s3 + s6) / total if total > 0 else 0.0

# --- Execution Engine ---

def limit_chaser(api, side, size, start_price):
    """
    Executes a Post-Only Limit Order, moving closer to market every minute.
    NO Market Orders.
    """
    if dry: return
    
    # Strict rounding to 4 decimals to avoid invalidSize
    size = round(size, 4)
    if size < MIN_TRADE_SIZE:
        log.warning(f"Chaser Skipped: Size {size} < Min {MIN_TRADE_SIZE}")
        return

    # Initial Price (Deep Maker)
    if side == "buy": limit_px = int(start_price - LIMIT_OFFSET_TICKS)
    else: limit_px = int(start_price + LIMIT_OFFSET_TICKS)
    
    log.info(f"Starting Limit Chaser: {side} {size} @ {limit_px}")
    
    order_id = None
    
    for i in range(int(LIMIT_CHASE_DURATION / CHASE_INTERVAL)):
        # 1. Place/Update Order
        try:
            if order_id:
                # Cancel old
                api.cancel_order({"orderId": order_id, "symbol": SYMBOL_FUTS})
            
            # Place new post-only
            payload = {
                "orderType": "lmt", 
                "symbol": SYMBOL_FUTS, 
                "side": side, 
                "size": size, 
                "limitPrice": limit_px,
                "postOnly": True
            }
            log.info(f"Sending Order: {payload}")
            resp = api.send_order(payload)
            log.info(f"API Response: {resp}")
            
            # Check response
            if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                order_id = resp["sendStatus"]["order_id"]
                log.info(f"Chaser [{i}]: Placed @ {limit_px} (ID: {order_id})")
            else:
                log.warning(f"Chaser rejected: {resp}")
                # Common reject for postOnly is 'would execute' -> Means we are crossing spread.
                # Since we want NO market orders, we should just break or retry.
                # For safety, break to re-evaluate state.
                break
                
        except Exception as e:
            log.error(f"Chaser error: {e}")
            break
            
        # 2. Wait
        time.sleep(CHASE_INTERVAL)
        
        # 3. Update Price (Converge)
        # Get current top of book
        tk = api.get_tickers()
        try:
            for t in tk["tickers"]:
                if t["symbol"] == SYMBOL_FUTS:
                    # Update limit to be closer
                    bid = float(t["bid"])
                    ask = float(t["ask"])
                    
                    if side == "buy":
                        limit_px = int(bid)
                    else:
                        limit_px = int(ask)
                    log.info(f"Updating limit price to: {limit_px} (Bid:{bid}/Ask:{ask})")
                    break
        except: pass
        
    # Cleanup
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass


def manage_virtual_stops(api, state, net_size, price, cap_per_strat):
    """
    Places independent stops for each strategy logic on the single netted position.
    Uses Reduce-Only.
    """
    net_size = round(net_size, 4)
    if dry or abs(net_size) < MIN_TRADE_SIZE: 
        log.info(f"Skipping stops: Net Size {net_size} < Min")
        return
    
    # Clear old stops
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass
    
    # 1. Planner Stops (Trailing)
    p_net_lev = run_planner(state["df_1d"], state, cap_per_strat) # Re-calc to get lev
    if abs(p_net_lev) > 0.01: 
        peak = state["planner"]["peak_equity"] or cap_per_strat
        # Direction
        side = "sell" if net_size > 0 else "buy"
        
        # Stop px
        stop_px = int(price * (1 - PLANNER_PARAMS["S1_STOP"])) if side == "sell" else int(price * (1 + PLANNER_PARAMS["S1_STOP"]))
        
        # Size: 1/3 of total position (approx)
        qty = round(abs(net_size) * 0.33, 4)
        if qty >= MIN_TRADE_SIZE:
            try:
                payload = {
                    "orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, 
                    "size": qty, "stopPrice": stop_px, "reduceOnly": True
                }
                log.info(f"Sending Stop: {payload}")
                resp = api.send_order(payload)
                log.info(f"Stop Response: {resp}")
            except Exception as e: log.error(f"Planner Stop failed: {e}")

    # 2. Tumbler Stop (Static)
    side = "sell" if net_size > 0 else "buy"
    stop_px = int(price * (1 - TUMBLER_PARAMS["STOP"])) if side == "sell" else int(price * (1 + TUMBLER_PARAMS["STOP"]))
    qty = round(abs(net_size) * 0.33, 4)
    if qty >= MIN_TRADE_SIZE:
        try:
             payload = {
                "orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, 
                "size": qty, "stopPrice": stop_px, "reduceOnly": True
            }
             log.info(f"Sending Tumbler Stop: {payload}")
             resp = api.send_order(payload)
             log.info(f"Tumbler Stop Resp: {resp}")
        except: pass


def run_cycle(api):
    log.info(">>> CYCLE START <<<")
    
    # 1. Data
    try:
        df_1h = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 60)
        df_1d = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 1440)
    except Exception as e:
        log.error(f"Failed to fetch OHLC data: {e}")
        return

    state = load_state()
    # Store df for internal use
    state["df_1d"] = df_1d 
    
    curr_price = get_market_price(api)
    if curr_price == 0: 
        log.warning("Market price 0, fallback to OHLC close")
        curr_price = df_1h['close'].iloc[-1]
    
    # 2. Capital
    try:
        accts = api.get_accounts()
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        strat_cap = total_pv * CAP_SPLIT
        log.info(f"Total PV: ${total_pv:.2f} | StratAlloc: ${strat_cap:.2f}")
    except Exception as e:
        log.error(f"Failed to get accounts: {e}")
        return
    
    # 3. Calculate Signals
    lev_planner = run_planner(df_1d, state, strat_cap)
    lev_tumbler = run_tumbler(df_1d, state, strat_cap)
    lev_gainer = run_gainer(df_1h, df_1d)
    
    log.info(f"Signals | Plan: {lev_planner:.2f} | Tumb: {lev_tumbler:.2f} | Gain: {lev_gainer:.2f}")
    
    # 4. Netting
    notional_planner = lev_planner * strat_cap
    notional_tumbler = lev_tumbler * strat_cap
    notional_gainer = lev_gainer * strat_cap
    
    net_notional = notional_planner + notional_tumbler + notional_gainer
    target_qty = net_notional / curr_price
    
    curr_qty = get_net_position(api)
    delta = target_qty - curr_qty
    
    log.info(f"Net Notional: ${net_notional:.2f} | Tgt: {target_qty:.4f} | Curr: {curr_qty:.4f} | Delta: {delta:.4f}")
    
    # 5. Execution (Limit Chaser)
    if abs(delta) >= MIN_TRADE_SIZE:
        side = "buy" if delta > 0 else "sell"
        limit_chaser(api, side, abs(delta), curr_price)
        
        # 6. Verification Sleep
        log.info("Sleeping 10s for position update...")
        time.sleep(10)
    else:
        log.info(f"Delta {delta:.4f} < Min {MIN_TRADE_SIZE}, skipping execution.")
    
    # 7. Stops
    final_pos = get_net_position(api)
    manage_virtual_stops(api, state, final_pos, curr_price, strat_cap)
    
    save_state(state)
    log.info(">>> CYCLE END <<<")


def wait_until_next_hour():
    now = datetime.now(timezone.utc)
    next_run = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    wait = (next_run - now).total_seconds()
    log.info(f"Sleeping until {next_run.strftime('%H:%M')} ({wait/60:.1f}m)")
    sys.stdout.flush()
    time.sleep(wait)

def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit("No API Keys")
    
    log.info("Initializing API...")
    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    if os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}:
        log.info("RUN_TRADE_NOW active")
        run_cycle(api)
        
    while True:
        wait_until_next_hour()
        run_cycle(api)

if __name__ == "__main__":
    main()
