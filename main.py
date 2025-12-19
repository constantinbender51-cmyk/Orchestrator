#!/usr/bin/env python3
"""
master_trader.py - Unified Orchestrator for Planner, Tumbler, and Gainer
Target: FF_XBTUSD_260130 (Fixed Jan 2030)
Features:
- Global Leverage Multiplier (4x on final position).
- Net Leverage Aggregation.
- Execution: Limit Chaser (Post-Only).
- Independent Virtual Stop Losses.
- FIX: 'Order Stacking' bug (strict cancel before replace).
- FIX: Strict size rounding and checks to prevent 'invalidSize' errors.
- LOGGING: Full API response logging for debugging.
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

# Capital Allocation (Must sum to 1.0)
CAP_SPLIT = 0.333  

# GLOBAL LEVERAGE CONTROL
# Updated: This is a MULTIPLIER. If strategies sum to 1.0x, we execute 2.0x.
GLOBAL_LEVERAGE_MULTIPLIER = 2.0

# Execution Settings
LIMIT_CHASE_DURATION = 720  # 12 minutes
CHASE_INTERVAL = 60         # Update every minute
MIN_TRADE_SIZE = 0.0002     # Safe minimum
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
    # Updated Levs: 0.079/4.327, 4.327/4.327, 3.868/4.327
    "LEVS": [0.079/4.327, 4.327/4.327, 3.868/4.327], 
    "III_TH": [0.058, 0.259]
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
        with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)
    except Exception as e: log.error(f"State save error: {e}")

# --- Helper Functions ---
def get_sma(prices, window):
    if len(prices) < window: return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def get_market_price(api):
    try:
        resp = api.get_tickers()
        # Truncate log to 200 chars to avoid spam
        log.info(f"[API_RES] get_tickers: {str(resp)[:200]}...")
        for t in resp.get("tickers", []):
            if t.get("symbol") == SYMBOL_FUTS: 
                return float(t.get("markPrice"))
    except Exception as e: log.error(f"Ticker fetch failed: {e}")
    return 0.0

def get_net_position(api):
    try:
        resp = api.get_open_positions()
        log.info(f"[API_RES] get_open_positions: {resp}")
        for p in resp.get("openPositions", []):
            if p.get('symbol') == SYMBOL_FUTS:
                size = float(p.get('size', 0.0))
                return size if p.get('side') == 'long' else -size
    except Exception as e: log.error(f"Pos fetch failed: {e}")
    return 0.0

# --- Strategy Modules ---

def run_planner(df_1d, state, capital):
    """Returns Raw Target Leverage (Divided by 2)"""
    s = state["planner"]
    if s["peak_equity"] < capital: s["peak_equity"] = capital 
    
    if s["peak_equity"] > 0:
        dd = (s["peak_equity"] - capital) / s["peak_equity"]
        if dd > PLANNER_PARAMS["S1_STOP"]: 
            if not s["stopped"]: log.info(f"Planner S1 Soft Stop (DD: {dd*100:.2f}%)")
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
    
    # Raw leverage from strategy logic
    raw_net = s1_lev + s2_lev
    
    # MODIFIED: Divide planner contribution by 2.
    return raw_net / 2.0

def run_tumbler(df_1d, state, capital):
    """Returns Target Leverage"""
    s = state["tumbler"]
    w = TUMBLER_PARAMS["III_WIN"]
    if len(df_1d) < w+1: return 0.0
    
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0).iloc[-1]
    
    # MODIFIED: Logic to use explicit tiers based on III
    lev = 0.0
    # Tier 1: Low Vol/Conviction
    if iii < TUMBLER_PARAMS["III_TH"][0]: 
        lev = TUMBLER_PARAMS["LEVS"][0] # 0.079 / 4.327
    # Tier 2: Mid
    elif iii < TUMBLER_PARAMS["III_TH"][1]: 
        lev = TUMBLER_PARAMS["LEVS"][1] # 4.327 / 4.327 (1.0)
    # Tier 3: High
    else:
        lev = TUMBLER_PARAMS["LEVS"][2] # 3.868 / 4.327

    # Flat Regime
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

    w = GAINER_PARAMS["WEIGHTS"]
    s1 = calc_macd(df_1h['close'], GAINER_PARAMS["MACD_1H"]['params'], GAINER_PARAMS["MACD_1H"]['weights']) * w[0]
    s3 = calc_macd(df_1d['close'], GAINER_PARAMS["MACD_1D"]['params'], GAINER_PARAMS["MACD_1D"]['weights']) * w[2]
    s6 = calc_sma(df_1d['close'], GAINER_PARAMS["SMA_1D"]['params'], GAINER_PARAMS["SMA_1D"]['weights']) * w[5]
    
    total = sum(w)
    raw = (s1 + s3 + s6) / total if total > 0 else 0.0
    
    return raw

# --- Execution Engine ---

def limit_chaser(api, side, size, start_price):
    """
    Executes a Post-Only Limit Order.
    FIX: Cancels previous order BEFORE placing new one to prevent stacking.
    """
    if dry: return
    
    size = round(size, 4)
    if size < MIN_TRADE_SIZE:
        log.warning(f"Chaser Skipped: Size {size} < Min {MIN_TRADE_SIZE}")
        return

    # Initial Price
    if side == "buy": limit_px = int(start_price - LIMIT_OFFSET_TICKS)
    else: limit_px = int(start_price + LIMIT_OFFSET_TICKS)
    
    log.info(f"Starting Limit Chaser: {side} {size} @ {limit_px}")
    
    order_id = None
    
    for i in range(int(LIMIT_CHASE_DURATION / CHASE_INTERVAL)):
        try:
            # 1. CRITICAL: Cancel Old Order First
            if order_id:
                try:
                    c_resp = api.cancel_order({"orderId": order_id, "symbol": SYMBOL_FUTS})
                    log.info(f"[API_RES] cancel_order: {c_resp}")
                except Exception as e:
                    # If cancel fails (e.g. already filled), we must stop chasing
                    log.warning(f"Cancel failed (Filled?): {e}")
                    break
            
            # Double check: Cancel All for this symbol just in case ID tracking failed
            if i == 0: 
                 try: 
                    ca_resp = api.cancel_all_orders({"symbol": SYMBOL_FUTS})
                    log.info(f"[API_RES] cancel_all_orders: {ca_resp}")
                 except: pass

            # 2. Place New Order
            payload = {
                "orderType": "lmt", 
                "symbol": SYMBOL_FUTS, 
                "side": side, 
                "size": size, 
                "limitPrice": limit_px,
                "postOnly": True
            }
            log.info(f"Chaser [{i}] Sending: {limit_px}")
            resp = api.send_order(payload)
            log.info(f"[API_RES] send_order: {resp}")
            
            if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                order_id = resp["sendStatus"]["order_id"]
            else:
                log.warning(f"Chaser Rejected: {resp}")
                # Likely "would execute". Since we strictly want post-only, we skip this cycle
                order_id = None 
                
        except Exception as e:
            log.error(f"Chaser Loop Error: {e}")
            break
            
        time.sleep(CHASE_INTERVAL)
        
        # 3. Update Price
        try:
            resp_tk = api.get_tickers()
            # Truncate log to 200 chars to avoid spam
            log.info(f"[API_RES] get_tickers (chaser): {str(resp_tk)[:200]}...")
            for t in resp_tk.get("tickers", []):
                if t["symbol"] == SYMBOL_FUTS:
                    bid = float(t["bid"])
                    ask = float(t["ask"])
                    
                    if side == "buy":
                        # Converge to Best Bid
                        limit_px = int(bid)
                    else:
                        limit_px = int(ask)
                    break
        except Exception as e: log.error(f"Chaser ticker update failed: {e}")
        
    # Cleanup at end
    try: 
        ca_resp_end = api.cancel_all_orders({"symbol": SYMBOL_FUTS})
        log.info(f"[API_RES] cancel_all_orders (cleanup): {ca_resp_end}")
    except: pass


def manage_virtual_stops(api, state, net_size, price, cap_per_strat):
    net_size = round(net_size, 4)
    if dry or abs(net_size) < MIN_TRADE_SIZE: return
    
    try: 
        ca_resp = api.cancel_all_orders({"symbol": SYMBOL_FUTS})
        log.info(f"[API_RES] cancel_all_orders (stops): {ca_resp}")
    except: pass
    
    # 1. Planner Stops
    p_net_lev = run_planner(state["df_1d"], state, cap_per_strat)
    if abs(p_net_lev) > 0.01: 
        side = "sell" if net_size > 0 else "buy"
        stop_px = int(price * (1 - PLANNER_PARAMS["S1_STOP"])) if side == "sell" else int(price * (1 + PLANNER_PARAMS["S1_STOP"]))
        qty = round(abs(net_size) * 0.33, 4)
        if qty >= MIN_TRADE_SIZE:
            try:
                s_resp = api.send_order({
                    "orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, 
                    "size": qty, "stopPrice": stop_px, "reduceOnly": True
                })
                log.info(f"[API_RES] send_order (planner stop): {s_resp}")
            except Exception as e: log.error(f"Planner Stop failed: {e}")

    # 2. Tumbler Stop
    side = "sell" if net_size > 0 else "buy"
    stop_px = int(price * (1 - TUMBLER_PARAMS["STOP"])) if side == "sell" else int(price * (1 + TUMBLER_PARAMS["STOP"]))
    qty = round(abs(net_size) * 0.33, 4)
    if qty >= MIN_TRADE_SIZE:
        try:
             s_resp_t = api.send_order({
                "orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, 
                "size": qty, "stopPrice": stop_px, "reduceOnly": True
            })
             log.info(f"[API_RES] send_order (tumbler stop): {s_resp_t}")
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
    
    curr_price = get_market_price(api)
    if curr_price == 0: curr_price = df_1h['close'].iloc[-1]
    
    try:
        accts = api.get_accounts()
        log.info(f"[API_RES] get_accounts: {accts}")
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        strat_cap = total_pv * CAP_SPLIT
        log.info(f"PV: ${total_pv:.2f} | StratAlloc: ${strat_cap:.2f} | GlobalMult: {GLOBAL_LEVERAGE_MULTIPLIER}x")
    except Exception as e:
        log.error(f"Account error: {e}")
        return
    
    # Calculate Signals
    lev_planner = run_planner(df_1d, state, strat_cap)
    lev_tumbler = run_tumbler(df_1d, state, strat_cap)
    lev_gainer = run_gainer(df_1h, df_1d)
    
    log.info(f"Sigs | Plan: {lev_planner:.2f} | Tumb: {lev_tumbler:.2f} | Gain: {lev_gainer:.2f}")
    
    notional_planner = lev_planner * strat_cap
    notional_tumbler = lev_tumbler * strat_cap
    notional_gainer = lev_gainer * strat_cap
    
    raw_net_notional = notional_planner + notional_tumbler + notional_gainer
    
    # MODIFIED: APPLY GLOBAL MULTIPLIER to the FINAL Aggregate Position
    final_net_notional = raw_net_notional * GLOBAL_LEVERAGE_MULTIPLIER
    
    target_qty = final_net_notional / curr_price
    
    curr_qty = get_net_position(api)
    delta = target_qty - curr_qty
    
    log.info(f"RawNotional: ${raw_net_notional:.2f} | FinalNotional (4x): ${final_net_notional:.2f} | Tgt: {target_qty:.4f} | Delta: {delta:.4f}")
    
    if abs(delta) >= MIN_TRADE_SIZE:
        side = "buy" if delta > 0 else "sell"
        limit_chaser(api, side, abs(delta), curr_price)
        log.info("Sleeping 10s for position update...")
        time.sleep(10)
    else:
        log.info(f"Delta < Min, skipping.")
    
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
