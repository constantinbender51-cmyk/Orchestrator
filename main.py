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
- Tumbler Corrected: SMA1/SMA2 Release + 12.6% Take Profit.
- Planner Final: Independent Virtual Equity & Symmetric Logic.
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
SYMBOL_FUTS = "FF_XBTUSD_260626"
SYMBOL_OHLC = "XBTUSD"
CAP_SPLIT = 0.333
LIMIT_CHASE_DURATION = 720
CHASE_INTERVAL = 60
MIN_TRADE_SIZE = 0.0002
LIMIT_OFFSET_TICKS = 1
GLOBAL_LEVERAGE=5

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
slow_h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M"))
log.addHandler(slow_h)

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

# --- Strategy Parameters ---
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

# --- State Management ---
def load_state():
    defaults = {
        "planner": {
            "s1_equity": 0.0, # 0.0 triggers init to strat_cap
            "s2_equity": 0.0,
            "last_price": 0.0,
            "last_lev_s1": 0.0,
            "last_lev_s2": 0.0,
            "s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False, "trend": 0},
            "s2": {"peak_equity": 0.0, "stopped": False, "trend": 0},
            "debug_levs": [0.0, 0.0] 
        },
        "tumbler": {"flat_regime": False}
    }
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                # Deep merge defaults
                for k, v in defaults.items():
                    if k not in saved: saved[k] = v
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if sub_k not in saved[k]: saved[k][sub_k] = sub_v
                return saved
        except Exception as e: log.error(f"State load error: {e}")
    return defaults

def save_state(state):
    try:
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

def calculate_decay(entry_date_str, decay_days):
    if not entry_date_str: return 1.0
    entry_dt = datetime.fromisoformat(entry_date_str)
    if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    days_since = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
    if days_since >= decay_days: return 0.0
    weight = 1.0 - (days_since / decay_days) ** 2
    return max(0.0, weight)

# --- Strategy Logic ---

def run_planner(df_1d, state, capital):
    """
    Corrected Planner Logic:
    1. Independent Virtual Equity for S1 and S2.
    2. Symmetric Short/Long Logic.
    3. Correct Proximity Sizing for S2.
    """
    p_state = state["planner"]
    price = df_1d['close'].iloc[-1]
    
    # --- 1. State Initialization (Independent Buckets) ---
    if p_state["s1_equity"] <= 0.0: p_state["s1_equity"] = capital
    if p_state["s2_equity"] <= 0.0: p_state["s2_equity"] = capital
    if p_state["last_price"] <= 0.0: p_state["last_price"] = price

    # --- 2. Update Virtual Equities Independently ---
    last_p = p_state["last_price"]
    if last_p > 0:
        pct_change = (price - last_p) / last_p
        
        # Update S1 Equity
        s1_pnl = pct_change * p_state["last_lev_s1"]
        p_state["s1_equity"] *= (1.0 + s1_pnl)
        
        # Update S2 Equity
        s2_pnl = pct_change * p_state["last_lev_s2"]
        p_state["s2_equity"] *= (1.0 + s2_pnl)

    # --- 3. Strategy 1: Tactical Trend (Symmetric) ---
    sma120 = get_sma(df_1d['close'], PLANNER_PARAMS["S1_SMA"])
    s1_trend = 1 if price > sma120 else -1
    s1 = p_state["s1"]
    
    # Trend Change / Reset Logic
    if s1.get("trend", 0) != s1_trend:
        s1["trend"] = s1_trend
        s1["entry_date"] = datetime.now(timezone.utc).isoformat()
        s1["stopped"] = False
        s1["peak_equity"] = p_state["s1_equity"] # Reset Peak
    
    # Trailing Stop Check
    if p_state["s1_equity"] > s1["peak_equity"]:
        s1["peak_equity"] = p_state["s1_equity"]
    
    dd_s1 = 0.0
    if s1["peak_equity"] > 0:
        dd_s1 = (s1["peak_equity"] - p_state["s1_equity"]) / s1["peak_equity"]
    
    if dd_s1 > PLANNER_PARAMS["S1_STOP"]:
        s1["stopped"] = True

    # Sizing Calculation
    s1_lev = 0.0
    if not s1["stopped"]:
        decay_w = calculate_decay(s1["entry_date"], PLANNER_PARAMS["S1_DECAY"])
        s1_lev = float(s1_trend) * decay_w
    
    # --- 4. Strategy 2: Core Trend (Symmetric + Prox) ---
    sma400 = get_sma(df_1d['close'], PLANNER_PARAMS["S2_SMA"])
    s2_trend = 1 if price > sma400 else -1
    s2 = p_state["s2"]
    
    # Trend Change Logic
    if s2.get("trend", 0) != s2_trend:
        s2["trend"] = s2_trend
        s2["stopped"] = False
        s2["peak_equity"] = p_state["s2_equity"]

    # Trailing Stop Check
    if p_state["s2_equity"] > s2["peak_equity"]:
        s2["peak_equity"] = p_state["s2_equity"]
        
    dd_s2 = 0.0
    if s2["peak_equity"] > 0:
        dd_s2 = (s2["peak_equity"] - p_state["s2_equity"]) / s2["peak_equity"]
        
    if dd_s2 > PLANNER_PARAMS["S2_STOP"]:
        s2["stopped"] = True

    # Proximity Check (Active & Re-entry)
    dist_pct = abs(price - sma400) / sma400
    is_prox = dist_pct < PLANNER_PARAMS["S2_PROX"]
    tgt_size = 0.5 if is_prox else 1.0

    s2_lev = 0.0
    if s2["stopped"]:
        # Re-entry allowed if Proximity is True
        if is_prox:
            s2["stopped"] = False 
            s2["peak_equity"] = p_state["s2_equity"] 
            s2_lev = float(s2_trend) * tgt_size
    else:
        s2_lev = float(s2_trend) * tgt_size

    # --- 5. Final State Update ---
    net_lev = max(-2.0, min(2.0, s1_lev + s2_lev))
    
    p_state["last_price"] = price
    p_state["last_lev_s1"] = s1_lev
    p_state["last_lev_s2"] = s2_lev
    p_state["debug_levs"] = [s1_lev, s2_lev]
    
    return net_lev

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
        sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
        curr = df_1d['close'].iloc[-1]
        band1, band2 = sma1 * TUMBLER_PARAMS["BAND"], sma2 * TUMBLER_PARAMS["BAND"]
        if abs(curr - sma1) <= band1 or abs(curr - sma2) <= band2:
            s["flat_regime"] = False
            
    if s["flat_regime"]: return 0.0
    
    sma1, sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"]), get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    curr = df_1d['close'].iloc[-1]
    return lev if (curr > sma1 and curr > sma2) else (-lev if (curr < sma1 and curr < sma2) else 0.0)

def run_gainer(df_1h, df_1d):
    def calc_macd_pos(prices, config):
        params, weights = config['params'], config['weights']
        composite = 0.0
        for (f, s, sig_p), w in zip(params, weights):
            fast = prices.ewm(span=f, adjust=False).mean()
            slow = prices.ewm(span=s, adjust=False).mean()
            macd = fast - slow
            sig_line = macd.ewm(span=sig_p, adjust=False).mean()
            composite += (1.0 if macd.iloc[-1] > sig_line.iloc[-1] else -1.0) * w
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else composite

    def calc_sma_pos(prices, config):
        params, weights = config['params'], config['weights']
        composite = 0.0
        current = prices.iloc[-1]
        for p, w in zip(params, weights):
            composite += (1.0 if current > get_sma(prices, p) else -1.0) * w
        total_w = sum(weights)
        return composite / total_w if total_w > 0 else composite

    m1h = calc_macd_pos(df_1h['close'], GAINER_PARAMS["MACD_1H"]) * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1H"]
    m1d = calc_macd_pos(df_1d['close'], GAINER_PARAMS["MACD_1D"]) * GAINER_PARAMS["GA_WEIGHTS"]["MACD_1D"]
    s1d = calc_sma_pos(df_1d['close'], GAINER_PARAMS["SMA_1D"]) * GAINER_PARAMS["GA_WEIGHTS"]["SMA_1D"]
    return (m1h + m1d + s1d) / sum(GAINER_PARAMS["GA_WEIGHTS"].values())

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

def manage_stops(api, net_size, price, state, strat_cap):
    if dry or abs(net_size) < MIN_TRADE_SIZE: return
    side = "sell" if net_size > 0 else "buy"
    direction = 1 if net_size > 0 else -1
    
    # 1. Planner Stops (Independent Virtual Equity Logic)
    p_state = state["planner"]
    s1_lev, s2_lev = p_state.get("debug_levs", [0.0, 0.0])
    
    def place_stop(label, qty, stop_px):
        qty = round(qty, 4)
        if qty < MIN_TRADE_SIZE: return
        stop_px_int = int(round(stop_px))
        try:
            api.send_order({"orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, "size": qty, "stopPrice": stop_px_int, "reduceOnly": True})
        except Exception as e: log.error(f"[{label}] Stop Fail: {e}")

    # Planner S1 Stop (13%)
    if abs(s1_lev) > 0.01 and not p_state["s1"]["stopped"]:
        # SIZE = Planner's Contribution
        s1_qty = (strat_cap * abs(s1_lev)) / price 
        
        # STOP PRICE Calculation based on S1 Specific Drawdown
        s = p_state["s1"]
        peak = s["peak_equity"]
        curr_eq = p_state["s1_equity"]
        current_dd_pct = (peak - curr_eq) / peak if peak > 0 else 0
        room_pct = PLANNER_PARAMS["S1_STOP"] - current_dd_pct
        
        if room_pct > 0:
            # Adjust distance by leverage: Tighter stop for higher leverage
            dist_pct = room_pct / abs(s1_lev)
            stop_px = price * (1 - dist_pct * direction)
            place_stop("PlanS1", s1_qty, stop_px)

    # Planner S2 Stop (27%)
    if abs(s2_lev) > 0.01 and not p_state["s2"]["stopped"]:
        s2_qty = (strat_cap * abs(s2_lev)) / price
        s = p_state["s2"]
        peak = s["peak_equity"]
        curr_eq = p_state["s2_equity"]
        current_dd_pct = (peak - curr_eq) / peak if peak > 0 else 0
        room_pct = PLANNER_PARAMS["S2_STOP"] - current_dd_pct
        
        if room_pct > 0:
            dist_pct = room_pct / abs(s2_lev)
            stop_px = price * (1 - dist_pct * direction)
            place_stop("PlanS2", s2_qty, stop_px)

    # 2. Tumbler Stop (4.3% Price Distance - Fixed)
    qty_tumb = round(abs(net_size) * 0.33, 4)
    t_stop = TUMBLER_PARAMS["STOP"]
    px_t = int(price * (1 - t_stop)) if side == "sell" else int(price * (1 + t_stop))
    place_stop("TumbStop", qty_tumb, px_t)

    # 3. Tumbler Take Profit (12.6%)
    t_tp = TUMBLER_PARAMS["TAKE_PROFIT"]
    px_tp = int(price * (1 + t_tp)) if side == "sell" else int(price * (1 - t_tp))
    try:
        api.send_order({"orderType": "lmt", "symbol": SYMBOL_FUTS, "side": side, "size": qty_tumb, "limitPrice": px_tp, "reduceOnly": True})
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
        strat_cap = float(accts["accounts"]["flex"]["portfolioValue"]*GLOBAL_LEVERAGE) * CAP_SPLIT
        log.info(f"Cap: ${strat_cap:.0f} | Price: ${curr_price:.1f}")
    except: return
    
    # Strategies
    r_p = run_planner(df_1d, state, strat_cap)
    r_t = run_tumbler(df_1d, state, strat_cap)
    r_g = run_gainer(df_1h, df_1d)
    
    # Normalization (Fair 2.0)
    n_p, n_t, n_g = r_p, r_t * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV), r_g * TARGET_STRAT_LEV
    log.info(f"Levs | P:{n_p:.2f} T:{n_t:.2f} G:{n_g:.2f}")

    target_qty = (n_p + n_t + n_g) * strat_cap / curr_price
    limit_chaser(api, target_qty)
    
    final_pos = get_net_position(api)
    manage_stops(api, final_pos, curr_price, state, strat_cap)
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
