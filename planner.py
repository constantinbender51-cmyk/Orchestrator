#!/usr/bin/env python3
"""
planner.py - Dual-Horizon Trend System (Optimized for Fixed Futures)
Strategy 1 (Tactical): SMA 120 with 40-day parabolic decay + 13% trailing stop
Strategy 2 (Core): SMA 400 with proximity sizing + 27% trailing stop + re-entry
Combined (S3): Net position = S1 + S2 (-2x to +2x)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

import kraken_futures as kf
import kraken_ohlc

# Configuration
dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
RUN_TRADE_NOW = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

# --- Instrument Settings ---
SYMBOL_FUTS_UC = "FF_XBTUSD_260327"
SYMBOL_FUTS_LC = "ff_xbtusd_260327"
SYMBOL_OHLC_KRAKEN = "XBTUSD" 
INTERVAL_KRAKEN = 1440

# --- Strategy Parameters ---
S1_SMA = 120
S1_DECAY_DAYS = 40
S1_STOP_PCT = 0.13  
S2_SMA = 400
S2_PROX_PCT = 0.05  
S2_STOP_PCT = 0.27  

STOP_WAIT_TIME = 600  
LIMIT_OFFSET_PCT = 0.0002 
MIN_TRADE_SIZE = 0.0001  

# Configure logger
log = logging.getLogger("planner")
if not log.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [PLANNER] %(message)s"))
    log.addHandler(ch)
    log.setLevel(logging.INFO)

STATE_FILE = Path("planner_state.json")

def load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load state: {e}")
    
    return {
        "s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False},
        "s2": {"peak_equity": 0.0, "stopped": False},
        "starting_capital": None, "performance": {}, "trades": []
    }

def save_state(state: Dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save state: {e}")

def get_sma(prices: pd.Series, window: int) -> float:
    if len(prices) < window: return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def calculate_decay_weight(entry_date_str: str) -> float:
    if not entry_date_str: return 1.0
    entry_dt = datetime.fromisoformat(entry_date_str)
    if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    days_since = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
    if days_since >= S1_DECAY_DAYS: return 0.0
    weight = 1.0 - (days_since / S1_DECAY_DAYS) ** 2
    return max(0.0, weight)

def update_trailing_stops(state: Dict, current_equity: float, s1_active: bool, s2_active: bool):
    if state["starting_capital"] is None: state["starting_capital"] = current_equity
    
    # S1
    s1 = state["s1"]
    if s1_active:
        if current_equity > s1["peak_equity"]: s1["peak_equity"] = current_equity
        drawdown = (s1["peak_equity"] - current_equity) / s1["peak_equity"] if s1["peak_equity"] > 0 else 0
        if drawdown > S1_STOP_PCT:
            log.warning(f"S1 STOPPED OUT. DD: {drawdown*100:.2f}%")
            s1["stopped"] = True; s1["entry_date"] = None
    else:
        if not s1["stopped"]: s1["peak_equity"] = current_equity

    # S2
    s2 = state["s2"]
    if s2_active:
        if current_equity > s2["peak_equity"]: s2["peak_equity"] = current_equity
        drawdown = (s2["peak_equity"] - current_equity) / s2["peak_equity"] if s2["peak_equity"] > 0 else 0
        if drawdown > S2_STOP_PCT:
            log.warning(f"S2 STOPPED OUT. DD: {drawdown*100:.2f}%")
            s2["stopped"] = True
    else:
        if not s2["stopped"]: s2["peak_equity"] = current_equity
            
    return state

def get_strategy_signals(price: float, sma120: float, sma400: float, state: Dict) -> Tuple[float, float, float, Dict]:
    # S1
    s1_lev = 0.0
    s1_state = state["s1"]
    if price > sma120:
        if not s1_state["stopped"]:
            if s1_state["entry_date"] is None: s1_state["entry_date"] = datetime.now(timezone.utc).isoformat()
            s1_lev = 1.0 * calculate_decay_weight(s1_state["entry_date"])
    else:
        if s1_state["stopped"]: s1_state["stopped"] = False; s1_state["peak_equity"] = 0.0
        s1_state["entry_date"] = datetime.now(timezone.utc).isoformat()
        s1_lev = -1.0 * calculate_decay_weight(s1_state["entry_date"])

    # S2
    s2_lev = 0.0
    s2_state = state["s2"]
    if price > sma400:
        if not s2_state["stopped"]: s2_lev = 1.0
        else:
            if (price - sma400) / sma400 < S2_PROX_PCT:
                s2_state["stopped"] = False; s2_state["peak_equity"] = 0.0; s2_lev = 0.5
    else:
        if s2_state["stopped"]: s2_state["stopped"] = False
        s2_lev = 0.0

    net_leverage = max(-2.0, min(2.0, s1_lev + s2_lev))
    return net_leverage, s1_lev, s2_lev, state

def get_current_net_position(api, symbol) -> float:
    try:
        resp = api.get_open_positions()
        for p in resp.get("openPositions", []):
            if p.get('symbol').upper() == symbol.upper():
                size = float(p.get('size', 0.0))
                return size if p.get('side') == 'long' else -size
        return 0.0
    except Exception as e:
        log.error(f"Error fetching positions: {e}")
        return 0.0

def get_market_price(api, symbol) -> float:
    try:
        resp = api.get_tickers()
        for t in resp.get("tickers", []):
            if t.get("symbol").upper() == symbol.upper(): return float(t.get("markPrice"))
        for t in resp.get("tickers", []): # Fallback
            if symbol.upper() in t.get("symbol").upper(): return float(t.get("markPrice"))
        raise ValueError(f"Ticker for {symbol} not found")
    except Exception as e:
        log.error(f"Error fetching price: {e}"); return 0.0

def execute_delta_order(api, symbol: str, delta_size: float, current_price: float):
    abs_size = round(abs(delta_size), 4)
    if abs_size < MIN_TRADE_SIZE:
        log.info(f"Delta {abs_size:.4f} < MIN. Skipped.")
        return "SKIPPED"

    side = "buy" if delta_size > 0 else "sell"
    limit_price = int(round(current_price * (1.0 - LIMIT_OFFSET_PCT) if side == "buy" else current_price * (1.0 + LIMIT_OFFSET_PCT)))
    
    log.info(f"EXEC DELTA: {side.upper()} {abs_size:.4f} @ {limit_price}")
    if dry: return "FILLED"

    try:
        api.send_order({"orderType": "lmt", "symbol": symbol, "side": side, "size": abs_size, "limitPrice": limit_price})
        time.sleep(STOP_WAIT_TIME)
        
        cancel_resp = api.cancel_all_orders({"symbol": symbol})
        if cancel_resp.get("cancelStatus", {}).get("status") == "cancelled" or len(cancel_resp.get("cancelledOrders", [])) > 0:
            return "CHECK_AGAIN"
        return "FILLED"
    except Exception as e:
        log.error(f"Order failed: {e}"); return "ERROR"

def manage_stop_loss_orders(api, symbol: str, current_price: float, collateral: float, net_size: float, s1_lev: float, s2_lev: float, state: Dict):
    log.info("--- Managing Safety Stops ---")
    if dry or abs(net_size) < MIN_TRADE_SIZE: return

    try: api.cancel_all_orders({"symbol": symbol})
    except: pass

    is_long = net_size > 0
    side = "sell" if is_long else "buy"
    
    def place_stop(label, qty, stop_px):
        qty = round(qty, 4)
        if qty < MIN_TRADE_SIZE: return
        stop_px_int = int(round(stop_px))
        log.info(f"Placing {label}: Stop {side.upper()} {qty:.4f} @ {stop_px_int}")
        try:
            api.send_order({"orderType": "stp", "symbol": symbol, "side": side, "size": qty, "stopPrice": stop_px_int, "reduceOnly": True})
        except Exception as e: log.error(f"Failed {label}: {e}")

    # S1 Stop
    if not state["s1"]["stopped"] and abs(s1_lev) > 0.01:
        peak_s1 = state["s1"]["peak_equity"]
        if peak_s1 == 0: peak_s1 = collateral
        loss_allowance = peak_s1 * (1 - S1_STOP_PCT) - collateral
        stop_price_s1 = current_price + (loss_allowance/abs(net_size) * (-1 if is_long else 1))
        place_stop("S1", (collateral * abs(s1_lev)) / current_price, stop_price_s1)

    # S2 Stop
    if not state["s2"]["stopped"] and abs(s2_lev) > 0.01:
        peak_s2 = state["s2"]["peak_equity"]
        if peak_s2 == 0: peak_s2 = collateral
        loss_allowance = peak_s2 * (1 - S2_STOP_PCT) - collateral
        stop_price_s2 = current_price + (loss_allowance/abs(net_size) * (-1 if is_long else 1))
        place_stop("S2", (collateral * abs(s2_lev)) / current_price, stop_price_s2)

def daily_trade(api):
    log.info("--- Starting Daily Cycle (Planner) ---")
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, interval=INTERVAL_KRAKEN)
    if df.empty: log.error("No OHLC"); return
        
    current_spot = df['close'].iloc[-1]
    sma120 = get_sma(df['close'], S1_SMA)
    sma400 = get_sma(df['close'], S2_SMA)
    log.info(f"Spot: {current_spot} | SMA120: {sma120:.1f} | SMA400: {sma400:.1f}")
    
    state = load_state()
    try:
        accts = api.get_accounts()
        collateral = float(accts["accounts"]["flex"]["portfolioValue"])
    except: collateral = 0.0; log.error("No Balance")

    log.info(f"Collateral: ${collateral:.2f}")
    
    state = update_trailing_stops(state, collateral, True, True)
    target_leverage, s1_lev, s2_lev, state = get_strategy_signals(current_spot, sma120, sma400, state)
    
    futs_price = get_market_price(api, SYMBOL_FUTS_UC)
    if futs_price == 0: futs_price = current_spot
    
    target_qty = (collateral * target_leverage) / futs_price
    current_qty = get_current_net_position(api, SYMBOL_FUTS_UC)
    delta_qty = target_qty - current_qty
    log.info(f"Pos: {current_qty:.4f} -> {target_qty:.4f} | Delta: {delta_qty:.4f}")
    
    res = execute_delta_order(api, SYMBOL_FUTS_UC, delta_qty, futs_price)
    
    if res == "CHECK_AGAIN" and not dry:
        time.sleep(2)
        rem_delta = round(target_qty - get_current_net_position(api, SYMBOL_FUTS_UC), 4)
        if abs(rem_delta) >= MIN_TRADE_SIZE:
            log.info(f"Fallback Market: {rem_delta:.4f}")
            try: api.send_order({"orderType": "mkt", "symbol": SYMBOL_FUTS_UC, "side": "buy" if rem_delta > 0 else "sell", "size": abs(rem_delta)})
            except Exception as e: log.error(f"Fallback failed: {e}")
    
    final_net_size = get_current_net_position(api, SYMBOL_FUTS_UC)
    manage_stop_loss_orders(api, SYMBOL_FUTS_UC, futs_price, collateral, final_net_size, s1_lev, s2_lev, state)

    state["trades"].append({"date": datetime.now().isoformat(), "price": futs_price, "leverage": target_leverage})
    save_state(state)
    log.info("Planner Cycle Complete.")

def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit(1)
    api = kf.KrakenFuturesApi(api_key, api_sec)
    daily_trade(api)

if __name__ == "__main__":
    main()
