#!/usr/bin/env python3
"""
planner.py - Fixed Maturity Trend System
Target: FF_XBTUSD_260327
Updates:
- Returns Status String for Main Report.
- Logs cleaner Order IDs instead of raw JSON.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import kraken_ohlc
import kraken_futures as kf

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

SYMBOL_FUTS_UC = "FF_XBTUSD_260327"
SYMBOL_OHLC_KRAKEN = "XBTUSD" 
INTERVAL_KRAKEN = 1440

S1_SMA = 120
S1_DECAY_DAYS = 40
S1_STOP_PCT = 0.13  
S2_SMA = 400
S2_PROX_PCT = 0.05  
S2_STOP_PCT = 0.27  

STOP_WAIT_TIME = 600  
LIMIT_OFFSET_PCT = 0.0002 
MIN_TRADE_SIZE = 0.0001  

log = logging.getLogger("planner")
STATE_FILE = Path("planner_state.json")

def load_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f: return json.load(f)
        except Exception: pass
    return {"s1": {"entry_date": None, "peak_equity": 0.0, "stopped": False}, "s2": {"peak_equity": 0.0, "stopped": False}, "starting_capital": None, "trades": []}

def save_state(state: Dict):
    try:
        with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)
    except Exception as e: log.error(f"State save error: {e}")

def get_sma(prices, window):
    if len(prices) < window: return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def calculate_decay_weight(entry_date_str):
    if not entry_date_str: return 1.0
    entry_dt = datetime.fromisoformat(entry_date_str)
    if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    days_since = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400
    if days_since >= S1_DECAY_DAYS: return 0.0
    weight = 1.0 - (days_since / S1_DECAY_DAYS) ** 2
    return max(0.0, weight)

def update_trailing_stops(state, current_equity, s1_active, s2_active):
    if state["starting_capital"] is None: state["starting_capital"] = current_equity
    # S1
    s1 = state["s1"]
    if s1_active:
        if current_equity > s1["peak_equity"]: s1["peak_equity"] = current_equity
        dd = (s1["peak_equity"] - current_equity) / s1["peak_equity"] if s1["peak_equity"] > 0 else 0
        if dd > S1_STOP_PCT:
            log.warning(f"S1 STOPPED. DD: {dd*100:.2f}%")
            s1["stopped"] = True; s1["entry_date"] = None
    else:
        if not s1["stopped"]: s1["peak_equity"] = current_equity
    # S2
    s2 = state["s2"]
    if s2_active:
        if current_equity > s2["peak_equity"]: s2["peak_equity"] = current_equity
        dd = (s2["peak_equity"] - current_equity) / s2["peak_equity"] if s2["peak_equity"] > 0 else 0
        if dd > S2_STOP_PCT:
            log.warning(f"S2 STOPPED. DD: {dd*100:.2f}%")
            s2["stopped"] = True
    else:
        if not s2["stopped"]: s2["peak_equity"] = current_equity
    return state

def get_strategy_signals(price, sma120, sma400, state):
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
    except Exception: return 0.0

def get_market_price(api, symbol) -> float:
    try:
        resp = api.get_tickers()
        for t in resp.get("tickers", []):
            if t.get("symbol").upper() == symbol.upper(): return float(t.get("markPrice"))
        raise ValueError("Ticker not found")
    except Exception as e:
        log.error(f"Price fetch error: {e}"); return 0.0

def execute_delta_order(api, symbol: str, delta_size: float, current_price: float):
    abs_size = round(abs(delta_size), 4)
    if abs_size < MIN_TRADE_SIZE:
        log.info(f"Already Correctly Positioned (Delta {abs_size} < Min).")
        return "SKIPPED"

    side = "buy" if delta_size > 0 else "sell"
    limit_price = int(round(current_price * (1.0 - LIMIT_OFFSET_PCT) if side == "buy" else current_price * (1.0 + LIMIT_OFFSET_PCT)))
    
    log.info(f"Placing LIMIT {side} {abs_size} @ {limit_price}")
    if dry: return "FILLED"

    try:
        resp = api.send_order({"orderType": "lmt", "symbol": symbol, "side": side, "size": abs_size, "limitPrice": limit_price})
        if "sendStatus" in resp: log.info(f"Limit Sent: {resp.get('sendStatus')}")
        else: log.info(f"Limit Resp: {resp}")
        
        log.info(f"Waiting {STOP_WAIT_TIME}s for fill...")
        time.sleep(STOP_WAIT_TIME)
        
        log.info(f"Cancelling pending orders...")
        api.cancel_all_orders({"symbol": symbol})
        return "CHECK_AGAIN"
    except Exception as e:
        log.error(f"Limit failed: {e}"); return "ERROR"

def manage_stop_loss_orders(api, symbol, current_price, allocated_collateral, net_size, s1_lev, s2_lev, state):
    if dry or abs(net_size) < MIN_TRADE_SIZE: return

    try: api.cancel_all_orders({"symbol": symbol})
    except: pass

    is_long = net_size > 0
    side = "sell" if is_long else "buy"
    direction = 1 if is_long else -1
    
    def place_stop(label, qty, stop_px):
        qty = round(qty, 4)
        if qty < MIN_TRADE_SIZE: return
        stop_px_int = int(round(stop_px))
        log.info(f"[{label}] Placing STOP {side} {qty} @ {stop_px_int}")
        try:
            resp = api.send_order({"orderType": "stp", "symbol": symbol, "side": side, "size": qty, "stopPrice": stop_px_int, "reduceOnly": True})
            if "sendStatus" in resp: log.info(f"[{label}] Stop Sent")
            else: log.info(f"[{label}] Stop Resp: {resp}")
        except Exception as e: log.error(f"[{label}] Failed: {e}")

    # S1
    if not state["s1"]["stopped"] and abs(s1_lev) > 0.01:
        peak_s1 = state["s1"]["peak_equity"]
        if peak_s1 == 0: peak_s1 = allocated_collateral
        loss_allowance = peak_s1 * (1 - S1_STOP_PCT) - allocated_collateral
        stop_price_s1 = current_price + (loss_allowance / abs(net_size)) * direction
        place_stop("S1", (allocated_collateral * abs(s1_lev)) / current_price, stop_price_s1)

    # S2
    if not state["s2"]["stopped"] and abs(s2_lev) > 0.01:
        peak_s2 = state["s2"]["peak_equity"]
        if peak_s2 == 0: peak_s2 = allocated_collateral
        loss_allowance = peak_s2 * (1 - S2_STOP_PCT) - allocated_collateral
        stop_price_s2 = current_price + (loss_allowance / abs(net_size)) * direction
        place_stop("S2", (allocated_collateral * abs(s2_lev)) / current_price, stop_price_s2)

def daily_trade(api, capital_pct=1.0) -> str:
    log.info("--- Starting Planner ---")
    
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, interval=INTERVAL_KRAKEN)
    if df.empty: return "ERROR: No OHLC"
        
    current_spot = df['close'].iloc[-1]
    sma120 = get_sma(df['close'], S1_SMA)
    sma400 = get_sma(df['close'], S2_SMA)
    
    state = load_state()
    try:
        accts = api.get_accounts()
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        collateral = total_pv * capital_pct
        log.info(f"Allocated Capital: ${collateral:.2f}")
    except Exception as e:
        log.error(f"Acct check failed: {e}"); return f"ERROR: {e}"

    state = update_trailing_stops(state, collateral, True, True)
    target_leverage, s1_lev, s2_lev, state = get_strategy_signals(current_spot, sma120, sma400, state)
    
    futs_price = get_market_price(api, SYMBOL_FUTS_UC)
    if futs_price == 0: futs_price = current_spot
    
    target_qty = (collateral * target_leverage) / futs_price
    current_qty = get_current_net_position(api, SYMBOL_FUTS_UC)
    delta_qty = target_qty - current_qty
    
    log.info(f"TgtLev: {target_leverage} | TgtQty: {target_qty:.4f} | Curr: {current_qty:.4f}")
    
    res = execute_delta_order(api, SYMBOL_FUTS_UC, delta_qty, futs_price)
    
    if res == "CHECK_AGAIN" and not dry:
        time.sleep(2)
        rem_delta = round(target_qty - get_current_net_position(api, SYMBOL_FUTS_UC), 4)
        if abs(rem_delta) >= MIN_TRADE_SIZE:
            log.info(f"Fallback MKT: {rem_delta}")
            try: 
                api.send_order({"orderType": "mkt", "symbol": SYMBOL_FUTS_UC, "side": "buy" if rem_delta > 0 else "sell", "size": abs(rem_delta)})
                log.info("Waiting 10s for propagation...")
                time.sleep(10)
            except Exception as e: log.error(f"Fallback failed: {e}")
    
    # Refresh & Stops
    final_net_size = get_current_net_position(api, SYMBOL_FUTS_UC)
    manage_stop_loss_orders(api, SYMBOL_FUTS_UC, futs_price, collateral, final_net_size, s1_lev, s2_lev, state)

    state["trades"].append({"date": datetime.now().isoformat(), "price": futs_price, "leverage": target_leverage})
    save_state(state)
    
    if abs(delta_qty) < MIN_TRADE_SIZE:
        return f"ALREADY POSITIONED ({current_qty:.4f})"
    return f"EXECUTED (Lev: {target_leverage})"
