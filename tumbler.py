#!/usr/bin/env python3
"""
tumbler.py - Perpetual Future Strategy
Target: PF_XBTUSD
Safe Mode: Accepts `capital_pct` to limit exposure.
UPDATES:
- Logs full API responses for debugging.
- cancel_all is strictly scoped to PF_XBTUSD.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import kraken_ohlc
import kraken_futures as kf

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_FUTS_LC = "pf_xbtusd"
SYMBOL_OHLC_KRAKEN = "XBTUSD"
INTERVAL_KRAKEN = 1440

# --- Strategy Parameters ---
SMA_PERIOD_1 = 32   
SMA_PERIOD_2 = 114  
STATIC_STOP_PCT = 0.043  
TAKE_PROFIT_PCT = 0.126  
LIMIT_OFFSET_PCT = 0.0002  
STOP_WAIT_TIME = 600  

III_WINDOW = 27  
III_T_LOW = 0.058  
III_T_HIGH = 0.259  
LEV_LOW = 0.04
LEV_MID = 2.15
LEV_HIGH = 1.935

FLAT_REGIME_THRESHOLD = 0.356  
BAND_WIDTH_PCT = 0.077  

STATE_FILE = Path("sma_state.json")

log = logging.getLogger("tumbler")

# --- Helpers ---
def calculate_iii(df):
    if len(df) < III_WINDOW + 1: return 0.0
    df_calc = df.copy()
    df_calc['log_ret'] = np.log(df_calc['close'] / df_calc['close'].shift(1))
    w = III_WINDOW
    iii_series = (df_calc['log_ret'].rolling(w).sum().abs() / df_calc['log_ret'].abs().rolling(w).sum()).fillna(0)
    return float(iii_series.iloc[-1])

def determine_leverage(iii):
    if iii < III_T_LOW: return LEV_LOW
    elif iii < III_T_HIGH: return LEV_MID
    else: return LEV_HIGH

def calculate_smas(df):
    df = df.copy()
    df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    return df

def check_flat_regime_trigger(iii, current_flat_regime):
    if iii < FLAT_REGIME_THRESHOLD:
        if not current_flat_regime: log.info(f"ENTERING FLAT REGIME: III={iii:.4f}")
        return True
    return current_flat_regime

def check_flat_regime_release(df, current_flat_regime):
    if not current_flat_regime: return False
    df_calc = calculate_smas(df)
    curr = df_calc.iloc[-1]
    if pd.isna(curr['sma_1']) or pd.isna(curr['sma_2']): return True
    
    diff_sma1 = abs(curr['close'] - curr['sma_1'])
    diff_sma2 = abs(curr['close'] - curr['sma_2'])
    thresh_sma1 = curr['sma_1'] * BAND_WIDTH_PCT
    thresh_sma2 = curr['sma_2'] * BAND_WIDTH_PCT
    
    if diff_sma1 <= thresh_sma1 or diff_sma2 <= thresh_sma2:
        log.info(f"RELEASING FLAT REGIME: Price ${curr['close']:.2f} entered band")
        return False
    return True

def mark_price(api):
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC: return float(t["markPrice"])
    raise RuntimeError("Mark-price for PF_XBTUSD not found")

def cancel_all_pf(api):
    """Safely cancel orders ONLY for PF_XBTUSD."""
    log.info(f"Cancelling all orders for {SYMBOL_FUTS_UC}...")
    try:
        api.cancel_all_orders({"symbol": SYMBOL_FUTS_UC})
    except Exception as e:
        log.warning(f"cancel_all_orders failed: {e}")

def get_current_position(api):
    try:
        pos = api.get_open_positions()
        for p in pos.get("openPositions", []):
            if p["symbol"] == SYMBOL_FUTS_UC:
                return {
                    "signal": "LONG" if p["side"] == "long" else "SHORT",
                    "side": p["side"],
                    "size_btc": abs(float(p["size"])),
                }
        return None
    except Exception as e:
        log.warning(f"Failed to get position: {e}"); return None

# --- Execution ---
def flatten_position_limit(api, current_price):
    pos = get_current_position(api)
    if not pos: return
    
    side = "sell" if pos["side"] == "long" else "buy"
    size = pos["size_btc"]
    
    limit_price = int(round(current_price * (1 + LIMIT_OFFSET_PCT) if side == "sell" else current_price * (1 - LIMIT_OFFSET_PCT)))
    log.info(f"Flattening LIMIT: {side} {size} @ {limit_price}")
    if not dry:
        try:
            resp = api.send_order({"orderType": "lmt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": size, "limitPrice": limit_price})
            log.info(f"Flatten Limit Resp: {resp}")
        except Exception as e: log.error(f"Flatten limit failed: {e}")

def flatten_position_market(api):
    pos = get_current_position(api)
    if not pos: return
    side = "sell" if pos["side"] == "long" else "buy"
    log.info(f"Flattening MARKET: {side} {pos['size_btc']}")
    if not dry:
        try:
            resp = api.send_order({"orderType": "mkt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": pos['size_btc']})
            log.info(f"Flatten MKT Resp: {resp}")
        except Exception as e: log.error(f"Flatten market failed: {e}")
    cancel_all_pf(api)

def open_position(api, signal, leverage, collateral, current_price):
    notional = collateral * leverage
    size_btc = round(notional / current_price, 4)
    side = "buy" if signal == "LONG" else "sell"
    
    log.info(f"Opening {signal} | Lev: {leverage}x | Size: {size_btc:.4f}")
    if dry: return size_btc, current_price
    
    # 1. Limit Entry
    limit_price = int(round(current_price * (1 - LIMIT_OFFSET_PCT) if side == "buy" else current_price * (1 + LIMIT_OFFSET_PCT)))
    try:
        resp = api.send_order({"orderType": "lmt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": size_btc, "limitPrice": limit_price})
        log.info(f"Entry Limit Resp: {resp}")
        time.sleep(STOP_WAIT_TIME)
    except Exception as e: log.error(f"Entry limit failed: {e}")
    
    # 2. Market Cleanup
    cancel_all_pf(api) 
    
    pos = get_current_position(api)
    filled_size = pos["size_btc"] if pos else 0.0
    
    # Fill remainder
    remaining = size_btc - filled_size
    if remaining > 0.0001:
        log.info(f"Filling remaining {remaining:.4f} via MARKET")
        try:
            resp = api.send_order({"orderType": "mkt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": remaining})
            log.info(f"Entry MKT Resp: {resp}")
        except Exception as e: log.error(f"Entry market failed: {e}")
        
    # 3. Stops & TP
    pos_final = get_current_position(api)
    if pos_final:
        final_size = pos_final["size_btc"]
        fill_price = current_price # Approx
        
        # Stop Loss
        sl_dist = fill_price * STATIC_STOP_PCT
        # FIX: Ensure int price
        sl_price = int(round(fill_price - sl_dist if side == "buy" else fill_price + sl_dist))
        sl_side = "sell" if side == "buy" else "buy"
        try:
            resp = api.send_order({"orderType": "stp", "symbol": SYMBOL_FUTS_LC, "side": sl_side, "size": final_size, "stopPrice": sl_price, "reduceOnly": True})
            log.info(f"Placed Stop @ {sl_price} | Resp: {resp}")
        except Exception as e: log.error(f"SL failed: {e}")
        
        # TP
        tp_price = int(round(fill_price * (1 + TAKE_PROFIT_PCT) if side == "buy" else fill_price * (1 - TAKE_PROFIT_PCT)))
        try:
            resp = api.send_order({"orderType": "lmt", "symbol": SYMBOL_FUTS_LC, "side": sl_side, "size": final_size, "limitPrice": tp_price, "reduceOnly": True})
            log.info(f"Placed TP @ {tp_price} | Resp: {resp}")
        except Exception as e: log.error(f"TP failed: {e}")
        
        return final_size, fill_price
    return 0, 0

def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"flat_regime_active": False, "trades": [], "starting_capital": None}

def save_state(st):
    STATE_FILE.write_text(json.dumps(st, indent=2))

def init_tumbler(api):
    log.info("Initializing Tumbler...")
    if not STATE_FILE.exists():
        df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
        iii = calculate_iii(df)
        is_flat = iii < FLAT_REGIME_THRESHOLD
        save_state({"flat_regime_active": is_flat, "trades": [], "starting_capital": None})

def daily_trade(api, capital_pct=1.0):
    log.info("--- Starting Tumbler Cycle ---")
    state = load_state()
    
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    curr_price = mark_price(api)
    
    try:
        accts = api.get_accounts()
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        collateral = total_pv * capital_pct
        log.info(f"Total PV: ${total_pv:.2f} | Tumbler Allocated: ${collateral:.2f}")
    except Exception as e:
        log.error(f"Account check failed: {e}"); return

    if state["starting_capital"] is None: state["starting_capital"] = collateral
    
    iii = calculate_iii(df)
    leverage = determine_leverage(iii)
    
    current_flat = state.get("flat_regime_active", False)
    is_flat = check_flat_regime_trigger(iii, current_flat)
    if is_flat: is_flat = check_flat_regime_release(df, is_flat)
    state["flat_regime_active"] = is_flat
    
    df_calc = calculate_smas(df)
    sma1 = df_calc['sma_1'].iloc[-1]
    sma2 = df_calc['sma_2'].iloc[-1]
    
    signal = "FLAT"
    if not is_flat:
        if curr_price > sma1 and curr_price > sma2: signal = "LONG"
        elif curr_price < sma1 and curr_price < sma2: signal = "SHORT"
    
    log.info(f"Signal: {signal} | FlatRegime: {is_flat} | Lev: {leverage}x")
    
    # 1. Clear OLD orders only
    cancel_all_pf(api)
    
    # 2. Close if signal changed or flat
    pos = get_current_position(api)
    needs_flatten = False
    if pos:
        if signal == "FLAT": needs_flatten = True
        elif signal == "LONG" and pos["side"] == "short": needs_flatten = True
        elif signal == "SHORT" and pos["side"] == "long": needs_flatten = True
        
    if needs_flatten:
        flatten_position_limit(api, curr_price)
        if not dry: time.sleep(STOP_WAIT_TIME)
        flatten_position_market(api)
    
    # 3. Open New
    if signal != "FLAT":
        # Check if we already have correct position
        pos = get_current_position(api)
        if not pos:
            size, price = open_position(api, signal, leverage, collateral, curr_price)
            if size > 0:
                state["trades"].append({"date": datetime.now().isoformat(), "signal": signal, "size": size, "price": price, "leverage": leverage})

    save_state(state)
    log.info("Tumbler Cycle Complete.")
