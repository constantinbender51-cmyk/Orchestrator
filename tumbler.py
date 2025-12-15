#!/usr/bin/env python3
"""
tumbler.py - Dual SMA Strategy with Flat Regime Detection + III Dynamic Leverage
SMA 1 (40 days): Primary logic
SMA 2 (120 days): Hard trend filter
III-Based Leverage: 0.5x (choppy) / 4.5x (trending) / 2.45x (overextended)
Flat Regime: Pauses trading when III < 0.16, resumes when price enters 4.5% bands
Trades daily at 00:01 UTC with 2% SL + 16% TP
Uses CURRENT data for all live trading decisions
Initializes flat regime state on startup from historical data
FIXED: III calculation now matches app.py exactly (no window slicing)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional
import subprocess
import numpy as np
import pandas as pd

import kraken_futures as kf
import kraken_ohlc
# import binance_ohlc # Removed if not strictly used to keep imports clean

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
RUN_TRADE_NOW = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_FUTS_LC = "pf_xbtusd"
SYMBOL_OHLC_KRAKEN = "XBTUSD"
INTERVAL_KRAKEN = 1440

# Strategy Parameters - OPTIMIZED FROM GENETIC ALGORITHM
SMA_PERIOD_1 = 32   
SMA_PERIOD_2 = 114  
STATIC_STOP_PCT = 0.043  
TAKE_PROFIT_PCT = 0.126  
LIMIT_OFFSET_PCT = 0.0002  
STOP_WAIT_TIME = 600  

# III Parameters - OPTIMIZED
III_WINDOW = 27  
III_T_LOW = 0.058  
III_T_HIGH = 0.259  
LEV_LOW = 0.079   
LEV_MID = 4.327   
LEV_HIGH = 3.868  

# Flat Regime Parameters - OPTIMIZED
FLAT_REGIME_THRESHOLD = 0.356  
BAND_WIDTH_PCT = 0.077  

STATE_FILE = Path("sma_state.json")

# Configure logger to not conflict if imported
log = logging.getLogger("tumbler")
if not log.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [TUMBLR] %(message)s"))
    log.addHandler(ch)
    log.setLevel(logging.INFO)


def calculate_iii(df: pd.DataFrame) -> float:
    if len(df) < III_WINDOW + 1:
        return 0.0
    df_calc = df.copy()
    df_calc['log_ret'] = np.log(df_calc['close'] / df_calc['close'].shift(1))
    w = III_WINDOW
    iii_series = (df_calc['log_ret'].rolling(w).sum().abs() / 
                  df_calc['log_ret'].abs().rolling(w).sum()).fillna(0)
    return float(iii_series.iloc[-1])


def determine_leverage(iii: float) -> float:
    if iii < III_T_LOW:
        return LEV_LOW
    elif iii < III_T_HIGH:
        return LEV_MID
    else:
        return LEV_HIGH


def calculate_smas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma_1'] = df['close'].rolling(window=SMA_PERIOD_1).mean()
    df['sma_2'] = df['close'].rolling(window=SMA_PERIOD_2).mean()
    return df


def check_flat_regime_trigger(iii: float, current_flat_regime: bool) -> bool:
    if iii < FLAT_REGIME_THRESHOLD:
        if not current_flat_regime:
            log.info(f"ENTERING FLAT REGIME: III={iii:.4f} < {FLAT_REGIME_THRESHOLD}")
        return True
    return current_flat_regime


def check_flat_regime_release(df: pd.DataFrame, current_flat_regime: bool) -> bool:
    if not current_flat_regime:
        return False
    
    df_calc = calculate_smas(df)
    current_close = df_calc['close'].iloc[-1]
    current_sma_1 = df_calc['sma_1'].iloc[-1]
    current_sma_2 = df_calc['sma_2'].iloc[-1]
    
    if pd.isna(current_sma_1) or pd.isna(current_sma_2):
        return True
    
    diff_sma1 = abs(current_close - current_sma_1)
    diff_sma2 = abs(current_close - current_sma_2)
    thresh_sma1 = current_sma_1 * BAND_WIDTH_PCT
    thresh_sma2 = current_sma_2 * BAND_WIDTH_PCT
    
    if diff_sma1 <= thresh_sma1 or diff_sma2 <= thresh_sma2:
        log.info(f"RELEASING FLAT REGIME: Price ${current_close:.2f} entered band")
        return False
    
    return True


def generate_signal(df: pd.DataFrame, current_price: float, is_flat_regime: bool) -> Tuple[str, float, float]:
    df_calc = calculate_smas(df)
    current_sma_1 = df_calc['sma_1'].iloc[-1]
    current_sma_2 = df_calc['sma_2'].iloc[-1]
    
    if pd.isna(current_sma_1) or pd.isna(current_sma_2):
        raise ValueError(f"Not enough historical data for SMAs")
    
    if is_flat_regime:
        log.info("FLAT REGIME ACTIVE: Forcing FLAT signal")
        return "FLAT", current_sma_1, current_sma_2
    
    signal = "FLAT"
    if current_price > current_sma_1 and current_price > current_sma_2:
        signal = "LONG"
    elif current_price < current_sma_1 and current_price < current_sma_2:
        signal = "SHORT"
    
    log.info(f"Price: ${current_price:.2f} | SMA1: ${current_sma_1:.2f} | SMA2: ${current_sma_2:.2f} | Sig: {signal}")
    return signal, current_sma_1, current_sma_2


def portfolio_usd(api: kf.KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])


def mark_price(api: kf.KrakenFuturesApi) -> float:
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC:
            return float(t["markPrice"])
    raise RuntimeError("Mark-price for PF_XBTUSD not found")


def cancel_all(api: kf.KrakenFuturesApi):
    """
    SAFETY UPDATE: Only cancel orders for PF_XBTUSD to avoid deleting Planner's orders.
    """
    log.info(f"Cancelling all orders for {SYMBOL_FUTS_UC}")
    try:
        api.cancel_all_orders({"symbol": SYMBOL_FUTS_UC})
    except Exception as e:
        log.warning("cancel_all_orders failed: %s", e)


def get_current_position(api: kf.KrakenFuturesApi) -> Optional[Dict]:
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
        log.warning(f"Failed to get position: {e}")
        return None


def flatten_position_limit(api: kf.KrakenFuturesApi, current_price: float):
    pos = get_current_position(api)
    if not pos: return
    
    side = "sell" if pos["side"] == "long" else "buy"
    size = pos["size_btc"]
    limit_price = current_price * (1 + LIMIT_OFFSET_PCT) if side == "sell" else current_price * (1 - LIMIT_OFFSET_PCT)
    
    log.info(f"Flatten Limit: {side} {size:.4f} @ ${limit_price:.2f}")
    try:
        api.send_order({
            "orderType": "lmt", "symbol": SYMBOL_FUTS_LC, "side": side,
            "size": round(size, 4), "limitPrice": int(round(limit_price)),
        })
    except Exception as e:
        log.warning(f"Flatten limit failed: {e}")


def flatten_position_market(api: kf.KrakenFuturesApi):
    pos = get_current_position(api)
    if not pos: return
    
    side = "sell" if pos["side"] == "long" else "buy"
    size = pos["size_btc"]
    log.info(f"Flatten Market: {side} {size:.4f}")
    try:
        api.send_order({
            "orderType": "mkt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": round(size, 4),
        })
    except Exception as e:
        log.warning(f"Flatten market failed: {e}")


def place_entry_limit(api: kf.KrakenFuturesApi, side: str, size_btc: float, current_price: float) -> float:
    limit_price = current_price * (1 - LIMIT_OFFSET_PCT) if side == "buy" else current_price * (1 + LIMIT_OFFSET_PCT)
    log.info(f"Entry Limit: {side} {size_btc:.4f} @ ${limit_price:.2f}")
    try:
        api.send_order({
            "orderType": "lmt", "symbol": SYMBOL_FUTS_LC, "side": side,
            "size": round(size_btc, 4), "limitPrice": int(round(limit_price)),
        })
        return limit_price
    except Exception as e:
        log.error(f"Entry limit failed: {e}")
        return current_price


def place_entry_market_remaining(api: kf.KrakenFuturesApi, side: str, intended_size: float, current_price: float) -> float:
    pos = get_current_position(api)
    
    if pos and pos["side"] == ("long" if side == "buy" else "short"):
        filled_size = pos["size_btc"]
        remaining = intended_size - filled_size
        if remaining > 0.0001:
            log.info(f"Entry Market Remaining: {side} {remaining:.4f}")
            try:
                api.send_order({
                    "orderType": "mkt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": round(remaining, 4),
                })
                return intended_size
            except Exception as e:
                log.warning(f"Entry market failed: {e}")
                return filled_size
        return filled_size
    else:
        log.warning("No fill on limit, placing full market order")
        try:
            api.send_order({
                "orderType": "mkt", "symbol": SYMBOL_FUTS_LC, "side": side, "size": round(intended_size, 4),
            })
            return intended_size
        except Exception as e:
            log.error(f"Full market failed: {e}")
            return 0


def place_stop(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float):
    stop_distance = fill_price * STATIC_STOP_PCT
    if side == "buy":
        stop_price = fill_price - stop_distance
        stop_side = "sell"
    else:
        stop_price = fill_price + stop_distance
        stop_side = "buy"
        
    log.info(f"Placing Stop: {stop_side} @ ${stop_price:.2f}")
    try:
        api.send_order({
            "orderType": "stp", "symbol": SYMBOL_FUTS_LC, "side": stop_side,
            "size": round(size_btc, 4), "stopPrice": int(round(stop_price)), "reduceOnly": True
        })
    except Exception as e:
        log.error(f"Stop order failed: {e}")


def place_take_profit(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float):
    if side == "buy":
        tp_price = fill_price * (1 + TAKE_PROFIT_PCT)
        tp_side = "sell"
    else:
        tp_price = fill_price * (1 - TAKE_PROFIT_PCT)
        tp_side = "buy"
    
    log.info(f"Placing TP: {tp_side} @ ${tp_price:.2f}")
    try:
        api.send_order({
            "orderType": "lmt", "symbol": SYMBOL_FUTS_LC, "side": tp_side,
            "size": round(size_btc, 4), "limitPrice": int(round(tp_price)), "reduceOnly": True
        })
    except Exception as e:
        log.error(f"TP order failed: {e}")


def initialize_flat_regime_state(api: kf.KrakenFuturesApi) -> bool:
    try:
        df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
        iii = calculate_iii(df)
        log.info(f"Init Check: III={iii:.4f}, Threshold={FLAT_REGIME_THRESHOLD}")
        
        if iii < FLAT_REGIME_THRESHOLD:
            # Simple check: if current III is low, we assume flat unless we find a breach
            log.info("III Low - Setting FLAT REGIME")
            return True
        return False
    except Exception as e:
        log.error(f"Error initializing flat regime: {e}")
        return False


def load_state() -> Dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {
        "trades": [], "starting_capital": None, "performance": {}, 
        "current_position": None, "flat_regime_active": False
    }


def save_state(st: Dict):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def update_state_with_current_position(api: kf.KrakenFuturesApi):
    state = load_state()
    current_pos = get_current_position(api)
    portfolio_value = portfolio_usd(api)
    state["current_position"] = current_pos
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    save_state(state)


def daily_trade(api: kf.KrakenFuturesApi):
    log.info("--- Starting Daily Cycle (Tumbler) ---")
    state = load_state()
    
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    current_price = mark_price(api)
    portfolio_value = portfolio_usd(api)
    
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    
    iii = calculate_iii(df)
    leverage = determine_leverage(iii)
    
    # Flat Regime Checks
    current_flat_regime = state.get("flat_regime_active", False)
    is_flat_regime = check_flat_regime_trigger(iii, current_flat_regime)
    if is_flat_regime:
        is_flat_regime = check_flat_regime_release(df, is_flat_regime)
    
    state["flat_regime_active"] = is_flat_regime
    
    signal, sma_1, sma_2 = generate_signal(df, current_price, is_flat_regime)
    
    # Execution
    flatten_position_limit(api, current_price)
    time.sleep(600)
    flatten_position_market(api)
    cancel_all(api) # Safe cancel
    time.sleep(2)
    
    collateral = portfolio_usd(api)
    
    if signal == "FLAT":
        log.info("Staying FLAT.")
    else:
        notional = collateral * leverage
        size_btc = round(notional / current_price, 4)
        side = "buy" if signal == "LONG" else "sell"
        
        log.info(f"Opening {signal} {size_btc} BTC ({leverage}x)")
        
        if not dry:
            place_entry_limit(api, side, size_btc, current_price)
            time.sleep(600)
            current_price = mark_price(api) # Update price
            final_size = place_entry_market_remaining(api, side, size_btc, current_price)
            
            cancel_all(api) # Safe cancel
            time.sleep(2)
            
            # Re-fetch fill price estimate (using current mark for simplicity in logs)
            fill_price = current_price 
            
            place_stop(api, side, final_size, fill_price)
            place_take_profit(api, side, final_size, fill_price)
            
            # Log Trade
            state["trades"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal": signal, "size": final_size, "price": fill_price,
                "iii": iii, "leverage": leverage
            })

    save_state(state)
    log.info("Tumbler Cycle Complete.")


def init_tumbler(api: kf.KrakenFuturesApi):
    """Called by the main orchestrator to setup initial state."""
    log.info("Initializing Tumbler State...")
    is_flat = initialize_flat_regime_state(api)
    state = load_state()
    state["flat_regime_active"] = is_flat
    save_state(state)
    update_state_with_current_position(api)


def main():
    # Standalone execution support
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit("No API Key")
    api = kf.KrakenFuturesApi(api_key, api_sec)
    init_tumbler(api)
    daily_trade(api)

if __name__ == "__main__":
    main()
