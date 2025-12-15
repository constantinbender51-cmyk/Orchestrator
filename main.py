#!/usr/bin/env python3
"""
main.py - Safe Orchestrator for Multi-Strategy Execution
Features:
- CAPITAL PARTITIONING: Splits account equity between strategies to prevent margin collisions.
- UNIFIED LOGGING: Prevents duplicate log entries.
- SEQUENTIAL RUN: Runs Planner, then Tumbler.
"""

import logging
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta, timezone

import kraken_futures as kf
import planner
import tumbler

# --- Configuration ---
# Capital Split: How much of the TOTAL Portfolio Value each strategy is allowed to see/use.
# Must sum to <= 1.0 (100%)
CAPITAL_SPLIT_PLANNER = 0.50
CAPITAL_SPLIT_TUMBLER = 0.50

# --- Unified Logging Setup ---
# This ensures logs appear once, with clear timestamps and module names.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_system.log") # Persist logs to file for debugging
    ]
)
log = logging.getLogger("MAIN")

def wait_until_00_01_utc():
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    
    wait_sec = (next_run - now).total_seconds()
    log.info(f"Next run scheduled for {next_run.strftime('%Y-%m-%d %H:%M UTC')} (sleeping {wait_sec/3600:.1f} hours)")
    time.sleep(wait_sec)

def log_account_health(api):
    try:
        accts = api.get_accounts()
        flex = accts.get("accounts", {}).get("flex", {})
        pv = float(flex.get("portfolioValue", 0))
        margin_equity = float(flex.get("marginEquity", 0))
        # Protect against div by zero if empty account
        curr_lev = 0.0
        if margin_equity > 0:
            initial_margin = float(flex.get("initialMargin", 0))
            curr_lev = initial_margin / margin_equity
            
        log.info(f"--- ACCOUNT HEALTH: PV=${pv:.2f} | Margin Equity=${margin_equity:.2f} | Current Margin Usage={curr_lev*100:.1f}% ---")
    except Exception as e:
        log.error(f"Could not fetch account health: {e}")

def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    
    if not api_key or not api_sec:
        log.critical("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET environment variables.")
        sys.exit(1)

    log.info("Initializing Kraken Futures API...")
    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    # 1. Initialize Strategies (State checks)
    log.info("Initializing Strategy State...")
    try:
        tumbler.init_tumbler(api)
    except Exception as e:
        log.error(f"Tumbler init failed: {e}")

    # 2. Check for Immediate Run Flag
    run_now = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

    if run_now:
        log.warning("RUN_TRADE_NOW is active. Executing strategies immediately...")
        log_account_health(api)
        
        # PLANNER RUN
        try:
            log.info(f"=== Running PLANNER (Alloc: {CAPITAL_SPLIT_PLANNER*100}%) ===")
            planner.daily_trade(api, capital_pct=CAPITAL_SPLIT_PLANNER)
        except Exception as e:
            log.exception(f"Planner crashed: {e}")
            
        # TUMBLER RUN
        try:
            log.info(f"=== Running TUMBLER (Alloc: {CAPITAL_SPLIT_TUMBLER*100}%) ===")
            tumbler.daily_trade(api, capital_pct=CAPITAL_SPLIT_TUMBLER)
        except Exception as e:
            log.exception(f"Tumbler crashed: {e}")

        log_account_health(api)

    # 3. Main Scheduler Loop
    while True:
        wait_until_00_01_utc()
        
        log.info(">>> STARTING DAILY CYCLE <<<")
        log_account_health(api)
        
        # Run Planner
        try:
            log.info(f"Executing Planner with {CAPITAL_SPLIT_PLANNER*100}% capital allocation.")
            planner.daily_trade(api, capital_pct=CAPITAL_SPLIT_PLANNER)
        except Exception as e:
            log.exception(f"CRITICAL: Planner execution failed: {e}")
            
        # Run Tumbler
        try:
            log.info(f"Executing Tumbler with {CAPITAL_SPLIT_TUMBLER*100}% capital allocation.")
            tumbler.daily_trade(api, capital_pct=CAPITAL_SPLIT_TUMBLER)
        except Exception as e:
            log.exception(f"CRITICAL: Tumbler execution failed: {e}")
            
        log.info(">>> CYCLE COMPLETE. Waiting for next schedule... <<<")

if __name__ == "__main__":
    main()
