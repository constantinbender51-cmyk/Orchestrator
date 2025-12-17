#!/usr/bin/env python3
"""
main.py - Safe Orchestrator
Features:
- CAPITAL PARTITIONING (50/50)
- STATUS REPORTING: Prints a clear summary of actions taken.
- LOGGING: Simplified for container output.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import kraken_futures as kf
import planner
import tumbler

CAPITAL_SPLIT_PLANNER = 0.50
CAPITAL_SPLIT_TUMBLER = 0.50

# Simplified logging (Container adds timestamp)
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("MAIN")

def flush_logs():
    sys.stdout.flush()

def wait_until_00_01_utc():
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    
    wait_sec = (next_run - now).total_seconds()
    log.info(f"--- SLEEPING --- Next run: {next_run.strftime('%Y-%m-%d %H:%M UTC')} ({wait_sec/3600:.1f}h)")
    flush_logs()
    time.sleep(wait_sec)

def log_account_health(api):
    try:
        accts = api.get_accounts()
        flex = accts.get("accounts", {}).get("flex", {})
        pv = float(flex.get("portfolioValue", 0))
        margin_equity = float(flex.get("marginEquity", 0))
        curr_lev = 0.0
        if margin_equity > 0:
            initial_margin = float(flex.get("initialMargin", 0))
            curr_lev = initial_margin / margin_equity
        log.info(f"HEALTH: PV=${pv:.2f} | Eq=${margin_equity:.2f} | Margin={curr_lev*100:.1f}%")
        flush_logs()
    except Exception as e:
        log.error(f"Health check failed: {e}")

def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key or not api_sec:
        sys.exit("Missing API Keys")

    log.info("Init API & State...")
    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    try: tumbler.init_tumbler(api)
    except Exception as e: log.error(f"Tumbler init failed: {e}")

    # Immediate Run Check
    if os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}:
        log.warning("RUN_TRADE_NOW=True. Executing...")
        log_account_health(api)
        
        status_planner = "FAILED"
        status_tumbler = "FAILED"
        
        try:
            log.info(f"=== PLANNER (50%) ===")
            status_planner = planner.daily_trade(api, capital_pct=CAPITAL_SPLIT_PLANNER)
        except Exception as e:
            log.exception(f"Planner Error: {e}")
            
        try:
            log.info(f"=== TUMBLER (50%) ===")
            status_tumbler = tumbler.daily_trade(api, capital_pct=CAPITAL_SPLIT_TUMBLER)
        except Exception as e:
            log.exception(f"Tumbler Error: {e}")

        log.info("="*30)
        log.info("   EXECUTION REPORT")
        log.info(f"PLANNER: {status_planner}")
        log.info(f"TUMBLER: {status_tumbler}")
        log.info("="*30)
        log_account_health(api)
        flush_logs()

    # Scheduler
    while True:
        wait_until_00_01_utc()
        log.info(">>> STARTING DAILY CYCLE <<<")
        log_account_health(api)
        
        try: planner.daily_trade(api, capital_pct=CAPITAL_SPLIT_PLANNER)
        except Exception as e: log.exception(f"Planner Failed: {e}")
            
        try: tumbler.daily_trade(api, capital_pct=CAPITAL_SPLIT_TUMBLER)
        except Exception as e: log.exception(f"Tumbler Failed: {e}")
            
        log.info(">>> CYCLE COMPLETE <<<")
        flush_logs()

if __name__ == "__main__":
    main()
