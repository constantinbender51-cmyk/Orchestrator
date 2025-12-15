#!/usr/bin/env python3
"""
main.py - Combined Orchestrator for Planner and Tumbler
Runs both strategies sequentially at 00:01 UTC every day.
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

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("MAIN")

def wait_until_00_01_utc():
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    
    wait_sec = (next_run - now).total_seconds()
    log.info("Next run at 00:01 UTC (%s), sleeping %.0f s", next_run.strftime("%Y-%m-%d"), wait_sec)
    time.sleep(wait_sec)

def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    
    if not api_key or not api_sec:
        log.error("Missing KRAKEN_API_KEY or KRAKEN_API_SECRET env vars")
        sys.exit(1)

    log.info("Initializing Kraken API...")
    api = kf.KrakenFuturesApi(api_key, api_sec)

    # 1. Initialize Strategies
    log.info("Setting up Tumbler...")
    try:
        tumbler.init_tumbler(api)
    except Exception as e:
        log.error(f"Tumbler initialization failed: {e}")

    # 2. Launch Optional Dashboard (from Tumbler)
    if os.path.exists("web_state.py"):
        log.info("Launching web dashboard...")
        subprocess.Popen([sys.executable, "web_state.py"])

    # 3. Main Loop
    run_now = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}
    
    if run_now:
        log.info("Executing immediate run...")
        try: planner.daily_trade(api)
        except Exception as e: log.error(f"Planner failed: {e}")
        
        try: tumbler.daily_trade(api)
        except Exception as e: log.error(f"Tumbler failed: {e}")

    while True:
        wait_until_00_01_utc()
        
        log.info(">>> Starting Daily Execution Cycle <<<")
        
        # Run Planner
        try:
            planner.daily_trade(api)
        except Exception as e:
            log.exception(f"CRITICAL: Planner execution failed: {e}")
            
        # Run Tumbler
        try:
            tumbler.daily_trade(api)
        except Exception as e:
            log.exception(f"CRITICAL: Tumbler execution failed: {e}")
            
        log.info(">>> Cycle Finished. Waiting for next day... <<<")

if __name__ == "__main__":
    main()
