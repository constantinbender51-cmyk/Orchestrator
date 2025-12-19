#!/usr/bin/env python3
"""
main.py - Hourly Orchestrator
Runs Planner, Tumbler, and Gainer every hour at XX:01.
Allocates 33% capital to each.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import kraken_futures as kf
import planner
import tumbler
import gainer

# --- 33% Split ---
CAP_SPLIT = 0.33

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("MAIN")

def flush_logs():
    sys.stdout.flush()

def wait_until_next_hour():
    """Sleeps until XX:01:00 of the NEXT hour."""
    now = datetime.now(timezone.utc)
    # Next hour, 1st minute
    next_run = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    
    wait_sec = (next_run - now).total_seconds()
    log.info(f"--- SLEEPING --- Next run: {next_run.strftime('%H:%M')} UTC ({wait_sec/60:.1f} mins)")
    flush_logs()
    time.sleep(wait_sec)

def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit("No API Keys")

    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    # Init States
    try: tumbler.init_tumbler(api)
    except: pass

    # Immediate Run Check
    if os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}:
        log.warning("RUN_TRADE_NOW=True. Executing...")
        run_cycle(api)

    # Hourly Loop
    while True:
        wait_until_next_hour()
        run_cycle(api)

def run_cycle(api):
    log.info(">>> STARTING HOURLY CYCLE <<<")
    
    # 1. PLANNER (Fixed Jan 2027) - 33%
    try:
        status = planner.daily_trade(api, capital_pct=CAP_SPLIT)
        log.info(f"PLANNER: {status}")
    except Exception as e: log.error(f"Planner Error: {e}")

    # 2. TUMBLER (Perp) - 33%
    try:
        status = tumbler.daily_trade(api, capital_pct=CAP_SPLIT)
        log.info(f"TUMBLER: {status}")
    except Exception as e: log.error(f"Tumbler Error: {e}")

    # 3. GAINER (Fixed Jan 2030) - 33%
    try:
        status = gainer.hourly_trade(api, capital_pct=CAP_SPLIT)
        log.info(f"GAINER: {status}")
    except Exception as e: log.error(f"Gainer Error: {e}")

    log.info(">>> CYCLE COMPLETE <<<")
    flush_logs()

if __name__ == "__main__":
    main()
