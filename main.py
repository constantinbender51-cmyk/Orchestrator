#!/usr/bin/env python3
"""
master_trader.py - Unified Trading Engine & Observation Deck
Features:
- "Fair 2.0" Strategy Normalization.
- Position-Aware Limit Chaser.
- Built-in Web Server (Port 8080).
- Immediate State Initialization (Fixes blank dashboard on startup).
- No-Flicker Dashboard Sync.
"""

import json
import logging
import os
import sys
import time
import threading
import collections
import http.server
import socketserver
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import kraken_futures as kf
import kraken_ohlc

# --- Global Configuration ---
SYMBOL_FUTS = "FF_XBTUSD_260130"
SYMBOL_OHLC = "XBTUSD"
CAP_SPLIT = 0.333
LIMIT_CHASE_DURATION = 720
CHASE_INTERVAL = 60
MIN_TRADE_SIZE = 0.0002
LIMIT_OFFSET_TICKS = 1

# Normalization
TUMBLER_MAX_LEV = 4.327
TARGET_STRAT_LEV = 2.0

# Paths & Server
STATE_FILE = Path("master_state.json")
PORT = int(os.getenv("PORT", 8080))

# --- Logging Setup ---
LOG_BUFFER = collections.deque(maxlen=200)

class BufferHandler(logging.Handler):
    """Stores logs with unique IDs to prevent frontend flickering."""
    def emit(self, record):
        try:
            msg = self.format(record)
            if hasattr(record, 'api_response'):
                try:
                    json_str = json.dumps(record.api_response, indent=2)
                    msg += (
                        f"<details style='margin:5px 0 10px 0; border-left:2px solid #30363d; padding-left:10px;'>"
                        f"<summary style='cursor:pointer; color:#58a6ff; font-size:0.85em; outline:none;'>API Response</summary>"
                        f"<pre style='margin:5px 0 0 0; font-size:0.8em; color:#8b949e; overflow-x:auto; background:#0d1117; padding:5px; border-radius:4px;'>{json_str}</pre>"
                        f"</details>"
                    )
                except: pass
            
            entry = {"id": str(time.time_ns()), "html": msg}
            LOG_BUFFER.append(entry)
        except Exception:
            self.handleError(record)

class SlowStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
            time.sleep(0.05)
        except Exception:
            self.handleError(record)

log = logging.getLogger("MASTER")
log.setLevel(logging.INFO)
fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M")

slow_h = SlowStreamHandler(sys.stdout)
slow_h.setFormatter(fmt)
log.addHandler(slow_h)

buff_h = BufferHandler()
buff_h.setFormatter(fmt)
log.addHandler(buff_h)

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

# --- State Management ---
def get_default_state():
    return {
        "strategies": {
            "planner": {"entry_date": None, "peak_equity": 0.0, "stopped": False, "raw_lev": 0, "fair_lev": 0},
            "tumbler": {"flat_regime": False, "raw_lev": 0, "fair_lev": 0},
            "gainer": {"raw_lev": 0, "fair_lev": 0}
        },
        "portfolio": {"total_equity": 0, "strat_cap": 0, "currency": "USD"},
        "position": {"current": 0, "target": 0, "delta": 0},
        "market": {"price": 0, "last_updated": datetime.now(timezone.utc).isoformat()},
        "logs": []
    }

def load_state():
    defaults = get_default_state()
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                for k, v in defaults.items():
                    if k not in saved: saved[k] = v
                return saved
        except Exception as e: log.error(f"State load error: {e}")
    return defaults

def save_state_rich(state, logs_deque=None):
    try:
        if logs_deque is not None:
            state["logs"] = list(logs_deque)
        state["market"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Strip DataFrames or non-serializable objects
        clean_state = {k: v for k, v in state.items() if k != "df_1d"}
        
        temp_file = STATE_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(clean_state, f, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        log.error(f"State save error: {e}")

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

# --- Strategies (Internal) ---
def run_planner(df_1d, state, capital):
    s = state["strategies"]["planner"]
    if s["peak_equity"] < capital: s["peak_equity"] = capital
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
    return max(-2.0, min(2.0, s1_lev + s2_lev))

def run_tumbler(df_1d, state, capital):
    s = state["strategies"]["tumbler"]
    w = TUMBLER_PARAMS["III_WIN"]
    if len(df_1d) < w+1: return 0.0
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    iii = (log_ret.rolling(w).sum().abs() / log_ret.abs().rolling(w).sum()).fillna(0).iloc[-1]
    lev = TUMBLER_PARAMS["LEVS"][2]
    if iii < TUMBLER_PARAMS["III_TH"][0]: lev = TUMBLER_PARAMS["LEVS"][0]
    elif iii < TUMBLER_PARAMS["III_TH"][1]: lev = TUMBLER_PARAMS["LEVS"][1]
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]:
        if not s["flat_regime"]: log.info(f"Tumb Flat | III={iii:.4f}")
        s["flat_regime"] = True
    if s["flat_regime"]:
        sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
        sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
        if abs(df_1d['close'].iloc[-1] - sma1) <= sma1*TUMBLER_PARAMS["BAND"]: s["flat_regime"] = False
    if s["flat_regime"]: return 0.0
    sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
    sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
    curr = df_1d['close'].iloc[-1]
    if curr > sma1 and curr > sma2: return lev
    if curr < sma1 and curr < sma2: return -lev
    return 0.0

def run_gainer(df_1h, df_1d):
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
    w = GAINER_PARAMS["WEIGHTS"]
    s1 = calc_macd(df_1h['close'], GAINER_PARAMS["MACD_1H"]['params'], GAINER_PARAMS["MACD_1H"]['weights']) * w[0]
    s3 = calc_macd(df_1d['close'], GAINER_PARAMS["MACD_1D"]['params'], GAINER_PARAMS["MACD_1D"]['weights']) * w[2]
    total = sum(w)
    return (s1 + s3) / total if total > 0 else 0.0

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
            log.info(f"Chase Done | Tgt Reached")
            break
        try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
        except: pass
        side = "buy" if delta > 0 else "sell"
        size = round(abs(delta), 4)
        tk = api.get_tickers()
        limit_px = 0
        for t in tk.get("tickers", []):
            if t["symbol"] == SYMBOL_FUTS:
                limit_px = int(float(t["bid"])) if side == "buy" else int(float(t["ask"]))
                limit_px += (-LIMIT_OFFSET_TICKS if side == "buy" else LIMIT_OFFSET_TICKS)
                break
        if limit_px == 0: continue
        try:
            resp = api.send_order({"orderType": "lmt", "symbol": SYMBOL_FUTS, "side": side, "size": size, "limitPrice": limit_px, "postOnly": True})
            log.info(f"Chase {i+1} | {side.upper()} {size} | {resp.get('result')}", extra={"api_response": resp})
        except: pass
        time.sleep(CHASE_INTERVAL)

def run_cycle(api):
    log.info("--- CYCLE ---")
    try:
        df_1h = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 60)
        df_1d = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 1440)
    except: return
    state = load_state()
    state["df_1d"] = df_1d 
    curr_price = get_market_price(api) or df_1h['close'].iloc[-1]
    state["market"]["price"] = curr_price
    try:
        accts = api.get_accounts()
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        strat_cap = total_pv * CAP_SPLIT
        state["portfolio"].update({"total_equity": total_pv, "strat_cap": strat_cap})
    except: return
    
    r_plan = run_planner(df_1d, state, strat_cap)
    r_tumb = run_tumbler(df_1d, state, strat_cap)
    r_gain = run_gainer(df_1h, df_1d)
    
    n_plan, n_tumb, n_gain = r_plan, r_tumb * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV), r_gain * TARGET_STRAT_LEV
    state["strategies"]["planner"].update({"raw_lev": r_plan, "fair_lev": n_plan})
    state["strategies"]["tumbler"].update({"raw_lev": r_tumb, "fair_lev": n_tumb})
    state["strategies"]["gainer"].update({"raw_lev": r_gain, "fair_lev": n_gain})

    target_qty = (n_plan + n_tumb + n_gain) * strat_cap / curr_price
    limit_chaser(api, target_qty)
    
    pos = get_net_position(api)
    state["position"].update({"current": pos, "target": target_qty, "delta": target_qty - pos})
    save_state_rich(state, LOG_BUFFER)
    log.info("--- END ---")

def wait_until_next_hour():
    now = datetime.now(timezone.utc)
    next_run = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    log.info(f"Next Run: {next_run.strftime('%H:%M')}")
    time.sleep((next_run - now).total_seconds())

# --- WEB DASHBOARD ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Trader</title>
    <style>
        :root { --bg-dark: #0d1117; --bg-card: #161b22; --border: #30363d; --text-main: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff; --success: #238636; --danger: #da3633; }
        body { background-color: var(--bg-dark); color: var(--text-main); font-family: 'Segoe UI', monospace; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 20px; }
        .metric { font-size: 2em; font-weight: bold; }
        .sub-metric { color: var(--text-muted); font-size: 0.9em; }
        .strat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 20px; }
        .strat-card { background: #21262d; padding: 15px; border-radius: 6px; text-align: center; }
        .pos { color: #3fb950; } .neg { color: #f85149; }
        .log-window { background: #000; border: 1px solid var(--border); height: 400px; overflow-y: scroll; padding: 10px; font-family: 'Consolas', monospace; font-size: 0.85em; color: #a5d6ff; margin-top: 20px; }
        .log-entry { margin-bottom: 4px; border-bottom: 1px solid #1f1f1f; padding-bottom: 2px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div><h1 style="margin:0">Master Trader</h1><small id="last-update">Connecting...</small></div>
            <div id="status" class="sub-metric">Initializing...</div>
        </div>
        <div class="grid">
            <div class="card"><h3>Equity</h3><div class="metric" id="total-equity">---</div><div class="sub-metric">Cap: <span id="strat-cap">---</span></div></div>
            <div class="card"><h3>Position</h3><div class="metric" id="current-pos">---</div><div class="sub-metric">Target: <span id="target-pos">---</span></div></div>
            <div class="card"><h3>Price</h3><div class="metric" id="market-price">---</div><div class="sub-metric">FF_XBTUSD_260130</div></div>
        </div>
        <div class="strat-grid">
            <div class="strat-card"><h3>Planner</h3><div class="metric" id="lev-planner">---</div><small>Raw: <span id="raw-planner">---</span></small></div>
            <div class="strat-card"><h3>Tumbler</h3><div class="metric" id="lev-tumbler">---</div><small>Raw: <span id="raw-tumbler">---</span></small></div>
            <div class="strat-card"><h3>Gainer</h3><div class="metric" id="lev-gainer">---</div><small>Raw: <span id="raw-gainer">---</span></small></div>
        </div>
        <div class="log-window" id="log-container"></div>
    </div>
    <script>
        function setVal(id, val, suffix='') { document.getElementById(id).innerText = val + suffix; }
        function colorize(id, val) { 
            const el = document.getElementById(id); 
            el.innerText = val.toFixed(2) + 'x';
            el.className = 'metric ' + (val >= 0 ? 'pos' : 'neg');
        }
        async function update() {
            try {
                const res = await fetch('/api/state');
                if (!res.ok) throw new Error();
                const d = await res.json();
                setVal('total-equity', '$' + d.portfolio.total_equity.toFixed(2));
                setVal('strat-cap', '$' + d.portfolio.strat_cap.toFixed(2));
                setVal('current-pos', d.position.current.toFixed(4));
                setVal('target-pos', d.position.target.toFixed(4));
                setVal('market-price', '$' + d.market.price.toFixed(1));
                setVal('last-update', 'Update: ' + new Date(d.market.last_updated).toLocaleTimeString());
                setVal('status', 'Bot Online');
                
                colorize('lev-planner', d.strategies.planner.fair_lev);
                setVal('raw-planner', d.strategies.planner.raw_lev.toFixed(2));
                colorize('lev-tumbler', d.strategies.tumbler.fair_lev);
                setVal('raw-tumbler', d.strategies.tumbler.raw_lev.toFixed(2));
                colorize('lev-gainer', d.strategies.gainer.fair_lev);
                setVal('raw-gainer', d.strategies.gainer.raw_lev.toFixed(2));
                
                const logCont = document.getElementById('log-container');
                const newLogs = d.logs.slice().reverse();
                const existingIds = Array.from(logCont.children).map(c => c.dataset.id);
                
                newLogs.forEach((l, i) => {
                    if (!existingIds.includes(l.id)) {
                        const div = document.createElement('div');
                        div.className = 'log-entry'; div.dataset.id = l.id; div.innerHTML = l.html;
                        if (i === 0) logCont.prepend(div); else logCont.children[i-1].after(div);
                    }
                });
            } catch { setVal('status', 'Waiting for state...'); }
        }
        setInterval(update, 2000); update();
    </script>
</body>
</html>
"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/index.html']:
            self.send_response(200); self.send_header('Content-type','text/html'); self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path == '/api/state':
            if STATE_FILE.exists():
                self.send_response(200); self.send_header('Content-type','application/json'); self.end_headers()
                with open(STATE_FILE, 'r') as f: self.wfile.write(f.read().encode())
            else: self.send_error(503)
        else: self.send_error(404)

def main():
    api_key, api_sec = os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit("No API Keys")
    
    # Initialize State File Immediately
    save_state_rich(get_default_state())
    
    threading.Thread(target=lambda: socketserver.TCPServer(("", PORT), Handler).serve_forever(), daemon=True).start()
    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    if os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}: run_cycle(api)
    while True:
        wait_until_next_hour()
        run_cycle(api)

if __name__ == "__main__": main()
