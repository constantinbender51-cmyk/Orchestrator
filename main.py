#!/usr/bin/env python3
"""
master_trader.py - Unified Trading Engine & Observation Deck
Features:
- "Fair 2.0" Strategy Normalization.
- Position-Aware Limit Chaser.
- Built-in Web Server (Port 8080) for Real-Time Monitoring.
- Runs on Railway (Single Process).
- Atomic State Persistence.
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
LOG_BUFFER = collections.deque(maxlen=200) # Keep last 200 logs for dashboard

class BufferHandler(logging.Handler):
    """Stores logs in memory for the dashboard."""
    def emit(self, record):
        try:
            msg = self.format(record)
            LOG_BUFFER.append(msg)
        except Exception:
            self.handleError(record)

class SlowStreamHandler(logging.StreamHandler):
    """Pauses to protect console."""
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
            time.sleep(0.1)
        except Exception:
            self.handleError(record)

log = logging.getLogger("MASTER")
log.setLevel(logging.INFO)
fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

# Add Handlers
slow_h = SlowStreamHandler(sys.stdout)
slow_h.setFormatter(fmt)
log.addHandler(slow_h)

buff_h = BufferHandler()
buff_h.setFormatter(fmt)
log.addHandler(buff_h)

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}

# --- Strategy Parameters ---
PLANNER_PARAMS = {
    "S1_SMA": 120, "S1_DECAY": 40, "S1_STOP": 0.13,
    "S2_SMA": 400, "S2_PROX": 0.05, "S2_STOP": 0.27
}
TUMBLER_PARAMS = {
    "SMA1": 32, "SMA2": 114, "STOP": 0.043, "TP": 0.126,
    "III_WIN": 27, "FLAT_THRESH": 0.356, "BAND": 0.077,
    "LEVS": [0.079, 4.327, 3.868], "III_TH": [0.058, 0.259]
}
GAINER_PARAMS = {
    "WEIGHTS": [0.8, 0.0, 0.4, 0.0, 0.0, 0.4],
    "MACD_1H": {'params': [(97, 366, 47), (15, 40, 11), (16, 55, 13)], 'weights': [0.45, 0.43, 0.01]},
    "MACD_1D": {'params': [(52, 64, 61), (5, 6, 4), (17, 18, 16)], 'weights': [0.87, 0.92, 0.73]},
    "SMA_1D":  {'params': [40, 120, 390], 'weights': [0.6, 0.8, 0.4]}
}

# --- State Management ---
def load_state():
    defaults = {
        "strategies": {
            "planner": {"entry_date": None, "peak_equity": 0.0, "stopped": False, "raw_lev": 0, "fair_lev": 0},
            "tumbler": {"flat_regime": False, "raw_lev": 0, "fair_lev": 0},
            "gainer": {"raw_lev": 0, "fair_lev": 0}
        },
        "portfolio": {"total_equity": 0, "strat_cap": 0, "currency": "USD"},
        "position": {"current": 0, "target": 0, "delta": 0},
        "market": {"price": 0, "last_updated": ""},
        "logs": []
    }
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                for k, v in defaults.items():
                    if k not in saved: saved[k] = v
                return saved
        except Exception as e: log.error(f"State load error: {e}")
    return defaults

def save_state_rich(state, logs_deque):
    try:
        state["logs"] = list(logs_deque)
        state["market"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        clean_state = {k: v for k, v in state.items() if k != "df_1d"}
        
        # Atomic Write
        temp_file = STATE_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(clean_state, f, indent=2)
        os.replace(temp_file, STATE_FILE)
    except Exception as e:
        log.error(f"State save error: {e}")

# --- Helper Functions ---
def get_sma(prices, window):
    if len(prices) < window: return 0.0
    return prices.rolling(window=window).mean().iloc[-1]

def get_market_price(api):
    try:
        resp = api.get_tickers()
        for t in resp.get("tickers", []):
            if t.get("symbol") == SYMBOL_FUTS: 
                return float(t.get("markPrice"))
    except Exception as e: log.error(f"Ticker fetch failed: {e}")
    return 0.0

def get_net_position(api):
    try:
        resp = api.get_open_positions()
        for p in resp.get("openPositions", []):
            if p.get('symbol') == SYMBOL_FUTS:
                size = float(p.get('size', 0.0))
                return size if p.get('side') == 'long' else -size
    except Exception as e: log.error(f"Pos fetch failed: {e}")
    return 0.0

# --- Strategy Modules ---
def run_planner(df_1d, state, capital):
    s = state["strategies"]["planner"]
    if s["peak_equity"] < capital: s["peak_equity"] = capital
    
    if s["peak_equity"] > 0:
        dd = (s["peak_equity"] - capital) / s["peak_equity"]
        if dd > PLANNER_PARAMS["S1_STOP"]: 
            if not s["stopped"]: log.info(f"Planner S1 Soft Stop Triggered (DD: {dd*100:.2f}%)")
            s["stopped"] = True
    else: s["peak_equity"] = capital

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
    if capital > s["peak_equity"]: s["peak_equity"] = capital
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
    
    s["last_iii"] = float(iii)
    
    if iii < TUMBLER_PARAMS["FLAT_THRESH"]:
        if not s["flat_regime"]: log.info(f"Tumbler: Entering Flat Regime (III={iii:.4f})")
        s["flat_regime"] = True
    
    if s["flat_regime"]:
        sma1 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA1"])
        sma2 = get_sma(df_1d['close'], TUMBLER_PARAMS["SMA2"])
        curr = df_1d['close'].iloc[-1]
        b = TUMBLER_PARAMS["BAND"]
        if abs(curr - sma1) <= sma1*b or abs(curr - sma2) <= sma2*b:
             log.info("Tumbler: Releasing Flat Regime")
             s["flat_regime"] = False
    
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
    def calc_sma(prices, params, weights):
        comp = 0.0
        curr = prices.iloc[-1]
        for p, w in zip(params, weights):
            if len(prices) < p: continue
            sma = prices.rolling(window=p).mean().iloc[-1]
            val = 1.0 if curr > sma else -1.0
            comp += val * w
        return comp
    w = GAINER_PARAMS["WEIGHTS"]
    s1 = calc_macd(df_1h['close'], GAINER_PARAMS["MACD_1H"]['params'], GAINER_PARAMS["MACD_1H"]['weights']) * w[0]
    s3 = calc_macd(df_1d['close'], GAINER_PARAMS["MACD_1D"]['params'], GAINER_PARAMS["MACD_1D"]['weights']) * w[2]
    s6 = calc_sma(df_1d['close'], GAINER_PARAMS["SMA_1D"]['params'], GAINER_PARAMS["SMA_1D"]['weights']) * w[5]
    total = sum(w)
    return (s1 + s3 + s6) / total if total > 0 else 0.0

# --- Execution Engine ---

def limit_chaser(api, target_qty):
    if dry: return
    log.info(f"Starting Limit Chaser. Target: {target_qty:.4f}")
    order_id = None
    
    for i in range(int(LIMIT_CHASE_DURATION / CHASE_INTERVAL)):
        curr_pos = get_net_position(api)
        delta = target_qty - curr_pos
        
        if abs(delta) < MIN_TRADE_SIZE:
            log.info(f"Chaser Complete: Target reached.")
            if order_id:
                try: 
                    c_resp = api.cancel_order({"orderId": order_id, "symbol": SYMBOL_FUTS})
                    log.info(f"Cleanup Cancel: {c_resp.get('result')}")
                except: pass
            break
            
        side = "buy" if delta > 0 else "sell"
        size = round(abs(delta), 4)
        limit_px = 0
        try:
            tk = api.get_tickers()
            for t in tk["tickers"]:
                if t["symbol"] == SYMBOL_FUTS:
                    limit_px = int(float(t["bid"])) if side == "buy" else int(float(t["ask"]))
                    limit_px += (-LIMIT_OFFSET_TICKS if side == "buy" else LIMIT_OFFSET_TICKS)
                    break
        except Exception as e:
            log.error(f"Ticker fetch failed: {e}")
            time.sleep(5)
            continue
            
        try:
            if order_id:
                c_resp = api.cancel_order({"orderId": order_id, "symbol": SYMBOL_FUTS})
                log.info(f"Chase {i+1} Cancel: {c_resp.get('result')}")
            
            payload = {"orderType": "lmt", "symbol": SYMBOL_FUTS, "side": side, "size": size, "limitPrice": limit_px, "postOnly": True}
            resp = api.send_order(payload)
            log.info(f"Chase {i+1} Place [{side} {size} @ {limit_px}]: {resp.get('result')}")
            if "sendStatus" in resp and "order_id" in resp["sendStatus"]:
                order_id = resp["sendStatus"]["order_id"]
        except Exception as e:
            log.error(f"Chaser Execution Error: {e}")
        time.sleep(CHASE_INTERVAL)
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass

def manage_virtual_stops(api, state, net_size, price, cap_per_strat):
    net_size = round(net_size, 4)
    if dry or abs(net_size) < MIN_TRADE_SIZE: return
    try: api.cancel_all_orders({"symbol": SYMBOL_FUTS})
    except: pass
    
    side = "sell" if net_size > 0 else "buy"
    qty = round(abs(net_size) * 0.33, 4)
    if qty < MIN_TRADE_SIZE: return

    stop_px_p = int(price * (1 - PLANNER_PARAMS["S1_STOP"])) if side == "sell" else int(price * (1 + PLANNER_PARAMS["S1_STOP"]))
    try:
        resp = api.send_order({"orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, "size": qty, "stopPrice": stop_px_p, "reduceOnly": True})
        log.info(f"Planner Stop: {resp.get('result')}")
    except: pass

    stop_px_t = int(price * (1 - TUMBLER_PARAMS["STOP"])) if side == "sell" else int(price * (1 + TUMBLER_PARAMS["STOP"]))
    try:
        resp = api.send_order({"orderType": "stp", "symbol": SYMBOL_FUTS, "side": side, "size": qty, "stopPrice": stop_px_t, "reduceOnly": True})
        log.info(f"Tumbler Stop: {resp.get('result')}")
    except: pass

def run_cycle(api):
    log.info(">>> CYCLE START <<<")
    try:
        df_1h = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 60)
        df_1d = kraken_ohlc.get_ohlc(SYMBOL_OHLC, 1440)
    except Exception as e:
        log.error(f"Failed to fetch OHLC: {e}")
        return

    state = load_state()
    state["df_1d"] = df_1d 
    curr_price = get_market_price(api) or df_1h['close'].iloc[-1]
    state["market"]["price"] = curr_price
    
    try:
        accts = api.get_accounts()
        total_pv = float(accts["accounts"]["flex"]["portfolioValue"])
        strat_cap = total_pv * CAP_SPLIT
        log.info(f"Total PV: ${total_pv:.2f} | StratAlloc: ${strat_cap:.2f}")
        state["portfolio"]["total_equity"] = total_pv
        state["portfolio"]["strat_cap"] = strat_cap
    except: return
    
    # Logic
    raw_planner = run_planner(df_1d, state, strat_cap)
    raw_tumbler = run_tumbler(df_1d, state, strat_cap)
    raw_gainer = run_gainer(df_1h, df_1d)
    
    # Fairness
    norm_planner = raw_planner
    norm_tumbler = raw_tumbler * (TARGET_STRAT_LEV / TUMBLER_MAX_LEV)
    norm_gainer  = raw_gainer  * TARGET_STRAT_LEV
    
    # State Update
    state["strategies"]["planner"].update({"raw_lev": raw_planner, "fair_lev": norm_planner})
    state["strategies"]["tumbler"].update({"raw_lev": raw_tumbler, "fair_lev": norm_tumbler})
    state["strategies"]["gainer"].update({"raw_lev": raw_gainer, "fair_lev": norm_gainer})

    log.info(f"Fair Levs | Plan: {norm_planner:.2f} | Tumb: {norm_tumbler:.2f} | Gain: {norm_gainer:.2f}")

    lev_total = norm_planner + norm_tumbler + norm_gainer
    target_qty = lev_total * strat_cap / curr_price
    
    # Execution
    limit_chaser(api, target_qty)
    
    final_pos = get_net_position(api)
    state["position"].update({"current": final_pos, "target": target_qty, "delta": target_qty - final_pos})
    
    manage_virtual_stops(api, state, final_pos, curr_price, strat_cap)
    save_state_rich(state, LOG_BUFFER)
    log.info(">>> CYCLE END <<<")

def wait_until_next_hour():
    now = datetime.now(timezone.utc)
    next_run = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
    wait = (next_run - now).total_seconds()
    log.info(f"Sleeping until {next_run.strftime('%H:%M')} ({wait/60:.1f}m)")
    time.sleep(wait)

# --- WEB SERVER (DASHBOARD) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Trader Observation</title>
    <style>
        :root { --bg-dark: #0d1117; --bg-card: #161b22; --border: #30363d; --text-main: #c9d1d9; --text-muted: #8b949e; --accent: #58a6ff; --success: #238636; --danger: #da3633; }
        body { background-color: var(--bg-dark); color: var(--text-main); font-family: 'Segoe UI', monospace; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; margin-bottom: 20px; }
        .status-badge { background: var(--success); color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 20px; }
        .card h3 { margin-top: 0; color: var(--text-muted); font-size: 0.9em; text-transform: uppercase; }
        .metric { font-size: 2em; font-weight: bold; color: var(--text-main); }
        .sub-metric { color: var(--text-muted); font-size: 0.9em; }
        .strat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 20px; }
        .strat-card { background: #21262d; padding: 15px; border-radius: 6px; text-align: center; }
        .strat-val { font-size: 1.5em; font-weight: bold; }
        .pos { color: #3fb950; }
        .neg { color: #f85149; }
        .log-window { background: #000; border: 1px solid var(--border); height: 300px; overflow-y: scroll; padding: 10px; font-family: 'Consolas', monospace; font-size: 0.85em; color: #a5d6ff; margin-top: 20px; }
        .log-entry { margin-bottom: 4px; border-bottom: 1px solid #1f1f1f; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div><h1 style="margin:0">Master Trader Node</h1><small id="last-update">Waiting...</small></div>
            <div class="status-badge">Online</div>
        </div>
        <div class="grid">
            <div class="card"><h3>Portfolio Equity</h3><div class="metric" id="total-equity">---</div><div class="sub-metric">Strategy Cap: <span id="strat-cap">---</span></div></div>
            <div class="card"><h3>Active Position</h3><div class="metric" id="current-pos">---</div><div class="sub-metric">Target: <span id="target-pos">---</span> | Delta: <span id="delta-pos">---</span></div></div>
            <div class="card"><h3>Market Price</h3><div class="metric" id="market-price">---</div><div class="sub-metric">Symbol: FF_XBTUSD_260130</div></div>
        </div>
        <h2 style="margin-top:30px;">Strategy Conviction (Fair 2.0)</h2>
        <div class="strat-grid">
            <div class="strat-card"><h3>Planner</h3><div class="strat-val" id="lev-planner">---</div><small>Raw: <span id="raw-planner">---</span></small></div>
            <div class="strat-card"><h3>Tumbler</h3><div class="strat-val" id="lev-tumbler">---</div><small>Raw: <span id="raw-tumbler">---</span></small></div>
            <div class="strat-card"><h3>Gainer</h3><div class="strat-val" id="lev-gainer">---</div><small>Raw: <span id="raw-gainer">---</span></small></div>
        </div>
        <h2 style="margin-top:30px;">System Logs</h2>
        <div class="log-window" id="log-container"></div>
    </div>
    <script>
        function colorize(val, id) {
            const el = document.getElementById(id);
            el.innerText = parseFloat(val).toFixed(2) + 'x';
            el.className = 'strat-val ' + (val >= 0 ? 'pos' : 'neg');
        }
        async function update() {
            try {
                const res = await fetch('/api/state');
                const d = await res.json();
                document.getElementById('total-equity').innerText = '$' + d.portfolio.total_equity.toFixed(2);
                document.getElementById('strat-cap').innerText = '$' + d.portfolio.strat_cap.toFixed(2);
                document.getElementById('current-pos').innerText = d.position.current.toFixed(4);
                document.getElementById('target-pos').innerText = d.position.target.toFixed(4);
                document.getElementById('delta-pos').innerText = d.position.delta.toFixed(4);
                document.getElementById('market-price').innerText = '$' + d.market.price.toFixed(1);
                document.getElementById('last-update').innerText = "Last: " + new Date(d.market.last_updated).toLocaleString();
                
                colorize(d.strategies.planner.fair_lev, 'lev-planner');
                document.getElementById('raw-planner').innerText = d.strategies.planner.raw_lev.toFixed(2);
                colorize(d.strategies.tumbler.fair_lev, 'lev-tumbler');
                document.getElementById('raw-tumbler').innerText = d.strategies.tumbler.raw_lev.toFixed(2);
                colorize(d.strategies.gainer.fair_lev, 'lev-gainer');
                document.getElementById('raw-gainer').innerText = d.strategies.gainer.raw_lev.toFixed(2);
                
                document.getElementById('log-container').innerHTML = d.logs.slice().reverse().map(l => `<div class="log-entry">${l}</div>`).join('');
            } catch(e) {}
        }
        setInterval(update, 2000); update();
    </script>
</body>
</html>
"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif self.path == '/api/state':
            try:
                if STATE_FILE.exists():
                    with open(STATE_FILE, 'r') as f: data = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(data.encode('utf-8'))
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Init..."}')
            except: self.send_error(500)
        else: self.send_error(404)

def start_server():
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        log.info(f"Dashboard running on port {PORT}")
        httpd.serve_forever()

def main():
    api_key, api_sec = os.getenv("KRAKEN_API_KEY"), os.getenv("KRAKEN_API_SECRET")
    if not api_key: sys.exit("No API Keys")
    api = kf.KrakenFuturesApi(api_key, api_sec)
    
    # Start Dashboard in Background Thread
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    
    if os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}:
        run_cycle(api)
        
    while True:
        wait_until_next_hour()
        run_cycle(api)

if __name__ == "__main__":
    main()
