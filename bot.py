"""
Bayse BTC Bot — v7 (simple)
============================
Two filters only:
  • Confidence ≥ 80%   (model must be at least 80% sure)
  • Payout   ≥ 0.30x  (win must return at least $1.30 per $1 staked)

Everything else is removed. Bot fires ~10–15 trades per day.
Auto-buys $1 flat. Balance synced from Bayse API.

Commands: /price /balance /stats /log /pause /play
          /play_silent /pause_silent /increase N /decrease N
          /start_session /stop_session /export /config
"""

import logging, requests, joblib, json, re, os, csv, sys, time
import hashlib, hmac as hmaclib, base64
import pandas as pd, numpy as np, ta
from datetime import datetime, timezone, timedelta
from collections import deque

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these two values only
# ─────────────────────────────────────────────────────────────
CONFIDENCE   = 0.80   # model confidence threshold
MIN_PAYOUT   = 0.30   # minimum profit per $1 stake (payout ≥ 0.30 → return ≥ $1.30)
STAKE        = 1.00   # default stake per trade
BALANCE_MIN  = 2.00   # pause auto-buy if balance drops below this
MAX_LOSSES   = 3      # circuit break after this many consecutive losses
SIGNAL_DELAY = 5      # minutes into the round before firing signal
FEE          = 0.05   # Bayse fee (5%)
LOG_FILE     = "trades.csv"

# ─────────────────────────────────────────────────────────────
# ENV VARS
# ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT    = os.environ.get("TELEGRAM_CHAT", "")
BAYSE_KEY        = os.environ.get("BAYSE_KEY", "")
BAYSE_SECRET_KEY = os.environ.get("BAYSE_SECRET_KEY", "")
BASE_URL         = "https://relay.bayse.markets/v1"
BAYSE_HEADERS    = {"X-Public-Key": BAYSE_KEY}
TG_API           = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
AUTO_BUY         = bool(BAYSE_SECRET_KEY)

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT or not BAYSE_KEY:
    print("ERROR: TELEGRAM_TOKEN, TELEGRAM_CHAT, BAYSE_KEY required", flush=True)
    sys.exit(1)

model    = joblib.load("btc_bayse_model_v2.joblib")
features = json.load(open("features_v2.json"))
print(f"✅ Bot v7 | conf≥{CONFIDENCE:.0%} | payout≥{MIN_PAYOUT:.2f}x | "
      f"stake=${STAKE} | auto_buy={AUTO_BUY}", flush=True)

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
CANDLES       = deque(maxlen=100)
paused        = False
silent_mode   = False
circuit_off   = False
consec_losses = 0
session_num   = 1
session_start = datetime.now(timezone.utc)
session_on    = True
balance       = 10.00        # will be synced from API
current_stake = STAKE
total_staked  = 0.0
total_won     = 0.0
last_update   = None
mkt_cache     = None

round_state = {
    "event_id": None, "prev_event_id": None,
    "start_time": None, "target": None,
    "fired": False, "signal": None, "order_id": None,
}
last_round = {
    "target": None, "close": None, "direction": None,
    "correct": None, "resolved": False,
}
stats = {
    "total": 0, "wins": 0, "losses": 0,
    "orders_ok": 0, "orders_fail": 0, "breaks": 0,
}
trade_log = []

# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────
def tg(text, chat=None):
    try:
        requests.post(f"{TG_API}/sendMessage",
            json={"chat_id": chat or TELEGRAM_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=10)
    except Exception as e:
        log.error(f"tg: {e}")

def tg_doc(filename, data, caption, chat=None):
    try:
        requests.post(f"{TG_API}/sendDocument",
            data={"chat_id": chat or TELEGRAM_CHAT, "caption": caption},
            files={"document": (filename, data, "text/csv")}, timeout=15)
    except Exception as e:
        log.error(f"tg_doc: {e}")

def tg_poll():
    global last_update
    try:
        p = {"timeout": 1, "allowed_updates": ["message"]}
        if last_update: p["offset"] = last_update + 1
        r = requests.get(f"{TG_API}/getUpdates", params=p, timeout=5)
        if r.ok:
            ups = r.json().get("result", [])
            for u in ups: last_update = u["update_id"]
            return ups
    except: pass
    return []

# ─────────────────────────────────────────────────────────────
# BAYSE API
# ─────────────────────────────────────────────────────────────
def fetch_market():
    try:
        r = requests.get(f"{BASE_URL}/pm/events", headers=BAYSE_HEADERS,
                         params={"limit": 50}, timeout=10)
        for ev in r.json().get("events", []):
            if "bitcoin" in ev.get("title","").lower() and "15" in ev.get("title",""):
                r2 = requests.get(f"{BASE_URL}/pm/events/{ev['id']}",
                                  headers=BAYSE_HEADERS, timeout=10)
                d = r2.json()
                ms = d.get("markets", [])
                if not ms: continue
                m = ms[0]

                def _f(obj, *keys):
                    for k in keys:
                        v = obj.get(k)
                        if v is not None:
                            try: return float(v)
                            except: pass
                    return None

                return {
                    "event_id"  : d["id"],
                    "yes_price" : m.get("outcome1Price", 0),
                    "no_price"  : m.get("outcome2Price", 0),
                    "liquidity" : d.get("liquidity", 0),
                    "orders"    : m.get("totalOrders", 0),
                    "rules"     : m.get("rules", ""),
                    "start_price": (
                        _f(d,"startPrice","start_price","targetPrice","target_price") or
                        _f(m,"startPrice","start_price","targetPrice","target_price")
                    ),
                    "outcome1Id": m.get("outcome1Id"),
                    "outcome2Id": m.get("outcome2Id"),
                    "market_id" : m.get("id"),
                    "raw"       : d,
                }
    except Exception as e:
        log.error(f"fetch_market: {e}")
    return None

def fetch_final_price(event_id):
    try:
        r = requests.get(f"{BASE_URL}/pm/events/{event_id}",
                         headers=BAYSE_HEADERS, timeout=10)
        if not r.ok: return None
        d = r.json(); ms = d.get("markets", []); m = ms[0] if ms else {}
        for k in ("finalPrice","final_price","resolutionPrice","resolution_price"):
            v = d.get(k) or m.get(k)
            if v is not None:
                try: return float(v)
                except: pass
    except: pass
    return None

def fetch_balance_api():
    try:
        r = requests.get(f"{BASE_URL}/wallet/assets", headers=BAYSE_HEADERS, timeout=8)
        if r.ok:
            d = r.json()
            assets = d if isinstance(d, list) else d.get("assets", d.get("data", []))
            if isinstance(assets, list):
                for a in assets:
                    if a.get("currency","").upper() in ("USD","USDT","USDC"):
                        v = a.get("available") or a.get("balance") or a.get("amount")
                        if v is not None: return float(v)
            for k in ("usd","USD","available","balance"):
                if k in d: return float(d[k])
    except: pass
    return None

def parse_end_dt(rules, now):
    m = re.search(r'(\d+:\d+:\d+\s?[AP]M)\s?GMT', rules)
    if m:
        try:
            end = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} {m.group(1).strip()}",
                "%Y-%m-%d %I:%M:%S %p").replace(tzinfo=timezone.utc)
            if (end - now).total_seconds() < -3600: end += timedelta(days=1)
            return end
        except: pass
    return None

# ─────────────────────────────────────────────────────────────
# KUCOIN CANDLES (for ML features only)
# ─────────────────────────────────────────────────────────────
def seed_candles():
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                         params={"symbol":"BTC-USDT","type":"1min"}, timeout=10)
        for c in reversed(r.json().get("data",[])):
            CANDLES.append({"open_time": pd.Timestamp(int(c[0]),unit="s",tz="UTC"),
                "open":float(c[1]),"close":float(c[2]),"high":float(c[3]),
                "low":float(c[4]),"volume":float(c[5])})
        log.info(f"Seeded {len(CANDLES)} candles | ${CANDLES[-1]['close']:,.2f}")
    except Exception as e: log.error(f"seed_candles: {e}")

def update_candles():
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                         params={"symbol":"BTC-USDT","type":"1min","pageSize":3}, timeout=5)
        data = r.json().get("data",[])
        if not data: return
        c = data[0]
        latest = {"open_time": pd.Timestamp(int(c[0]),unit="s",tz="UTC"),
                  "open":float(c[1]),"close":float(c[2]),"high":float(c[3]),
                  "low":float(c[4]),"volume":float(c[5])}
        if not CANDLES or latest["open_time"] > CANDLES[-1]["open_time"]:
            CANDLES.append(latest)
    except: pass

def btc_price():
    return CANDLES[-1]["close"] if CANDLES else None

# ─────────────────────────────────────────────────────────────
# ML FEATURES
# ─────────────────────────────────────────────────────────────
def compute_features(target):
    if len(CANDLES) < 60 or not target: return None
    df = pd.DataFrame(list(CANDLES)).sort_values("open_time").reset_index(drop=True)
    try:
        df["rsi_14"]      = ta.momentum.RSIIndicator(df["close"],14).rsi()
        df["rsi_7"]       = ta.momentum.RSIIndicator(df["close"],7).rsi()
        df["rsi_21"]      = ta.momentum.RSIIndicator(df["close"],21).rsi()
        df["stoch"]       = ta.momentum.StochasticOscillator(df["high"],df["low"],df["close"]).stoch()
        df["macd"]        = ta.trend.MACD(df["close"]).macd_diff()
        df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
        df["ema_9"]       = ta.trend.EMAIndicator(df["close"],9).ema_indicator()
        df["ema_21"]      = ta.trend.EMAIndicator(df["close"],21).ema_indicator()
        df["ema_cross"]   = df["ema_9"] - df["ema_21"]
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_position"] = (df["close"]-bb.bollinger_mavg())/bb.bollinger_wband()
        df["bb_width"]    = bb.bollinger_wband()
        df["atr"]         = ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
        df["vol_ratio_15"]= df["volume"]/df["volume"].rolling(15).mean()
        df["vol_ratio_60"]= df["volume"]/df["volume"].rolling(60).mean()
        df["obv"]         = ta.volume.OnBalanceVolumeIndicator(df["close"],df["volume"]).on_balance_volume()
        df["obv_slope"]   = df["obv"].diff(5)
        df["candle_body"] = (df["close"]-df["open"])/df["open"]
        df["momentum_1"]  = df["close"].pct_change(1)
        df["momentum_3"]  = df["close"].pct_change(3)
        df["momentum_5"]  = df["close"].pct_change(5)
        df["rolling_std_15"] = df["close"].pct_change().rolling(15).std()
        df["rolling_std_60"] = df["close"].pct_change().rolling(60).std()
        df["hour"]        = df["open_time"].dt.hour
        df["dayofweek"]   = df["open_time"].dt.dayofweek
        latest = df.iloc[-1].copy()
        atr = max(float(latest["atr"]),1e-8)
        pvt = (latest["close"]-target)/target
        latest["price_vs_target"] = pvt
        latest["price_gap_usd"]   = latest["close"]-target
        latest["gap_vs_atr"]      = (latest["close"]-target)/atr
        latest["gap_pct_abs"]     = abs(pvt)
        latest["early_momentum"]  = pvt
        latest["rsi_vs_neutral"]  = float(latest["rsi_14"])-50
        return latest[features].values
    except Exception as e:
        log.error(f"features: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# ORDER PLACEMENT
# ─────────────────────────────────────────────────────────────
def sign_request(method, path, body_str):
    ts = str(int(datetime.now(timezone.utc).timestamp()))
    bh = hashlib.sha256(body_str.encode()).hexdigest()
    payload = f"{ts}.{method}.{path}.{bh}"
    sig = base64.b64encode(
        hmaclib.new(BAYSE_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).digest()
    ).decode()
    return ts, sig

def place_order(market, direction_up, amount):
    if not BAYSE_SECRET_KEY: return None
    event_id = market["raw"].get("id")
    market_id = market["market_id"]
    outcome_id = market["outcome1Id"] if direction_up else market["outcome2Id"]
    if not all([event_id, market_id]):
        log.error("place_order: missing event_id or market_id"); return None

    sign_path = f"/v1/pm/events/{event_id}/markets/{market_id}/orders"
    req_path  = f"/pm/events/{event_id}/markets/{market_id}/orders"
    payload   = {"side":"BUY","amount":round(amount,2),"type":"MARKET",
                 "currency":"USD","outcome":"YES" if direction_up else "NO"}
    if outcome_id: payload["outcomeId"] = outcome_id

    body_str = json.dumps(payload, separators=(",",":"))
    ts, sig  = sign_request("POST", sign_path, body_str)

    try:
        r = requests.post(f"{BASE_URL}{req_path}",
            headers={"X-Public-Key":BAYSE_KEY,"X-Timestamp":ts,
                     "X-Signature":sig,"Content-Type":"application/json"},
            data=body_str, timeout=10)
        log.info(f"order response: HTTP {r.status_code} | {r.text[:300]}")
        if r.ok:
            d = r.json(); o = d.get("order",{})
            return {"order_id": str(o.get("id","?")), "status": o.get("status","?"),
                    "amount": o.get("amount", amount), "price": o.get("price"),
                    "quantity": o.get("quantity"), "engine": d.get("engine","?")}
        else:
            log.error(f"order failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        log.error(f"place_order: {e}")
    return None

# ─────────────────────────────────────────────────────────────
# CSV LOG
# ─────────────────────────────────────────────────────────────
FIELDS = ["timestamp","time","target","close","direction","actual",
          "correct","conf","payout","op","gap_pct","stake","order_id","session"]

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w",newline="") as f: csv.writer(f).writerow(FIELDS)

def write_log(row):
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([row.get(k,"") for k in FIELDS])

# ─────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────
def handle_commands():
    global paused, circuit_off, consec_losses, silent_mode, current_stake
    global session_num, session_start, session_on, balance, total_staked, total_won
    global stats, trade_log, last_round, round_state

    for u in tg_poll():
        msg  = u.get("message",{})
        text = msg.get("text","").strip().lower()
        chat = str(msg.get("chat",{}).get("id",""))

        if text.startswith("/pause"):
            paused = True
            tg("⏸ *Paused.* No signals or orders.", chat)

        elif text.startswith("/play_silent"):
            silent_mode = True
            tg(f"🔇→📢 *Silent mode ON* — betting ${current_stake:.2f} on all signals "
               f"that pass conf+payout filters.\n/pause_silent to stop.", chat)

        elif text.startswith("/pause_silent"):
            silent_mode = False
            tg("🔇 *Silent mode OFF.*", chat)

        elif text.startswith("/increase"):
            parts = text.split()
            try:
                current_stake = round(current_stake + float(parts[1]), 2)
                tg(f"📈 Stake → *${current_stake:.2f}*", chat)
            except:
                tg(f"Usage: /increase 2\nCurrent stake: ${current_stake:.2f}", chat)

        elif text.startswith("/decrease"):
            parts = text.split()
            try:
                new = round(current_stake - float(parts[1]), 2)
                if new < 1.00: tg(f"❌ Can't go below $1.00 (would be ${new:.2f})", chat)
                else:
                    current_stake = new
                    tg(f"📉 Stake → *${current_stake:.2f}*", chat)
            except:
                tg(f"Usage: /decrease 1\nCurrent stake: ${current_stake:.2f}", chat)

        elif text.startswith("/play"):
            paused = False; circuit_off = False; consec_losses = 0
            tg("▶️ *Resumed.* Circuit break cleared.", chat)

        elif text.startswith("/balance"):
            global balance
            api = fetch_balance_api()
            if api is not None: balance = api; src = "Bayse API"
            else: src = "internal"
            wr = f"{stats['wins']/(stats['wins']+stats['losses'])*100:.1f}%" if (stats['wins']+stats['losses'])>0 else "N/A"
            tg(f"💰 *Balance*\n"
               f"  Amount : *${balance:.2f}*  _({src})_\n"
               f"  P&L    : ${balance-10:+.2f}\n"
               f"  Stake  : ${current_stake:.2f} | Silent: {'ON' if silent_mode else 'OFF'}\n"
               f"  Wins   : {stats['wins']} | Losses: {stats['losses']} | WR: {wr}\n"
               f"  Orders : {stats['orders_ok']} ok, {stats['orders_fail']} failed", chat)

        elif text.startswith("/stats"):
            total = stats["wins"] + stats["losses"]
            wr = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            tg(f"📊 *Stats — Session #{session_num}*\n"
               f"  Trades : {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} | ❌ {stats['losses']}\n"
               f"  Balance: ${balance:.2f}\n"
               f"  Stake  : ${current_stake:.2f}\n"
               f"  Circuit: {'⚡ ACTIVE' if circuit_off else f'✅ {consec_losses}/{MAX_LOSSES}'}\n"
               f"  Silent : {'ON' if silent_mode else 'OFF'}\n"
               f"  Status : {'⏸ Paused' if paused else '▶️ Running'}", chat)

        elif text.startswith("/log"):
            if not trade_log:
                tg("No trades yet.", chat); continue
            lines = [f"📋 *Last trades*"]
            for t in trade_log[-10:]:
                e = "✅" if t["correct"] else "❌"
                lines.append(f"{e} {t['time']} | {t['direction']} | "
                              f"conf:{t['conf']:.0%} | pay:{t['payout']:.2f}x")
            tg("\n".join(lines), chat)

        elif text.startswith("/price"):
            btc = btc_price(); tgt = round_state.get("target")
            if btc and tgt:
                diff = btc-tgt
                tg(f"💰 BTC: ${btc:,.2f}\n🎯 Target: ${tgt:,.2f}\n"
                   f"💹 Gap: ${diff:+.2f} ({diff/tgt*100:+.3f}%)", chat)
            else:
                tg("No round data yet.", chat)

        elif text.startswith("/config"):
            tg(f"⚙️ *Config*\n"
               f"  Confidence : ≥{CONFIDENCE:.0%}\n"
               f"  Min payout : ≥{MIN_PAYOUT:.2f}x (return ≥${1+MIN_PAYOUT:.2f})\n"
               f"  Stake      : ${current_stake:.2f}\n"
               f"  Auto-buy   : {'✅ ON' if AUTO_BUY else '⚠️ OFF'}\n"
               f"  Balance min: ${BALANCE_MIN:.2f}\n"
               f"  Circuit    : after {MAX_LOSSES} losses", chat)

        elif text.startswith("/export"):
            if not os.path.exists(LOG_FILE):
                tg("No log yet.", chat); continue
            with open(LOG_FILE,"rb") as f: data = f.read()
            fname = f"session_{session_num}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
            tg_doc(fname, data, f"Session #{session_num} trades", chat)

        elif text.startswith("/start_session"):
            session_num += 1; session_start = datetime.now(timezone.utc)
            session_on = True; paused = False; circuit_off = False
            consec_losses = 0; current_stake = STAKE; silent_mode = False
            balance = 10.00; total_staked = 0.0; total_won = 0.0
            stats = {k:0 for k in stats}; trade_log = []
            last_round = {k:None for k in last_round}; last_round["resolved"]=False
            round_state = {"event_id":None,"prev_event_id":None,"start_time":None,
                           "target":None,"fired":False,"signal":None,"order_id":None}
            tg(f"🆕 *Session #{session_num} started*\n"
               f"  conf≥{CONFIDENCE:.0%} | payout≥{MIN_PAYOUT:.2f}x | stake=${current_stake:.2f}\n"
               f"  Auto-buy: {'ON' if AUTO_BUY else 'OFF'}", chat)

        elif text.startswith("/stop_session"):
            session_on = False; paused = True
            total = stats["wins"]+stats["losses"]
            wr = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            tg(f"🛑 *Session #{session_num} ended*\n"
               f"  Trades: {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} | ❌ {stats['losses']}\n"
               f"  Balance: ${balance:.2f} | P&L: ${balance-10:+.2f}", chat)

        elif text.startswith("/start"):
            tg(f"🤖 *Bayse BTC Bot v7*\n"
               f"  conf≥{CONFIDENCE:.0%} + payout≥{MIN_PAYOUT:.2f}x\n"
               f"  Auto-buy: {'✅ ON' if AUTO_BUY else '⚠️ OFF (set BAYSE_SECRET_KEY)'}\n\n"
               f"Commands:\n"
               f"  /balance  /stats  /log  /price  /config  /export\n"
               f"  /pause  /play  /play_silent  /pause_silent\n"
               f"  /increase N  /decrease N\n"
               f"  /start_session  /stop_session", chat)

# ─────────────────────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────────────────────
def get_signal(market, target, now):
    btc = btc_price()
    if not btc: return None

    feat = compute_features(target)
    if feat is None: return None

    raw_proba = model.predict_proba(pd.DataFrame([feat], columns=features))[0][1]
    up        = raw_proba > 0.5
    conf      = raw_proba if up else (1.0 - raw_proba)
    direction = "UP" if up else "DOWN"

    yes_p   = market["yes_price"]
    op      = yes_p if up else (1.0 - yes_p)
    op      = max(op, 0.01)
    payout  = (1.0/op - 1) * (1 - FEE)
    gap_pct = (btc - target) / target * 100

    # ── The only two filters ──────────────────────────────────
    skip = []
    if conf < CONFIDENCE:
        skip.append(f"Conf {conf:.1%} < {CONFIDENCE:.0%}")
    if payout < MIN_PAYOUT:
        skip.append(f"Payout {payout:.2f}x < {MIN_PAYOUT:.2f}x min")
    if circuit_off:
        skip.append(f"Circuit break ({consec_losses}/{MAX_LOSSES} losses)")
    if AUTO_BUY and balance < BALANCE_MIN:
        skip.append(f"Balance ${balance:.2f} < ${BALANCE_MIN:.2f} min")

    trade = not skip
    return {
        "trade"    : trade,
        "direction": direction,
        "up"       : up,
        "conf"     : conf,
        "payout"   : round(payout, 3),
        "op"       : op,
        "gap_pct"  : round(gap_pct, 4),
        "btc"      : btc,
        "target"   : target,
        "yes_p"    : yes_p,
        "no_p"     : 1.0-yes_p,
        "skip"     : " | ".join(skip),
        "liq"      : market.get("liquidity",0),
        "orders"   : market.get("orders",0),
    }

# ─────────────────────────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────────────────────────
def msg_open(market, btc, target, now, mins):
    sep = "─"*28
    diff = btc - target
    yes = market["yes_price"]; no = 1.0-yes

    if last_round["resolved"] and last_round["target"]:
        lr = last_round
        l_dir = "UP 📈" if (lr["close"] or 0) >= (lr["target"] or 0) else "DOWN 📉"
        if lr["correct"] is True:    res = "✅ WON"
        elif lr["correct"] is False: res = "❌ LOST"
        else:                        res = "— no trade"
        total = stats["wins"]+stats["losses"]
        wr = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
        last_blk = (f"{sep}\n📋 *Last Round*\n"
                    f"  Target: ${lr['target']:,.2f} → ${lr['close']:,.2f} {l_dir}\n"
                    f"  Bot: {lr['direction'] or '—'} → {res} | WR: {wr}\n")
    else:
        last_blk = f"{sep}\n📋 *Last Round:* no data yet\n"

    bal_line = f"\n💰 Balance: *${balance:.2f}*" if AUTO_BUY else ""
    status = ""
    if circuit_off: status = f"\n⚡ *Circuit break — {consec_losses} losses in a row* | /play to resume"
    elif paused:    status = "\n⏸ *Paused*"

    return (f"🔔 *NEW ROUND — Session #{session_num}*\n{sep}\n"
            f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ ~{mins} mins\n{sep}\n"
            f"💰 BTC    : ${btc:,.2f}\n"
            f"🎯 Target : ${target:,.2f}\n"
            f"💹 Gap    : ${diff:+.2f} ({diff/target*100:+.3f}%)\n"
            f"📊 YES {yes:.0%} | NO {no:.0%}\n"
            + last_blk + f"{sep}\n⏳ Signal in ~{SIGNAL_DELAY} mins"
            + bal_line + status)


def msg_signal(sig, order_id=None, silent=False):
    sep = "─"*28
    win_amt  = round(current_stake * sig["payout"], 2)
    total_ret = round(current_stake + win_amt, 2)

    if order_id:
        trade_blk = (f"\n{sep}\n🤖 *Order placed*\n"
                     f"  Stake : ${current_stake:.2f}\n"
                     f"  Win → +${win_amt:.2f}  (get back ${total_ret:.2f})\n"
                     f"  Lose → -${current_stake:.2f}\n"
                     f"  Order : `{order_id}`\n"
                     f"  Balance: ${balance:.2f}")
    elif AUTO_BUY:
        trade_blk = f"\n{sep}\n⚠️ *Order FAILED* — check logs"
    else:
        trade_blk = (f"\n{sep}\n💡 Stake ${current_stake:.2f} on *{sig['direction']}*\n"
                     f"  Win → +${win_amt:.2f} | Lose → -${current_stake:.2f}")

    prefix = "🔇 *SILENT TRADE*" if silent else "✅ *TRADE*"
    return (f"🤖 *SIGNAL*\n{sep}\n"
            f"  BTC    : ${sig['btc']:,.2f}\n"
            f"  Target : ${sig['target']:,.2f}\n"
            f"  Gap    : {sig['gap_pct']:+.3f}%\n{sep}\n"
            f"  {prefix}\n"
            f"  Direction : *{sig['direction']}*\n"
            f"  Confidence: {sig['conf']:.1%}\n"
            f"  Payout    : {sig['payout']:.2f}x  (odds {sig['op']:.0%})\n"
            f"  Liquidity : ${sig['liq']:,.0f} | Orders: {sig['orders']}"
            + trade_blk)


def msg_receipt(order, sig):
    sep = "─"*28
    win_amt   = round(current_stake * sig["payout"], 2)
    total_ret = round(current_stake + win_amt, 2)
    price = order.get("price"); qty = order.get("quantity")
    return (f"🧾 *ORDER RECEIPT*\n{sep}\n"
            f"  Direction : *{sig['direction']}*\n"
            f"  Spent     : ${order.get('amount', current_stake):.2f}\n"
            f"  Fill price: {'${:.4f}/share'.format(price) if price else 'market'}\n"
            f"  Shares    : {'{:.4f}'.format(qty) if qty else '—'}\n"
            f"  Engine    : {order.get('engine','?')}\n"
            f"  Status    : {order.get('status','?')}\n"
            f"  Order ID  : `{order['order_id']}`\n{sep}\n"
            f"  Win  → +${win_amt:.2f} (total ${total_ret:.2f})\n"
            f"  Lose → -${current_stake:.2f}\n"
            f"  Balance   : ${balance:.2f}")

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main():
    global balance, round_state, last_round, stats, trade_log
    global circuit_off, consec_losses

    init_log()
    seed_candles()

    mode = f"AUTO-BUY ${STAKE:.2f}" if AUTO_BUY else "NOTIFY ONLY"
    tg(f"🟢 *Bot v7 online — Session #{session_num}*\n"
       f"  conf≥{CONFIDENCE:.0%} | payout≥{MIN_PAYOUT:.2f}x | {mode}\n"
       f"  ~10–16 trades/day expected")
    log.info(f"Bot v7 started | auto_buy={AUTO_BUY}")

    while True:
        try:
            handle_commands()
            update_candles()

            if not session_on:
                time.sleep(30); continue

            now    = datetime.now(timezone.utc)
            market = fetch_market()
            if not market:
                time.sleep(30); continue

            mkt_cache = market
            event_id  = market["event_id"]
            btc       = btc_price()
            end_dt    = parse_end_dt(market["rules"], now)
            mins_left = round((end_dt-now).total_seconds()/60, 1) if end_dt else None

            # ── NEW ROUND ─────────────────────────────────────
            if event_id != round_state["event_id"]:
                log.info(f"New round: {event_id}")

                prev_sig    = round_state.get("signal")
                prev_target = round_state.get("target")

                # ── RESOLVE PREVIOUS ROUND ────────────────────
                if prev_target:
                    prev_eid = round_state.get("prev_event_id")
                    final    = None
                    if prev_eid:
                        final = fetch_final_price(prev_eid)
                        if final: log.info(f"Final price from Bayse: ${final:,.2f}")
                    if not final: final = market.get("start_price") or btc
                    if not final: log.error("No final price available")
                    else:
                        actual_up  = final >= prev_target
                        actual_dir = "UP" if actual_up else "DOWN"

                        if prev_sig:
                            correct = (prev_sig["up"] == actual_up)
                            was_trade = prev_sig.get("was_trade", False)
                            used_stake = prev_sig.get("stake_used", current_stake)

                            if was_trade:
                                stats["total"] += 1
                                if correct:
                                    stats["wins"] += 1; consec_losses = 0
                                    earned = round(used_stake * prev_sig["payout"], 2)
                                    balance = round(balance + earned - used_stake, 2)
                                    total_won += earned
                                    if circuit_off:
                                        circuit_off = False
                                        tg("✅ *Win — circuit break cleared.*")
                                else:
                                    stats["losses"] += 1; consec_losses += 1
                                    balance = round(balance - used_stake, 2)
                                    if consec_losses >= MAX_LOSSES and not circuit_off:
                                        circuit_off = True; stats["breaks"] += 1
                                        tg(f"⚡ *Circuit break — {MAX_LOSSES} losses in a row.*\n"
                                           f"Balance: ${balance:.2f} | /play to resume")

                                # Sync balance from API after each resolved trade
                                api_bal = fetch_balance_api()
                                if api_bal is not None:
                                    balance = api_bal
                                    log.info(f"Balance synced: ${balance:.2f}")

                                entry = {
                                    "timestamp": now.isoformat(), "time": now.strftime("%H:%M"),
                                    "target": prev_target, "close": final,
                                    "direction": prev_sig["direction"], "actual": actual_dir,
                                    "correct": correct, "conf": prev_sig["conf"],
                                    "payout": prev_sig["payout"], "op": prev_sig["op"],
                                    "gap_pct": prev_sig.get("gap_pct",""),
                                    "stake": used_stake,
                                    "order_id": round_state.get("order_id",""),
                                    "session": session_num,
                                }
                                write_log(entry)
                                trade_log.append(entry)
                                log.info(f"Resolved: {'✅' if correct else '❌'} | "
                                         f"balance=${balance:.2f}")

                        last_round.update({
                            "target": prev_target, "close": final,
                            "direction": prev_sig["direction"] if prev_sig else None,
                            "correct": correct if (prev_sig and prev_sig.get("was_trade")) else None,
                            "resolved": True,
                        })

                # ── SET NEW ROUND TARGET ──────────────────────
                new_target = market.get("start_price") or btc
                if not new_target: log.error("No target available")

                round_state.update({
                    "event_id"     : event_id,
                    "prev_event_id": round_state.get("event_id"),
                    "start_time"   : now,
                    "target"       : new_target,
                    "fired"        : False,
                    "signal"       : None,
                    "order_id"     : None,
                })

                if new_target and btc and mins_left:
                    tg(msg_open(market, btc, new_target, now, mins_left))
                    log.info(f"Open | target=${new_target:,.2f} | btc=${btc:,.2f}")

            # ── FIRE SIGNAL ───────────────────────────────────
            start_t = round_state.get("start_time")
            if (not round_state["fired"] and start_t and round_state["target"]
                    and not paused):
                mins_in = (now - start_t).total_seconds() / 60
                if mins_in >= SIGNAL_DELAY:
                    sig = get_signal(market, round_state["target"], now)
                    if sig:
                        round_state["fired"] = True

                        should_trade = (sig["trade"] or
                                        (silent_mode and sig["conf"] >= CONFIDENCE
                                         and sig["payout"] >= MIN_PAYOUT))
                        is_silent = silent_mode and not sig["trade"]

                        if should_trade:
                            sig["was_trade"]   = True
                            sig["stake_used"]  = current_stake
                            round_state["signal"] = sig
                            order = None; order_id = None

                            if AUTO_BUY and balance >= BALANCE_MIN:
                                order = place_order(market, sig["up"], current_stake)
                                if order:
                                    order_id = order["order_id"]
                                    round_state["order_id"] = order_id
                                    stats["orders_ok"] += 1
                                    # Sync balance immediately
                                    api_bal = fetch_balance_api()
                                    if api_bal is not None:
                                        balance = api_bal
                                else:
                                    stats["orders_fail"] += 1

                            tg(msg_signal(sig, order_id, silent=is_silent))
                            if order: tg(msg_receipt(order, sig))
                            log.info(f"Trade | {sig['direction']} | conf {sig['conf']:.1%} | "
                                     f"pay {sig['payout']:.2f}x | order={'ok' if order else 'none'}")
                        else:
                            sig["was_trade"] = False
                            round_state["signal"] = sig
                            log.info(f"Skip | {sig['direction']} | {sig['skip']}")
                    else:
                        log.warning("Features not ready")

            time.sleep(30)

        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
