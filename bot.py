"""
Bayse BTC Bot — v9
===================
Key improvements:
  • Session config wizard — /startsession asks you to set confidence, gap,
    min payout, blocked hours, stake interactively. Each session can have
    different settings.
  • /default resets session config to global defaults mid-session.
  • /config shows current session vs default settings side by side.
  • Target price always from Bayse API (startPrice field). Gap = KuCoin BTC
    vs Bayse target (KuCoin for live price because Bayse has no live price).
  • Every round sends signal to Telegram whether trading or not.
  • Confidence default lowered to 70%.
  • Min payout filter is optional (disabled by default, session can enable).
  • Hour blocking is optional (disabled by default, session can enable).

Commands:
  /start        — welcome + command list
  /startsession — begin wizard to configure new session
  /stopsession  — end session + final report
  /default      — reset session config to defaults
  /config       — show current session config vs defaults
  /balance      — live balance (from Bayse API when available)
  /price        — BTC vs Bayse target + odds
  /stats        — session performance
  /log          — last 10 signals
  /hours        — win rate by UTC hour
  /export       — download CSV
  /pause        — pause all trading
  /play         — resume + clear circuit break
  /playsilent   — also trade skipped (filtered-out) signals
  /pausesilent  — stop silent trading
  /increase N   — add $N to current stake
  /decrease N   — remove $N from stake (min $1)
  /stake        — show current stake
"""

import logging, requests, joblib, json, re, os, csv, sys, time
import hashlib, hmac as _hmac, base64
import pandas as pd, numpy as np, ta
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# GLOBAL DEFAULTS — never changed at runtime
# (Edit these in Railway env or here for permanent changes)
# ─────────────────────────────────────────────────────────────
DEFAULT = {
    "confidence"   : 0.70,    # model confidence threshold
    "min_gap"      : 0.05,    # % BTC must have moved from Bayse target
    "min_payout"   : 0.0,     # 0 = disabled. e.g. 0.30 = need ≥$1.30 return
    "blocked_hours": [],      # e.g. [9, 2, 13] — hours UTC to skip
    "stake"        : 1.00,    # $ per trade
    "max_losses"   : 3,       # circuit break after N straight losses
}

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT    = os.environ.get("TELEGRAM_CHAT", "")
BAYSE_KEY        = os.environ.get("BAYSE_KEY", "")
BAYSE_SECRET_KEY = os.environ.get("BAYSE_SECRET_KEY", "")
BASE_URL         = "https://relay.bayse.markets/v1"
BAYSE_HEADERS    = {"X-Public-Key": BAYSE_KEY}
TG_API           = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
AUTO_BUY         = bool(BAYSE_SECRET_KEY)
SIGNAL_DELAY     = 5     # minutes into round before firing signal
FEE              = 0.05  # Bayse fee
LOG_FILE         = "trades.csv"

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT or not BAYSE_KEY:
    print("ERROR: TELEGRAM_TOKEN, TELEGRAM_CHAT, BAYSE_KEY required", flush=True)
    sys.exit(1)

model    = joblib.load("btc_bayse_model_v2.joblib")
features = json.load(open("features_v2.json"))
print(f"✅ v9 | auto_buy={AUTO_BUY}", flush=True)

# ─────────────────────────────────────────────────────────────
# SESSION CONFIG — copy of DEFAULT, overridden by wizard
# ─────────────────────────────────────────────────────────────
cfg = DEFAULT.copy()   # current session settings

# ─────────────────────────────────────────────────────────────
# MUTABLE STATE
# ─────────────────────────────────────────────────────────────
CANDLES       = deque(maxlen=100)
paused        = False
silent_mode   = False
circuit_off   = False
consec_losses = 0
session_num   = 0         # starts at 0; first /startsession makes it 1
session_start = datetime.now(timezone.utc)
session_on    = False     # off until /startsession
balance       = 10.00
current_stake = DEFAULT["stake"]
total_staked  = 0.0
total_won     = 0.0
last_update   = None

# Wizard state — multi-step /startsession flow
wizard_active  = False
wizard_step    = 0
wizard_chat    = None
wizard_data    = {}

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
    "total": 0, "signalled": 0, "wins": 0, "losses": 0,
    "orders_ok": 0, "orders_fail": 0, "breaks": 0,
}
trade_log  = []
hour_stats = defaultdict(lambda: {"total": 0, "correct": 0})

# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────
def tg(text, chat=None):
    try:
        requests.post(f"{TG_API}/sendMessage",
            json={"chat_id": chat or TELEGRAM_CHAT,
                  "text": text, "parse_mode": "Markdown"},
            timeout=10)
    except Exception as e:
        log.error(f"tg: {e}")

def tg_doc(filename, data, caption, chat=None):
    try:
        requests.post(f"{TG_API}/sendDocument",
            data={"chat_id": chat or TELEGRAM_CHAT, "caption": caption},
            files={"document": (filename, data, "text/csv")},
            timeout=15)
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

def sep(): return "─" * 30

# ─────────────────────────────────────────────────────────────
# BAYSE API
# ─────────────────────────────────────────────────────────────
def _tryf(obj, *keys):
    for k in keys:
        v = obj.get(k)
        if v is not None:
            try: return float(v)
            except: pass
    return None

def fetch_market():
    try:
        r = requests.get(f"{BASE_URL}/pm/events", headers=BAYSE_HEADERS,
                         params={"limit": 50}, timeout=10)
        for ev in r.json().get("events", []):
            if "bitcoin" in ev.get("title","").lower() and "15" in ev.get("title",""):
                r2 = requests.get(f"{BASE_URL}/pm/events/{ev['id']}",
                                  headers=BAYSE_HEADERS, timeout=10)
                d  = r2.json(); ms = d.get("markets", [])
                if not ms: continue
                m = ms[0]
                # Bayse target price — the definitive reference price for the round
                bayse_target = (
                    _tryf(d,"startPrice","start_price","targetPrice","target_price") or
                    _tryf(m,"startPrice","start_price","targetPrice","target_price")
                )
                return {
                    "event_id"   : d["id"],
                    "yes_price"  : float(m.get("outcome1Price", 0.5)),
                    "no_price"   : float(m.get("outcome2Price", 0.5)),
                    "liquidity"  : float(d.get("liquidity", 0)),
                    "orders"     : int(m.get("totalOrders", 0)),
                    "rules"      : m.get("rules", ""),
                    "bayse_target": bayse_target,   # ← Bayse Price Target
                    "outcome1Id" : m.get("outcome1Id"),
                    "outcome2Id" : m.get("outcome2Id"),
                    "market_id"  : m.get("id"),
                    "raw"        : d,
                }
    except Exception as e:
        log.error(f"fetch_market: {e}")
    return None

def fetch_final_price(event_id):
    try:
        r = requests.get(f"{BASE_URL}/pm/events/{event_id}",
                         headers=BAYSE_HEADERS, timeout=10)
        if not r.ok: return None
        d = r.json(); ms = d.get("markets",[]); m = ms[0] if ms else {}
        for k in ("finalPrice","final_price","resolutionPrice","resolution_price"):
            v = d.get(k) or m.get(k)
            if v is not None:
                try: return float(v)
                except: pass
    except: pass
    return None

def fetch_balance_api():
    try:
        r = requests.get(f"{BASE_URL}/wallet/assets",
                         headers=BAYSE_HEADERS, timeout=8)
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
# KUCOIN — live BTC price only (for gap calculation vs Bayse target)
# ─────────────────────────────────────────────────────────────
def seed_candles():
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                         params={"symbol":"BTC-USDT","type":"1min"}, timeout=10)
        for c in reversed(r.json().get("data",[])):
            CANDLES.append({
                "open_time": pd.Timestamp(int(c[0]),unit="s",tz="UTC"),
                "open": float(c[1]), "close": float(c[2]),
                "high": float(c[3]), "low":   float(c[4]), "volume": float(c[5]),
            })
        log.info(f"Seeded {len(CANDLES)} candles | ${CANDLES[-1]['close']:,.2f}")
    except Exception as e: log.error(f"seed_candles: {e}")

def update_candles():
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                         params={"symbol":"BTC-USDT","type":"1min","pageSize":3}, timeout=5)
        data = r.json().get("data",[])
        if not data: return
        c = data[0]
        latest = {"open_time":pd.Timestamp(int(c[0]),unit="s",tz="UTC"),
                  "open":float(c[1]),"close":float(c[2]),
                  "high":float(c[3]),"low":float(c[4]),"volume":float(c[5])}
        if not CANDLES or latest["open_time"] > CANDLES[-1]["open_time"]:
            CANDLES.append(latest)
    except: pass

def btc_live():
    """Live BTC price from KuCoin — used for gap vs Bayse target and ML features."""
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
        df["atr"]         = ta.volatility.AverageTrueRange(
                                df["high"],df["low"],df["close"],14).average_true_range()
        df["vol_ratio_15"]= df["volume"]/df["volume"].rolling(15).mean()
        df["vol_ratio_60"]= df["volume"]/df["volume"].rolling(60).mean()
        df["obv"]         = ta.volume.OnBalanceVolumeIndicator(
                                df["close"],df["volume"]).on_balance_volume()
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
def _sign(method, path, body_str):
    ts  = str(int(datetime.now(timezone.utc).timestamp()))
    bh  = hashlib.sha256(body_str.encode()).hexdigest()
    sig = base64.b64encode(
        _hmac.new(BAYSE_SECRET_KEY.encode(),
                  f"{ts}.{method}.{path}.{bh}".encode(),
                  hashlib.sha256).digest()
    ).decode()
    return ts, sig

def place_order(market, direction_up, amount):
    if not BAYSE_SECRET_KEY: return None
    event_id  = market["raw"].get("id")
    market_id = market["market_id"]
    outcome_id = market["outcome1Id"] if direction_up else market["outcome2Id"]
    if not event_id or not market_id:
        log.error("place_order: missing event_id or market_id"); return None

    sign_path = f"/v1/pm/events/{event_id}/markets/{market_id}/orders"
    req_path  = f"/pm/events/{event_id}/markets/{market_id}/orders"
    payload   = {"side":"BUY","amount":round(amount,2),"type":"MARKET",
                 "currency":"USD","outcome":"YES" if direction_up else "NO"}
    if outcome_id: payload["outcomeId"] = outcome_id

    body_str = json.dumps(payload, separators=(",",":"))
    ts, sig  = _sign("POST", sign_path, body_str)
    try:
        r = requests.post(f"{BASE_URL}{req_path}",
            headers={"X-Public-Key":BAYSE_KEY,"X-Timestamp":ts,
                     "X-Signature":sig,"Content-Type":"application/json"},
            data=body_str, timeout=10)
        log.info(f"order: HTTP {r.status_code} | {r.text[:200]}")
        if r.ok:
            d = r.json()
            o = d.get("order") or d.get("clobOrder") or d.get("ammOrder") or {}
            oid = o.get("id") or d.get("id","?")
            return {"order_id":str(oid),"status":o.get("status","?"),
                    "amount":o.get("amount",amount),
                    "price":o.get("price") or o.get("avgFillPrice"),
                    "quantity":o.get("quantity") or o.get("filledSize"),
                    "engine":d.get("engine","?")}
    except Exception as e:
        log.error(f"place_order: {e}")
    return None

# ─────────────────────────────────────────────────────────────
# CSV LOG
# ─────────────────────────────────────────────────────────────
FIELDS = ["timestamp","time","hour","bayse_target","btc_live","close",
          "direction","actual","correct","conf","payout","op","gap_pct",
          "traded","stake","order_id","session","balance_after"]

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w",newline="") as f:
            csv.writer(f).writerow(FIELDS)

def write_log(row):
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([row.get(k,"") for k in FIELDS])

# ─────────────────────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────────────────────
def get_signal(market, bayse_target, now):
    btc = btc_live()
    if not btc or not bayse_target: return None

    feat = compute_features(bayse_target)
    if feat is None: return None

    raw_proba = model.predict_proba(pd.DataFrame([feat], columns=features))[0][1]
    up        = raw_proba > 0.5
    conf      = raw_proba if up else (1.0 - raw_proba)
    direction = "UP" if up else "DOWN"

    yes_p  = market["yes_price"]
    op     = yes_p if up else (1.0 - yes_p)
    op     = max(op, 0.01)
    payout = round((1.0/op - 1) * (1 - FEE), 3)

    # Gap: KuCoin live price vs Bayse target
    gap_pct = (btc - bayse_target) / bayse_target * 100

    # Apply session config filters
    skip = []
    if conf < cfg["confidence"]:
        skip.append(f"conf {conf:.1%} < {cfg['confidence']:.0%}")
    if abs(gap_pct) < cfg["min_gap"]:
        skip.append(f"gap {abs(gap_pct):.3f}% < {cfg['min_gap']:.2f}%")
    if cfg["min_payout"] > 0 and payout < cfg["min_payout"]:
        skip.append(f"payout {payout:.2f}x < {cfg['min_payout']:.2f}x min")
    if now.hour in cfg["blocked_hours"]:
        skip.append(f"hour {now.hour:02d}:00 blocked")
    if circuit_off:
        skip.append(f"circuit break ({consec_losses}/{cfg['max_losses']} losses)")
    if AUTO_BUY and balance < DEFAULT.get("balance_min", 2.00):
        skip.append(f"balance ${balance:.2f} low")

    return {
        "trade"    : len(skip) == 0,
        "skip"     : " | ".join(skip) if skip else None,
        "direction": direction,
        "up"       : up,
        "conf"     : round(conf, 4),
        "payout"   : payout,
        "op"       : op,
        "gap_pct"  : round(gap_pct, 4),
        "abs_gap"  : abs(gap_pct),
        "btc"      : btc,
        "target"   : bayse_target,
        "yes_p"    : yes_p,
        "no_p"     : 1.0 - yes_p,
        "liq"      : market.get("liquidity",0),
        "orders"   : market.get("orders",0),
        "hour"     : now.hour,
    }

# ─────────────────────────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────────────────────────
def fmt_cfg(c):
    """Format a config dict as a short readable string."""
    ph = f"blocked {c['blocked_hours']}" if c["blocked_hours"] else "no hour block"
    mp = f"pay≥{c['min_payout']:.2f}x" if c["min_payout"] > 0 else "no payout filter"
    return (f"conf≥{c['confidence']:.0%} | gap≥{c['min_gap']:.2f}% | "
            f"{mp} | {ph} | stake=${c['stake']:.2f} | circuit@{c['max_losses']}")

def msg_open(market, btc, target, now, mins):
    diff     = btc - target
    diff_pct = diff / target * 100
    yes      = market["yes_price"]; no = 1.0 - yes

    if last_round["resolved"] and last_round["target"]:
        lr    = last_round
        close = lr["close"] or 0
        l_dir = "UP 📈" if close >= (lr["target"] or 0) else "DOWN 📉"
        if lr["correct"] is True:    res = "✅ WON"
        elif lr["correct"] is False: res = "❌ LOST"
        else:                        res = "— no trade"
        total = stats["wins"] + stats["losses"]
        wr    = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
        last_blk = (f"{sep()}\n📋 *Last Round*\n"
                    f"  Target : ${lr['target']:,.2f}\n"
                    f"  Closed : ${close:,.2f} → {l_dir}\n"
                    f"  Bot    : {lr['direction'] or '—'} → {res}\n"
                    f"  Signals: {stats['signalled']} | WR: {wr}\n")
    else:
        last_blk = f"{sep()}\n📋 *Last Round:* No data yet\n"

    status = ""
    if circuit_off:
        status = f"\n⚡ *Circuit break — {consec_losses} straight losses* | /play to reset"
    elif paused:
        status = "\n⏸ *Paused*"
    bal_line = f"\n💰 Balance: *${balance:.2f}*" if AUTO_BUY else ""

    return (f"🔔 *NEW ROUND — Session #{session_num}*\n{sep()}\n"
            f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ ~{mins} mins\n{sep()}\n"
            f"💰 BTC (KuCoin live): ${btc:,.2f}\n"
            f"🎯 Bayse Target     : ${target:,.2f}\n"
            f"💹 Gap              : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
            f"📊 Bayse Odds       : YES {yes:.0%} | NO {no:.0%}\n"
            + last_blk + f"{sep()}\n⏳ Signal in ~{SIGNAL_DELAY} mins"
            + bal_line + status)


def msg_signal_card(sig, traded, order_id=None, silent=False):
    win_amt   = round(current_stake * sig["payout"], 2)
    total_ret = round(current_stake + win_amt, 2)

    if traded:
        if silent: status_line = f"🔇 *SILENT TRADE* — ${current_stake:.2f} placed"
        else:      status_line = f"✅ *TRADE* — ${current_stake:.2f} staked"
        if AUTO_BUY:
            if order_id:
                trade_blk = (f"\n{sep()}\n🤖 *Auto-buy confirmed*\n"
                             f"  Stake   : ${current_stake:.2f}\n"
                             f"  Win  →  +${win_amt:.2f}  (get back ${total_ret:.2f})\n"
                             f"  Lose →  -${current_stake:.2f}\n"
                             f"  Order   : `{order_id}`\n"
                             f"  Balance : ${balance:.2f}")
            else:
                trade_blk = f"\n{sep()}\n⚠️ *Order FAILED* — check Railway logs"
        else:
            trade_blk = (f"\n{sep()}\n💡 Stake ${current_stake:.2f} on *{sig['direction']}*\n"
                         f"  Win → +${win_amt:.2f} | Lose → -${current_stake:.2f}")
    else:
        status_line = f"⛔ *NO TRADE* — {sig['skip']}"
        trade_blk   = ""

    conv = "⚡ HIGH" if sig["conf"] >= 0.85 else "💛 MED" if sig["conf"] >= 0.70 else "⚠️ LOW"
    return (f"🤖 *SIGNAL — {datetime.now(timezone.utc).strftime('%H:%M UTC')}*\n{sep()}\n"
            f"💰 BTC     : ${sig['btc']:,.2f}\n"
            f"🎯 Target  : ${sig['target']:,.2f}\n"
            f"💹 Gap     : ${sig['btc']-sig['target']:+.2f} ({sig['gap_pct']:+.3f}%)\n{sep()}\n"
            f"📊 *Bayse Odds*\n"
            f"  YES (UP)  : {sig['yes_p']:.0%}\n"
            f"  NO (DOWN) : {sig['no_p']:.0%}\n"
            f"  Payout    : {sig['payout']:.2f}x  → win returns ${total_ret:.2f}\n"
            f"  Liq ${sig['liq']:,.0f} | Orders: {sig['orders']}\n{sep()}\n"
            f"{status_line}\n"
            f"  Direction : *{sig['direction']}*\n"
            f"  Conf      : {sig['conf']:.1%}  [{conv}]\n"
            + trade_blk)


def msg_receipt(order, sig):
    win_amt   = round(current_stake * sig["payout"], 2)
    total_ret = round(current_stake + win_amt, 2)
    price = order.get("price"); qty = order.get("quantity")
    return (f"🧾 *RECEIPT*\n{sep()}\n"
            f"  Direction : *{sig['direction']}*\n"
            f"  Spent     : ${order.get('amount', current_stake):.2f} USD\n"
            f"  Fill      : {'${:.4f}/share'.format(price) if price else 'market'}\n"
            f"  Shares    : {'{:.4f}'.format(qty) if qty else '—'}\n"
            f"  Engine    : {order.get('engine','?')}\n"
            f"  Status    : {order.get('status','?')}\n"
            f"  Order ID  : `{order['order_id']}`\n{sep()}\n"
            f"  Win  → +${win_amt:.2f}  (total back ${total_ret:.2f})\n"
            f"  Lose → -${current_stake:.2f}\n"
            f"  Balance   : ${balance:.2f}")

# ─────────────────────────────────────────────────────────────
# SESSION WIZARD
# ─────────────────────────────────────────────────────────────
WIZARD_STEPS = [
    {
        "key"    : "confidence",
        "prompt" : ("🎯 *Step 1/6 — Confidence threshold*\n"
                    "Minimum model confidence to trade.\n\n"
                    f"Default: {DEFAULT['confidence']:.0%}\n"
                    "Examples: 70% | 75% | 80% | 85%\n\n"
                    "Reply with a number (e.g. `70`) or `skip` for default."),
        "parse"  : lambda v: float(v)/100 if float(v) > 1 else float(v),
        "valid"  : lambda v: 0.50 <= v <= 0.99,
        "err"    : "Must be between 50 and 99 (e.g. 70 for 70%)",
    },
    {
        "key"    : "min_gap",
        "prompt" : ("📏 *Step 2/6 — Minimum gap %*\n"
                    "BTC must have moved at least this % from the Bayse target.\n\n"
                    f"Default: {DEFAULT['min_gap']:.2f}%\n"
                    "Examples: 0.02 | 0.05 | 0.08 | 0.10\n\n"
                    "Reply with a number (e.g. `0.05`) or `skip` for default."),
        "parse"  : float,
        "valid"  : lambda v: 0.0 <= v <= 1.0,
        "err"    : "Must be between 0.00 and 1.00 (e.g. 0.05)",
    },
    {
        "key"    : "min_payout",
        "prompt" : ("💰 *Step 3/6 — Minimum payout (optional)*\n"
                    "Minimum profit per $1 staked. Trades with lower payout are skipped.\n"
                    "Set to 0 to disable this filter entirely.\n\n"
                    f"Default: {DEFAULT['min_payout']:.2f} (disabled)\n"
                    "Example: 0.30 means you need at least $1.30 back per $1 win\n\n"
                    "Reply with a number (e.g. `0.30`) or `skip`/`0` to disable."),
        "parse"  : float,
        "valid"  : lambda v: 0.0 <= v <= 2.0,
        "err"    : "Must be between 0.00 and 2.00 (0 = disabled)",
    },
    {
        "key"    : "blocked_hours",
        "prompt" : ("🚫 *Step 4/6 — Block bad hours (optional)*\n"
                    "UTC hours where the model performs poorly. These rounds are skipped.\n\n"
                    f"Default: none\n"
                    "Confirmed bad: 2, 4, 9, 13 (all ≤52% WR across 4+ days)\n"
                    "Example: `2 4 9 13` to block those four hours\n\n"
                    "Reply with space-separated hours, or `skip`/`none` to disable."),
        "parse"  : lambda v: [] if v.lower() in ("none","skip","0","") else [int(x) for x in v.split()],
        "valid"  : lambda v: all(0 <= h <= 23 for h in v),
        "err"    : "Must be space-separated hours 0–23 (e.g. 2 9 13)",
    },
    {
        "key"    : "stake",
        "prompt" : ("💵 *Step 5/6 — Stake per trade*\n"
                    "How much to bet on each qualifying signal.\n\n"
                    f"Default: ${DEFAULT['stake']:.2f}\n"
                    "Examples: 1 | 2 | 3 | 5\n\n"
                    "Reply with a number or `skip` for default."),
        "parse"  : float,
        "valid"  : lambda v: 1.0 <= v <= 1000.0,
        "err"    : "Must be at least $1.00",
    },
    {
        "key"    : "max_losses",
        "prompt" : ("⚡ *Step 6/6 — Circuit break*\n"
                    "Pause auto-buy after this many consecutive losses.\n\n"
                    f"Default: {DEFAULT['max_losses']}\n"
                    "Examples: 2 | 3 | 5 | 0 (disable)\n\n"
                    "Reply with a number or `skip` for default."),
        "parse"  : int,
        "valid"  : lambda v: 0 <= v <= 20,
        "err"    : "Must be between 0 (disabled) and 20",
    },
]

def wizard_start(chat):
    global wizard_active, wizard_step, wizard_chat, wizard_data
    wizard_active = True
    wizard_step   = 0
    wizard_chat   = chat
    wizard_data   = {}
    tg(f"🆕 *New Session Setup*\n{sep()}\n"
       f"Answer 6 quick questions to configure your session.\n"
       f"Type `skip` at any step to use the default value.\n"
       f"Type `cancel` to abort.\n{sep()}", chat)
    time.sleep(0.5)
    tg(WIZARD_STEPS[0]["prompt"], chat)

def wizard_handle(text, chat):
    """Process wizard input. Returns True if wizard consumed the message."""
    global wizard_active, wizard_step, wizard_data, wizard_chat
    global session_num, session_start, session_on, cfg, balance
    global current_stake, total_staked, total_won, paused, circuit_off
    global consec_losses, silent_mode, stats, trade_log, hour_stats
    global last_round, round_state

    if not wizard_active: return False
    if chat != wizard_chat: return False  # only respond to wizard initiator

    if text.lower() == "cancel":
        wizard_active = False
        tg("❌ Session setup cancelled. Send /startsession to try again.", chat)
        return True

    step = WIZARD_STEPS[wizard_step]

    if text.lower() in ("skip","default",""):
        # Use default value
        wizard_data[step["key"]] = DEFAULT[step["key"]]
        tg(f"✅ Using default: `{DEFAULT[step['key']]}`", chat)
    else:
        try:
            val = step["parse"](text)
            if not step["valid"](val):
                tg(f"❌ {step['err']}\n\nTry again or type `skip` for default.", chat)
                return True
            wizard_data[step["key"]] = val
            tg(f"✅ Set to: `{val}`", chat)
        except Exception as e:
            tg(f"❌ Couldn't parse that. {step['err']}\nTry again or type `skip`.", chat)
            return True

    wizard_step += 1

    if wizard_step < len(WIZARD_STEPS):
        time.sleep(0.3)
        tg(WIZARD_STEPS[wizard_step]["prompt"], chat)
        return True

    # All steps done — apply config and start session
    wizard_active = False
    cfg           = wizard_data.copy()
    current_stake = cfg["stake"]

    # Reset session state
    session_num   += 1
    session_start  = datetime.now(timezone.utc)
    session_on     = True
    paused         = False
    circuit_off    = False
    consec_losses  = 0
    silent_mode    = False
    balance        = 10.00
    total_staked   = 0.0
    total_won      = 0.0
    stats          = {k: 0 for k in stats}
    trade_log      = []
    hour_stats     = defaultdict(lambda: {"total": 0, "correct": 0})
    last_round     = {"target":None,"close":None,"direction":None,
                      "correct":None,"resolved":False}
    round_state    = {"event_id":None,"prev_event_id":None,"start_time":None,
                      "target":None,"fired":False,"signal":None,"order_id":None}

    # Summary
    ph = f"blocked {cfg['blocked_hours']}" if cfg["blocked_hours"] else "no hour blocking"
    mp = f"payout≥{cfg['min_payout']:.2f}x" if cfg["min_payout"] > 0 else "no payout filter"
    mode = f"AUTO-BUY ${cfg['stake']:.2f}/trade" if AUTO_BUY else "NOTIFY ONLY"
    tg(f"🚀 *Session #{session_num} started!*\n{sep()}\n"
       f"📅 {session_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
       f"Mode: *{mode}*\n{sep()}\n"
       f"  Confidence : ≥{cfg['confidence']:.0%}\n"
       f"  Min gap    : ≥{cfg['min_gap']:.2f}%\n"
       f"  Min payout : {mp}\n"
       f"  Hours      : {ph}\n"
       f"  Stake      : ${cfg['stake']:.2f}\n"
       f"  Circuit    : after {cfg['max_losses']} straight losses\n{sep()}\n"
       f"Balance reset to $10.00. Bot is running. Good luck! 🎯", chat)
    return True

# ─────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────
def handle_commands():
    global paused, circuit_off, consec_losses, silent_mode, current_stake
    global session_on, balance, total_staked, total_won
    global stats, trade_log, hour_stats, last_round, round_state, cfg

    for u in tg_poll():
        msg  = u.get("message",{})
        raw  = msg.get("text","").strip()
        text = raw.lower()
        chat = str(msg.get("chat",{}).get("id",""))

        # Wizard takes priority
        if wizard_active:
            if wizard_handle(raw, chat): continue

        if text.startswith("/startsession"):
            wizard_start(chat)

        elif text.startswith("/stopsession"):
            session_on = False; paused = True
            total = stats["wins"] + stats["losses"]
            wr    = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            pnl   = total_won - total_staked
            tg(f"🛑 *Session #{session_num} ended*\n{sep()}\n"
               f"  Signals : {stats['signalled']}\n"
               f"  Trades  : {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} wins | ❌ {stats['losses']} losses\n"
               f"  Balance : ${balance:.2f}\n"
               f"  P&L     : ${pnl:+.2f}\n{sep()}\n"
               f"Send /startsession to begin a new session.", chat)

        elif text.startswith("/default"):
            cfg = DEFAULT.copy()
            current_stake = cfg["stake"]
            tg(f"↩️ *Config reset to defaults*\n{sep()}\n"
               f"  {fmt_cfg(cfg)}\n{sep()}\n"
               f"These will apply from the next signal.", chat)

        elif text.startswith("/pause"):
            paused = True
            tg("⏸ *Paused.* No signals or auto-buy.\n/play to resume.", chat)

        elif text.startswith("/play"):
            paused = False; circuit_off = False; consec_losses = 0
            tg("▶️ *Resumed.* Signals active. Circuit break cleared.", chat)

        elif text.startswith("/playsilent"):
            silent_mode = True
            tg(f"🔇→📢 *Silent mode ON*\nAlso auto-buying rounds that pass "
               f"conf+gap but were filtered out.\n/pausesilent to stop.", chat)

        elif text.startswith("/pausesilent"):
            silent_mode = False
            tg("🔇 *Silent mode OFF.*", chat)

        elif text.startswith("/increase"):
            parts = raw.split()
            try:
                current_stake = round(current_stake + float(parts[1]), 2)
                tg(f"📈 Stake → *${current_stake:.2f}*", chat)
            except:
                tg(f"Usage: /increase 2\nCurrent stake: ${current_stake:.2f}", chat)

        elif text.startswith("/decrease"):
            parts = raw.split()
            try:
                new = round(current_stake - float(parts[1]), 2)
                if new < 1.00: tg(f"❌ Minimum stake is $1.00", chat)
                else:
                    current_stake = new
                    tg(f"📉 Stake → *${current_stake:.2f}*", chat)
            except:
                tg(f"Usage: /decrease 1\nCurrent stake: ${current_stake:.2f}", chat)

        elif text.startswith("/stake"):
            tg(f"💵 *Stake*\n  Current : ${current_stake:.2f}\n"
               f"  Default : ${DEFAULT['stake']:.2f}\n"
               f"  /increase N  |  /decrease N", chat)

        elif text.startswith("/balance"):
            global balance
            api = fetch_balance_api()
            if api is not None: balance = api; src = "Bayse API ✅"
            else: src = "internal tracker ⚠️"
            total = stats["wins"] + stats["losses"]
            wr    = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            pnl   = total_won - total_staked
            tg(f"💰 *Balance — Session #{session_num}*\n{sep()}\n"
               f"  Balance : *${balance:.2f}*  _({src})_\n"
               f"  Staked  : ${total_staked:.2f} | Won: ${total_won:.2f}\n"
               f"  P&L     : ${pnl:+.2f}\n{sep()}\n"
               f"  Trades  : {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} | ❌ {stats['losses']}\n"
               f"  Stake   : ${current_stake:.2f} | Silent: {'ON' if silent_mode else 'OFF'}\n"
               f"  Orders  : {stats['orders_ok']} ok | {stats['orders_fail']} failed", chat)

        elif text.startswith("/stats"):
            total = stats["wins"] + stats["losses"]
            wr    = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            lines = []
            for t in trade_log[-10:]:
                e  = "✅" if t.get("correct") is True else ("❌" if t.get("correct") is False else "—")
                f_ = "💵" if t.get("traded") else "⛔"
                lines.append(f"{e}{f_} {t['time']} | {t['direction']} | "
                             f"conf:{t['conf']:.0%} | pay:{t['payout']:.2f}x")
            circuit_str = "⚡ ACTIVE" if circuit_off else f"✅ clear ({consec_losses}/{cfg['max_losses']})"
            tg(f"📊 *Stats — Session #{session_num}*\n{sep()}\n"
               f"  Signals : {stats['signalled']} | Trades: {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} | ❌ {stats['losses']}\n"
               f"  Balance : ${balance:.2f}\n"
               f"  Stake   : ${current_stake:.2f}\n"
               f"  Circuit : {circuit_str}\n"
               f"  Status  : {'⏸ Paused' if paused else '▶️ Running'}\n{sep()}\n"
               + ("\n".join(lines) if lines else "No signals yet"), chat)

        elif text.startswith("/log"):
            if not trade_log:
                tg("No signals yet this session.", chat); continue
            lines = [f"📋 *Last 10 signals — Session #{session_num}*\n{sep()}"]
            for t in trade_log[-10:]:
                e  = "✅" if t.get("correct") is True else ("❌" if t.get("correct") is False else "—")
                f_ = "💵" if t.get("traded") else "⛔"
                lines.append(f"{e}{f_} {t['time']} | {t['direction']} "
                             f"conf:{t['conf']:.0%} | pay:{t['payout']:.2f}x")
            tg("\n".join(lines), chat)

        elif text.startswith("/hours"):
            if not hour_stats:
                tg("No hour data yet.", chat); continue
            lines = [f"🕐 *Win Rate by Hour — Session #{session_num}*\n{sep()}"]
            for h in sorted(hour_stats.keys()):
                d  = hour_stats[h]
                wr = d["correct"]/d["total"]*100 if d["total"] else 0
                flag = "🔥" if wr==100 else "✅" if wr>=75 else "⚠️" if wr>=52.6 else "❌"
                blk  = " 🚫" if h in cfg["blocked_hours"] else ""
                bar  = "█" * int(wr/10)
                lines.append(f"  {h:02d}:00{blk} {flag} {bar:<10} {wr:.0f}% ({d['correct']}/{d['total']})")
            lines.append(f"{sep()}\nBreak-even ≈ 52.6%")
            tg("\n".join(lines), chat)

        elif text.startswith("/config"):
            ph_d = f"blocked {DEFAULT['blocked_hours']}" if DEFAULT["blocked_hours"] else "no hour block"
            mp_d = f"pay≥{DEFAULT['min_payout']:.2f}x" if DEFAULT["min_payout"] > 0 else "no payout filter"
            ph_c = f"blocked {cfg['blocked_hours']}" if cfg["blocked_hours"] else "no hour block"
            mp_c = f"pay≥{cfg['min_payout']:.2f}x" if cfg["min_payout"] > 0 else "no payout filter"
            tg(f"⚙️ *Config — Session #{session_num}*\n{sep()}\n"
               f"*Session settings:*\n"
               f"  Confidence : ≥{cfg['confidence']:.0%}\n"
               f"  Min gap    : ≥{cfg['min_gap']:.2f}%\n"
               f"  Min payout : {mp_c}\n"
               f"  Hours      : {ph_c}\n"
               f"  Stake      : ${cfg['stake']:.2f}\n"
               f"  Circuit    : after {cfg['max_losses']} losses\n{sep()}\n"
               f"*Global defaults:*\n"
               f"  Confidence : ≥{DEFAULT['confidence']:.0%}\n"
               f"  Min gap    : ≥{DEFAULT['min_gap']:.2f}%\n"
               f"  Min payout : {mp_d}\n"
               f"  Hours      : {ph_d}\n"
               f"  Stake      : ${DEFAULT['stake']:.2f}\n"
               f"  Circuit    : after {DEFAULT['max_losses']} losses\n{sep()}\n"
               f"  Auto-buy   : {'✅ ON' if AUTO_BUY else '⚠️ OFF'}\n"
               f"  Silent     : {'ON' if silent_mode else 'OFF'}\n"
               f"  Status     : {'⏸ Paused' if paused else '▶️ Running'}\n\n"
               f"/default — reset session to defaults\n"
               f"/startsession — configure new session", chat)

        elif text.startswith("/price"):
            btc = btc_live(); tgt = round_state.get("target")
            mkt = None
            if btc and tgt:
                diff = btc - tgt
                tg(f"💰 *BTC Snapshot*\n{sep()}\n"
                   f"  BTC (KuCoin) : ${btc:,.2f}\n"
                   f"  Bayse Target : ${tgt:,.2f}\n"
                   f"  Gap          : ${diff:+.2f} ({diff/tgt*100:+.3f}%)\n"
                   f"  Status       : {'📈 ABOVE' if diff > 0 else '📉 BELOW'} target", chat)
            else:
                tg("No active round data yet. Wait for next round open.", chat)

        elif text.startswith("/export"):
            if not os.path.exists(LOG_FILE):
                tg("No CSV log yet.", chat); continue
            with open(LOG_FILE,"rb") as f: data = f.read()
            rows  = data.count(b"\n") - 1
            fname = f"session_{session_num}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
            tg_doc(fname, data, f"Session #{session_num} — {rows} signals", chat)

        elif text.startswith("/start"):
            mode = f"AUTO-BUY ${current_stake:.2f}/trade" if AUTO_BUY else "NOTIFY ONLY"
            tg(f"🤖 *Bayse BTC Bot v9*\n"
               f"Mode: *{mode}*\n{sep()}\n"
               f"Every round sends a signal — trade or not.\n"
               f"Target price is from Bayse API. Gap vs live BTC.\n{sep()}\n"
               f"*Commands:*\n"
               f"  /startsession  — configure + start new session\n"
               f"  /stopsession   — end session\n"
               f"  /default       — reset session config to defaults\n"
               f"  /config        — show current vs default config\n"
               f"  /balance       — live balance\n"
               f"  /price         — BTC vs Bayse target\n"
               f"  /stats         — session stats\n"
               f"  /log           — last 10 signals\n"
               f"  /hours         — WR by hour\n"
               f"  /export        — download CSV\n"
               f"  /pause  /play\n"
               f"  /playsilent  /pausesilent\n"
               f"  /increase N  /decrease N  /stake\n{sep()}\n"
               f"Send /startsession to begin!", chat)

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main():
    global balance, total_staked, total_won, round_state, last_round
    global stats, trade_log, hour_stats, circuit_off, consec_losses

    init_log()
    seed_candles()

    tg(f"🟢 *Bot v9 online*\n"
       f"  Auto-buy: {'✅ ON' if AUTO_BUY else '⚠️ OFF'}\n"
       f"  Default config: {fmt_cfg(DEFAULT)}\n\n"
       f"Send /startsession to configure and start.\n"
       f"Send /start for full command list.")
    log.info(f"v9 started | auto_buy={AUTO_BUY}")

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

            event_id     = market["event_id"]
            btc          = btc_live()
            bayse_target = market.get("bayse_target")  # Bayse's own price target
            end_dt       = parse_end_dt(market["rules"], now)
            mins_left    = round((end_dt - now).total_seconds()/60, 1) if end_dt else None

            # ── NEW ROUND ─────────────────────────────────────
            if event_id != round_state["event_id"]:
                log.info(f"New round: {event_id} | bayse_target=${bayse_target}")

                prev_sig    = round_state.get("signal")
                prev_target = round_state.get("target")
                prev_eid    = round_state.get("prev_event_id")

                # ── RESOLVE PREVIOUS ROUND ────────────────────
                if prev_target:
                    final = None
                    if prev_eid:
                        final = fetch_final_price(prev_eid)
                        if final: log.info(f"Bayse final price: ${final:,.2f}")
                    if not final:
                        final = btc
                        log.warning(f"Bayse finalPrice unavailable — using live btc ${final:,.2f}")

                    if final and prev_sig:
                        actual_up  = final >= prev_target
                        actual_dir = "UP" if actual_up else "DOWN"
                        correct    = (prev_sig["up"] == actual_up)
                        traded     = prev_sig.get("was_trade", False)
                        used_stake = prev_sig.get("stake_used", current_stake)
                        hour_h     = prev_sig.get("hour", now.hour)

                        hour_stats[hour_h]["total"] += 1
                        if correct: hour_stats[hour_h]["correct"] += 1

                        if traded:
                            stats["total"] += 1
                            if correct:
                                stats["wins"] += 1; consec_losses = 0
                                earned        = round(used_stake * prev_sig["payout"], 2)
                                total_won    += earned
                                total_staked += used_stake
                                balance       = round(balance + earned - used_stake, 2)
                                if circuit_off:
                                    circuit_off = False
                                    tg("✅ *Win — circuit break cleared. Signals resuming.*")
                            else:
                                stats["losses"] += 1; consec_losses += 1
                                total_staked += used_stake
                                balance       = round(balance - used_stake, 2)
                                if consec_losses >= cfg["max_losses"] and not circuit_off:
                                    circuit_off = True; stats["breaks"] += 1
                                    tg(f"⚡ *Circuit break — {cfg['max_losses']} straight losses*\n"
                                       f"Balance: ${balance:.2f}\n/play to resume manually")

                            # Sync from Bayse API
                            api_bal = fetch_balance_api()
                            if api_bal is not None:
                                balance = api_bal
                                log.info(f"Balance synced: ${balance:.2f}")

                        last_round.update({
                            "target"   : prev_target,
                            "close"    : final,
                            "direction": prev_sig["direction"],
                            "correct"  : correct if traded else None,
                            "resolved" : True,
                        })
                        write_log({
                            "timestamp"    : now.isoformat(),
                            "time"         : now.strftime("%H:%M"),
                            "hour"         : hour_h,
                            "bayse_target" : prev_target,
                            "btc_live"     : prev_sig.get("btc",""),
                            "close"        : final,
                            "direction"    : prev_sig["direction"],
                            "actual"       : actual_dir,
                            "correct"      : correct if traded else "",
                            "conf"         : prev_sig["conf"],
                            "payout"       : prev_sig["payout"],
                            "op"           : prev_sig["op"],
                            "gap_pct"      : prev_sig["gap_pct"],
                            "traded"       : traded,
                            "stake"        : used_stake if traded else "",
                            "order_id"     : round_state.get("order_id",""),
                            "session"      : session_num,
                            "balance_after": balance if traded else "",
                        })
                        trade_log.append({
                            "time"      : prev_sig.get("time", now.strftime("%H:%M")),
                            "direction" : prev_sig["direction"],
                            "conf"      : prev_sig["conf"],
                            "payout"    : prev_sig["payout"],
                            "traded"    : traded,
                            "correct"   : correct if traded else None,
                        })
                        log.info(f"Resolved: {'✅' if correct else '❌'} | "
                                 f"traded={traded} | balance=${balance:.2f}")

                # ── SET NEW ROUND ─────────────────────────────
                # Target is always from Bayse API (bayse_target field)
                new_target = bayse_target
                if not new_target:
                    new_target = btc
                    log.warning(f"Bayse target not available — using live btc ${new_target:,.2f}")

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
                    log.info(f"Open | bayse_target=${new_target:,.2f} | btc=${btc:,.2f}")

            # ── FIRE SIGNAL ───────────────────────────────────
            start_t = round_state.get("start_time")
            if (not round_state["fired"] and start_t
                    and round_state["target"] and not paused):
                mins_in = (now - start_t).total_seconds() / 60
                if mins_in >= SIGNAL_DELAY:
                    sig = get_signal(market, round_state["target"], now)
                    if sig:
                        sig["time"] = now.strftime("%H:%M")
                        round_state["fired"] = True
                        stats["signalled"] += 1

                        silent_trade = (silent_mode and not sig["trade"]
                                        and sig["conf"] >= cfg["confidence"]
                                        and sig["abs_gap"] >= cfg["min_gap"])
                        should_trade = sig["trade"] or silent_trade

                        sig["was_trade"]  = should_trade
                        sig["stake_used"] = current_stake
                        round_state["signal"] = sig

                        order = None; order_id = None
                        if should_trade and AUTO_BUY and balance >= DEFAULT.get("balance_min", 2.00):
                            order = place_order(market, sig["up"], current_stake)
                            if order:
                                order_id = order["order_id"]
                                round_state["order_id"] = order_id
                                stats["orders_ok"] += 1
                                api_bal = fetch_balance_api()
                                if api_bal is not None:
                                    balance = api_bal
                            else:
                                stats["orders_fail"] += 1

                        # Always send signal
                        tg(msg_signal_card(sig, traded=should_trade,
                                           order_id=order_id, silent=silent_trade))
                        if order: tg(msg_receipt(order, sig))

                        log.info(f"Signal | {sig['direction']} | conf {sig['conf']:.1%} | "
                                 f"gap {sig['abs_gap']:.3f}% | pay {sig['payout']:.2f}x | "
                                 f"trade={should_trade} | order={'ok' if order else 'none'}")
                    else:
                        log.warning("Features not ready")

            time.sleep(30)

        except Exception as e:
            log.error(f"Loop: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
