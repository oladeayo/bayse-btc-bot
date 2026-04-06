"""
Bayse BTC Bot — v8
===================
Working config (backed by 396 trades):
  conf ≥ 80%   — model confidence threshold
  gap  ≥ 0.05% — price must have moved at least 0.05% from target
  No payout filter — removed, was blocking almost all trades
  No hour blocking — removed for simplicity

Every round sends a signal to Telegram whether trading or not.
Auto-buys when conditions met. Balance tracked from API or internally.

Commands:
  /price /balance /stats /log /hours /config /export
  /pause /play /playsilent /pausesilent
  /increase N /decrease N /stake
  /startsession /stopsession
"""

import logging, requests, joblib, json, re, os, csv, sys, time
import hashlib, hmac as _hmac, base64
import pandas as pd, numpy as np, ta
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CONFIDENCE   = 0.70    # conf≥80%
MIN_GAP_PCT  = 0.05    # gap≥0.05% from target
MAX_LOSSES   = 3       # circuit break after N consecutive losses
SIGNAL_DELAY = 5       # minutes into round before firing signal
FEE          = 0.05    # Bayse fee
STAKE        = 1.00    # default $1 stake
BALANCE_MIN  = 2.00    # stop auto-buy if balance below this
LOG_FILE     = "trades.csv"

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
print(f"✅ v8 | conf≥{CONFIDENCE:.0%} | gap≥{MIN_GAP_PCT:.2f}% | "
      f"stake=${STAKE} | auto_buy={AUTO_BUY}", flush=True)

# ─────────────────────────────────────────────────────────────
# MUTABLE STATE — all global, modified via global declarations
# ─────────────────────────────────────────────────────────────
CANDLES       = deque(maxlen=100)
paused        = False
silent_mode   = False
circuit_off   = False
consec_losses = 0
session_num   = 1
session_start = datetime.now(timezone.utc)
session_on    = True
balance       = 10.00
current_stake = STAKE
total_staked  = 0.0
total_won     = 0.0
last_update   = None

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
trade_log  = []   # all rounds that produced a signal
hour_stats = defaultdict(lambda: {"total": 0, "correct": 0})

# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────
def tg(text, chat=None):
    try:
        requests.post(f"{TG_API}/sendMessage",
            json={"chat_id": chat or TELEGRAM_CHAT, "text": text,
                  "parse_mode": "Markdown"},
            timeout=10)
    except Exception as e:
        log.error(f"tg send: {e}")

def tg_doc(filename, data, caption, chat=None):
    try:
        requests.post(f"{TG_API}/sendDocument",
            data={"chat_id": chat or TELEGRAM_CHAT, "caption": caption},
            files={"document": (filename, data, "text/csv")},
            timeout=15)
    except Exception as e:
        log.error(f"tg doc: {e}")

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
                d  = r2.json()
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

                start_price = (
                    _f(d,"startPrice","start_price","targetPrice","target_price") or
                    _f(m,"startPrice","start_price")
                )
                return {
                    "event_id"  : d["id"],
                    "yes_price" : float(m.get("outcome1Price", 0.5)),
                    "no_price"  : float(m.get("outcome2Price", 0.5)),
                    "liquidity" : float(d.get("liquidity", 0)),
                    "orders"    : int(m.get("totalOrders", 0)),
                    "rules"     : m.get("rules", ""),
                    "start_price": start_price,
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
# KUCOIN CANDLES (ML features only — not for price decisions)
# ─────────────────────────────────────────────────────────────
def seed_candles():
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                         params={"symbol":"BTC-USDT","type":"1min"}, timeout=10)
        for c in reversed(r.json().get("data",[])):
            CANDLES.append({"open_time":pd.Timestamp(int(c[0]),unit="s",tz="UTC"),
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
        latest = {"open_time":pd.Timestamp(int(c[0]),unit="s",tz="UTC"),
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
                    "amount":o.get("amount",amount),"price":o.get("price") or o.get("avgFillPrice"),
                    "quantity":o.get("quantity") or o.get("filledSize"),"engine":d.get("engine","?")}
    except Exception as e:
        log.error(f"place_order: {e}")
    return None

# ─────────────────────────────────────────────────────────────
# CSV LOG
# ─────────────────────────────────────────────────────────────
FIELDS = ["timestamp","time","hour","target","btc","close","direction",
          "actual","correct","conf","payout","op","gap_pct","traded",
          "stake","order_id","session","balance_after"]

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w",newline="") as f:
            csv.writer(f).writerow(FIELDS)

def write_log(row):
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([row.get(k,"") for k in FIELDS])

# ─────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
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

    yes_p  = market["yes_price"]
    op     = yes_p if up else (1.0 - yes_p)
    op     = max(op, 0.01)
    payout = round((1.0/op - 1) * (1 - FEE), 3)
    gap_pct = (btc - target) / target * 100

    # Determine if this should be a trade
    skip_reasons = []
    if conf < CONFIDENCE:
        skip_reasons.append(f"conf {conf:.1%} < {CONFIDENCE:.0%}")
    if abs(gap_pct) < MIN_GAP_PCT:
        skip_reasons.append(f"gap {abs(gap_pct):.3f}% < {MIN_GAP_PCT:.2f}%")
    if circuit_off:
        skip_reasons.append(f"circuit break ({consec_losses}/{MAX_LOSSES} losses)")
    if AUTO_BUY and balance < BALANCE_MIN:
        skip_reasons.append(f"balance ${balance:.2f} < ${BALANCE_MIN:.2f}")

    should_trade = len(skip_reasons) == 0

    return {
        "trade"    : should_trade,
        "skip"     : " | ".join(skip_reasons) if skip_reasons else None,
        "direction": direction,
        "up"       : up,
        "conf"     : round(conf, 4),
        "payout"   : payout,
        "op"       : op,
        "gap_pct"  : round(gap_pct, 4),
        "abs_gap"  : abs(gap_pct),
        "btc"      : btc,
        "target"   : target,
        "yes_p"    : yes_p,
        "no_p"     : 1.0 - yes_p,
        "liq"      : market.get("liquidity",0),
        "orders"   : market.get("orders",0),
        "hour"     : now.hour,
    }

# ─────────────────────────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────────────────────────
def sep(): return "─"*30

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
        last_blk = (
            f"{sep()}\n📋 *Last Round*\n"
            f"  Target : ${lr['target']:,.2f}\n"
            f"  Closed : ${close:,.2f} → {l_dir}\n"
            f"  Bot    : {lr['direction'] or '—'} → {res}\n"
            f"  Signals: {stats['signalled']} | WR: {wr}\n"
        )
    else:
        last_blk = f"{sep()}\n📋 *Last Round:* No data yet\n"

    status = ""
    if circuit_off:
        status = f"\n⚡ *Circuit break — {consec_losses} straight losses* | /play to resume"
    elif paused:
        status = "\n⏸ *Paused*"
    bal_line = f"\n💰 Balance: *${balance:.2f}*" if AUTO_BUY else ""

    return (
        f"🔔 *NEW ROUND — Session #{session_num}*\n{sep()}\n"
        f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ ~{mins} mins\n{sep()}\n"
        f"💰 BTC Now : ${btc:,.2f}\n"
        f"🎯 Target  : ${target:,.2f}\n"
        f"💹 Gap     : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
        f"📊 Bayse   : YES {yes:.0%} | NO {no:.0%}\n"
        + last_blk
        + f"{sep()}\n⏳ Signal in ~{SIGNAL_DELAY} mins"
        + bal_line + status
    )


def msg_signal_card(sig, traded, order_id=None, silent=False):
    """Always sent — shows signal details whether trading or not."""
    win_amt   = round(current_stake * sig["payout"], 2)
    total_ret = round(current_stake + win_amt, 2)
    conviction = "⚡ HIGH" if sig["conf"] >= 0.85 else "⚠️ MED"

    if traded:
        if silent:
            status_line = f"🔇 *SILENT TRADE* — ${current_stake:.2f} placed"
        else:
            status_line = f"✅ *TRADE* — ${current_stake:.2f} placed"
        if AUTO_BUY:
            if order_id:
                trade_blk = (f"\n{sep()}\n🤖 *Auto-buy confirmed*\n"
                             f"  Stake  : ${current_stake:.2f}\n"
                             f"  Win  → +${win_amt:.2f}  (get back ${total_ret:.2f})\n"
                             f"  Lose → -${current_stake:.2f}\n"
                             f"  Order  : `{order_id}`\n"
                             f"  Balance: ${balance:.2f}")
            else:
                trade_blk = f"\n{sep()}\n⚠️ *Order FAILED* — check Railway logs"
        else:
            trade_blk = (f"\n{sep()}\n💡 *Place manually:* ${current_stake:.2f} on *{sig['direction']}*\n"
                         f"  Win → +${win_amt:.2f} | Lose → -${current_stake:.2f}")
    else:
        status_line = f"⛔ *NO TRADE* — {sig['skip']}"
        trade_blk   = ""

    return (
        f"🤖 *SIGNAL — {datetime.now(timezone.utc).strftime('%H:%M UTC')}*\n{sep()}\n"
        f"💰 BTC    : ${sig['btc']:,.2f}\n"
        f"🎯 Target : ${sig['target']:,.2f}\n"
        f"💹 Gap    : ${sig['btc']-sig['target']:+.2f} ({sig['gap_pct']:+.3f}%)\n"
        f"⏱ Left   : ~{SIGNAL_DELAY} mins after\n{sep()}\n"
        f"📊 *Bayse Odds*\n"
        f"  YES (UP)  : {sig['yes_p']:.0%}\n"
        f"  NO (DOWN) : {sig['no_p']:.0%}\n"
        f"  Payout    : {sig['payout']:.2f}x → win returns ${total_ret:.2f}\n"
        f"  Liq: ${sig['liq']:,.0f} | Orders: {sig['orders']}\n{sep()}\n"
        f"{status_line}\n"
        f"  Direction : *{sig['direction']}*\n"
        f"  Conf      : {sig['conf']:.1%}  [{conviction}]\n"
        + trade_blk
    )


def msg_receipt(order, sig):
    win_amt   = round(current_stake * sig["payout"], 2)
    total_ret = round(current_stake + win_amt, 2)
    price = order.get("price"); qty = order.get("quantity")
    return (
        f"🧾 *RECEIPT — {datetime.now(timezone.utc).strftime('%H:%M UTC')}*\n{sep()}\n"
        f"  Direction : *{sig['direction']}*\n"
        f"  Spent     : ${order.get('amount', current_stake):.2f} USD\n"
        f"  Fill      : {'${:.4f}/share'.format(price) if price else 'market fill'}\n"
        f"  Shares    : {'{:.4f}'.format(qty) if qty else '—'}\n"
        f"  Engine    : {order.get('engine','?')}\n"
        f"  Status    : {order.get('status','?')}\n"
        f"  Order ID  : `{order['order_id']}`\n{sep()}\n"
        f"  Win  → +${win_amt:.2f}  (get back ${total_ret:.2f})\n"
        f"  Lose → -${current_stake:.2f}\n"
        f"  Balance   : ${balance:.2f}"
    )

# ─────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────
def handle_commands():
    global paused, circuit_off, consec_losses, silent_mode, current_stake
    global session_num, session_start, session_on, balance, total_staked, total_won
    global stats, trade_log, hour_stats, last_round, round_state

    for u in tg_poll():
        msg  = u.get("message",{})
        raw  = msg.get("text","").strip()
        text = raw.lower()
        chat = str(msg.get("chat",{}).get("id",""))

        if text.startswith("/pause"):
            paused = True
            tg("⏸ *Paused.* No signals or auto-buy.\n/play to resume.", chat)

        elif text.startswith("/play"):
            paused = False; circuit_off = False; consec_losses = 0
            tg("▶️ *Resumed.* Signals active. Circuit break cleared.", chat)

        elif text.startswith("/playsilent"):
            silent_mode = True
            tg(f"🔇→📢 *Silent mode ON*\nBot will also auto-buy rounds that "
               f"pass conf+gap but were skipped for other reasons.\n"
               f"/pausesilent to stop.", chat)

        elif text.startswith("/pausesilent"):
            silent_mode = False
            tg("🔇 *Silent mode OFF.*", chat)

        elif text.startswith("/increase"):
            parts = raw.split()
            try:
                current_stake = round(current_stake + float(parts[1]), 2)
                tg(f"📈 Stake → *${current_stake:.2f}*\n"
                   f"(default ${STAKE:.2f} | /decrease N to reduce)", chat)
            except:
                tg(f"Usage: /increase 2\nCurrent stake: ${current_stake:.2f}", chat)

        elif text.startswith("/decrease"):
            parts = raw.split()
            try:
                new = round(current_stake - float(parts[1]), 2)
                if new < 1.00:
                    tg(f"❌ Minimum stake is $1.00 (would be ${new:.2f})", chat)
                else:
                    current_stake = new
                    tg(f"📉 Stake → *${current_stake:.2f}*", chat)
            except:
                tg(f"Usage: /decrease 1\nCurrent stake: ${current_stake:.2f}", chat)

        elif text.startswith("/stake"):
            tg(f"💵 Current stake: *${current_stake:.2f}*\n"
               f"Default: ${STAKE:.2f}\n"
               f"  /increase N — add N dollars\n"
               f"  /decrease N — remove N dollars", chat)

        elif text.startswith("/balance"):
            global balance
            api = fetch_balance_api()
            if api is not None:
                balance = api; src = "Bayse API ✅"
            else:
                src = "internal tracker ⚠️"
            total = stats["wins"] + stats["losses"]
            wr    = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            tg(f"💰 *Balance — Session #{session_num}*\n{sep()}\n"
               f"  Balance : *${balance:.2f}*  _({src})_\n"
               f"  Staked  : ${total_staked:.2f} | Won: ${total_won:.2f}\n"
               f"  P&L     : ${total_won - total_staked:+.2f}\n{sep()}\n"
               f"  Trades  : {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} wins | ❌ {stats['losses']} losses\n"
               f"  Stake   : ${current_stake:.2f} | Silent: {'ON' if silent_mode else 'OFF'}\n"
               f"  Orders  : {stats['orders_ok']} ok | {stats['orders_fail']} failed\n"
               f"  Stop at : ${BALANCE_MIN:.2f}", chat)

        elif text.startswith("/stats"):
            total = stats["wins"] + stats["losses"]
            wr    = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            recent = trade_log[-10:]
            lines  = []
            for t in recent:
                e = "✅" if t.get("correct") else "❌"
                lines.append(f"{e} {t['time']} | {t['direction']} | "
                             f"conf:{t['conf']:.0%} | pay:{t['payout']:.2f}x | "
                             f"{'TRADE' if t['traded'] else 'skip'}")
            tg(f"📊 *Stats — Session #{session_num}*\n{sep()}\n"
               f"  Signals : {stats['signalled']} | Trades: {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} | ❌ {stats['losses']}\n"
               f"  Balance : ${balance:.2f}\n"
               f"  Stake   : ${current_stake:.2f}\n"
               f"  Circuit : {'⚡ ACTIVE' if circuit_off else f'✅ clear ({consec_losses}/{MAX_LOSSES})'}\n"
               f"  Silent  : {'ON' if silent_mode else 'OFF'}\n"
               f"  Status  : {'⏸ Paused' if paused else '▶️ Running'}\n{sep()}\n"
               + ("\n".join(lines) if lines else "No signals yet"), chat)

        elif text.startswith("/log"):
            if not trade_log:
                tg("No signals yet.", chat); continue
            lines = [f"📋 *Last 10 signals — Session #{session_num}*\n{sep()}"]
            for t in trade_log[-10:]:
                e = "✅" if t.get("correct") else ("❌" if t.get("correct") is False else "—")
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
                bar  = "█" * int(wr/10)
                lines.append(f"  {h:02d}:00 {flag} {bar:<10} {wr:.0f}% ({d['correct']}/{d['total']})")
            lines.append(f"{sep()}\nBreak-even ≈ 52.6%")
            tg("\n".join(lines), chat)

        elif text.startswith("/config"):
            tg(f"⚙️ *Config — v8*\n{sep()}\n"
               f"  Confidence : ≥{CONFIDENCE:.0%}\n"
               f"  Min gap    : ≥{MIN_GAP_PCT:.2f}%\n"
               f"  Stake      : ${current_stake:.2f} (default ${STAKE:.2f})\n"
               f"  Auto-buy   : {'✅ ON' if AUTO_BUY else '⚠️ OFF'}\n"
               f"  Silent mode: {'ON' if silent_mode else 'OFF'}\n"
               f"  Balance min: ${BALANCE_MIN:.2f}\n"
               f"  Circuit    : after {MAX_LOSSES} consecutive losses\n"
               f"  Signal at  : {SIGNAL_DELAY} mins into round\n"
               f"  Status     : {'⏸ Paused' if paused else '▶️ Running'}", chat)

        elif text.startswith("/price"):
            btc = btc_price(); tgt = round_state.get("target")
            if btc and tgt:
                diff     = btc - tgt
                diff_pct = diff / tgt * 100
                tg(f"💰 *BTC Snapshot*\n{sep()}\n"
                   f"  BTC Now : ${btc:,.2f}\n"
                   f"  Target  : ${tgt:,.2f}\n"
                   f"  Gap     : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
                   f"  Status  : {'📈 ABOVE' if diff > 0 else '📉 BELOW'} target", chat)
            else:
                tg("No active round data yet.", chat)

        elif text.startswith("/export"):
            if not os.path.exists(LOG_FILE):
                tg("No CSV log yet.", chat); continue
            with open(LOG_FILE,"rb") as f: data = f.read()
            rows = data.count(b"\n") - 1
            fname = f"session_{session_num}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
            tg_doc(fname, data, f"Session #{session_num} — {rows} signals", chat)

        elif text.startswith("/startsession"):
            session_num += 1; session_start = datetime.now(timezone.utc)
            session_on = True; paused = False; circuit_off = False
            consec_losses = 0; current_stake = STAKE; silent_mode = False
            balance = 10.00; total_staked = 0.0; total_won = 0.0
            stats = {k: 0 for k in stats}; trade_log = []
            hour_stats = defaultdict(lambda: {"total": 0, "correct": 0})
            last_round = {k: None for k in last_round}; last_round["resolved"] = False
            round_state = {"event_id":None,"prev_event_id":None,"start_time":None,
                           "target":None,"fired":False,"signal":None,"order_id":None}
            tg(f"🆕 *Session #{session_num} started*\n"
               f"📅 {session_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
               f"  conf≥{CONFIDENCE:.0%} | gap≥{MIN_GAP_PCT:.2f}% | stake=${current_stake:.2f}\n"
               f"  Auto-buy: {'ON' if AUTO_BUY else 'OFF'}\n"
               f"  Balance reset to ${balance:.2f}", chat)

        elif text.startswith("/stopsession"):
            session_on = False; paused = True
            total = stats["wins"] + stats["losses"]
            wr    = f"{stats['wins']/total*100:.1f}%" if total else "N/A"
            tg(f"🛑 *Session #{session_num} ended*\n"
               f"  Signals : {stats['signalled']} | Trades: {total} | WR: {wr}\n"
               f"  ✅ {stats['wins']} | ❌ {stats['losses']}\n"
               f"  Balance : ${balance:.2f} | P&L: ${total_won-total_staked:+.2f}", chat)

        elif text.startswith("/start"):
            mode = f"AUTO-BUY ${current_stake:.2f}/trade" if AUTO_BUY else "NOTIFY ONLY"
            tg(f"🤖 *Bayse BTC Bot v8*\n"
               f"Mode: *{mode}*\n{sep()}\n"
               f"  conf≥{CONFIDENCE:.0%} | gap≥{MIN_GAP_PCT:.2f}% | ~20–30 signals/day\n"
               f"  Every round shows signal even if not traded\n{sep()}\n"
               f"Commands:\n"
               f"  /balance  /price  /stats  /log  /hours  /config  /export\n"
               f"  /pause  /play\n"
               f"  /playsilent  /pausesilent\n"
               f"  /increase N  /decrease N  /stake\n"
               f"  /startsession  /stopsession", chat)

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main():
    global balance, total_staked, total_won, round_state, last_round
    global stats, trade_log, hour_stats, circuit_off, consec_losses

    init_log()
    seed_candles()

    mode = f"AUTO-BUY ${STAKE:.2f}" if AUTO_BUY else "NOTIFY ONLY"
    tg(f"🟢 *Bot v8 online — Session #{session_num}*\n"
       f"  conf≥{CONFIDENCE:.0%} | gap≥{MIN_GAP_PCT:.2f}% | {mode}\n"
       f"  Every round sends signal to Telegram\n"
       f"  ~20–30 signals/day expected")
    log.info(f"v8 started | auto_buy={AUTO_BUY}")

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

            event_id = market["event_id"]
            btc      = btc_price()
            end_dt   = parse_end_dt(market["rules"], now)
            mins_left = round((end_dt - now).total_seconds()/60, 1) if end_dt else None

            # ── NEW ROUND ─────────────────────────────────────
            if event_id != round_state["event_id"]:
                log.info(f"New round: {event_id}")

                prev_sig    = round_state.get("signal")
                prev_target = round_state.get("target")
                prev_eid    = round_state.get("prev_event_id")

                # ── RESOLVE PREVIOUS ROUND ────────────────────
                if prev_target:
                    final = None
                    if prev_eid:
                        final = fetch_final_price(prev_eid)
                        if final: log.info(f"Bayse final price: ${final:,.2f}")
                    if not final and market.get("start_price"):
                        final = btc  # best available fallback
                    if not final:
                        final = btc

                    if final and prev_sig:
                        actual_up  = final >= prev_target
                        actual_dir = "UP" if actual_up else "DOWN"
                        correct    = (prev_sig["up"] == actual_up)
                        traded     = prev_sig.get("was_trade", False)
                        used_stake = prev_sig.get("stake_used", current_stake)
                        hour_h     = prev_sig.get("hour", now.hour)

                        # Update hour stats
                        hour_stats[hour_h]["total"] += 1
                        if correct: hour_stats[hour_h]["correct"] += 1

                        if traded:
                            stats["total"] += 1
                            if correct:
                                stats["wins"] += 1
                                consec_losses = 0
                                earned = round(used_stake * prev_sig["payout"], 2)
                                total_won    += earned
                                total_staked += used_stake
                                balance       = round(balance + earned - used_stake, 2)
                                if circuit_off:
                                    circuit_off = False
                                    tg("✅ *Win — circuit break cleared. Signals resuming.*")
                            else:
                                stats["losses"] += 1
                                consec_losses += 1
                                total_staked += used_stake
                                balance       = round(balance - used_stake, 2)
                                if consec_losses >= MAX_LOSSES and not circuit_off:
                                    circuit_off = True; stats["breaks"] += 1
                                    tg(f"⚡ *Circuit break — {MAX_LOSSES} straight losses*\n"
                                       f"Balance: ${balance:.2f}\n/play to resume manually")

                            # Sync from API after each resolved trade
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
                            "target"       : prev_target,
                            "btc"          : prev_sig.get("btc",""),
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
                        log.info(f"Resolved: {'✅' if correct else '❌'} | "
                                 f"traded={traded} | balance=${balance:.2f}")

                # ── SET NEW ROUND ─────────────────────────────
                new_target = market.get("start_price") or btc
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

            # ── FIRE SIGNAL (5 mins into round) ──────────────
            start_t = round_state.get("start_time")
            if (not round_state["fired"] and start_t
                    and round_state["target"] and not paused):
                mins_in = (now - start_t).total_seconds() / 60
                if mins_in >= SIGNAL_DELAY:
                    sig = get_signal(market, round_state["target"], now)
                    if sig:
                        round_state["fired"] = True
                        stats["signalled"] += 1

                        # Determine whether to trade
                        # Trade if: (normal signal) or (silent mode + passes conf+gap)
                        silent_trade = (silent_mode and not sig["trade"]
                                        and sig["conf"] >= CONFIDENCE
                                        and sig["abs_gap"] >= MIN_GAP_PCT)
                        should_trade = sig["trade"] or silent_trade

                        sig["was_trade"]  = should_trade
                        sig["stake_used"] = current_stake
                        round_state["signal"] = sig

                        order = None; order_id = None
                        if should_trade and AUTO_BUY and balance >= BALANCE_MIN:
                            order = place_order(market, sig["up"], current_stake)
                            if order:
                                order_id = order["order_id"]
                                round_state["order_id"] = order_id
                                stats["orders_ok"] += 1
                                # Immediate balance sync
                                api_bal = fetch_balance_api()
                                if api_bal is not None:
                                    balance = api_bal
                            else:
                                stats["orders_fail"] += 1

                        # Always send signal to Telegram
                        tg(msg_signal_card(sig, traded=should_trade,
                                           order_id=order_id, silent=silent_trade))
                        if order:
                            tg(msg_receipt(order, sig))

                        log.info(f"Signal | {sig['direction']} | conf {sig['conf']:.1%} | "
                                 f"gap {sig['abs_gap']:.3f}% | pay {sig['payout']:.2f}x | "
                                 f"trade={should_trade} | order={'ok' if order else 'none'}")
                    else:
                        log.warning("Features not ready yet")

            time.sleep(30)

        except Exception as e:
            log.error(f"Loop: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
