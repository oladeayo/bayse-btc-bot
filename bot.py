"""
Bayse BTC Prediction Bot — v6
==============================
Final production version. All changes backed by 295 trades across 4 days.

NEW in v6:
  1. AUTO-BUY — places real $1 orders on Bayse when ✅ TRADE fires.
     Requires BAYSE_KEY (public) + BAYSE_SECRET_KEY (secret) for HMAC write auth.
     Falls back to notification-only mode if key is missing or read-only.

  2. HOUR BLOCKING — 09:00 UTC blocked (35.7% WR across 4 days, confirmed bad).
     02:00 UTC optionally blockable — currently watch-only.

  3. BANKROLL TRACKING — bot tracks its own running balance, checks it before
     every order, and refuses to over-commit. /balance command shows live state.

  4. BALANCE-AWARE CIRCUIT BREAK — pauses if balance drops below BALANCE_MIN
     in addition to the consecutive-loss circuit break.

Carried from v5.1:
  • Confidence ≥ 80%  (60–80% band = 41–59% WR)
  • Gap ≥ 0.05%        (below = 47–62% WR)
  • Outcome price ≤ 90% (above = payout < 0.11x, need 9+ wins per loss)
  • Bayse-native prices (start_price / finalPrice from API, not KuCoin)
  • Silent CSV logging (all rounds, Telegram only on ✅ TRADE)
  • Directional confidence display  ("86.2% DOWN" not "13.8%")
  • WON/LOST/no-signal correct display (explicit is True/is False checks)
  • Circuit break after MAX_CONSEC_LOSSES
  • Tiered stake guide in signal messages
"""

import logging, requests, joblib, json, re, os, csv, sys, time, io
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone, timedelta
from collections import deque

logging.basicConfig(format="%(asctime)s — %(levelname)s — %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT     = os.environ.get("TELEGRAM_CHAT", "")
BAYSE_KEY         = os.environ.get("BAYSE_KEY", "")           # read-only (market data)
BAYSE_SECRET_KEY  = os.environ.get("BAYSE_SECRET_KEY", "")    # secret key for HMAC write auth
BASE_URL          = "https://relay.bayse.markets/v1"
BAYSE_HEADERS     = {"X-Public-Key": BAYSE_KEY}
TELEGRAM_API      = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ── Strategy ──────────────────────────────────────────────────
CONFIDENCE        = 0.80    # 60–80% band = 41–59% WR — dead zone
MIN_GAP_PCT       = 0.05    # below 0.05% = 47–62% WR
MAX_OUTCOME_PRICE = 0.90    # above 90% odds → payout <0.11x, need 9+ wins per loss
MIN_LIQUIDITY     = 50.0
MIN_ORDERS        = 10
FEE               = 0.05
SIGNAL_DELAY_MINS = 5
MAX_CONSEC_LOSSES = 3

# ── Hour blocking ─────────────────────────────────────────────
# 09:00 UTC: 35.7% WR across 4 days — confirmed bad, hard block
# 02:00 UTC: 41.7% WR across 3 days — approaching block threshold
BLOCKED_HOURS     = {9}     # add 2 when you have 4 days confirming it

# ── Auto-buy / bankroll ───────────────────────────────────────
STAKE             = 1.00    # $ per trade — flat $1
STARTING_BALANCE  = 10.00   # initial deposit
BALANCE_MIN       = 2.00    # stop auto-buy if balance drops to this
AUTO_BUY_ENABLED  = bool(BAYSE_SECRET_KEY)  # off until BAYSE_SECRET_KEY is set

# ── Other ─────────────────────────────────────────────────────
REPORT_HOURS      = 12
LOG_FILE          = "trade_log.csv"

# ─────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT or not BAYSE_KEY:
    print("ERROR: Missing env vars (TELEGRAM_TOKEN, TELEGRAM_CHAT, BAYSE_KEY)", flush=True)
    sys.exit(1)
print(f"✅ Token: {TELEGRAM_TOKEN[:10]}...", flush=True)
print(f"✅ Chat: {TELEGRAM_CHAT}", flush=True)
if AUTO_BUY_ENABLED:
    print(f"✅ Auto-buy: ENABLED | stake ${STAKE:.2f} | balance ${STARTING_BALANCE:.2f}", flush=True)
else:
    print("⚠️  Auto-buy: DISABLED — set BAYSE_SECRET_KEY env var to enable", flush=True)

model    = joblib.load("btc_bayse_model_v2.joblib")
features = json.load(open("features_v2.json"))
print(f"✅ Model loaded | {len(features)} features", flush=True)

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
CANDLE_WINDOW  = deque(maxlen=100)
bot_paused     = False
circuit_broken = False
consec_losses  = 0
session_active = True
session_start  = datetime.now(timezone.utc)
session_num    = 1

# Running bankroll — updated on every auto-buy result
# If auto-buy is off this tracks what WOULD have happened (paper mode)
balance        = STARTING_BALANCE
total_staked   = 0.0
total_profit   = 0.0

current_round = {
    "event_id"         : None,
    "prev_event_id"    : None,
    "start_time"       : None,
    "round_start_price": None,
    "signal_fired"     : False,
    "pending_signal"   : None,
    "order_id"         : None,   # Bayse order ID if auto-buy placed
}

last_completed = {
    "round_start_price": None,
    "btc_at_end"       : None,
    "actual_direction" : None,
    "bot_direction"    : None,
    "correct"          : None,
    "resolved"         : False,
}

last_report_time     = datetime.now(timezone.utc)
last_update_id       = None
current_market_cache = None

stats = {
    "total"            : 0,
    "signalled"        : 0,
    "correct"          : 0,
    "incorrect"        : 0,
    "high_conv_total"  : 0,
    "high_conv_correct": 0,
    "skipped_conf"     : 0,
    "skipped_gap"      : 0,
    "skipped_max_odds" : 0,
    "skipped_liquidity": 0,
    "skipped_hour"     : 0,
    "circuit_breaks"   : 0,
    "orders_placed"    : 0,
    "orders_failed"    : 0,
}
trade_log  = []
hour_stats = {}

# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────
def tg_send(text, chat_id=None):
    try:
        r = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id or TELEGRAM_CHAT,
                  "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not r.ok:
            log.error(f"TG send: {r.status_code} {r.text[:100]}")
    except Exception as e:
        log.error(f"TG exception: {e}")

def tg_send_document(filename, content_bytes, caption, chat_id=None):
    try:
        r = requests.post(
            f"{TELEGRAM_API}/sendDocument",
            data={"chat_id": chat_id or TELEGRAM_CHAT, "caption": caption},
            files={"document": (filename, content_bytes, "text/csv")},
            timeout=15,
        )
        if not r.ok:
            log.error(f"TG doc: {r.status_code}")
    except Exception as e:
        log.error(f"TG doc exception: {e}")

def tg_get_updates():
    global last_update_id
    try:
        params = {"timeout": 1, "allowed_updates": ["message"]}
        if last_update_id is not None:
            params["offset"] = last_update_id + 1
        r = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=5)
        if r.ok:
            updates = r.json().get("result", [])
            for u in updates:
                last_update_id = u["update_id"]
            return updates
    except Exception as e:
        log.error(f"TG poll: {e}")
    return []

# ─────────────────────────────────────────────────────────────
# AUTO-BUY
# ─────────────────────────────────────────────────────────────
#
# Bayse write auth uses HMAC-SHA256 — NOT a Bearer token.
# You need your API key PAIR: public key (pk_live_...) + secret key (sk_live_...).
# The public key is already BAYSE_KEY. Add BAYSE_SECRET_KEY as a new env var.
#
# Signing process (from Bayse docs):
#   payload = f"{timestamp}.{METHOD}.{url_path}.{sha256_hex(body)}"
#   signature = base64( hmac_sha256(secret_key, payload) )
#
# Order endpoint: POST /v1/pm/events/{eventId}/markets/{marketId}/orders
# Required fields: side, outcomeId (UUID), amount, type
# outcomeId comes from the event detail: outcome1Id (YES) or outcome2Id (NO)

import hashlib, hmac, base64

def _build_signature(secret_key, method, url_path, body_str):
    """
    Build Bayse HMAC-SHA256 write signature.
    payload = "{timestamp}.{METHOD}.{path}.{sha256_hex_of_body}"
    """
    timestamp  = str(int(datetime.now(timezone.utc).timestamp()))
    body_hash  = hashlib.sha256(body_str.encode()).hexdigest()
    signing_payload = f"{timestamp}.{method}.{url_path}.{body_hash}"
    sig = hmac.new(
        secret_key.encode(),
        signing_payload.encode(),
        hashlib.sha256,
    ).digest()
    signature = base64.b64encode(sig).decode()
    return timestamp, signature


def get_order_ids(market, direction_up):
    """
    Extract eventId, marketId, and the outcomeId UUID for the chosen side.
    outcomeId is NOT "YES"/"NO" — it is a UUID from the event detail object.
    BUY UP  → outcome1Id (YES side)
    BUY DOWN → outcome2Id (NO side)
    """
    detail    = market.get("raw_detail", {})
    event_id  = detail.get("id")
    ms        = detail.get("markets", [])
    if not ms or not event_id:
        return None, None, None
    m          = ms[0]
    market_id  = m.get("id")
    outcome_id = m.get("outcome1Id") if direction_up else m.get("outcome2Id")
    return event_id, market_id, outcome_id


def place_order(market, direction_up, amount):
    """
    Place a MARKET BUY order on Bayse using correct HMAC write authentication.

    Requires two env vars:
      BAYSE_KEY        — your public key  (pk_live_...)  [already set]
      BAYSE_SECRET_KEY — your secret key  (sk_live_...)  [new — add to Railway]

    How to get these if you don't have them yet:
      1. curl -X POST https://relay.bayse.markets/v1/user/login
            -H "Content-Type: application/json"
            -d '{"email":"you@email.com","password":"yourpassword"}'
         → copy token and deviceId from response

      2. curl -X POST https://relay.bayse.markets/v1/user/me/api-keys
            -H "x-auth-token: TOKEN" -H "x-device-id: DEVICEID"
            -H "Content-Type: application/json"
            -d '{"name":"BTC Bot"}'
         → copy publicKey and secretKey (secret shown ONCE, store it now)

      3. Set in Railway:
            BAYSE_KEY        = pk_live_...   (public key)
            BAYSE_SECRET_KEY = sk_live_...   (secret key)

    Returns order_id string on success, None on failure.
    """
    if not BAYSE_SECRET_KEY:
        log.warning("place_order: BAYSE_SECRET_KEY not set — running notify-only. "
                    "See docstring for how to create your API key pair.")
        return None

    event_id, market_id, outcome_id = get_order_ids(market, direction_up)
    if not all([event_id, market_id, outcome_id]):
        log.error(f"place_order: missing IDs — event={event_id} market={market_id} outcome={outcome_id}")
        return None

    url_path = f"/v1/pm/events/{event_id}/markets/{market_id}/orders"
    payload  = {
        "side"      : "BUY",
        "outcomeId" : outcome_id,
        "amount"    : round(amount, 2),
        "type"      : "MARKET",
        "currency"  : "USD",
    }
    body_str  = json.dumps(payload, separators=(",", ":"))   # compact, no spaces
    timestamp, signature = _build_signature(BAYSE_SECRET_KEY, "POST", url_path, body_str)

    try:
        headers = {
            "X-Public-Key" : BAYSE_KEY,
            "X-Timestamp"  : timestamp,
            "X-Signature"  : signature,
            "Content-Type" : "application/json",
        }
        r = requests.post(
            f"{BASE_URL}{url_path}",
            headers=headers,
            data=body_str,   # use data= not json= to keep body identical to what was signed
            timeout=10,
        )
        if r.ok:
            data     = r.json()
            order    = data.get("order", {})
            order_id = order.get("id") or "unknown"
            status   = order.get("status", "?")
            log.info(f"Order placed ✅ | event={event_id} | outcome={outcome_id} | "
                     f"${amount:.2f} | id={order_id} | status={status}")
            return str(order_id)
        else:
            log.error(f"Order failed: HTTP {r.status_code} | {r.text[:300]}")
            if r.status_code == 401:
                log.error("→ Signature invalid. Check BAYSE_KEY and BAYSE_SECRET_KEY match "
                          "the same API key pair.")
            elif r.status_code == 403:
                log.error("→ Forbidden. Your API key may not have trading permissions.")
            return None
    except Exception as e:
        log.error(f"place_order exception: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────────────────────
def handle_commands():
    global bot_paused, circuit_broken, consec_losses
    global session_active, session_start, session_num
    global stats, trade_log, hour_stats, last_completed, current_round
    global balance, total_staked, total_profit

    for update in tg_get_updates():
        msg     = update.get("message", {})
        text    = msg.get("text", "").strip().lower()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if text.startswith("/pause"):
            bot_paused = True
            tg_send("⏸ *Bot paused.* No signals or orders will fire.\nUse /play to resume.", chat_id)

        elif text.startswith("/play"):
            bot_paused     = False
            circuit_broken = False
            consec_losses  = 0
            tg_send("▶️ *Bot resumed.* Signals active. Circuit break cleared.", chat_id)

        elif text.startswith("/balance"):
            cmd_balance(chat_id)

        elif text.startswith("/start_session"):
            session_num   += 1
            session_start  = datetime.now(timezone.utc)
            session_active = True
            bot_paused     = False
            circuit_broken = False
            consec_losses  = 0
            balance        = STARTING_BALANCE
            total_staked   = 0.0
            total_profit   = 0.0
            stats = {k: 0 for k in stats}
            trade_log  = []
            hour_stats = {}
            last_completed = {k: None for k in last_completed}
            last_completed["resolved"] = False
            current_round = {
                "event_id": None, "prev_event_id": None, "start_time": None,
                "round_start_price": None, "signal_fired": False,
                "pending_signal": None, "order_id": None,
            }
            mode = f"AUTO-BUY ${STAKE:.2f}/trade" if AUTO_BUY_ENABLED else "NOTIFY ONLY (set BAYSE_SECRET_KEY to enable)"
            tg_send(
                f"🆕 *Session {session_num} started*\n"
                f"📅 {session_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Mode: *{mode}*\n"
                f"Balance reset to ${STARTING_BALANCE:.2f}\n\n"
                f"Active filters:\n"
                f"  • Conf ≥ {CONFIDENCE:.0%}\n"
                f"  • Gap ≥ {MIN_GAP_PCT:.2f}%\n"
                f"  • Outcome ≤ {MAX_OUTCOME_PRICE:.0%}\n"
                f"  • Blocked hours: {sorted(BLOCKED_HOURS)}\n"
                f"  • Circuit break after {MAX_CONSEC_LOSSES} losses\n"
                f"  • Stop trading if balance < ${BALANCE_MIN:.2f}",
                chat_id,
            )

        elif text.startswith("/stop_session"):
            session_active = False
            bot_paused     = True
            tg_send(build_final_report(), chat_id)
            tg_send("🛑 *Session stopped.* Use /start_session to begin a new one.", chat_id)

        elif text.startswith("/export"):
            cmd_export(chat_id)

        elif text.startswith("/price"):
            cmd_price(chat_id)

        elif text.startswith("/trade"):
            cmd_price(chat_id)

        elif text.startswith("/stats"):
            tg_send(build_stats_message(), chat_id)

        elif text.startswith("/log"):
            tg_send(build_log_message(), chat_id)

        elif text.startswith("/hours"):
            tg_send(build_hour_stats(), chat_id)

        elif text.startswith("/config"):
            tg_send(build_config_message(), chat_id)

        elif text.startswith("/start"):
            tg_send(start_message(), chat_id)


def start_message():
    mode = f"AUTO-BUY ${STAKE:.2f}/trade" if AUTO_BUY_ENABLED else "NOTIFY ONLY (set BAYSE_SECRET_KEY)"
    return (
        f"🤖 *Bayse BTC Bot v6*\n"
        f"Mode: *{mode}*\n\n"
        "Commands:\n"
        "  /balance       — live bankroll + P&L\n"
        "  /price         — BTC vs target + Bayse odds\n"
        "  /stats         — full performance summary\n"
        "  /log           — last 10 signalled trades\n"
        "  /hours         — win rate by UTC hour\n"
        "  /config        — current strategy settings\n"
        "  /export        — download full session CSV\n"
        "  /pause         — pause signals + orders\n"
        "  /play          — resume + reset circuit break\n"
        "  /start_session — new session (resets balance)\n"
        "  /stop_session  — end session + final report\n\n"
        f"v6 | conf≥{CONFIDENCE:.0%} | gap≥{MIN_GAP_PCT:.2f}% | "
        f"op≤{MAX_OUTCOME_PRICE:.0%} | block {sorted(BLOCKED_HOURS)} | "
        f"circuit @{MAX_CONSEC_LOSSES}"
    )


def build_config_message():
    mode = f"AUTO-BUY ${STAKE:.2f}" if AUTO_BUY_ENABLED else "NOTIFY ONLY"
    cb   = (f"⚡ ACTIVE ({consec_losses}/{MAX_CONSEC_LOSSES})"
            if circuit_broken else f"✅ Clear ({consec_losses}/{MAX_CONSEC_LOSSES})")
    return (
        f"⚙️ *Bot v6 Config*\n"
        f"{'─'*30}\n"
        f"  Mode           : *{mode}*\n"
        f"  Stake/trade    : ${STAKE:.2f}\n"
        f"  Starting bal   : ${STARTING_BALANCE:.2f}\n"
        f"  Stop threshold : ${BALANCE_MIN:.2f}\n"
        f"{'─'*30}\n"
        f"  Confidence     : ≥{CONFIDENCE:.0%}\n"
        f"  Min gap        : ≥{MIN_GAP_PCT:.2f}%\n"
        f"  Max odds       : ≤{MAX_OUTCOME_PRICE:.0%}\n"
        f"  Blocked hours  : {sorted(BLOCKED_HOURS)} UTC\n"
        f"  Min liquidity  : ${MIN_LIQUIDITY:.0f}\n"
        f"  Signal delay   : {SIGNAL_DELAY_MINS} mins\n"
        f"  Fee            : {FEE:.0%}\n"
        f"  Circuit break  : after {MAX_CONSEC_LOSSES} losses\n"
        f"{'─'*30}\n"
        f"  Status         : {'⏸ Paused' if bot_paused else '▶️ Running'}\n"
        f"  Circuit        : {cb}\n"
        f"  Session        : #{session_num}\n"
        f"  Since          : {session_start.strftime('%Y-%m-%d %H:%M UTC')}"
    )


def cmd_balance(chat_id):
    mode   = "AUTO-BUY" if AUTO_BUY_ENABLED else "Paper (notify only)"
    roi    = f"{(balance - STARTING_BALANCE) / STARTING_BALANCE * 100:+.1f}%" if STARTING_BALANCE else "N/A"
    wr     = f"{stats['correct']/stats['signalled']*100:.1f}%" if stats["signalled"] > 0 else "N/A"
    tg_send(
        f"💰 *Balance — Session #{session_num}*\n"
        f"{'─'*28}\n"
        f"  Mode       : {mode}\n"
        f"  Balance    : *${balance:.2f}*\n"
        f"  Started    : ${STARTING_BALANCE:.2f}\n"
        f"  P&L        : ${balance - STARTING_BALANCE:+.2f} ({roi})\n"
        f"  Total in   : ${total_staked:.2f}\n"
        f"  Total out  : ${total_profit:.2f}\n"
        f"{'─'*28}\n"
        f"  Signals    : {stats['signalled']} | WR {wr}\n"
        f"  Orders ok  : {stats['orders_placed']}\n"
        f"  Orders fail: {stats['orders_failed']}\n"
        f"  Stop at    : ${BALANCE_MIN:.2f}",
        chat_id,
    )


def cmd_price(chat_id):
    btc    = get_btc_price()
    target = current_round.get("round_start_price")
    m      = current_market_cache
    now    = datetime.now(timezone.utc)
    if btc and target and m:
        diff     = btc - target
        diff_pct = (diff / target) * 100
        yes      = m.get("yes_price", 0)
        no       = m.get("no_price", 0)
        op_yes   = yes; op_no = 1.0 - yes
        cb_note  = f"\n  ⚡ Circuit: {consec_losses}/{MAX_CONSEC_LOSSES} losses" if circuit_broken else ""
        bal_note = f"\n  💰 Balance: ${balance:.2f}" if AUTO_BUY_ENABLED else ""
        tg_send(
            f"💰 *BTC Snapshot*\n"
            f"{'─'*28}\n"
            f"  BTC now  : ${btc:,.2f}\n"
            f"  Target   : ${target:,.2f}\n"
            f"  Gap      : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
            f"  Status   : {'📈 ABOVE' if diff > 0 else '📉 BELOW'} target\n"
            f"{'─'*28}\n"
            f"  YES (UP) : {op_yes:.0%} {'✅' if op_yes <= MAX_OUTCOME_PRICE else '⛔ cap'}\n"
            f"  NO (DOWN): {op_no:.0%} {'✅' if op_no <= MAX_OUTCOME_PRICE else '⛔ cap'}\n"
            f"  Orders   : {m.get('total_orders', 0)} | Liq: ${m.get('liquidity', 0):,.2f}\n"
            f"  Time     : {now.strftime('%H:%M:%S UTC')}\n"
            f"{'─'*28}\n"
            f"  Bot      : {'⏸ Paused' if bot_paused else '▶️ Running'}"
            + cb_note + bal_note,
            chat_id,
        )
    else:
        tg_send("⚠️ No active round data yet.", chat_id)


def cmd_export(chat_id):
    if not os.path.exists(LOG_FILE):
        tg_send("No CSV log found yet.", chat_id)
        return
    with open(LOG_FILE, "rb") as f:
        content = f.read()
    count    = content.count(b"\n") - 1
    filename = f"session_{session_num}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
    tg_send_document(
        filename, content,
        f"Session #{session_num} — {count} rounds (all, including silent)",
        chat_id,
    )

# ─────────────────────────────────────────────────────────────
# CSV LOG
# ─────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "timestamp", "time", "round_start_price", "btc_at_signal",
    "btc_at_end", "bot_direction", "actual_direction", "correct",
    "confidence", "conviction", "yes_price", "no_price",
    "gap_pct", "odds_filter_pass", "liquidity", "orders", "session",
    "signalled", "order_placed", "order_id", "stake", "payout_earned",
]

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)

def write_log(entry):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([entry.get(k, "") for k in CSV_FIELDS])

# ─────────────────────────────────────────────────────────────
# KUCOIN CANDLES (for ML model features only)
# ─────────────────────────────────────────────────────────────
def seed_candles():
    try:
        r = requests.get(
            "https://api.kucoin.com/api/v1/market/candles",
            params={"symbol": "BTC-USDT", "type": "1min"},
            timeout=10,
        )
        for c in reversed(r.json().get("data", [])):
            CANDLE_WINDOW.append({
                "open_time": pd.Timestamp(int(c[0]), unit="s", tz="UTC"),
                "open": float(c[1]), "close": float(c[2]),
                "high": float(c[3]), "low": float(c[4]), "volume": float(c[5]),
            })
        log.info(f"Seeded {len(CANDLE_WINDOW)} candles | ${CANDLE_WINDOW[-1]['close']:,.2f}")
    except Exception as e:
        log.error(f"seed_candles: {e}")

def update_candles():
    try:
        r = requests.get(
            "https://api.kucoin.com/api/v1/market/candles",
            params={"symbol": "BTC-USDT", "type": "1min", "pageSize": 3},
            timeout=5,
        )
        data = r.json().get("data", [])
        if not data:
            return
        c = data[0]
        latest = {
            "open_time": pd.Timestamp(int(c[0]), unit="s", tz="UTC"),
            "open": float(c[1]), "close": float(c[2]),
            "high": float(c[3]), "low": float(c[4]), "volume": float(c[5]),
        }
        if not CANDLE_WINDOW or latest["open_time"] > CANDLE_WINDOW[-1]["open_time"]:
            CANDLE_WINDOW.append(latest)
    except Exception as e:
        log.error(f"update_candles: {e}")

def get_btc_price():
    return CANDLE_WINDOW[-1]["close"] if CANDLE_WINDOW else None

# ─────────────────────────────────────────────────────────────
# BAYSE API
# ─────────────────────────────────────────────────────────────
def fetch_btc_market():
    """
    Fetch current BTC 15-min event. Extracts start_price and final_price
    directly from Bayse — KuCoin is never used for these values.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/pm/events",
            headers=BAYSE_HEADERS,
            params={"limit": 50},
            timeout=10,
        )
        for event in r.json().get("events", []):
            if "bitcoin" in event.get("title", "").lower() and "15" in event.get("title", ""):
                r2     = requests.get(f"{BASE_URL}/pm/events/{event['id']}",
                                      headers=BAYSE_HEADERS, timeout=10)
                detail = r2.json()
                ms     = detail.get("markets", [])
                if ms:
                    m = ms[0]

                    def _price(obj, *keys):
                        for k in keys:
                            v = obj.get(k)
                            if v is not None:
                                try: return float(v)
                                except (TypeError, ValueError): pass
                        return None

                    start_price = (
                        _price(detail, "startPrice","start_price","targetPrice","target_price")
                        or _price(m, "startPrice","start_price","targetPrice","target_price")
                        or parse_target(m.get("rules", ""))
                    )
                    final_price = (
                        _price(detail, "finalPrice","final_price","resolutionPrice","resolution_price")
                        or _price(m, "finalPrice","final_price","resolutionPrice","resolution_price")
                    )

                    log.debug(f"Bayse | event={detail['id']} | start={start_price} | final={final_price}")

                    return {
                        "event_id"    : detail["id"],
                        "yes_price"   : m.get("outcome1Price", 0),
                        "no_price"    : m.get("outcome2Price", 0),
                        "total_orders": m.get("totalOrders", 0),
                        "liquidity"   : detail.get("liquidity", 0),
                        "created_at"  : detail.get("createdAt", ""),
                        "rules"       : m.get("rules", ""),
                        "start_price" : start_price,
                        "final_price" : final_price,
                        "raw_detail"  : detail,
                    }
    except Exception as e:
        log.error(f"fetch_btc_market: {e}")
    return None


def fetch_event_final_price(event_id):
    """Fetch a resolved event's finalPrice from Bayse."""
    try:
        r = requests.get(f"{BASE_URL}/pm/events/{event_id}",
                         headers=BAYSE_HEADERS, timeout=10)
        if not r.ok:
            return None
        detail = r.json()
        ms     = detail.get("markets", [])
        m      = ms[0] if ms else {}
        for k in ("finalPrice","final_price","resolutionPrice","resolution_price"):
            v = detail.get(k) or m.get(k)
            if v is not None:
                try: return float(v)
                except (TypeError, ValueError): pass
        return None
    except Exception as e:
        log.error(f"fetch_event_final_price({event_id}): {e}")
        return None


def parse_target(rules):
    m = re.findall(r'\$([\d,]+\.?\d*)', rules)
    if m:
        try: return float(m[0].replace(",", ""))
        except ValueError: pass
    return None


def parse_end_dt(rules, now):
    m = re.search(r'(\d+:\d+:\d+\s?[AP]M)\s?GMT', rules)
    if m:
        try:
            end = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} {m.group(1).strip()}",
                "%Y-%m-%d %I:%M:%S %p",
            ).replace(tzinfo=timezone.utc)
            if (end - now).total_seconds() < -3600:
                end += timedelta(days=1)
            return end
        except Exception:
            pass
    return None

# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────
def compute_features(round_start_price):
    if len(CANDLE_WINDOW) < 60 or not round_start_price:
        return None, None
    df = pd.DataFrame(list(CANDLE_WINDOW)).sort_values("open_time").reset_index(drop=True)
    try:
        df["rsi_14"]         = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi_7"]          = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
        df["rsi_21"]         = ta.momentum.RSIIndicator(df["close"], window=21).rsi()
        df["stoch"]          = ta.momentum.StochasticOscillator(df["high"],df["low"],df["close"]).stoch()
        df["macd"]           = ta.trend.MACD(df["close"]).macd_diff()
        df["macd_signal"]    = ta.trend.MACD(df["close"]).macd_signal()
        df["ema_9"]          = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema_21"]         = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
        df["ema_cross"]      = df["ema_9"] - df["ema_21"]
        bb                   = ta.volatility.BollingerBands(df["close"])
        df["bb_position"]    = (df["close"] - bb.bollinger_mavg()) / bb.bollinger_wband()
        df["bb_width"]       = bb.bollinger_wband()
        df["atr"]            = ta.volatility.AverageTrueRange(
                                   df["high"],df["low"],df["close"],window=14).average_true_range()
        df["vol_ratio_15"]   = df["volume"] / df["volume"].rolling(15).mean()
        df["vol_ratio_60"]   = df["volume"] / df["volume"].rolling(60).mean()
        df["obv"]            = ta.volume.OnBalanceVolumeIndicator(
                                   df["close"],df["volume"]).on_balance_volume()
        df["obv_slope"]      = df["obv"].diff(5)
        df["candle_body"]    = (df["close"] - df["open"]) / df["open"]
        df["momentum_1"]     = df["close"].pct_change(1)
        df["momentum_3"]     = df["close"].pct_change(3)
        df["momentum_5"]     = df["close"].pct_change(5)
        df["rolling_std_15"] = df["close"].pct_change().rolling(15).std()
        df["rolling_std_60"] = df["close"].pct_change().rolling(60).std()
        df["hour"]           = df["open_time"].dt.hour
        df["dayofweek"]      = df["open_time"].dt.dayofweek

        latest  = df.iloc[-1].copy()
        atr_val = max(float(latest["atr"]), 1e-8)
        pvt     = (latest["close"] - round_start_price) / round_start_price
        pgd     = latest["close"] - round_start_price

        latest["price_vs_target"] = pvt
        latest["price_gap_usd"]   = pgd
        latest["gap_vs_atr"]      = pgd / atr_val
        latest["gap_pct_abs"]     = abs(pvt)
        latest["early_momentum"]  = pvt
        latest["rsi_vs_neutral"]  = float(latest["rsi_14"]) - 50

        return latest[features].values, float(latest["rolling_std_60"])
    except Exception as e:
        log.error(f"compute_features: {e}")
        return None, None


def kelly_fraction(win_prob, payout):
    if payout <= 0: return 0.05
    k = (win_prob * payout - (1 - win_prob)) / payout
    return round(max(0.01, min(k, 0.25)), 3)

# ─────────────────────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────────────────────
def get_signal(market, round_start_price, now):
    """
    Compute signal. Always returns a result dict.
    Caller sends to Telegram only if result["signal"] == "✅ TRADE".
    Caller places order only if auto-buy is enabled and signal is ✅ TRADE.
    """
    rules          = market.get("rules", "")
    end_dt         = parse_end_dt(rules, now)
    btc            = get_btc_price()
    secs_remaining = (end_dt - now).total_seconds() if end_dt else 999
    mins_remaining = round(secs_remaining / 60, 1) if end_dt else None

    price_diff     = (btc - round_start_price) if btc else 0
    price_diff_pct = (price_diff / round_start_price * 100) if round_start_price else 0
    abs_gap        = abs(price_diff_pct)

    yes_p  = market["yes_price"]
    no_p   = market["no_price"]
    liq    = market.get("liquidity", 0)
    orders = market.get("total_orders", 0)
    hour   = now.hour

    feat_vals, vol_regime = compute_features(round_start_price)
    if feat_vals is None:
        return None

    feat_df      = pd.DataFrame([feat_vals], columns=features)
    raw_proba    = model.predict_proba(feat_df)[0][1]
    direction_up = raw_proba > 0.5
    direction_s  = "UP" if direction_up else "DOWN"
    direction    = "BUY YES (UP) 📈" if direction_up else "BUY NO (DOWN) 📉"

    # Directional confidence — always from the predicted side's view
    dir_conf  = raw_proba if direction_up else (1.0 - raw_proba)
    high_conv = dir_conf >= CONFIDENCE

    outcome_price = yes_p if direction_up else (1.0 - yes_p)
    outcome_price = max(outcome_price, 0.01)
    payout        = (1.0 / outcome_price - 1) * (1 - FEE)
    ev            = dir_conf * payout - (1 - dir_conf)
    kelly         = kelly_fraction(dir_conf, payout)

    # Tiered stake guide
    if abs_gap >= 0.20:   stake_tier, stake_note = "3× base", "gap ≥ 0.2%"
    elif abs_gap >= 0.10: stake_tier, stake_note = "2× base", "gap 0.1–0.2%"
    else:                 stake_tier, stake_note = "1× base", "gap 0.05–0.1%"

    # ── Filters ───────────────────────────────────────────────
    skip_reasons = []

    if hour in BLOCKED_HOURS:
        skip_reasons.append(f"Hour {hour:02d}:00 UTC blocked ({hour:02d}:00 = confirmed bad WR)")
        stats["skipped_hour"] += 1

    if not high_conv:
        skip_reasons.append(f"Conf {dir_conf:.1%} {direction_s} < {CONFIDENCE:.0%}")
        stats["skipped_conf"] += 1

    if abs_gap < MIN_GAP_PCT:
        skip_reasons.append(f"Gap {abs_gap:.3f}% < {MIN_GAP_PCT:.2f}% min")
        stats["skipped_gap"] += 1

    if outcome_price > MAX_OUTCOME_PRICE:
        skip_reasons.append(
            f"Odds {outcome_price:.0%} > {MAX_OUTCOME_PRICE:.0%} cap "
            f"(payout only {payout:.2f}x)"
        )
        stats["skipped_max_odds"] += 1

    if liq < MIN_LIQUIDITY:
        skip_reasons.append(f"Liq ${liq:.0f} < ${MIN_LIQUIDITY:.0f}")
        stats["skipped_liquidity"] += 1

    if orders < MIN_ORDERS:
        skip_reasons.append(f"Only {orders} orders (min {MIN_ORDERS})")

    if vol_regime and vol_regime > 0.003:
        skip_reasons.append("Volatility too high")

    if 0 < secs_remaining < 180:
        skip_reasons.append(f"Only {secs_remaining:.0f}s left")

    if circuit_broken:
        skip_reasons.append(f"Circuit break — {consec_losses} straight losses. /play to reset.")

    if AUTO_BUY_ENABLED and balance < BALANCE_MIN:
        skip_reasons.append(f"Balance ${balance:.2f} below stop threshold ${BALANCE_MIN:.2f}")

    signal = "⛔ NO TRADE" if skip_reasons else "✅ TRADE"
    reason = (
        " | ".join(skip_reasons) if skip_reasons else
        f"Model: {dir_conf:.1%} {direction_s} | "
        f"Gap: ${price_diff:+.2f} ({price_diff_pct:+.3f}%) | "
        f"Odds {outcome_price:.0%} → payout {payout:.2f}x | "
        f"EV: {ev:+.3f}"
    )

    return {
        "signal"           : signal,
        "direction"        : direction,
        "direction_short"  : direction_s,
        "conviction_label" : f"{'⚡ HIGH' if high_conv else '⚠️ LOW'} — {dir_conf:.1%} {direction_s}",
        "high_conviction"  : high_conv and (not skip_reasons),
        "confidence"       : round(dir_conf, 4),
        "raw_proba"        : round(raw_proba, 4),
        "btc_price"        : btc,
        "round_start_price": round_start_price,
        "price_diff"       : round(price_diff, 2),
        "price_diff_pct"   : round(price_diff_pct, 4),
        "abs_gap_pct"      : abs_gap,
        "mins_remaining"   : mins_remaining,
        "yes_price"        : yes_p,
        "no_price"         : no_p,
        "outcome_price"    : outcome_price,
        "payout"           : round(payout, 3),
        "ev"               : round(ev, 4),
        "kelly"            : kelly,
        "stake_tier"       : stake_tier,
        "stake_note"       : stake_note,
        "total_orders"     : orders,
        "liquidity"        : liq,
        "odds_filter_pass" : (outcome_price <= MAX_OUTCOME_PRICE),
        "reason"           : reason,
        "timestamp"        : now.isoformat(),
        "hour_utc"         : now.hour,
    }

# ─────────────────────────────────────────────────────────────
# MESSAGES
# ─────────────────────────────────────────────────────────────
def msg_open_alert(market, btc, target, now, mins_remaining):
    diff     = btc - target
    diff_pct = (diff / target * 100) if target else 0
    yes      = market.get("yes_price", 0)
    no       = market.get("no_price", 0)

    if last_completed["resolved"] and last_completed["round_start_price"]:
        lc = last_completed
        l_diff = (lc["btc_at_end"] - lc["round_start_price"]) if lc["btc_at_end"] else 0
        l_dir  = "UP 📈" if l_diff >= 0 else "DOWN 📉"
        if lc["correct"] is True:    result_txt = "✅ WON"
        elif lc["correct"] is False: result_txt = "❌ LOST"
        else:                        result_txt = "— no signal"
        bot_dir = lc.get("bot_direction") or "—"
        wr      = f"{stats['correct']/stats['signalled']*100:.1f}%" if stats["signalled"] > 0 else "N/A"
        last_block = (
            f"{'─'*30}\n"
            f"📋 *Last Round*\n"
            f"  Target : ${lc['round_start_price']:,.2f}\n"
            f"  Closed : ${lc['btc_at_end']:,.2f} → {l_dir}\n"
            f"  Bot    : {bot_dir} → {result_txt}\n"
            f"  Signals: {stats['signalled']} | WR {wr}\n"
        )
    else:
        last_block = f"{'─'*30}\n📋 *Last Round:* No data yet\n"

    bal_line = f"\n💰 Balance: *${balance:.2f}*" if AUTO_BUY_ENABLED else ""
    extra = ""
    if circuit_broken:
        extra = f"\n⚡ *Circuit break — {consec_losses} straight losses*\n/play to resume."
    elif bot_paused:
        extra = "\n⏸ *Bot paused*"
    elif AUTO_BUY_ENABLED and balance < BALANCE_MIN:
        extra = f"\n🛑 *Balance ${balance:.2f} below stop threshold — no orders will fire*"

    return (
        f"🔔 *NEW BTC ROUND — Session #{session_num}*\n"
        f"{'─'*30}\n"
        f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ ~{mins_remaining} mins\n"
        f"{'─'*30}\n"
        f"💰 BTC Now : ${btc:,.2f}\n"
        f"🎯 Target  : ${target:,.2f}\n"
        f"💹 Gap     : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
        f"📊 Bayse   : YES {yes:.0%} | NO {no:.0%}\n"
        f"{'─'*30}\n"
        + last_block
        + f"{'─'*30}\n"
        f"⏳ Signal in ~{SIGNAL_DELAY_MINS} mins..."
        + bal_line
        + extra
    )


def msg_signal(result, now, order_id=None):
    """Only called for ✅ TRADE."""
    reason_lines = "\n".join([f"  • {x}" for x in result["reason"].split(" | ")])
    multiplier   = 3 if "3×" in result["stake_tier"] else 2 if "2×" in result["stake_tier"] else 1

    if AUTO_BUY_ENABLED:
        if order_id:
            order_block = (
                f"\n{'─'*30}\n"
                f"🤖 *Auto-buy placed*\n"
                f"  Stake    : ${STAKE:.2f}\n"
                f"  Balance  : ${balance:.2f}\n"
                f"  Order ID : `{order_id}`"
            )
        else:
            order_block = (
                f"\n{'─'*30}\n"
                f"⚠️ *Auto-buy FAILED* — placed manually if needed\n"
                f"  Stake guide: ${STAKE:.2f} on {result['direction']}"
            )
    else:
        order_block = (
            f"\n{'─'*30}\n"
            f"💡 *Stake guide*  _({result['stake_note']})_\n"
            f"  {result['stake_tier']}\n"
            f"  $10 base → *${10 * multiplier}*  |  $20 base → *${20 * multiplier}*\n"
            f"  Kelly: {result['kelly']:.1%} of bankroll\n"
            f"  EV/dollar: {result['ev']:+.3f}"
        )

    return (
        f"🤖 *SIGNAL — {now.strftime('%H:%M UTC')}*\n"
        f"{'─'*30}\n"
        f"💰 BTC    : ${result['btc_price']:,.2f}\n"
        f"🎯 Target : ${result['round_start_price']:,.2f}\n"
        f"💹 Gap    : ${result['price_diff']:+.2f} ({result['price_diff_pct']:+.3f}%)\n"
        f"⏱ Left   : {result['mins_remaining']} mins\n"
        f"{'─'*30}\n"
        f"📊 *Bayse*\n"
        f"   YES (UP)  : {result['yes_price']:.0%}\n"
        f"   NO (DOWN) : {result['no_price']:.0%}\n"
        f"   Outcome   : {result['outcome_price']:.0%} → payout {result['payout']:.2f}x\n"
        f"   Orders    : {result['total_orders']} | Liq: ${result['liquidity']:,.2f}\n"
        f"{'─'*30}\n"
        f"🔔 *✅ TRADE*\n"
        f"📈 Direction  : {result['direction']}\n"
        f"⚡ Conviction : {result['conviction_label']}\n"
        f"{'─'*30}\n"
        f"💡 *Why:*\n{reason_lines}"
        + order_block
    )


def build_log_message():
    if not trade_log:
        return "📋 *Trade Log*\nNo signalled trades yet this session."
    lines = [f"📋 *Last 10 Signals — Session #{session_num}*\n" + "─"*30]
    for t in trade_log[-10:]:
        e    = "✅" if t["correct"] else "❌"
        conv = "⚡" if t.get("high_conviction") else "⚠️"
        tgt  = t.get("round_start_price", 0) or 0
        end  = t.get("btc_at_end", 0) or 0
        diff = end - tgt if (end and tgt) else 0
        op   = t.get("outcome_price", 0) or 0
        pay  = t.get("payout", 0) or 0
        pe   = t.get("payout_earned", "") 
        lines.append(
            f"{e}{conv} *{t['time']}*\n"
            f"  {t['bot_direction']} ({t['confidence']:.0%}) | Actual: {t['actual_direction']}\n"
            f"  Target ${tgt:,.2f} → Closed ${end:,.2f} ({diff:+.0f})\n"
            f"  Odds {op:.0%} | Pay {pay:.2f}x"
            + (f" | Earned ${pe}" if pe else "")
        )
    wr = f"{stats['correct']/stats['signalled']*100:.1f}%" if stats["signalled"] > 0 else "N/A"
    bal_line = f"\n💰 Balance: ${balance:.2f} | P&L: ${balance-STARTING_BALANCE:+.2f}" if AUTO_BUY_ENABLED else ""
    lines.append("─"*30)
    lines.append(f"📊 {stats['signalled']} signals | {wr} WR | ✅{stats['correct']} ❌{stats['incorrect']}" + bal_line)
    return "\n".join(lines)


def build_hour_stats():
    if not hour_stats:
        return "📊 *Hourly stats*\nNo signalled trades yet."
    lines = [f"🕐 *Win Rate by Hour (UTC) — Session #{session_num}*\n" + "─"*30]
    for h in sorted(hour_stats.keys()):
        d   = hour_stats[h]
        wr  = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        bar = "█" * int(wr / 10)
        blk = " 🚫" if h in BLOCKED_HOURS else ""
        flag = "🔥" if wr==100 else "✅" if wr>=75 else "⚠️" if wr>=52.6 else "❌"
        lines.append(f"  {h:02d}:00{blk} {flag} {bar:<10} {wr:.0f}% ({d['correct']}/{d['total']})")
    lines.append("─"*30)
    lines.append(f"Break-even ≈ 52.6%  |  Blocked: {sorted(BLOCKED_HOURS)} UTC")
    return "\n".join(lines)


def build_stats_message():
    now        = datetime.now(timezone.utc)
    sig_wr     = f"{stats['correct']/stats['signalled']*100:.1f}%" if stats["signalled"] > 0 else "N/A"
    all_wr     = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    hcwr       = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
                  if stats["high_conv_total"] > 0 else "N/A")
    last_10    = trade_log[-10:]
    l10c       = sum(1 for t in last_10 if t["correct"])
    l10r       = f"{l10c/len(last_10)*100:.1f}%" if last_10 else "N/A"
    one_hr_ago = now - timedelta(hours=1)
    last_1h    = [t for t in trade_log if datetime.fromisoformat(t["timestamp"]) > one_hr_ago]
    l1hc       = sum(1 for t in last_1h if t["correct"])
    l1hr       = f"{l1hc/len(last_1h)*100:.1f}%" if last_1h else "N/A"
    lines = []
    for t in last_10:
        e = "✅" if t["correct"] else "❌"
        c = "⚡" if t.get("high_conviction") else "⚠️"
        lines.append(f"{e}{c} {t['time']} | {t['bot_direction']} | {t['confidence']:.0%} | {t['actual_direction']}")
    table = "\n".join(lines) if lines else "No trades yet"
    cb    = (f"⚡ Active ({consec_losses}/{MAX_CONSEC_LOSSES})"
             if circuit_broken else f"✅ Clear ({consec_losses}/{MAX_CONSEC_LOSSES})")
    mode  = f"AUTO-BUY ${STAKE:.2f}" if AUTO_BUY_ENABLED else "Notify only"
    bal_block = (
        f"{'─'*30}\n"
        f"💰 *Bankroll*\n"
        f"  Balance  : ${balance:.2f} (started ${STARTING_BALANCE:.2f})\n"
        f"  P&L      : ${balance-STARTING_BALANCE:+.2f} ({(balance-STARTING_BALANCE)/STARTING_BALANCE*100:+.1f}%)\n"
        f"  Orders ok: {stats['orders_placed']} | Failed: {stats['orders_failed']}\n"
    ) if AUTO_BUY_ENABLED else ""

    return (
        f"📊 *BOT STATS — Session #{session_num}*\n"
        f"{'─'*30}\n"
        f"🕐 {now.strftime('%H:%M UTC')} | Since {session_start.strftime('%m-%d %H:%M UTC')}\n"
        f"Mode: {mode}\n"
        f"{'─'*30}\n"
        f"📈 Signals   : {stats['signalled']} | {sig_wr} WR\n"
        f"   ✅ {stats['correct']} correct | ❌ {stats['incorrect']} wrong\n"
        f"📊 All rounds: {stats['total']} computed | {all_wr} WR\n"
        f"⏱ Last hour : {len(last_1h)} | {l1hr} WR\n"
        f"🔟 Last 10   : {l10r} WR\n"
        f"⚡ High conv : {stats['high_conv_total']} | {hcwr} WR\n"
        f"{'─'*30}\n"
        + bal_block +
        f"{'─'*30}\n"
        f"🚫 Skipped — conf:{stats['skipped_conf']} gap:{stats['skipped_gap']} "
        f"odds:{stats['skipped_max_odds']} liq:{stats['skipped_liquidity']} "
        f"hour:{stats['skipped_hour']}\n"
        f"⚡ Circuit breaks: {stats['circuit_breaks']} | Status: {cb}\n"
        f"{'─'*30}\n"
        f"⏸ Paused: {'Yes' if bot_paused else 'No'}\n"
        f"{'─'*30}\n"
        f"`{table}`"
    )


def build_final_report():
    wr   = f"{stats['correct']/stats['signalled']*100:.1f}%" if stats["signalled"] > 0 else "N/A"
    hcwr = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
            if stats["high_conv_total"] > 0 else "N/A")
    duration = datetime.now(timezone.utc) - session_start
    hours    = int(duration.total_seconds() / 3600)
    mins     = int((duration.total_seconds() % 3600) / 60)
    lines = []
    for t in trade_log[-30:]:
        e = "✅" if t["correct"] else "❌"
        c = "⚡" if t.get("high_conviction") else "⚠️"
        lines.append(f"{e}{c} {t['time']} | {t['bot_direction']} | {t['confidence']:.0%} | {t['actual_direction']}")
    pnl_block = (
        f"  Balance     : ${balance:.2f} (started ${STARTING_BALANCE:.2f})\n"
        f"  P&L         : ${balance-STARTING_BALANCE:+.2f} ({(balance-STARTING_BALANCE)/STARTING_BALANCE*100:+.1f}%)\n"
        f"  Orders      : {stats['orders_placed']} placed | {stats['orders_failed']} failed\n"
    ) if AUTO_BUY_ENABLED else ""
    return (
        f"📊 *SESSION #{session_num} FINAL REPORT*\n"
        f"{'─'*30}\n"
        f"📅 {session_start.strftime('%Y-%m-%d %H:%M')} → "
        f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
        f"⏱ Duration   : {hours}h {mins}m\n"
        f"{'─'*30}\n"
        f"  Signals     : {stats['signalled']}\n"
        f"  All rounds  : {stats['total']}\n"
        f"  Win rate    : {wr}\n"
        f"  ✅ {stats['correct']} | ❌ {stats['incorrect']}\n"
        f"  High conv   : {stats['high_conv_total']} | {hcwr} WR\n"
        f"  Skipped     : conf {stats['skipped_conf']} | gap {stats['skipped_gap']} | "
        f"odds {stats['skipped_max_odds']} | liq {stats['skipped_liquidity']} | "
        f"hour {stats['skipped_hour']}\n"
        f"  Circ breaks : {stats['circuit_breaks']}\n"
        + pnl_block +
        f"{'─'*30}\n"
        f"📋 *Last 30*\n`{'chr(10)'.join(lines) if lines else 'No trades'}`"
    )

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main():
    global current_market_cache, last_report_time
    global circuit_broken, consec_losses, balance, total_staked, total_profit

    init_log()
    seed_candles()
    tg_send(start_message())
    mode = f"AUTO-BUY ${STAKE:.2f}/trade | buffer ${STARTING_BALANCE:.2f}" if AUTO_BUY_ENABLED else "NOTIFY ONLY — set BAYSE_SECRET_KEY to enable auto-buy"
    tg_send(
        f"🟢 *Bot v6 online — Session #{session_num}*\n"
        f"📅 {session_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Mode: *{mode}*\n\n"
        f"Active filters:\n"
        f"  ✅ Conf ≥ {CONFIDENCE:.0%}\n"
        f"  ✅ Gap ≥ {MIN_GAP_PCT:.2f}%\n"
        f"  ✅ Outcome ≤ {MAX_OUTCOME_PRICE:.0%}\n"
        f"  ✅ Block hours {sorted(BLOCKED_HOURS)} UTC\n"
        f"  ✅ Bayse-native prices (start/finalPrice)\n"
        f"  ✅ Silent CSV | Circuit break @{MAX_CONSEC_LOSSES}"
    )
    log.info(f"Bot v6 started | auto_buy={AUTO_BUY_ENABLED} | stake=${STAKE:.2f}")

    while True:
        try:
            handle_commands()
            update_candles()

            if not session_active:
                time.sleep(30)
                continue

            now    = datetime.now(timezone.utc)
            market = fetch_btc_market()

            if not market:
                log.info("No BTC market — retrying in 30s")
                time.sleep(30)
                continue

            current_market_cache = market
            event_id  = market["event_id"]
            btc       = get_btc_price()
            rules     = market.get("rules", "")
            end_dt    = parse_end_dt(rules, now)
            mins_left = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None

            # 12h periodic report
            if ((now - last_report_time).total_seconds() / 3600 >= REPORT_HOURS
                    and stats["signalled"] > 0):
                tg_send(build_final_report())
                last_report_time = now

            # ── NEW ROUND ─────────────────────────────────────
            if event_id != current_round["event_id"]:
                log.info(f"New round: {event_id}")

                prev_signal = current_round.get("pending_signal")
                prev_target = current_round.get("round_start_price")

                # ── RESOLVE PREVIOUS ROUND ───────────────────
                if prev_target:
                    prev_eid  = current_round.get("prev_event_id")
                    final_btc = None

                    if prev_eid:
                        final_btc = fetch_event_final_price(prev_eid)
                        if final_btc:
                            log.info(f"Resolved via Bayse finalPrice=${final_btc:,.2f} (event {prev_eid})")
                        else:
                            log.warning(f"Bayse finalPrice not yet available for {prev_eid}")

                    if not final_btc:
                        final_btc = market.get("final_price")
                        if final_btc:
                            log.info(f"Resolved via market.final_price=${final_btc:,.2f}")

                    if not final_btc:
                        final_btc = btc
                        log.warning(f"Bayse finalPrice unavailable — KuCoin fallback=${final_btc:,.2f}")

                    if not final_btc:
                        log.error("No final price — skipping resolution")
                    else:
                        actual_up  = final_btc >= prev_target
                        actual_dir = "UP" if actual_up else "DOWN"

                        if prev_signal:
                            predicted_up  = prev_signal["direction_short"] == "UP"
                            correct       = predicted_up == actual_up
                            high_conv     = prev_signal["high_conviction"]
                            hour_h        = prev_signal.get("hour_utc", now.hour)
                            was_signalled = (prev_signal["signal"] == "✅ TRADE")
                            order_was_placed = bool(current_round.get("order_id"))

                            # ── Update bankroll ───────────────
                            if was_signalled and AUTO_BUY_ENABLED and order_was_placed:
                                earned = round(STAKE * prev_signal["payout"], 2) if correct else 0
                                pnl_trade = round(earned - STAKE, 2)
                                balance   = round(balance + pnl_trade, 2)
                                total_staked  += STAKE
                                total_profit  += earned
                                log.info(f"Bankroll: ${balance:.2f} | trade P&L ${pnl_trade:+.2f}")
                            else:
                                earned = None

                            if was_signalled:
                                stats["signalled"] += 1
                                stats["correct" if correct else "incorrect"] += 1
                                if high_conv:
                                    stats["high_conv_total"] += 1
                                    if correct: stats["high_conv_correct"] += 1
                                if hour_h not in hour_stats:
                                    hour_stats[hour_h] = {"total": 0, "correct": 0}
                                hour_stats[hour_h]["total"] += 1
                                if correct: hour_stats[hour_h]["correct"] += 1

                                # Circuit breaker
                                if correct:
                                    consec_losses = 0
                                    if circuit_broken:
                                        circuit_broken = False
                                        tg_send("✅ *Win — circuit break cleared. Signals resuming.*")
                                else:
                                    consec_losses += 1
                                    if consec_losses >= MAX_CONSEC_LOSSES and not circuit_broken:
                                        circuit_broken = True
                                        stats["circuit_breaks"] += 1
                                        tg_send(
                                            f"⚡ *Circuit break triggered*\n"
                                            f"{'─'*28}\n"
                                            f"{MAX_CONSEC_LOSSES} consecutive losses.\n"
                                            f"Signals and orders paused.\n"
                                            f"Use /play to resume manually."
                                            + (f"\n💰 Balance: ${balance:.2f}" if AUTO_BUY_ENABLED else "")
                                        )

                            last_completed.update({
                                "round_start_price": prev_target,
                                "btc_at_end"       : final_btc,
                                "actual_direction" : actual_dir,
                                "bot_direction"    : prev_signal["direction_short"],
                                "correct"          : correct if was_signalled else None,
                                "resolved"         : True,
                            })

                            stats["total"] += 1
                            entry = {
                                "timestamp"        : now.isoformat(),
                                "time"             : now.strftime("%H:%M"),
                                "round_start_price": prev_target,
                                "btc_at_signal"    : prev_signal.get("btc_price", ""),
                                "btc_at_end"       : final_btc,
                                "bot_direction"    : prev_signal["direction_short"],
                                "actual_direction" : actual_dir,
                                "correct"          : correct,
                                "confidence"       : prev_signal["confidence"],
                                "conviction"       : "HIGH" if high_conv else "LOW",
                                "yes_price"        : prev_signal.get("yes_price", ""),
                                "no_price"         : prev_signal.get("no_price", ""),
                                "gap_pct"          : prev_signal.get("price_diff_pct", ""),
                                "odds_filter_pass" : prev_signal.get("odds_filter_pass", ""),
                                "liquidity"        : prev_signal.get("liquidity", ""),
                                "orders"           : prev_signal.get("total_orders", ""),
                                "session"          : session_num,
                                "signalled"        : was_signalled,
                                "order_placed"     : order_was_placed,
                                "order_id"         : current_round.get("order_id", ""),
                                "stake"            : STAKE if (was_signalled and order_was_placed) else "",
                                "payout_earned"    : earned if earned is not None else "",
                                "high_conviction"  : high_conv,
                                "outcome_price"    : prev_signal.get("outcome_price", ""),
                                "payout"           : prev_signal.get("payout", ""),
                            }
                            write_log(entry)
                            if was_signalled:
                                trade_log.append(entry)
                            log.info(
                                f"Closed: {'✅' if correct else '❌'} | "
                                f"signalled={was_signalled} | order={order_was_placed} | "
                                f"consec={consec_losses} | balance=${balance:.2f}"
                            )

                        else:
                            # No pending signal this round
                            last_completed.update({
                                "round_start_price": prev_target,
                                "btc_at_end"       : final_btc,
                                "actual_direction" : actual_dir,
                                "bot_direction"    : None,
                                "correct"          : None,
                                "resolved"         : True,
                            })
                            stats["total"] += 1
                            write_log({
                                "timestamp"        : now.isoformat(),
                                "time"             : now.strftime("%H:%M"),
                                "round_start_price": prev_target,
                                "btc_at_signal"    : "",
                                "btc_at_end"       : final_btc,
                                "bot_direction"    : "",
                                "actual_direction" : actual_dir,
                                "correct"          : "",
                                "confidence"       : "",
                                "conviction"       : "",
                                "yes_price"        : "",
                                "no_price"         : "",
                                "gap_pct"          : "",
                                "odds_filter_pass" : "",
                                "liquidity"        : "",
                                "orders"           : "",
                                "session"          : session_num,
                                "signalled"        : False,
                                "order_placed"     : False,
                                "order_id"         : "",
                                "stake"            : "",
                                "payout_earned"    : "",
                            })

                # ── SET UP NEW ROUND ─────────────────────────
                new_target = market.get("start_price")
                if new_target:
                    log.info(f"Target from Bayse start_price: ${new_target:,.2f}")
                elif btc:
                    new_target = btc
                    log.warning(f"Bayse start_price missing — KuCoin fallback: ${new_target:,.2f}")
                else:
                    new_target = parse_target(rules)
                    log.warning(f"All price sources failed — parse_target: {new_target}")

                if not new_target:
                    log.error(f"Could not determine target for event {event_id}")

                current_round.update({
                    "event_id"         : event_id,
                    "prev_event_id"    : current_round.get("event_id"),
                    "start_time"       : now,
                    "round_start_price": new_target,
                    "signal_fired"     : False,
                    "pending_signal"   : None,
                    "order_id"         : None,
                })

                if new_target and btc and mins_left:
                    tg_send(msg_open_alert(market, btc, new_target, now, mins_left))
                    log.info(f"Open alert | target=${new_target:,.2f} | BTC=${btc:,.2f}")

            # ── FIRE SIGNAL (5 mins into round) ──────────────
            start_time = current_round.get("start_time")
            can_fire   = (
                not current_round["signal_fired"]
                and start_time is not None
                and current_round["round_start_price"] is not None
            )
            if can_fire:
                mins_in = (now - start_time).total_seconds() / 60
                if mins_in >= SIGNAL_DELAY_MINS:
                    result = get_signal(market, current_round["round_start_price"], now)
                    if result:
                        current_round["pending_signal"] = result
                        current_round["signal_fired"]   = True

                        if result["signal"] == "✅ TRADE" and not bot_paused and not circuit_broken:
                            order_id = None

                            # ── AUTO-BUY ─────────────────────
                            if AUTO_BUY_ENABLED and balance >= BALANCE_MIN:
                                market_id = get_market_id(market)
                                if market_id:
                                    outcome  = "YES" if result["direction_short"] == "UP" else "NO"
                                    order_id = place_order(market_id, outcome, STAKE)
                                    if order_id:
                                        current_round["order_id"] = order_id
                                        stats["orders_placed"] += 1
                                    else:
                                        stats["orders_failed"] += 1
                                        log.warning("Auto-buy failed — signal sent but no order placed")
                                else:
                                    log.warning("Could not extract market_id — order skipped")
                                    stats["orders_failed"] += 1

                            tg_send(msg_signal(result, now, order_id))
                            log.info(
                                f"Signal ✅ | {result['direction_short']} | "
                                f"conf {result['confidence']:.1%} | "
                                f"gap {result['abs_gap_pct']:.3f}% | "
                                f"op {result['outcome_price']:.0%} | "
                                f"payout {result['payout']:.2f}x | "
                                f"order={'placed '+str(order_id) if order_id else 'not placed'}"
                            )
                        else:
                            log.info(
                                f"Signal ⛔ (silent) | {result['direction_short']} | "
                                f"conf {result['confidence']:.1%} | {result['reason'][:60]}"
                            )
                    else:
                        log.warning("Features not ready — not enough candles yet")

            time.sleep(30)

        except Exception as e:
            log.error(f"Main loop: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
