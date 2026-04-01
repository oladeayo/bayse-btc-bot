"""
Bayse BTC Prediction Bot — v5
==============================
Changes from v3 (uploaded):

1.  CONFIDENCE raised 0.70 → 0.80
    The 60–80% band was 41–59% WR across 189 trades — worse than random after fee.

2.  MIN_GAP_PCT raised 0.02 → 0.05
    Below 0.05% gap = 47–62% WR. Above 0.05% = 74–96% WR.

3.  OLD odds filter (ODDS_MIN/ODDS_MAX) REPLACED by MAX_OUTCOME_PRICE = 0.90
    The old filter blocked low-odds trades (your best payouts).
    The new cap blocks high-odds trades where payout ≈ $0.05 per $1.
    At 90% odds you need 9.5 wins to recover 1 loss — not worth it.

4.  SILENT CSV LOGGING — every round logged, Telegram only on ✅ TRADE
    All 189 rounds per session are written to CSV for analysis.
    Only the ~40–50% that pass all filters send a Telegram notification.
    This kills the noise and keeps data complete.

5.  TARGET PRICE BUG FIXED
    parse_target() called once at new event_id, frozen for the round.
    Never re-parsed mid-round — prevents target drift between rounds.

6.  DIRECTIONAL CONFIDENCE DISPLAY FIXED
    DOWN signal with raw_proba=0.138 now shows "86.2% DOWN" not "13.8%".

7.  CIRCUIT BREAKER — pause after MAX_CONSEC_LOSSES straight losses
    Auto-clears on next win. /play also resets it.

8.  TIERED STAKE GUIDE in every ✅ TRADE signal
    Based on gap size (the strongest predictor in all data):
    gap 0.05–0.1% → 1× base | gap 0.1–0.2% → 2× base | gap ≥0.2% → 3× base

9.  Stats: removed skipped_odds, added skipped_conf + skipped_max_odds + circuit_breaks

10. /play resets circuit break in addition to unpausing

No timezone/hour blocking — not enough data yet to confirm consistent pattern.
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
BAYSE_KEY         = os.environ.get("BAYSE_KEY", "")
BASE_URL          = "https://relay.bayse.markets/v1"
BAYSE_HEADERS     = {"X-Public-Key": BAYSE_KEY}
TELEGRAM_API      = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Strategy — every value here is backed by data across 300+ trades
CONFIDENCE        = 0.80    # raised from 0.70; 60–80% band = 41–59% WR
MIN_GAP_PCT       = 0.05    # raised from 0.02; below 0.05% = 47–62% WR
MAX_OUTCOME_PRICE = 0.90    # new; above 90% odds → payout <0.11x, need 9+ wins/loss
MIN_LIQUIDITY     = 50.0
MIN_ORDERS        = 10
FEE               = 0.05
SIGNAL_DELAY_MINS = 5
MAX_CONSEC_LOSSES = 3       # circuit breaker threshold
REPORT_HOURS      = 6      # send auto-report every N hours
LOG_FILE          = "trade_log.csv"

# ─────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT or not BAYSE_KEY:
    print("ERROR: Missing env vars (TELEGRAM_TOKEN, TELEGRAM_CHAT, BAYSE_KEY)", flush=True)
    sys.exit(1)
print(f"✅ Token: {TELEGRAM_TOKEN[:10]}...", flush=True)
print(f"✅ Chat: {TELEGRAM_CHAT}", flush=True)

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

# Target price fix: round_start_price is set ONCE when event_id changes,
# then frozen for the entire round. Never re-parsed mid-round.
current_round = {
    "event_id"         : None,
    "start_time"       : None,
    "round_start_price": None,
    "signal_fired"     : False,
    "pending_signal"   : None,
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

# Stats track both signalled trades AND all-rounds for calibration
stats = {
    "total"            : 0,      # rounds where a signal was computed (all filtered)
    "signalled"        : 0,      # rounds where ✅ TRADE was sent
    "correct"          : 0,
    "incorrect"        : 0,
    "high_conv_total"  : 0,
    "high_conv_correct": 0,
    "skipped_conf"     : 0,
    "skipped_gap"      : 0,
    "skipped_max_odds" : 0,
    "skipped_liquidity": 0,
    "circuit_breaks"   : 0,
}
trade_log  = []   # only signalled (✅ TRADE) trades — for in-memory display
hour_stats = {}   # {hour: {"total": N, "correct": N}} — all signalled trades

# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────
def tg_send(text, chat_id=None):
    try:
        r = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id or TELEGRAM_CHAT,
                  "text": text, "parse_mode": "Markdown"},
            timeout=10
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
            timeout=15
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
# COMMANDS
# ─────────────────────────────────────────────────────────────
def handle_commands():
    global bot_paused, circuit_broken, consec_losses
    global session_active, session_start, session_num
    global stats, trade_log, hour_stats, last_completed, current_round

    for update in tg_get_updates():
        msg     = update.get("message", {})
        text    = msg.get("text", "").strip().lower()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if text.startswith("/pause"):
            bot_paused = True
            tg_send("⏸ *Bot paused.* No signals will fire.\nUse /play to resume.", chat_id)

        elif text.startswith("/play"):
            bot_paused     = False
            circuit_broken = False
            consec_losses  = 0
            tg_send("▶️ *Bot resumed.* Signals active. Circuit break cleared.", chat_id)

        elif text.startswith("/start_session"):
            session_num   += 1
            session_start  = datetime.now(timezone.utc)
            session_active = True
            bot_paused     = False
            circuit_broken = False
            consec_losses  = 0
            stats = {k: 0 for k in stats}
            trade_log  = []
            hour_stats = {}
            last_completed = {k: None for k in last_completed}
            last_completed["resolved"] = False
            current_round = {"event_id": None, "start_time": None,
                             "round_start_price": None,
                             "signal_fired": False, "pending_signal": None}
            tg_send(
                f"🆕 *Session {session_num} started*\n"
                f"📅 {session_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Stats and log cleared.\n\n"
                f"Active filters:\n"
                f"  • Conf ≥ {CONFIDENCE:.0%}\n"
                f"  • Gap ≥ {MIN_GAP_PCT:.2f}%\n"
                f"  • Outcome price ≤ {MAX_OUTCOME_PRICE:.0%}\n"
                f"  • Circuit break after {MAX_CONSEC_LOSSES} losses\n"
                f"  • All rounds logged to CSV silently",
                chat_id
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
    return (
        "🤖 *Bayse BTC Bot v5*\n\n"
        "Commands:\n"
        "  /price         — BTC vs target + Bayse odds\n"
        "  /stats         — full performance summary\n"
        "  /log           — last 10 signalled trades\n"
        "  /hours         — win rate by UTC hour\n"
        "  /config        — current strategy settings\n"
        "  /export        — download full session CSV\n"
        "  /pause         — pause signals\n"
        "  /play          — resume + reset circuit break\n"
        "  /start_session — new session (clears stats)\n"
        "  /stop_session  — end session + final report\n\n"
        f"v5 | conf≥{CONFIDENCE:.0%} | gap≥{MIN_GAP_PCT:.2f}% | "
        f"op≤{MAX_OUTCOME_PRICE:.0%} | silent CSV | circuit @{MAX_CONSEC_LOSSES}"
    )


def build_config_message():
    cb = (f"⚡ ACTIVE ({consec_losses}/{MAX_CONSEC_LOSSES})"
          if circuit_broken else f"✅ Clear ({consec_losses}/{MAX_CONSEC_LOSSES})")
    return (
        f"⚙️ *Bot v5 Config*\n"
        f"{'─'*30}\n"
        f"  Confidence     : ≥{CONFIDENCE:.0%}  _(was 70%)_\n"
        f"  Min gap        : ≥{MIN_GAP_PCT:.2f}%  _(was 0.02%)_\n"
        f"  Max odds (op)  : ≤{MAX_OUTCOME_PRICE:.0%}  _(replaces old range filter)_\n"
        f"  Min liquidity  : ${MIN_LIQUIDITY:.0f}\n"
        f"  Min orders     : {MIN_ORDERS}\n"
        f"  Signal delay   : {SIGNAL_DELAY_MINS} mins into round\n"
        f"  Fee            : {FEE:.0%}\n"
        f"  Circuit break  : after {MAX_CONSEC_LOSSES} straight losses\n"
        f"  CSV logging    : all rounds (silent)\n"
        f"{'─'*30}\n"
        f"  Status         : {'⏸ Paused' if bot_paused else '▶️ Running'}\n"
        f"  Circuit        : {cb}\n"
        f"  Session        : #{session_num}\n"
        f"  Since          : {session_start.strftime('%Y-%m-%d %H:%M UTC')}"
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
        op_up    = yes
        op_dn    = 1.0 - yes
        odds_note = (
            f"{'✅' if op_up <= MAX_OUTCOME_PRICE else '⛔'} YES odds {op_up:.0%} "
            f"| {'✅' if op_dn <= MAX_OUTCOME_PRICE else '⛔'} NO odds {op_dn:.0%}"
        )
        cb_note = f"\n  ⚡ Circuit break: {consec_losses}/{MAX_CONSEC_LOSSES} losses" if circuit_broken else ""
        tg_send(
            f"💰 *BTC Snapshot*\n"
            f"{'─'*28}\n"
            f"  BTC now  : ${btc:,.2f}\n"
            f"  Target   : ${target:,.2f}\n"
            f"  Gap      : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
            f"  Status   : {'📈 ABOVE' if diff > 0 else '📉 BELOW'} target\n"
            f"{'─'*28}\n"
            f"  {odds_note}\n"
            f"  Orders   : {m.get('total_orders', 0)} | Liq: ${m.get('liquidity', 0):,.2f}\n"
            f"  Time     : {now.strftime('%H:%M:%S UTC')}\n"
            f"{'─'*28}\n"
            f"  Bot      : {'⏸ Paused' if bot_paused else '▶️ Running'}"
            + cb_note,
            chat_id
        )
    else:
        tg_send("⚠️ No active round data yet.", chat_id)


def cmd_export(chat_id):
    """Export the full CSV log (all rounds, including silent ones)."""
    if not os.path.exists(LOG_FILE):
        tg_send("No CSV log found yet.", chat_id)
        return
    with open(LOG_FILE, "rb") as f:
        content = f.read()
    count = content.count(b"\n") - 1  # subtract header
    filename = f"session_{session_num}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
    tg_send_document(
        filename, content,
        f"Session #{session_num} — {count} rounds (all, including silent)",
        chat_id
    )

# ─────────────────────────────────────────────────────────────
# CSV LOG  (all rounds, not just signalled ones)
# ─────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "timestamp", "time", "round_start_price", "btc_at_signal",
    "btc_at_end", "bot_direction", "actual_direction", "correct",
    "confidence", "conviction", "yes_price", "no_price",
    "gap_pct", "odds_filter_pass", "liquidity", "orders", "session",
    "signalled",   # NEW: True if ✅ TRADE was sent to Telegram
]

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)

def write_log(entry):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([entry.get(k, "") for k in CSV_FIELDS])

# ─────────────────────────────────────────────────────────────
# KUCOIN CANDLES
# ─────────────────────────────────────────────────────────────
def seed_candles():
    try:
        r = requests.get(
            "https://api.kucoin.com/api/v1/market/candles",
            params={"symbol": "BTC-USDT", "type": "1min"},
            timeout=10
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
            timeout=5
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
    try:
        r = requests.get(
            f"{BASE_URL}/pm/events",
            headers=BAYSE_HEADERS,
            params={"limit": 50},
            timeout=10
        )
        for event in r.json().get("events", []):
            if "bitcoin" in event.get("title", "").lower() and "15" in event.get("title", ""):
                r2     = requests.get(f"{BASE_URL}/pm/events/{event['id']}",
                                      headers=BAYSE_HEADERS, timeout=10)
                detail = r2.json()
                ms     = detail.get("markets", [])
                if ms:
                    m = ms[0]
                    return {
                        "event_id"    : detail["id"],
                        "yes_price"   : m.get("outcome1Price", 0),
                        "no_price"    : m.get("outcome2Price", 0),
                        "total_orders": m.get("totalOrders", 0),
                        "liquidity"   : detail.get("liquidity", 0),
                        "created_at"  : detail.get("createdAt", ""),
                        "rules"       : m.get("rules", ""),
                    }
    except Exception as e:
        log.error(f"fetch_btc_market: {e}")
    return None

def parse_target(rules):
    """
    Extract target price from Bayse rules string.
    Called ONCE per new event_id. Result frozen for the whole round.
    This prevents the target from drifting when the next round's rules
    are served before the event_id has updated.
    """
    m = re.findall(r'\$([\d,]+\.?\d*)', rules)
    if m:
        try:
            return float(m[0].replace(",", ""))
        except ValueError:
            pass
    return None

def parse_end_dt(rules, now):
    m = re.search(r'(\d+:\d+:\d+\s?[AP]M)\s?GMT', rules)
    if m:
        try:
            end = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} {m.group(1).strip()}",
                "%Y-%m-%d %I:%M:%S %p"
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
        df["stoch"]          = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
        df["macd"]           = ta.trend.MACD(df["close"]).macd_diff()
        df["macd_signal"]    = ta.trend.MACD(df["close"]).macd_signal()
        df["ema_9"]          = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema_21"]         = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
        df["ema_cross"]      = df["ema_9"] - df["ema_21"]
        bb                   = ta.volatility.BollingerBands(df["close"])
        df["bb_position"]    = (df["close"] - bb.bollinger_mavg()) / bb.bollinger_wband()
        df["bb_width"]       = bb.bollinger_wband()
        df["atr"]            = ta.volatility.AverageTrueRange(
                                   df["high"], df["low"], df["close"], window=14
                               ).average_true_range()
        df["vol_ratio_15"]   = df["volume"] / df["volume"].rolling(15).mean()
        df["vol_ratio_60"]   = df["volume"] / df["volume"].rolling(60).mean()
        df["obv"]            = ta.volume.OnBalanceVolumeIndicator(
                                   df["close"], df["volume"]
                               ).on_balance_volume()
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

        pvt = (latest["close"] - round_start_price) / round_start_price
        pgd = latest["close"] - round_start_price

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
    if payout <= 0:
        return 0.05
    k = (win_prob * payout - (1 - win_prob)) / payout
    return round(max(0.01, min(k, 0.25)), 3)

# ─────────────────────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────────────────────
def get_signal(market, round_start_price, now):
    """
    Compute signal for the current round.
    Always returns a result dict — whether ✅ TRADE or ⛔ NO TRADE.
    The caller decides whether to send to Telegram based on result["signal"].
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

    feat_vals, vol_regime = compute_features(round_start_price)
    if feat_vals is None:
        return None

    feat_df      = pd.DataFrame([feat_vals], columns=features)
    raw_proba    = model.predict_proba(feat_df)[0][1]   # P(UP)
    direction_up = raw_proba > 0.5
    direction_s  = "UP" if direction_up else "DOWN"
    direction    = "BUY YES (UP) 📈" if direction_up else "BUY NO (DOWN) 📉"

    # Directional confidence — always from the predicted side's perspective
    # DOWN signal: raw_proba=0.138 → dir_conf=0.862 → shows "86.2% DOWN"
    dir_conf  = raw_proba if direction_up else (1.0 - raw_proba)
    high_conv = dir_conf >= CONFIDENCE

    # Outcome price — the side you're actually buying
    outcome_price = yes_p if direction_up else (1.0 - yes_p)
    outcome_price = max(outcome_price, 0.01)
    payout        = (1.0 / outcome_price - 1) * (1 - FEE)
    ev            = dir_conf * payout - (1 - dir_conf)
    kelly         = kelly_fraction(dir_conf, payout)

    # Tiered stake guide based on gap size (strongest predictor in data)
    if abs_gap >= 0.20:
        stake_tier = "3× base"
        stake_note = "gap ≥ 0.2% — top tier"
    elif abs_gap >= 0.10:
        stake_tier = "2× base"
        stake_note = "gap 0.1–0.2% — strong"
    else:
        stake_tier = "1× base"
        stake_note = "gap 0.05–0.1% — standard"

    # ── Skip reasons (filters) ────────────────────────────────
    skip_reasons = []

    if not high_conv:
        skip_reasons.append(f"Conf {dir_conf:.1%} {direction_s} < {CONFIDENCE:.0%}")
        stats["skipped_conf"] += 1

    if abs_gap < MIN_GAP_PCT:
        skip_reasons.append(f"Gap {abs_gap:.3f}% < {MIN_GAP_PCT:.2f}% min")
        stats["skipped_gap"] += 1

    if outcome_price > MAX_OUTCOME_PRICE:
        skip_reasons.append(
            f"Outcome {outcome_price:.0%} > {MAX_OUTCOME_PRICE:.0%} cap "
            f"(payout only {payout:.2f}x)"
        )
        stats["skipped_max_odds"] += 1

    if liq < MIN_LIQUIDITY:
        skip_reasons.append(f"Liq ${liq:.0f} < ${MIN_LIQUIDITY:.0f} min")
        stats["skipped_liquidity"] += 1

    if orders < MIN_ORDERS:
        skip_reasons.append(f"Only {orders} orders (min {MIN_ORDERS})")

    if vol_regime and vol_regime > 0.003:
        skip_reasons.append("Volatility too high")

    if 0 < secs_remaining < 180:
        skip_reasons.append(f"Only {secs_remaining:.0f}s left")

    if circuit_broken:
        skip_reasons.append(
            f"Circuit break — {consec_losses} straight losses. /play to reset."
        )

    signal = "⛔ NO TRADE" if skip_reasons else "✅ TRADE"

    reason = (
        " | ".join(skip_reasons) if skip_reasons else
        f"Model: {dir_conf:.1%} {direction_s} | "
        f"Gap: ${price_diff:+.2f} ({price_diff_pct:+.3f}%) | "
        f"Outcome {outcome_price:.0%} → payout {payout:.2f}x | "
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
        lc     = last_completed
        l_diff = (lc["btc_at_end"] - lc["round_start_price"]) if lc["btc_at_end"] else 0
        l_dir  = "UP 📈" if l_diff >= 0 else "DOWN 📉"
        result = "✅ WON" if lc["correct"] else "❌ LOST"
        wr     = f"{stats['correct']/stats['signalled']*100:.1f}%" if stats["signalled"] > 0 else "N/A"
        last_block = (
            f"{'─'*30}\n"
            f"📋 *Last Round*\n"
            f"  Target : ${lc['round_start_price']:,.2f}\n"
            f"  Closed : ${lc['btc_at_end']:,.2f} → {l_dir}\n"
            f"  Bot    : {lc['bot_direction']} → {result}\n"
            f"  Signals: {stats['signalled']} | WR {wr}\n"
        )
    else:
        last_block = f"{'─'*30}\n📋 *Last Round:* No data yet\n"

    extra = ""
    if circuit_broken:
        extra = f"\n⚡ *Circuit break — {consec_losses} losses in a row*\nUse /play to resume."
    elif bot_paused:
        extra = "\n⏸ *Bot paused*"

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
        + extra
    )


def msg_signal(result, now):
    """Only called for ✅ TRADE signals."""
    reason_lines = "\n".join([f"  • {x}" for x in result["reason"].split(" | ")])
    multiplier   = 3 if "3×" in result["stake_tier"] else 2 if "2×" in result["stake_tier"] else 1

    trade_block = (
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
        + trade_block
    )


def build_log_message():
    """Shows last 10 signalled (✅ TRADE) trades."""
    if not trade_log:
        return "📋 *Trade Log*\nNo signalled trades yet this session."
    lines = [f"📋 *Last 10 Signals — Session #{session_num}*\n" + "─" * 30]
    for t in trade_log[-10:]:
        e    = "✅" if t["correct"] else "❌"
        conv = "⚡" if t.get("high_conviction") else "⚠️"
        tgt  = t.get("round_start_price", 0) or 0
        end  = t.get("btc_at_end", 0) or 0
        diff = end - tgt if (end and tgt) else 0
        lines.append(
            f"{e}{conv} *{t['time']}*\n"
            f"  {t['bot_direction']} ({t['confidence']:.0%}) | Actual: {t['actual_direction']}\n"
            f"  Target ${tgt:,.2f} → Closed ${end:,.2f} ({diff:+.0f})\n"
            f"  Outcome {t.get('outcome_price', 0):.0%} | "
            f"Payout {t.get('payout', 0):.2f}x | "
            f"YES {t.get('yes_price', 0):.0%}"
        )
    wr = f"{stats['correct']/stats['signalled']*100:.1f}%" if stats["signalled"] > 0 else "N/A"
    lines.append("─" * 30)
    lines.append(
        f"📊 {stats['signalled']} signals | {wr} WR | "
        f"✅{stats['correct']} ❌{stats['incorrect']}"
    )
    return "\n".join(lines)


def build_hour_stats():
    if not hour_stats:
        return "📊 *Hourly stats*\nNo signalled trades yet."
    lines = [f"🕐 *Win Rate by Hour (UTC) — Session #{session_num}*\n" + "─" * 30]
    for h in sorted(hour_stats.keys()):
        d   = hour_stats[h]
        wr  = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        bar = "█" * int(wr / 10)
        flag = "🔥" if wr == 100 else "✅" if wr >= 75 else "⚠️" if wr >= 52.6 else "❌"
        lines.append(f"  {h:02d}:00 {flag} {bar:<10} {wr:.0f}% ({d['correct']}/{d['total']})")
    lines.append("─" * 30)
    lines.append("Break-even ≈ 52.6%  |  Signals only (not all rounds)")
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
    last_1h    = [t for t in trade_log
                  if datetime.fromisoformat(t["timestamp"]) > one_hr_ago]
    l1hc       = sum(1 for t in last_1h if t["correct"])
    l1hr       = f"{l1hc/len(last_1h)*100:.1f}%" if last_1h else "N/A"
    lines = []
    for t in last_10:
        e = "✅" if t["correct"] else "❌"
        c = "⚡" if t.get("high_conviction") else "⚠️"
        lines.append(f"{e}{c} {t['time']} | {t['bot_direction']} | {t['confidence']:.0%} | {t['actual_direction']}")
    table = "\n".join(lines) if lines else "No trades yet"
    cb = (f"⚡ Active ({consec_losses}/{MAX_CONSEC_LOSSES})"
          if circuit_broken else f"✅ Clear ({consec_losses}/{MAX_CONSEC_LOSSES})")
    return (
        f"📊 *BOT STATS — Session #{session_num}*\n"
        f"{'─'*30}\n"
        f"🕐 {now.strftime('%H:%M UTC')} | Since {session_start.strftime('%m-%d %H:%M UTC')}\n"
        f"{'─'*30}\n"
        f"📈 Signals    : {stats['signalled']} | {sig_wr} WR\n"
        f"   ✅ {stats['correct']} correct | ❌ {stats['incorrect']} wrong\n"
        f"📊 All rounds : {stats['total']} computed | {all_wr} WR (inc silent)\n"
        f"⏱ Last hour  : {len(last_1h)} | {l1hr} WR\n"
        f"🔟 Last 10   : {l10r} WR\n"
        f"⚡ High conv  : {stats['high_conv_total']} | {hcwr} WR\n"
        f"{'─'*30}\n"
        f"🚫 Skipped — conf:{stats['skipped_conf']} gap:{stats['skipped_gap']} "
        f"odds:{stats['skipped_max_odds']} liq:{stats['skipped_liquidity']}\n"
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
    log_text = "\n".join(lines) if lines else "No trades"
    return (
        f"📊 *SESSION #{session_num} FINAL REPORT*\n"
        f"{'─'*30}\n"
        f"📅 {session_start.strftime('%Y-%m-%d %H:%M')} → "
        f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
        f"⏱ Duration    : {hours}h {mins}m\n"
        f"{'─'*30}\n"
        f"  Signals sent: {stats['signalled']} ✅TRADE\n"
        f"  All rounds  : {stats['total']} computed\n"
        f"  Win rate    : {wr}\n"
        f"  ✅ {stats['correct']} | ❌ {stats['incorrect']}\n"
        f"  High conv   : {stats['high_conv_total']} signals | {hcwr} WR\n"
        f"  Skipped     : conf {stats['skipped_conf']} | gap {stats['skipped_gap']} | "
        f"odds {stats['skipped_max_odds']} | liq {stats['skipped_liquidity']}\n"
        f"  Circ breaks : {stats['circuit_breaks']}\n"
        f"{'─'*30}\n"
        f"📋 *Last 30 signals*\n`{log_text}`"
    )

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def main():
    global current_market_cache, last_report_time
    global circuit_broken, consec_losses

    init_log()
    seed_candles()
    tg_send(start_message())
    tg_send(
        f"🟢 *Bot v5 online — Session #{session_num}*\n"
        f"📅 {session_start.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Active filters:\n"
        f"  ✅ Conf ≥ {CONFIDENCE:.0%}  _(raised from 70%)_\n"
        f"  ✅ Gap ≥ {MIN_GAP_PCT:.2f}%  _(raised from 0.02%)_\n"
        f"  ✅ Outcome ≤ {MAX_OUTCOME_PRICE:.0%}  _(new — blocks near-zero payout)_\n"
        f"  ✅ Silent CSV logging  _(all rounds, Telegram only on ✅ TRADE)_\n"
        f"  ✅ Target price freeze  _(set once at round open, never re-parsed)_\n"
        f"  ✅ Circuit break after {MAX_CONSEC_LOSSES} straight losses"
    )
    log.info("Bot v5 started")

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

            # Periodic 12h report
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
                if prev_target and btc:
                    # BTC at new round open = closing price of previous round
                    final_btc  = btc
                    actual_up  = final_btc >= prev_target
                    actual_dir = "UP" if actual_up else "DOWN"

                    if prev_signal:
                        predicted_up = prev_signal["direction_short"] == "UP"
                        correct      = predicted_up == actual_up
                        high_conv    = prev_signal["high_conviction"]
                        hour_h       = prev_signal.get("hour_utc", now.hour)
                        was_signalled = (prev_signal["signal"] == "✅ TRADE")

                        # Update stats only for signalled trades
                        if was_signalled:
                            stats["signalled"] += 1
                            stats["correct" if correct else "incorrect"] += 1
                            if high_conv:
                                stats["high_conv_total"] += 1
                                if correct:
                                    stats["high_conv_correct"] += 1
                            if hour_h not in hour_stats:
                                hour_stats[hour_h] = {"total": 0, "correct": 0}
                            hour_stats[hour_h]["total"] += 1
                            if correct:
                                hour_stats[hour_h]["correct"] += 1

                            # Circuit breaker update
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
                                        f"Signals paused to protect bankroll.\n"
                                        f"Use /play to resume manually."
                                    )

                        last_completed.update({
                            "round_start_price": prev_target,
                            "btc_at_end"       : final_btc,
                            "actual_direction" : actual_dir,
                            "bot_direction"    : prev_signal["direction_short"],
                            "correct"          : correct if was_signalled else None,
                            "resolved"         : True,
                        })

                        # ── SILENT CSV WRITE (ALL rounds) ────
                        # Every round is written regardless of signal.
                        # `signalled` field distinguishes ✅ TRADE from silent.
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
                            "conviction"       : "HIGH" if prev_signal.get("high_conviction") else "LOW",
                            "yes_price"        : prev_signal.get("yes_price", ""),
                            "no_price"         : prev_signal.get("no_price", ""),
                            "gap_pct"          : prev_signal.get("price_diff_pct", ""),
                            "odds_filter_pass" : prev_signal.get("odds_filter_pass", ""),
                            "liquidity"        : prev_signal.get("liquidity", ""),
                            "orders"           : prev_signal.get("total_orders", ""),
                            "session"          : session_num,
                            "signalled"        : was_signalled,
                            "high_conviction"  : prev_signal.get("high_conviction", False),
                            "outcome_price"    : prev_signal.get("outcome_price", ""),
                            "payout"           : prev_signal.get("payout", ""),
                        }
                        write_log(entry)
                        if was_signalled:
                            trade_log.append(entry)
                        log.info(
                            f"Closed: {'✅' if correct else '❌'} | "
                            f"signalled={was_signalled} | "
                            f"consec={consec_losses} | "
                            f"sig_total={stats['signalled']}"
                        )

                    else:
                        # Round had no pending signal (paused or features unavailable)
                        # Still write a minimal CSV row so data is complete
                        stats["total"] += 1
                        write_log({
                            "timestamp"        : now.isoformat(),
                            "time"             : now.strftime("%H:%M"),
                            "round_start_price": prev_target,
                            "btc_at_signal"    : "",
                            "btc_at_end"       : btc,
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
                        })

                # ── SET UP NEW ROUND ─────────────────────────
                # parse_target called ONCE here — result frozen for round
                new_target = parse_target(rules)
                if not new_target:
                    log.warning(f"Could not parse target: {rules[:60]}")

                current_round.update({
                    "event_id"         : event_id,
                    "start_time"       : now,
                    "round_start_price": new_target,
                    "signal_fired"     : False,
                    "pending_signal"   : None,
                })

                if new_target and btc and mins_left:
                    tg_send(msg_open_alert(market, btc, new_target, now, mins_left))
                    log.info(f"Open alert | target ${new_target:,.2f} | BTC ${btc:,.2f}")

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
                            # Send to Telegram only for ✅ TRADE
                            tg_send(msg_signal(result, now))
                            log.info(
                                f"Signal ✅ | {result['direction_short']} | "
                                f"conf {result['confidence']:.1%} | "
                                f"gap {result['abs_gap_pct']:.3f}% | "
                                f"op {result['outcome_price']:.0%} | "
                                f"payout {result['payout']:.2f}x | "
                                f"ev {result['ev']:+.3f}"
                            )
                        else:
                            # Silent — just log
                            reason_short = result["reason"][:60]
                            log.info(
                                f"Signal ⛔ (silent) | {result['direction_short']} | "
                                f"conf {result['confidence']:.1%} | {reason_short}"
                            )
                    else:
                        log.warning("Features not ready — not enough candles yet")

            time.sleep(30)

        except Exception as e:
            log.error(f"Main loop: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
