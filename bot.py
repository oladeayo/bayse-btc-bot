import logging
import requests
import joblib
import json
import re
import os
import csv
import sys
import time
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone, timedelta
from collections import deque

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
TELEGRAM_TOKEN        = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT         = os.environ.get("TELEGRAM_CHAT", "")
BAYSE_KEY             = os.environ.get("BAYSE_KEY", "")
BASE_URL              = "https://relay.bayse.markets/v1"
BAYSE_HEADERS         = {"X-Public-Key": BAYSE_KEY}
CONFIDENCE            = 0.70
FEE                   = 0.05
REPORT_INTERVAL_HOURS = 12
SIGNAL_DELAY_MINS     = 5
LOG_FILE              = "trade_log.csv"
TELEGRAM_API          = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ── Startup validation ────────────────────────────────────────
if not TELEGRAM_TOKEN:
    print("ERROR: TELEGRAM_TOKEN not set!", flush=True)
    sys.exit(1)
if not TELEGRAM_CHAT:
    print("ERROR: TELEGRAM_CHAT not set!", flush=True)
    sys.exit(1)
if not BAYSE_KEY:
    print("ERROR: BAYSE_KEY not set!", flush=True)
    sys.exit(1)

print(f"✅ Token: {TELEGRAM_TOKEN[:10]}...", flush=True)
print(f"✅ Chat: {TELEGRAM_CHAT}", flush=True)

# ── Load model ───────────────────────────────────────────────
model    = joblib.load("btc_bayse_model_v2.joblib")
features = json.load(open("features_v2.json"))
print(f"✅ Model loaded | {len(features)} features", flush=True)

# ── State ─────────────────────────────────────────────────────
CANDLE_WINDOW        = deque(maxlen=100)
current_round        = {
    "event_id"         : None,
    "start_time"       : None,
    "round_start_price": None,   # Bayse target price
    "signal_fired"     : False,
    "open_alert_sent"  : False,
    "pending_signal"   : None,   # signal result dict
}
prev_round           = {
    "event_id"         : None,
    "round_start_price": None,
    "final_price"      : None,
    "bot_direction"    : None,   # "UP" or "DOWN"
    "actual_direction" : None,   # "UP" or "DOWN"
    "would_have_won"   : None,
    "resolved"         : False,
}
last_report_time     = datetime.now(timezone.utc)
last_update_id       = None
current_market_cache = None

stats = {
    "total"            : 0,
    "correct"          : 0,
    "incorrect"        : 0,
    "high_conv_total"  : 0,
    "high_conv_correct": 0,
}
trade_log = []

# ── Telegram helpers ──────────────────────────────────────────
def tg_send(text, chat_id=None):
    try:
        r = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id"   : chat_id or TELEGRAM_CHAT,
                "text"      : text,
                "parse_mode": "Markdown"
            },
            timeout=10
        )
        if not r.ok:
            log.error(f"TG send error: {r.status_code} {r.text[:150]}")
    except Exception as e:
        log.error(f"TG send exception: {e}")

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
        log.error(f"TG poll error: {e}")
    return []

def handle_commands():
    for update in tg_get_updates():
        msg     = update.get("message", {})
        text    = msg.get("text", "").strip().lower()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if   text.startswith("/price"): cmd_price(chat_id)
        elif text.startswith("/trade"): cmd_trade(chat_id)
        elif text.startswith("/stats"): tg_send(build_stats_message(), chat_id)
        elif text.startswith("/start"): tg_send(start_message(), chat_id)

def start_message():
    return (
        "🤖 *Bayse BTC Bot*\n\n"
        "I monitor every BTC round on Bayse and send:\n"
        "  1️⃣ Open alert when round starts\n"
        "  2️⃣ Trade signal 5 mins in\n\n"
        "Commands:\n"
        "  /price — current BTC vs target\n"
        "  /trade — current Yes/No % on Bayse\n"
        "  /stats — performance summary\n\n"
        f"Confidence threshold: {CONFIDENCE:.0%} | Fee: {FEE:.0%}"
    )

def cmd_price(chat_id):
    btc = CANDLE_WINDOW[-1]["close"] if CANDLE_WINDOW else None
    target = current_round.get("round_start_price")
    now = datetime.now(timezone.utc)
    if btc and target:
        diff     = btc - target
        diff_pct = (diff / target) * 100
        arrow    = "📈 ABOVE" if diff > 0 else "📉 BELOW"
        tg_send(
            f"💰 *BTC vs Target*\n"
            f"{'─'*25}\n"
            f"  Current : ${btc:,.2f}\n"
            f"  Target  : ${target:,.2f}\n"
            f"  Gap     : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
            f"  Status  : {arrow} target\n"
            f"  Time    : {now.strftime('%H:%M:%S UTC')}",
            chat_id
        )
    else:
        tg_send("⚠️ No active round data yet.", chat_id)

def cmd_trade(chat_id):
    m = current_market_cache
    if m:
        yes = m.get("yes_price", 0)
        no  = m.get("no_price", 0)
        tg_send(
            f"📊 *Bayse Market Now*\n"
            f"{'─'*25}\n"
            f"  YES (UP)  : {yes:.0%}\n"
            f"  NO (DOWN) : {no:.0%}\n"
            f"  Orders    : {m.get('total_orders', 0)}\n"
            f"  Liquidity : ${m.get('liquidity', 0):,.2f}\n"
            f"  Time      : {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
            chat_id
        )
    else:
        tg_send("⚠️ No active round data yet.", chat_id)

# ── Trade log ─────────────────────────────────────────────────
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "round_start_price", "final_price",
                "direction", "confidence", "conviction",
                "actual_outcome", "correct",
                "price_vs_target_pct", "edge"
            ])

def write_log(entry):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            entry["timestamp"],
            entry.get("round_start_price", ""),
            entry.get("final_price", ""),
            entry.get("direction_short", ""),
            entry.get("confidence", ""),
            "HIGH" if entry.get("high_conviction") else "LOW",
            entry.get("actual", ""),
            entry.get("correct", ""),
            entry.get("price_vs_target_pct", ""),
            entry.get("edge", ""),
        ])

# ── KuCoin ────────────────────────────────────────────────────
def seed_candles():
    try:
        r    = requests.get("https://api.kucoin.com/api/v1/market/candles",
                            params={"symbol": "BTC-USDT", "type": "1min"}, timeout=10)
        data = r.json().get("data", [])
        for c in reversed(data):
            CANDLE_WINDOW.append({
                "open_time": pd.Timestamp(int(c[0]), unit="s", tz="UTC"),
                "open": float(c[1]), "close": float(c[2]),
                "high": float(c[3]), "low": float(c[4]), "volume": float(c[5])
            })
        log.info(f"Seeded {len(CANDLE_WINDOW)} candles | ${CANDLE_WINDOW[-1]['close']:,.2f}")
    except Exception as e:
        log.error(f"Seed error: {e}")

def update_candles():
    try:
        r    = requests.get("https://api.kucoin.com/api/v1/market/candles",
                            params={"symbol": "BTC-USDT", "type": "1min", "pageSize": 3}, timeout=5)
        data = r.json().get("data", [])
        if not data:
            return
        c = data[0]
        latest = {
            "open_time": pd.Timestamp(int(c[0]), unit="s", tz="UTC"),
            "open": float(c[1]), "close": float(c[2]),
            "high": float(c[3]), "low": float(c[4]), "volume": float(c[5])
        }
        if not CANDLE_WINDOW or latest["open_time"] > CANDLE_WINDOW[-1]["open_time"]:
            CANDLE_WINDOW.append(latest)
    except Exception as e:
        log.error(f"Update candles error: {e}")

def get_btc_price():
    return CANDLE_WINDOW[-1]["close"] if CANDLE_WINDOW else None

# ── Bayse ─────────────────────────────────────────────────────
def fetch_btc_market():
    try:
        r      = requests.get(f"{BASE_URL}/pm/events", headers=BAYSE_HEADERS,
                              params={"limit": 50}, timeout=10)
        events = r.json().get("events", [])
        for event in events:
            if "bitcoin" in event.get("title", "").lower() and "15" in event.get("title", ""):
                r2     = requests.get(f"{BASE_URL}/pm/events/{event['id']}",
                                      headers=BAYSE_HEADERS, timeout=10)
                detail = r2.json()
                ms     = detail.get("markets", [])
                if ms:
                    m = ms[0]
                    return {
                        "event_id"        : detail["id"],
                        "yes_price"       : m.get("outcome1Price", 0),
                        "no_price"        : m.get("outcome2Price", 0),
                        "total_orders"    : m.get("totalOrders", 0),
                        "liquidity"       : detail.get("liquidity", 0),
                        "created_at"      : detail.get("createdAt", ""),
                        "rules"           : m.get("rules", ""),
                        "resolved_outcome": m.get("resolvedOutcome", ""),
                        "status"          : m.get("status", ""),
                    }
    except Exception as e:
        log.error(f"Bayse fetch error: {e}")
    return None

def fetch_market_by_id(event_id):
    try:
        r      = requests.get(f"{BASE_URL}/pm/events/{event_id}",
                              headers=BAYSE_HEADERS, timeout=10)
        detail = r.json()
        ms     = detail.get("markets", [])
        if ms:
            m = ms[0]
            return {
                "resolved_outcome": m.get("resolvedOutcome", ""),
                "status"          : m.get("status", ""),
            }
    except Exception as e:
        log.error(f"Fetch by ID error: {e}")
    return None

def parse_target(rules):
    m = re.findall(r'\$([\d,]+\.?\d*)', rules)
    return float(m[0].replace(",", "")) if m else None

def parse_end_dt(rules, now):
    m = re.search(r'(\d+:\d+:\d+\s?[AP]M)\s?GMT', rules)
    if m:
        try:
            end = datetime.strptime(
                f"{now.strftime('%Y-%m-%d')} {m.group(1).strip()}",
                "%Y-%m-%d %I:%M:%S %p"
            ).replace(tzinfo=timezone.utc)
            # Handle midnight crossover
            if (end - now).total_seconds() < -3600:
                end += timedelta(days=1)
            return end
        except Exception:
            pass
    return None

# ── Features ──────────────────────────────────────────────────
def compute_features(round_start_price):
    if len(CANDLE_WINDOW) < 60:
        return None, None
    df = pd.DataFrame(list(CANDLE_WINDOW)).sort_values("open_time").reset_index(drop=True)
    try:
        df["rsi_14"]       = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi_7"]        = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
        df["rsi_21"]       = ta.momentum.RSIIndicator(df["close"], window=21).rsi()
        df["stoch"]        = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
        df["macd"]         = ta.trend.MACD(df["close"]).macd_diff()
        df["macd_signal"]  = ta.trend.MACD(df["close"]).macd_signal()
        df["ema_9"]        = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
        df["ema_21"]       = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
        df["ema_cross"]    = df["ema_9"] - df["ema_21"]
        bb                 = ta.volatility.BollingerBands(df["close"])
        df["bb_position"]  = (df["close"] - bb.bollinger_mavg()) / bb.bollinger_wband()
        df["bb_width"]     = bb.bollinger_wband()
        df["atr"]          = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        df["vol_ratio_15"] = df["volume"] / df["volume"].rolling(15).mean()
        df["vol_ratio_60"] = df["volume"] / df["volume"].rolling(60).mean()
        df["obv"]          = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["obv_slope"]    = df["obv"].diff(5)
        df["candle_body"]  = (df["close"] - df["open"]) / df["open"]
        df["momentum_1"]   = df["close"].pct_change(1)
        df["momentum_3"]   = df["close"].pct_change(3)
        df["momentum_5"]   = df["close"].pct_change(5)
        df["rolling_std_15"] = df["close"].pct_change().rolling(15).std()
        df["rolling_std_60"] = df["close"].pct_change().rolling(60).std()
        df["hour"]         = df["open_time"].dt.hour
        df["dayofweek"]    = df["open_time"].dt.dayofweek

        latest = df.iloc[-1].copy()

        # Bayse-specific features using actual round start price
        if round_start_price and round_start_price > 0:
            price_vs_target     = (latest["close"] - round_start_price) / round_start_price
            price_gap_usd       = latest["close"] - round_start_price
            gap_vs_atr          = price_gap_usd / (latest["atr"] + 1e-8)
            gap_pct_abs         = abs(price_vs_target)
            early_momentum      = price_vs_target
            rsi_vs_neutral      = latest["rsi_14"] - 50
        else:
            price_vs_target = price_gap_usd = gap_vs_atr = gap_pct_abs = early_momentum = rsi_vs_neutral = 0.0

        latest["price_vs_target"]  = price_vs_target
        latest["price_gap_usd"]    = price_gap_usd
        latest["gap_vs_atr"]       = gap_vs_atr
        latest["gap_pct_abs"]      = gap_pct_abs
        latest["early_momentum"]   = early_momentum
        latest["rsi_vs_neutral"]   = rsi_vs_neutral

        return latest[features].values, latest["rolling_std_60"]

    except Exception as e:
        log.error(f"Feature error: {e}")
        return None, None

# ── Signal ────────────────────────────────────────────────────
def get_signal(market, round_start_price, now):
    rules          = market.get("rules", "")
    end_dt         = parse_end_dt(rules, now)
    btc            = get_btc_price()
    mins_remaining = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None
    secs_remaining = (end_dt - now).total_seconds() if end_dt else 999

    price_diff     = (btc - round_start_price) if (btc and round_start_price) else 0
    price_diff_pct = (price_diff / round_start_price * 100) if round_start_price else 0

    feat_vals, vol_regime = compute_features(round_start_price)
    if feat_vals is None:
        return None

    feat_df = pd.DataFrame([feat_vals], columns=features)
    proba   = model.predict_proba(feat_df)[0][1]
    edge    = abs(proba - market["yes_price"])

    direction  = "BUY YES (UP) 📈" if proba > 0.5 else "BUY NO (DOWN) 📉"
    conviction = "⚡ HIGH" if (proba > CONFIDENCE or proba < (1 - CONFIDENCE)) else "⚠️ LOW"

    if vol_regime and vol_regime > 0.003:
        signal = "⛔ NO TRADE"
        reason = "Volatility too high"
    elif 0 < secs_remaining < 120:
        signal = "⛔ NO TRADE"
        reason = f"Only {secs_remaining:.0f}s left"
    elif proba > CONFIDENCE or proba < (1 - CONFIDENCE):
        signal = "✅ TRADE"
        gap_status = f"${price_diff:+.2f} ({price_diff_pct:+.3f}%) vs target"
        reasons = [
            f"Model: {proba:.1%} {'UP' if proba > 0.5 else 'DOWN'}",
            f"BTC {gap_status}",
            f"Gap/ATR ratio: {price_diff/(feat_vals[features.index('atr')] + 1e-8):.2f}x",
            f"Edge: {edge:.1%}"
        ]
        reason = " | ".join(reasons)
    else:
        signal = "⛔ NO TRADE"
        reason = f"Confidence {proba:.1%} below {CONFIDENCE:.0%}"

    return {
        "signal"           : signal,
        "direction"        : direction,
        "direction_short"  : "UP" if proba > 0.5 else "DOWN",
        "conviction"       : conviction,
        "confidence"       : round(proba, 4),
        "edge"             : round(edge, 4),
        "btc_price"        : btc,
        "round_start_price": round_start_price,
        "price_diff"       : round(price_diff, 2),
        "price_diff_pct"   : round(price_diff_pct, 4),
        "mins_remaining"   : mins_remaining,
        "yes_price"        : market["yes_price"],
        "no_price"         : market["no_price"],
        "reason"           : reason,
        "timestamp"        : now.isoformat(),
        "high_conviction"  : signal == "✅ TRADE",
    }

# ── Messages ──────────────────────────────────────────────────
def msg_open_alert(market, btc, target, now, mins_remaining):
    """Fires immediately when new round opens."""
    diff     = btc - target if (btc and target) else 0
    diff_pct = (diff / target * 100) if target else 0

    if prev_round["resolved"] and prev_round["round_start_price"]:
        p_target = prev_round["round_start_price"]
        p_final  = prev_round["final_price"] or btc
        p_diff   = p_final - p_target
        p_dir    = "UP 📈" if p_diff >= 0 else "DOWN 📉"
        won_txt  = "✅ WON" if prev_round["would_have_won"] else "❌ LOST"
        wr       = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
        last_block = (
            f"{'─'*30}\n"
            f"📋 *Last Round*\n"
            f"  Target → Final: ${p_target:,.2f} → ${p_final:,.2f}\n"
            f"  Result: {p_dir}\n"
            f"  Bot said: {prev_round['bot_direction']} → {won_txt}\n"
            f"  Session: {stats['total']} rounds | {wr} WR\n"
        )
    else:
        last_block = f"{'─'*30}\n📋 *Last Round:* No data yet\n"

    return (
        f"🔔 *NEW BTC ROUND OPEN*\n"
        f"{'─'*30}\n"
        f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ {mins_remaining} mins\n"
        f"{'─'*30}\n"
        f"💰 BTC Now  : ${btc:,.2f}\n"
        f"🎯 Target   : ${target:,.2f}\n"
        f"💹 Gap      : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
        f"{'─'*30}\n"
        + last_block +
        f"{'─'*30}\n"
        f"⏳ Signal in ~{SIGNAL_DELAY_MINS} mins..."
    )

def msg_signal(result, now):
    """Fires 5 mins into round."""
    return (
        f"🤖 *SIGNAL — {now.strftime('%H:%M UTC')}*\n"
        f"{'─'*30}\n"
        f"💰 BTC    : ${result['btc_price']:,.2f}\n"
        f"🎯 Target : ${result['round_start_price']:,.2f}\n"
        f"💹 Gap    : ${result['price_diff']:+.2f} ({result['price_diff_pct']:+.3f}%)\n"
        f"⏱ Left   : {result['mins_remaining']} mins\n"
        f"{'─'*30}\n"
        f"🔔 *{result['signal']}*\n"
        f"📈 Direction  : {result['direction']}\n"
        f"⚡ Conviction : {result['conviction']} ({result['confidence']:.1%})\n"
        f"📉 Edge       : {result['edge']:.1%}\n"
        f"{'─'*30}\n"
        f"💡 *Why:*\n"
        + "\n".join([f"  • {x}" for x in result["reason"].split(" | ")])
    )

def msg_report():
    wr   = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    hcwr = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
            if stats["high_conv_total"] > 0 else "N/A")
    lines = []
    for t in trade_log[-25:]:
        e = "✅" if t["correct"] else "❌"
        c = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(f"{e}{c} {t['time']} | {t['direction_short']} | {t['confidence']:.0%} | {'✓' if t['correct'] else '✗'}")
    return (
        f"📊 *{REPORT_INTERVAL_HOURS}H REPORT*\n"
        f"{'─'*30}\n"
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"  All: {stats['total']} rounds | {wr} WR\n"
        f"  High conv: {stats['high_conv_total']} | {hcwr} WR\n"
        f"{'─'*30}\n"
        f"`{'chr(10)'.join(lines) if lines else 'No trades yet'}`"
    )

def build_stats_message():
    now  = datetime.now(timezone.utc)
    wr   = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    hcwr = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
            if stats["high_conv_total"] > 0 else "N/A")
    last_10         = trade_log[-10:]
    l10c            = sum(1 for t in last_10 if t["correct"])
    l10r            = f"{l10c/len(last_10)*100:.1f}%" if last_10 else "N/A"
    one_hr_ago      = now - timedelta(hours=1)
    last_1h         = [t for t in trade_log if datetime.fromisoformat(t["timestamp"]) > one_hr_ago]
    l1hc            = sum(1 for t in last_1h if t["correct"])
    l1hr            = f"{l1hc/len(last_1h)*100:.1f}%" if last_1h else "N/A"
    lines = []
    for t in last_10:
        e = "✅" if t["correct"] else "❌"
        c = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(f"{e}{c} {t['time']} | {t['direction_short']} | {t['confidence']:.0%} | {'✓' if t['correct'] else '✗'}")
    table = "\n".join(lines) if lines else "No trades yet"
    return (
        f"📊 *BOT STATS*\n"
        f"{'─'*30}\n"
        f"🕐 {now.strftime('%H:%M UTC')}\n"
        f"{'─'*30}\n"
        f"📈 All time : {stats['total']} rounds | {wr} WR\n"
        f"  ✅ {stats['correct']} | ❌ {stats['incorrect']}\n"
        f"⏱ Last hour : {len(last_1h)} rounds | {l1hr} WR\n"
        f"🔟 Last 10  : {l10r} WR\n"
        f"⚡ High conv : {stats['high_conv_total']} | {hcwr} WR\n"
        f"{'─'*30}\n"
        f"`{table}`"
    )

# ── Main loop ─────────────────────────────────────────────────
def main():
    global current_market_cache, last_report_time

    init_log()
    seed_candles()
    tg_send(start_message())
    log.info("Bot started")

    while True:
        try:
            handle_commands()
            update_candles()

            now    = datetime.now(timezone.utc)
            market = fetch_btc_market()

            if not market:
                log.info("No BTC market")
                time.sleep(30)
                continue

            current_market_cache = market
            event_id   = market["event_id"]
            btc        = get_btc_price()
            rules      = market.get("rules", "")
            target     = parse_target(rules)
            end_dt     = parse_end_dt(rules, now)
            mins_left  = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None

            # ── 12h report ────────────────────────────────────
            if (now - last_report_time).total_seconds() / 3600 >= REPORT_INTERVAL_HOURS and stats["total"] > 0:
                tg_send(msg_report())
                last_report_time = now

            # ── New round ─────────────────────────────────────
            if event_id != current_round["event_id"]:
                log.info(f"New round: {event_id}")

                # ── Resolve previous round ────────────────────
                prev_id     = current_round["event_id"]
                prev_signal = current_round["pending_signal"]
                prev_target = current_round["round_start_price"]

                if prev_id and prev_signal:
                    resolved_data = fetch_market_by_id(prev_id)
                    resolved      = resolved_data.get("resolved_outcome", "") if resolved_data else ""

                    if resolved:
                        final_price  = btc
                        predicted_up = prev_signal["direction_short"] == "UP"
                        actual_up    = resolved.lower() in ["up", "yes"]
                        correct      = predicted_up == actual_up
                        high_conv    = prev_signal["high_conviction"]

                        stats["total"] += 1
                        stats["correct" if correct else "incorrect"] += 1
                        if high_conv:
                            stats["high_conv_total"] += 1
                            if correct:
                                stats["high_conv_correct"] += 1

                        # Update prev_round for display in open alert
                        prev_round.update({
                            "event_id"         : prev_id,
                            "round_start_price": prev_target,
                            "final_price"      : final_price,
                            "bot_direction"    : f"{'UP 📈' if predicted_up else 'DOWN 📉'}",
                            "actual_direction" : "UP" if actual_up else "DOWN",
                            "would_have_won"   : correct,
                            "resolved"         : True,
                        })

                        entry = {
                            "timestamp"        : now.isoformat(),
                            "time"             : now.strftime("%H:%M"),
                            "direction_short"  : prev_signal["direction_short"],
                            "confidence"       : prev_signal["confidence"],
                            "correct"          : correct,
                            "high_conviction"  : high_conv,
                            "actual"           : resolved,
                            "btc_price"        : final_price,
                            "round_start_price": prev_target,
                            "price_vs_target_pct": prev_signal.get("price_diff_pct", ""),
                            "edge"             : prev_signal.get("edge", ""),
                        }
                        trade_log.append(entry)
                        write_log(entry)
                        log.info(f"Resolved: {'✅' if correct else '❌'} | {stats}")
                    else:
                        log.warning(f"No resolved_outcome for {prev_id}")

                # ── Reset and send open alert ─────────────────
                current_round.update({
                    "event_id"         : event_id,
                    "start_time"       : now,
                    "round_start_price": target,
                    "signal_fired"     : False,
                    "open_alert_sent"  : False,
                    "pending_signal"   : None,
                })

                if target and btc and mins_left:
                    tg_send(msg_open_alert(market, btc, target, now, mins_left))
                    current_round["open_alert_sent"] = True

            # ── Signal after 5 mins ───────────────────────────
            start_time = current_round["start_time"]
            if (not current_round["signal_fired"] and
                    start_time and
                    current_round["round_start_price"]):

                mins_in = (now - start_time).total_seconds() / 60
                if mins_in >= SIGNAL_DELAY_MINS:
                    result = get_signal(market, current_round["round_start_price"], now)
                    if result:
                        current_round["pending_signal"] = result
                        current_round["signal_fired"]   = True
                        tg_send(msg_signal(result, now))
                        log.info(f"Signal: {result['signal']} {result['direction']} {result['confidence']:.1%}")
                    else:
                        log.warning("Features not ready")

            time.sleep(30)

        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
