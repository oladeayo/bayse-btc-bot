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
CONFIDENCE            = 0.58
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
model    = joblib.load("btc_bayse_model_6m.joblib")
features = json.load(open("features.json"))
print("✅ Model loaded", flush=True)

# ── State ─────────────────────────────────────────────────────
CANDLE_WINDOW        = deque(maxlen=100)
last_event_id        = None       # event ID of current round
prev_event_id        = None       # event ID of previous round (for result lookup)
pending_signal       = None       # signal sent 5 mins in
signal_fired         = False      # has signal been sent this round
round_start_time     = None       # when current round was detected
open_alert_sent      = False      # has the open alert been sent this round
last_report_time     = datetime.now(timezone.utc)
last_update_id       = None
current_market_cache = None       # cache of current market for /price /trade

# Last round result for displaying in open alert
last_round = {
    "target"        : None,
    "final_price"   : None,
    "direction"     : None,   # UP or DOWN (what actually happened)
    "bot_prediction": None,   # UP or DOWN (what bot said)
    "would_have_won": None,   # True/False
    "resolved"      : False
}

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
        cid = chat_id or TELEGRAM_CHAT
        r = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": cid, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
        if not r.ok:
            log.error(f"Telegram send error: {r.status_code} {r.text[:200]}")
    except Exception as e:
        log.error(f"Telegram send exception: {e}")

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
        log.error(f"Telegram poll error: {e}")
    return []

def handle_commands():
    updates = tg_get_updates()
    for update in updates:
        msg     = update.get("message", {})
        text    = msg.get("text", "").strip().lower()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if text.startswith("/price"):
            handle_price(chat_id)
        elif text.startswith("/trade"):
            handle_trade_cmd(chat_id)
        elif text.startswith("/stats"):
            tg_send(build_stats_message(), chat_id=chat_id)
        elif text.startswith("/start"):
            tg_send(
                "🤖 *Bayse BTC Bot*\n\n"
                "Commands:\n"
                "  /price — current BTC vs target\n"
                "  /trade — current Yes/No % on Bayse\n"
                "  /stats — performance summary\n\n"
                f"Confidence threshold: {CONFIDENCE:.0%}\n"
                f"Signal fires {SIGNAL_DELAY_MINS} mins into each round.",
                chat_id=chat_id
            )

def handle_price(chat_id):
    now = datetime.now(timezone.utc)
    btc = CANDLE_WINDOW[-1]["close"] if CANDLE_WINDOW else None
    if current_market_cache and btc:
        rules        = current_market_cache.get("rules", "")
        price_match  = re.findall(r'\$([\d,]+\.?\d*)', rules)
        price_target = float(price_match[0].replace(",", "")) if price_match else None
        if price_target:
            diff     = btc - price_target
            diff_pct = (diff / price_target) * 100
            status   = "📈 ABOVE target" if diff > 0 else "📉 BELOW target"
            tg_send(
                f"💰 *BTC Price Check*\n"
                f"{'─' * 25}\n"
                f"  Current:  ${btc:,.2f}\n"
                f"  Target:   ${price_target:,.2f}\n"
                f"  Diff:     ${diff:+.2f} ({diff_pct:+.3f}%)\n"
                f"  Status:   {status}\n"
                f"  Time:     {now.strftime('%H:%M:%S UTC')}",
                chat_id=chat_id
            )
            return
    tg_send("⚠️ No active round data yet.", chat_id=chat_id)

def handle_trade_cmd(chat_id):
    if current_market_cache:
        yes = current_market_cache.get("yes_price", 0)
        no  = current_market_cache.get("no_price", 0)
        orders = current_market_cache.get("total_orders", 0)
        liq    = current_market_cache.get("liquidity", 0)
        now = datetime.now(timezone.utc)
        tg_send(
            f"📊 *Current Bayse Market*\n"
            f"{'─' * 25}\n"
            f"  YES (UP):  {yes:.0%}\n"
            f"  NO (DOWN): {no:.0%}\n"
            f"  Orders:    {orders}\n"
            f"  Liquidity: ${liq:,.2f}\n"
            f"  Time:      {now.strftime('%H:%M:%S UTC')}",
            chat_id=chat_id
        )
    else:
        tg_send("⚠️ No active round data yet.", chat_id=chat_id)

# ── Trade log CSV ─────────────────────────────────────────────
def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "direction", "confidence", "conviction",
                "actual_outcome", "correct", "btc_price", "price_target",
                "yes_price", "edge", "mins_remaining_at_signal"
            ])

def append_to_log(entry):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            entry["timestamp"], entry["direction_short"], entry["confidence"],
            "HIGH" if entry["high_conviction"] else "LOW",
            entry["actual"], entry["correct"],
            entry.get("btc_price", ""), entry.get("price_target", ""),
            entry.get("yes_price", ""), entry.get("edge", ""),
            entry.get("mins_remaining", "")
        ])

# ── KuCoin ────────────────────────────────────────────────────
def seed_candle_window():
    url    = "https://api.kucoin.com/api/v1/market/candles"
    params = {"symbol": "BTC-USDT", "type": "1min"}
    r      = requests.get(url, params=params, timeout=10)
    data   = r.json().get("data", [])
    for c in reversed(data):
        CANDLE_WINDOW.append({
            "open_time": pd.Timestamp(int(c[0]), unit="s", tz="UTC"),
            "open"     : float(c[1]),
            "close"    : float(c[2]),
            "high"     : float(c[3]),
            "low"      : float(c[4]),
            "volume"   : float(c[5])
        })
    log.info(f"Seeded {len(CANDLE_WINDOW)} candles | BTC: ${CANDLE_WINDOW[-1]['close']:,.2f}")

def update_candle_window():
    try:
        url    = "https://api.kucoin.com/api/v1/market/candles"
        params = {"symbol": "BTC-USDT", "type": "1min", "pageSize": 3}
        r      = requests.get(url, params=params, timeout=5)
        data   = r.json().get("data", [])
        if not data:
            return
        latest = {
            "open_time": pd.Timestamp(int(data[0][0]), unit="s", tz="UTC"),
            "open"     : float(data[0][1]),
            "close"    : float(data[0][2]),
            "high"     : float(data[0][3]),
            "low"      : float(data[0][4]),
            "volume"   : float(data[0][5])
        }
        if not CANDLE_WINDOW or latest["open_time"] > CANDLE_WINDOW[-1]["open_time"]:
            CANDLE_WINDOW.append(latest)
    except Exception as e:
        log.error(f"KuCoin update error: {e}")

# ── Bayse ─────────────────────────────────────────────────────
def fetch_btc_bayse_market():
    try:
        r      = requests.get(f"{BASE_URL}/pm/events", headers=BAYSE_HEADERS,
                              params={"limit": 50}, timeout=10)
        events = r.json().get("events", [])
        for event in events:
            title = event.get("title", "").lower()
            if "bitcoin" in title and "15" in title:
                r2      = requests.get(f"{BASE_URL}/pm/events/{event['id']}",
                                       headers=BAYSE_HEADERS, timeout=10)
                detail  = r2.json()
                markets = detail.get("markets", [])
                if markets:
                    m = markets[0]
                    return {
                        "event_id"        : detail["id"],
                        "event_title"     : detail["title"],
                        "market_id"       : m["id"],
                        "yes_price"       : m.get("outcome1Price", 0),
                        "no_price"        : m.get("outcome2Price", 0),
                        "total_orders"    : m.get("totalOrders", 0),
                        "status"          : m.get("status"),
                        "liquidity"       : detail.get("liquidity", 0),
                        "created_at"      : detail.get("createdAt", ""),
                        "rules"           : m.get("rules", ""),
                        "resolved_outcome": m.get("resolvedOutcome", "")
                    }
    except Exception as e:
        log.error(f"Bayse fetch error: {e}")
    return None

def fetch_resolved_outcome(event_id):
    """Fetch a specific past event to get its resolved outcome."""
    try:
        r      = requests.get(f"{BASE_URL}/pm/events/{event_id}",
                              headers=BAYSE_HEADERS, timeout=10)
        detail = r.json()
        markets = detail.get("markets", [])
        if markets:
            m = markets[0]
            return {
                "resolved_outcome": m.get("resolvedOutcome", ""),
                "status"          : m.get("status", ""),
                "yes_price"       : m.get("outcome1Price", 0),
            }
    except Exception as e:
        log.error(f"Fetch resolved error: {e}")
    return None

def parse_price_target(rules):
    price_match = re.findall(r'\$([\d,]+\.?\d*)', rules)
    return float(price_match[0].replace(",", "")) if price_match else None

def parse_end_time(rules, now):
    end_match = re.search(r'(\d+:\d+:\d+\s?[AP]M)\s?GMT', rules)
    if end_match:
        today  = now.strftime("%Y-%m-%d")
        end_dt = datetime.strptime(f"{today} {end_match.group(1).strip()}", "%Y-%m-%d %I:%M:%S %p")
        return end_dt.replace(tzinfo=timezone.utc)
    return None

# ── Features ──────────────────────────────────────────────────
def compute_features():
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
        df["upper_wick"]   = (df["high"] - df[["close","open"]].max(axis=1)) / df["open"]
        df["lower_wick"]   = (df[["close","open"]].min(axis=1) - df["low"]) / df["open"]
        df["momentum_1"]   = df["close"].pct_change(1)
        df["momentum_3"]   = df["close"].pct_change(3)
        df["momentum_5"]   = df["close"].pct_change(5)
        df["momentum_15"]  = df["close"].pct_change(15)
        df["momentum_30"]  = df["close"].pct_change(30)
        df["rolling_std_15"] = df["close"].pct_change().rolling(15).std()
        df["rolling_std_60"] = df["close"].pct_change().rolling(60).std()
        df["hour"]         = df["open_time"].dt.hour
        df["dayofweek"]    = df["open_time"].dt.dayofweek
        df["minute"]       = df["open_time"].dt.minute
        latest = df.iloc[-1]
        return latest[features].values, latest["rolling_std_60"]
    except Exception as e:
        log.error(f"Feature error: {e}")
        return None, None

# ── Signal ────────────────────────────────────────────────────
def get_signal(market, feature_values, vol_regime, now):
    current_price = CANDLE_WINDOW[-1]["close"]
    rules         = market.get("rules", "")
    price_target  = parse_price_target(rules)
    end_dt        = parse_end_time(rules, now)

    mins_remaining = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None
    secs_remaining = (end_dt - now).total_seconds() if end_dt else 999
    price_diff     = (current_price - price_target) if price_target else 0
    price_diff_pct = (price_diff / price_target * 100) if price_target else 0

    feat_df = pd.DataFrame([feature_values], columns=features)
    proba   = model.predict_proba(feat_df)[0][1]
    edge    = abs(proba - market["yes_price"])

    direction  = "BUY YES (UP) 📈" if proba > 0.5 else "BUY NO (DOWN) 📉"
    conviction = "⚡ HIGH CONVICTION" if (proba > CONFIDENCE or proba < (1 - CONFIDENCE)) else "⚠️ LOW CONVICTION"

    if vol_regime > 0.003:
        signal = "⛔ NO TRADE"
        reason = "Volatility too high — model unreliable"
    elif 0 < secs_remaining < 120:
        signal = "⛔ NO TRADE"
        reason = f"Only {secs_remaining:.0f}s left — too late"
    elif proba > CONFIDENCE or proba < (1 - CONFIDENCE):
        signal = "✅ TRADE"
        reasons = [
            f"Model confidence: {proba:.1%} {'UP' if proba > 0.5 else 'DOWN'}",
            f"RSI & momentum align with {'bullish' if proba > 0.5 else 'bearish'} signal",
            f"BTC ${price_diff:+.2f} ({price_diff_pct:+.3f}%) vs round start price",
            f"Edge over market: {edge:.1%}"
        ]
        reason = " | ".join(reasons)
    else:
        signal = "⛔ NO TRADE"
        reason = f"Confidence {proba:.1%} below {CONFIDENCE:.0%} threshold"

    return {
        "signal"        : signal,
        "direction"     : direction,
        "conviction"    : conviction,
        "confidence"    : round(proba, 4),
        "edge"          : round(edge, 4),
        "current_price" : current_price,
        "price_target"  : price_target,
        "price_diff"    : round(price_diff, 2),
        "price_diff_pct": round(price_diff_pct, 4),
        "mins_remaining": mins_remaining,
        "yes_price"     : market["yes_price"],
        "no_price"      : market["no_price"],
        "total_orders"  : market["total_orders"],
        "liquidity"     : market["liquidity"],
        "reason"        : reason,
        "event_id"      : market["event_id"],
        "timestamp"     : now.isoformat()
    }

# ── Message formatters ────────────────────────────────────────
def format_open_alert(market, btc_price, price_target, now):
    """Message 1 — fires immediately when new round detected."""
    end_dt         = parse_end_time(market.get("rules", ""), now)
    mins_remaining = round((end_dt - now).total_seconds() / 60, 1) if end_dt else "?"
    diff           = btc_price - price_target if price_target else 0
    diff_pct       = (diff / price_target * 100) if price_target else 0

    # Last round result block
    if last_round["resolved"] and last_round["target"] and last_round["final_price"]:
        actual_diff  = last_round["final_price"] - last_round["target"]
        actual_dir   = "UP 📈" if actual_diff >= 0 else "DOWN 📉"
        won_icon     = "✅ WON" if last_round["would_have_won"] else "❌ LOST"
        bot_said     = last_round["bot_prediction"] or "N/A"
        last_block = (
            f"{'─' * 30}\n"
            f"📋 *Last Round*\n"
            f"  Target:    ${last_round['target']:,.2f}\n"
            f"  Final BTC: ${last_round['final_price']:,.2f} → {actual_dir}\n"
            f"  Bot said:  {bot_said} → {won_icon}\n"
            f"  📊 Session: {stats['total']} rounds | "
            f"{stats['correct']/stats['total']*100:.1f}% win rate\n"
        )
    else:
        last_block = (
            f"{'─' * 30}\n"
            f"📋 *Last Round:* No data yet\n"
        )

    return (
        f"🔔 *NEW BTC ROUND OPEN*\n"
        f"{'─' * 30}\n"
        f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ {mins_remaining} mins left\n"
        f"{'─' * 30}\n"
        f"💰 *Current BTC:* ${btc_price:,.2f}\n"
        f"🎯 *Target Price:* ${price_target:,.2f}\n"
        f"💹 *Diff:* ${diff:+.2f} ({diff_pct:+.3f}%)\n"
        f"{'─' * 30}\n"
        + last_block +
        f"{'─' * 30}\n"
        f"⏳ Signal in ~{SIGNAL_DELAY_MINS} mins..."
    )

def format_signal_message(result, now):
    """Message 2 — fires 5 mins into round."""
    return (
        f"🤖 *SIGNAL — {now.strftime('%H:%M UTC')}*\n"
        f"{'─' * 30}\n"
        f"💰 *BTC:* ${result['current_price']:,.2f}\n"
        f"🎯 *Target:* ${result['price_target']:,.2f}\n"
        f"💹 *Diff:* ${result['price_diff']:+.2f} ({result['price_diff_pct']:+.3f}%)\n"
        f"⏱ *Time Left:* {result['mins_remaining']} mins\n"
        f"{'─' * 30}\n"
        f"🔔 *Signal:* {result['signal']}\n"
        f"📈 *Direction:* {result['direction']}\n"
        f"⚡ *Conviction:* {result['conviction']}\n"
        f"🎯 *Confidence:* {result['confidence']:.1%}\n"
        f"📉 *Edge:* {result['edge']:.1%}\n"
        f"{'─' * 30}\n"
        f"💡 *Reasons:*\n"
        + "\n".join([f"  • {x}" for x in result['reason'].split(' | ')])
    )

def format_report_message():
    win_rate = f"{stats['correct']/stats['total']*100:.1f}%" if stats['total'] > 0 else "N/A"
    hc_rate  = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
                if stats['high_conv_total'] > 0 else "N/A")
    lines = []
    for t in trade_log[-25:]:
        emoji = "✅" if t["correct"] else "❌"
        conv  = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(f"{emoji}{conv} {t['time']} | {t['direction_short']} | {t['confidence']:.0%} | {'RIGHT' if t['correct'] else 'WRONG'}")
    log_text = "\n".join(lines) if lines else "No completed trades yet"
    return (
        f"📊 *{REPORT_INTERVAL_HOURS}H REPORT*\n"
        f"{'─' * 30}\n"
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{'─' * 30}\n"
        f"  All rounds: {stats['total']} | ✅ {stats['correct']} | ❌ {stats['incorrect']} | {win_rate} WR\n"
        f"  High conv:  {stats['high_conv_total']} signals | {hc_rate} WR\n"
        f"{'─' * 30}\n"
        f"📋 *Last 25*\n`{log_text}`"
    )

def build_stats_message():
    now      = datetime.now(timezone.utc)
    win_rate = f"{stats['correct']/stats['total']*100:.1f}%" if stats['total'] > 0 else "N/A"
    hc_rate  = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
                if stats['high_conv_total'] > 0 else "N/A")
    last_10         = trade_log[-10:]
    last_10_correct = sum(1 for t in last_10 if t["correct"])
    last_10_rate    = f"{last_10_correct/len(last_10)*100:.1f}%" if last_10 else "N/A"
    one_hour_ago    = now - timedelta(hours=1)
    last_1h         = [t for t in trade_log if datetime.fromisoformat(t["timestamp"]) > one_hour_ago]
    last_1h_correct = sum(1 for t in last_1h if t["correct"])
    last_1h_rate    = f"{last_1h_correct/len(last_1h)*100:.1f}%" if last_1h else "N/A"
    lines = []
    for t in last_10:
        emoji = "✅" if t["correct"] else "❌"
        conv  = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(f"{emoji}{conv} {t['time']} | {t['direction_short']} | {t['confidence']:.0%} | {'RIGHT' if t['correct'] else 'WRONG'}")
    table = "\n".join(lines) if lines else "No trades yet"
    return (
        f"📊 *BOT STATS*\n"
        f"{'─' * 30}\n"
        f"🕐 {now.strftime('%H:%M:%S UTC')}\n"
        f"{'─' * 30}\n"
        f"📈 *All Time:* {stats['total']} rounds | {win_rate} WR\n"
        f"  ✅ {stats['correct']} correct | ❌ {stats['incorrect']} wrong\n"
        f"{'─' * 30}\n"
        f"⏱ *Last Hour* ({len(last_1h)} rounds): {last_1h_rate} WR\n"
        f"🔟 *Last 10:* {last_10_rate} WR\n"
        f"{'─' * 30}\n"
        f"`{table}`\n"
        f"{'─' * 30}\n"
        f"⚡ *High Conviction:* {stats['high_conv_total']} signals | {hc_rate} WR"
    )

# ── Main loop ─────────────────────────────────────────────────
def main():
    global last_event_id, prev_event_id, pending_signal
    global signal_fired, round_start_time, open_alert_sent
    global last_report_time, current_market_cache, last_round

    init_log_file()
    seed_candle_window()

    tg_send(
        "🤖 *Bayse BTC Bot is live!*\n\n"
        "Commands:\n"
        "  /price — BTC vs target\n"
        "  /trade — Yes/No % on Bayse\n"
        "  /stats — performance\n\n"
        f"Signal fires {SIGNAL_DELAY_MINS} mins into each round."
    )
    log.info("Bot started")

    while True:
        try:
            handle_commands()
            update_candle_window()

            now    = datetime.now(timezone.utc)
            market = fetch_btc_bayse_market()

            if not market:
                log.info("No BTC market — waiting...")
                time.sleep(30)
                continue

            # Cache market for /price and /trade commands
            current_market_cache = market
            current_event_id     = market["event_id"]
            btc_price            = CANDLE_WINDOW[-1]["close"] if CANDLE_WINDOW else 0
            price_target         = parse_price_target(market.get("rules", ""))

            # ── 12h report ────────────────────────────────────
            hours_since = (now - last_report_time).total_seconds() / 3600
            if hours_since >= REPORT_INTERVAL_HOURS and stats["total"] > 0:
                tg_send(format_report_message())
                last_report_time = now

            # ── New round detected ────────────────────────────
            if current_event_id != last_event_id:
                log.info(f"New round detected: {current_event_id}")

                # ── Resolve previous round BEFORE sending open alert ──
                if prev_event_id and pending_signal:
                    log.info(f"Resolving previous round: {prev_event_id}")
                    prev = fetch_resolved_outcome(prev_event_id)

                    if prev and prev.get("resolved_outcome"):
                        resolved     = prev["resolved_outcome"]
                        final_price  = btc_price
                        predicted_up = pending_signal["direction"].startswith("BUY YES")
                        actual_up    = resolved.lower() in ["up", "yes"]
                        correct      = predicted_up == actual_up
                        high_conv    = pending_signal["signal"] == "✅ TRADE"

                        # Update stats
                        stats["total"] += 1
                        if correct:
                            stats["correct"] += 1
                        else:
                            stats["incorrect"] += 1
                        if high_conv:
                            stats["high_conv_total"] += 1
                            if correct:
                                stats["high_conv_correct"] += 1

                        # Update last_round for open alert display
                        last_round["target"]         = pending_signal.get("price_target")
                        last_round["final_price"]    = final_price
                        last_round["direction"]      = "UP" if actual_up else "DOWN"
                        last_round["bot_prediction"] = "UP 📈" if predicted_up else "DOWN 📉"
                        last_round["would_have_won"] = correct
                        last_round["resolved"]       = True

                        # Log to CSV
                        entry = {
                            "timestamp"      : now.isoformat(),
                            "time"           : now.strftime("%H:%M"),
                            "direction_short": "UP" if predicted_up else "DOWN",
                            "confidence"     : pending_signal["confidence"],
                            "correct"        : correct,
                            "high_conviction": high_conv,
                            "actual"         : resolved,
                            "btc_price"      : final_price,
                            "price_target"   : pending_signal.get("price_target"),
                            "yes_price"      : pending_signal.get("yes_price"),
                            "edge"           : pending_signal.get("edge"),
                            "mins_remaining" : pending_signal.get("mins_remaining")
                        }
                        trade_log.append(entry)
                        append_to_log(entry)
                        log.info(f"Round resolved: {'✅' if correct else '❌'} | Stats: {stats}")
                    else:
                        log.warning(f"Could not resolve {prev_event_id} — no resolved_outcome yet")

                # Send open alert (includes last round result)
                if price_target:
                    tg_send(format_open_alert(market, btc_price, price_target, now))

                # Reset state
                prev_event_id    = current_event_id
                last_event_id    = current_event_id
                signal_fired     = False
                open_alert_sent  = True
                round_start_time = now
                pending_signal   = None

            # ── Fire signal 5 mins into round ─────────────────
            if not signal_fired and round_start_time:
                mins_into_round = (now - round_start_time).total_seconds() / 60
                if mins_into_round >= SIGNAL_DELAY_MINS:
                    feat_vals, vol_regime = compute_features()
                    if feat_vals is not None:
                        result         = get_signal(market, feat_vals, vol_regime, now)
                        pending_signal = result
                        signal_fired   = True
                        tg_send(format_signal_message(result, now))
                        log.info(
                            f"Signal at {mins_into_round:.1f} mins: "
                            f"{result['signal']} | {result['direction']} | {result['confidence']:.1%}"
                        )
                    else:
                        log.warning("Not enough candles for features")

            time.sleep(30)

        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
