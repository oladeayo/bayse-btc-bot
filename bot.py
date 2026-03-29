import asyncio
import logging
import requests
import joblib
import json
import re
import os
import csv
import sys
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone, timedelta
from collections import deque
from telegram import Bot
from telegram.constants import ParseMode

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
HEADERS               = {"X-Public-Key": BAYSE_KEY}
CONFIDENCE            = 0.58
FEE                   = 0.05
REPORT_INTERVAL_HOURS = 12
SIGNAL_DELAY_MINS     = 5    # wait 5 mins into round before firing signal
LOG_FILE              = "trade_log.csv"

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
CANDLE_WINDOW      = deque(maxlen=100)
last_event_id      = None
pending_signal     = None   # signal waiting for result
pending_event_id   = None   # event ID of pending signal
signal_fired       = False  # has signal been sent for current round
round_start_time   = None   # when current round started
last_report_time   = datetime.now(timezone.utc)

stats = {
    "total"            : 0,
    "correct"          : 0,
    "incorrect"        : 0,
    "no_trade"         : 0,
    "high_conv_total"  : 0,
    "high_conv_correct": 0,
}

trade_log = []

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
    r      = requests.get(url, params=params)
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
    url    = "https://api.kucoin.com/api/v1/market/candles"
    params = {"symbol": "BTC-USDT", "type": "1min", "pageSize": 3}
    r      = requests.get(url, params=params)
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

# ── Bayse ─────────────────────────────────────────────────────
def fetch_btc_bayse_market():
    """Fetch current BTC 15-min market with full detail."""
    try:
        r      = requests.get(f"{BASE_URL}/pm/events", headers=HEADERS,
                              params={"limit": 50}, timeout=10)
        events = r.json().get("events", [])
        for event in events:
            title = event.get("title", "").lower()
            if "bitcoin" in title and "15" in title:
                r2      = requests.get(f"{BASE_URL}/pm/events/{event['id']}",
                                       headers=HEADERS, timeout=10)
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

def fetch_event_by_id(event_id):
    """Fetch a specific event by ID to check resolution."""
    try:
        r      = requests.get(f"{BASE_URL}/pm/events/{event_id}",
                              headers=HEADERS, timeout=10)
        detail = r.json()
        markets = detail.get("markets", [])
        if markets:
            m = markets[0]
            return {
                "event_id"        : detail["id"],
                "resolved_outcome": m.get("resolvedOutcome", ""),
                "status"          : m.get("status", ""),
                "yes_price"       : m.get("outcome1Price", 0),
                "no_price"        : m.get("outcome2Price", 0),
            }
    except Exception as e:
        log.error(f"Event fetch error: {e}")
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
def get_signal(market, feature_values, vol_regime):
    now           = datetime.now(timezone.utc)
    current_price = CANDLE_WINDOW[-1]["close"]
    rules         = market.get("rules", "")

    # Parse price target
    price_match  = re.findall(r'\$([\d,]+\.?\d*)', rules)
    price_target = float(price_match[0].replace(",", "")) if price_match else None

    # Parse end time
    end_match = re.search(r'(\d+:\d+:\d+\s?[AP]M)\s?GMT', rules)
    end_dt    = None
    if end_match:
        today  = now.strftime("%Y-%m-%d")
        end_dt = datetime.strptime(f"{today} {end_match.group(1).strip()}", "%Y-%m-%d %I:%M:%S %p")
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    mins_remaining = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None
    secs_remaining = (end_dt - now).total_seconds() if end_dt else 999
    price_diff     = (current_price - price_target) if price_target else 0
    price_diff_pct = (price_diff / price_target * 100) if price_target else 0

    # Model prediction
    feat_df = pd.DataFrame([feature_values], columns=features)
    proba   = model.predict_proba(feat_df)[0][1]
    edge    = abs(proba - market["yes_price"])

    direction  = "BUY YES (UP) 📈" if proba > 0.5 else "BUY NO (DOWN) 📉"
    conviction = "⚡ HIGH CONVICTION" if (proba > CONFIDENCE or proba < (1 - CONFIDENCE)) else "⚠️ LOW CONVICTION"

    # Guards
    if vol_regime > 0.003:
        signal = "⛔ NO TRADE"
        reason = "Volatility too high — model unreliable"
    elif 0 < secs_remaining < 120:
        signal = "⛔ NO TRADE"
        reason = f"Only {secs_remaining:.0f}s left — too late"
    elif proba > CONFIDENCE or proba < (1 - CONFIDENCE):
        signal = "✅ TRADE"
        # ── Clean reasons — model only, no Bayse crowd ──────
        reasons = [
            f"Model confidence: {proba:.1%} {'UP' if proba > 0.5 else 'DOWN'}",
            f"RSI & momentum align with {'bullish' if proba > 0.5 else 'bearish'} signal",
            f"BTC ${price_diff:+.2f} ({price_diff_pct:+.3f}%) vs round start price",
            f"Edge over market price: {edge:.1%}"
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
        "end_time"      : end_dt.strftime("%H:%M:%S UTC") if end_dt else "unknown",
        "yes_price"     : market["yes_price"],
        "no_price"      : market["no_price"],
        "total_orders"  : market["total_orders"],
        "liquidity"     : market["liquidity"],
        "reason"        : reason,
        "event_id"      : market["event_id"],
        "market_id"     : market["market_id"],
        "created_at"    : market["created_at"],
        "timestamp"     : now.isoformat()
    }

# ── Message formatters ────────────────────────────────────────
def format_signal_message(r, last_trade=None):
    win_rate = f"{stats['correct']/stats['total']*100:.1f}%" if stats['total'] > 0 else "N/A"

    # Last round result block
    if last_trade:
        emoji     = "✅" if last_trade["correct"] else "❌"
        conv_icon = "⚡" if last_trade["high_conviction"] else "⚠️"
        actual    = last_trade["actual"].upper()
        last_block = (
            f"\n{'─' * 30}\n"
            f"📋 *Last Round Result*\n"
            f"  {emoji} Predicted: *{last_trade['direction_short']}* | Actual: *{actual}*\n"
            f"  {conv_icon} Confidence: {last_trade['confidence']:.0%} | "
            f"{'CORRECT ✅' if last_trade['correct'] else 'WRONG ❌'}\n"
            f"  📊 Session: {stats['total']} rounds | {win_rate} win rate"
        )
    else:
        last_block = (
            f"\n{'─' * 30}\n"
            f"📋 *Last Round:* First round of session"
        )

    return (
        f"🔔 *NEW BAYSE BTC ROUND*\n"
        f"{'─' * 30}\n"
        f"📍 *Time:* {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}\n"
        f"💰 *BTC Price:* ${r['current_price']:,.2f}\n"
        f"🎯 *Round Start Price:* ${r['price_target']:,.2f}\n"
        f"💹 *Price diff:* ${r['price_diff']:+.2f} ({r['price_diff_pct']:+.3f}%)\n"
        f"⏱ *Time Left:* {r['mins_remaining']} mins\n"
        f"{'─' * 30}\n"
        f"🔔 *Signal:* {r['signal']}\n"
        f"📈 *Direction:* {r['direction']}\n"
        f"⚡ *Conviction:* {r['conviction']}\n"
        f"🎯 *Confidence:* {r['confidence']:.1%}\n"
        f"📉 *Edge vs Market:* {r['edge']:.1%}\n"
        f"{'─' * 30}\n"
        f"💡 *Reasons:*\n"
        + "\n".join([f"  • {x}" for x in r['reason'].split(' | ')]) +
        last_block
    )

def format_result_message(signal_r, resolved_outcome, final_price):
    predicted_up = signal_r["direction"].startswith("BUY YES")
    actual_up    = resolved_outcome.lower() in ["up", "yes"]
    correct      = predicted_up == actual_up
    emoji        = "✅" if correct else "❌"
    win_rate     = f"{stats['correct']/stats['total']*100:.1f}%" if stats['total'] > 0 else "N/A"
    hc_rate      = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
                    if stats['high_conv_total'] > 0 else "N/A")

    return (
        f"{emoji} *ROUND RESULT*\n"
        f"{'─' * 30}\n"
        f"🤖 *Predicted:* {signal_r['direction']}\n"
        f"📊 *Actual outcome:* {'UP 📈' if actual_up else 'DOWN 📉'}\n"
        f"🎯 *Confidence was:* {signal_r['confidence']:.1%} ({signal_r['conviction']})\n"
        f"💰 *Final BTC:* ${final_price:,.2f}\n"
        f"🎯 *Round start was:* ${signal_r['price_target']:,.2f}\n"
        f"{'─' * 30}\n"
        f"📊 *All signals:* {stats['total']} | ✅ {stats['correct']} | ❌ {stats['incorrect']} | {win_rate} WR\n"
        f"⚡ *High conviction:* {stats['high_conv_total']} signals | {hc_rate} WR"
    )

def format_report_message():
    win_rate = f"{stats['correct']/stats['total']*100:.1f}%" if stats['total'] > 0 else "N/A"
    hc_rate  = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
                if stats['high_conv_total'] > 0 else "N/A")

    log_lines = []
    for t in trade_log[-25:]:
        emoji = "✅" if t["correct"] else "❌"
        conv  = "⚡" if t["high_conviction"] else "⚠️"
        log_lines.append(
            f"{emoji}{conv} {t['time']} | {t['direction_short']} | "
            f"{t['confidence']:.0%} | {'RIGHT' if t['correct'] else 'WRONG'}"
        )

    log_text = "\n".join(log_lines) if log_lines else "No completed trades yet"

    return (
        f"📊 *{REPORT_INTERVAL_HOURS}H PERFORMANCE REPORT*\n"
        f"{'─' * 30}\n"
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{'─' * 30}\n"
        f"📈 *All Signals*\n"
        f"  • Total rounds: {stats['total']}\n"
        f"  • Correct: {stats['correct']} | Wrong: {stats['incorrect']}\n"
        f"  • Win rate: {win_rate}\n"
        f"{'─' * 30}\n"
        f"⚡ *High Conviction Only*\n"
        f"  • Signals: {stats['high_conv_total']}\n"
        f"  • Correct: {stats['high_conv_correct']}\n"
        f"  • Win rate: {hc_rate}\n"
        f"{'─' * 30}\n"
        f"📋 *Last 25 trades*\n"
        f"`{log_text}`"
    )

# ── Main bot loop ─────────────────────────────────────────────
async def run_bot():
    global last_event_id, pending_signal, pending_event_id
    global signal_fired, round_start_time, last_report_time

    init_log_file()
    bot = Bot(token=TELEGRAM_TOKEN)

    await bot.send_message(
        chat_id=TELEGRAM_CHAT,
        text=(
            "🤖 *Bayse BTC Bot is live!*\n"
            f"Confidence threshold: {CONFIDENCE:.0%}\n"
            f"Signal fires {SIGNAL_DELAY_MINS} mins into each round.\n"
            f"Reports every {REPORT_INTERVAL_HOURS} hours."
        ),
        parse_mode=ParseMode.MARKDOWN
    )
    log.info("Bot started")
    seed_candle_window()

    while True:
        try:
            update_candle_window()
            now    = datetime.now(timezone.utc)
            market = fetch_btc_bayse_market()

            if not market:
                log.info("No BTC market — waiting...")
                await asyncio.sleep(30)
                continue

            current_event_id = market["event_id"]

            # ── Send 12h report if due ────────────────────────
            hours_since = (now - last_report_time).total_seconds() / 3600
            if hours_since >= REPORT_INTERVAL_HOURS and stats["total"] > 0:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT,
                    text=format_report_message(),
                    parse_mode=ParseMode.MARKDOWN
                )
                last_report_time = now
                log.info("Report sent")

            # ── New round detected ────────────────────────────
            if current_event_id != last_event_id:
                log.info(f"New round: {current_event_id}")

                # ── Check result of previous round ────────────
                # Fetch the PREVIOUS event by ID to get resolved outcome
                if pending_signal and pending_event_id:
                    log.info(f"Checking result for previous event: {pending_event_id}")
                    prev = fetch_event_by_id(pending_event_id)

                    if prev and prev.get("resolved_outcome"):
                        resolved     = prev["resolved_outcome"]
                        final_price  = CANDLE_WINDOW[-1]["close"]
                        predicted_up = pending_signal["direction"].startswith("BUY YES")
                        actual_up    = resolved.lower() in ["up", "yes"]
                        correct      = predicted_up == actual_up
                        high_conv    = pending_signal["signal"] == "✅ TRADE"

                        stats["total"] += 1
                        if correct:
                            stats["correct"] += 1
                        else:
                            stats["incorrect"] += 1

                        if high_conv:
                            stats["high_conv_total"] += 1
                            if correct:
                                stats["high_conv_correct"] += 1
                        else:
                            stats["no_trade"] += 1

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

                        result_msg = format_result_message(pending_signal, resolved, final_price)
                        await bot.send_message(
                            chat_id=TELEGRAM_CHAT,
                            text=result_msg,
                            parse_mode=ParseMode.MARKDOWN
                        )
                        log.info(f"Result: {'✅' if correct else '❌'} | {stats}")
                    else:
                        log.warning(f"No resolved outcome for {pending_event_id} yet")

                # Reset for new round
                last_event_id   = current_event_id
                signal_fired    = False
                round_start_time = now
                pending_event_id = current_event_id  # track this round for result later
                log.info(f"New round started at {now.strftime('%H:%M:%S UTC')}")

            # ── Fire signal after SIGNAL_DELAY_MINS ──────────
            if not signal_fired and round_start_time:
                mins_into_round = (now - round_start_time).total_seconds() / 60
                if mins_into_round >= SIGNAL_DELAY_MINS:
                    feat_vals, vol_regime = compute_features()
                    if feat_vals is not None:
                        result        = get_signal(market, feat_vals, vol_regime)
                        pending_signal = result
                        signal_fired  = True

                        last_trade = trade_log[-1] if trade_log else None
                        msg = format_signal_message(result, last_trade)
                        await bot.send_message(
                            chat_id=TELEGRAM_CHAT,
                            text=msg,
                            parse_mode=ParseMode.MARKDOWN
                        )
                        log.info(f"Signal fired at {mins_into_round:.1f} mins into round: "
                                 f"{result['signal']} | {result['direction']} | {result['confidence']:.1%}")
                    else:
                        log.warning("Not enough candles")

            await asyncio.sleep(30)

        except Exception as e:
            log.error(f"Bot error: {e}", exc_info=True)
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run_bot())
