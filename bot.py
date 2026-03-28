import asyncio
import logging
import requests
import joblib
import json
import re
import os
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone, timedelta
from collections import deque
from telegram import Bot
from telegram.constants import ParseMode


# ── Startup validation ───────────────────────────────────────
import sys

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT", "")
BAYSE_KEY      = os.environ.get("BAYSE_KEY", "")

if not TELEGRAM_TOKEN:
    print("ERROR: TELEGRAM_TOKEN environment variable is not set!", flush=True)
    sys.exit(1)
if not TELEGRAM_CHAT:
    print("ERROR: TELEGRAM_CHAT environment variable is not set!", flush=True)
    sys.exit(1)
if not BAYSE_KEY:
    print("ERROR: BAYSE_KEY environment variable is not set!", flush=True)
    sys.exit(1)

print(f"✅ Token loaded: {TELEGRAM_TOKEN[:10]}...", flush=True)
print(f"✅ Chat ID: {TELEGRAM_CHAT}", flush=True)
print(f"✅ Bayse key: {BAYSE_KEY[:10]}...", flush=True)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Config from environment variables ────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT", "")
BAYSE_KEY      = os.environ.get("BAYSE_KEY", "")
BASE_URL       = "https://relay.bayse.markets/v1"
HEADERS        = {"X-Public-Key": BAYSE_KEY}
CONFIDENCE     = 0.58
FEE            = 0.05
REPORT_INTERVAL_HOURS = 12  # send report every 12 hours

# ── Load model ───────────────────────────────────────────────
model    = joblib.load("btc_bayse_model_6m.joblib")
features = json.load(open("features.json"))

# ── State ────────────────────────────────────────────────────
CANDLE_WINDOW      = deque(maxlen=100)
last_event_id      = None
last_signal_result = None
consecutive_losses = 0
last_report_time   = datetime.now(timezone.utc)

stats = {
    "total"    : 0,
    "correct"  : 0,
    "incorrect": 0,
    "no_trade" : 0,
    "high_conv_total"  : 0,
    "high_conv_correct": 0,
}

# Trade log for daily report
trade_log = []

# ── KuCoin ───────────────────────────────────────────────────
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
    r      = requests.get(f"{BASE_URL}/pm/events", headers=HEADERS, params={"limit": 50})
    events = r.json().get("events", [])
    for event in events:
        title = event.get("title", "").lower()
        if "bitcoin" in title and "15" in title:
            r2      = requests.get(f"{BASE_URL}/pm/events/{event['id']}", headers=HEADERS)
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
                    "total_volume"    : detail.get("totalVolume", 0),
                    "created_at"      : detail.get("createdAt", ""),
                    "rules"           : m.get("rules", ""),
                    "resolved_outcome": m.get("resolvedOutcome", "")
                }
    return None

# ── Features ─────────────────────────────────────────────────
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

    price_match  = re.findall(r'\$([\d,]+\.?\d*)', rules)
    price_target = float(price_match[0].replace(",", "")) if price_match else None

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

    feat_df = pd.DataFrame([feature_values], columns=features)
    proba   = model.predict_proba(feat_df)[0][1]
    edge    = abs(proba - market["yes_price"])

    direction  = "BUY YES (UP) 📈" if proba > 0.5 else "BUY NO (DOWN) 📉"
    conviction = "⚡ HIGH CONVICTION" if (proba > CONFIDENCE or proba < (1 - CONFIDENCE)) else "⚠️ LOW CONVICTION"

    if vol_regime > 0.003:
        signal = "⛔ NO TRADE"
        reason = "Volatility too high"
    elif 0 < secs_remaining < 120:
        signal = "⛔ NO TRADE"
        reason = f"Only {secs_remaining:.0f}s left — too late"
    elif proba > CONFIDENCE or proba < (1 - CONFIDENCE):
        signal = "✅ TRADE"
        reason = (f"Model: {proba:.1%} {'UP' if proba > 0.5 else 'DOWN'} | "
                  f"Bayse crowd: YES {market['yes_price']:.0%} | "
                  f"Edge: {edge:.1%} | "
                  f"BTC ${price_diff:+.2f} ({price_diff_pct:+.3f}%) vs target")
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
    
    # Build last trade result summary
    if last_trade:
        emoji  = "✅" if last_trade["correct"] else "❌"
        conv   = "⚡" if last_trade["high_conviction"] else "⚠️"
        last_trade_text = (
            f"\n{'─' * 30}\n"
            f"📋 *Last Round Result*\n"
            f"  {emoji} Predicted: {'UP' if last_trade['direction_short'] == 'UP' else 'DOWN'} | "
            f"Actual: {'UP 📈' if last_trade['actual'].lower() in ['up','yes'] else 'DOWN 📉'}\n"
            f"  {conv} Confidence was: {last_trade['confidence']:.0%} | "
            f"{'CORRECT' if last_trade['correct'] else 'WRONG'}\n"
            f"  📊 Session: {stats['total']} rounds | {win_rate} win rate"
        )
    else:
        last_trade_text = (
            f"\n{'─' * 30}\n"
            f"📋 *Last Round:* No previous round yet"
        )

    return (
        f"🔔 *NEW BAYSE BTC ROUND*\n"
        f"{'─' * 30}\n"
        f"📍 *Time:* {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}\n"
        f"💰 *BTC Price:* ${r['current_price']:,.2f}\n"
        f"🎯 *Price Target:* ${r['price_target']:,.2f}\n"
        f"📊 *Bayse:* YES {r['yes_price']:.0%} | NO {r['no_price']:.0%}\n"
        f"⏱ *Time Left:* {r['mins_remaining']} mins\n"
        f"{'─' * 30}\n"
        f"🔔 *Signal:* {r['signal']}\n"
        f"📈 *Direction:* {r['direction']}\n"
        f"⚡ *Conviction:* {r['conviction']}\n"
        f"🎯 *Confidence:* {r['confidence']:.1%}\n"
        f"📉 *Edge vs Bayse:* {r['edge']:.1%}\n"
        f"{'─' * 30}\n"
        f"💡 *Reasons:*\n"
        + "\n".join([f"  • {x}" for x in r['reason'].split(' | ')]) +
        last_trade_text
    )

def format_result_message(signal_r, resolved_outcome, final_price):
    predicted_up = signal_r["direction"].startswith("BUY YES")
    actual_up    = resolved_outcome.lower() in ["up", "yes"]
    correct      = predicted_up == actual_up
    emoji        = "✅" if correct else "❌"
    win_rate     = f"{stats['correct']/stats['total']*100:.1f}%" if stats['total'] > 0 else "N/A"

    return (
        f"{emoji} *ROUND RESULT*\n"
        f"{'─' * 30}\n"
        f"🤖 *Predicted:* {signal_r['direction']}\n"
        f"📊 *Actual outcome:* {'UP 📈' if actual_up else 'DOWN 📉'}\n"
        f"🎯 *Confidence was:* {signal_r['confidence']:.1%} ({signal_r['conviction']})\n"
        f"💰 *Final BTC:* ${final_price:,.2f}\n"
        f"🎯 *Target was:* ${signal_r['price_target']:,.2f}\n"
        f"{'─' * 30}\n"
        f"📊 *Session:* {stats['total']} rounds | "
        f"✅ {stats['correct']} correct | "
        f"❌ {stats['incorrect']} wrong | "
        f"{win_rate} win rate"
    )

def format_report_message(period_hours):
    """Generate 12-hour or 24-hour performance report."""
    now         = datetime.now(timezone.utc)
    win_rate    = f"{stats['correct']/stats['total']*100:.1f}%" if stats['total'] > 0 else "N/A"
    hc_win_rate = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
                   if stats['high_conv_total'] > 0 else "N/A")

    # Build trade log table
    log_lines = []
    recent_trades = trade_log[-20:]  # last 20 trades
    for t in recent_trades:
        emoji  = "✅" if t["correct"] else "❌"
        conv   = "⚡" if t["high_conviction"] else "⚠️"
        line   = (f"{emoji}{conv} {t['time']} | "
                  f"{t['direction_short']} | "
                  f"{t['confidence']:.0%} conf | "
                  f"{'RIGHT' if t['correct'] else 'WRONG'}")
        log_lines.append(line)

    log_text = "\n".join(log_lines) if log_lines else "No trades recorded yet"

    return (
        f"📊 *{period_hours}H PERFORMANCE REPORT*\n"
        f"{'─' * 30}\n"
        f"🕐 *Period:* Last {period_hours} hours\n"
        f"📅 *Generated:* {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"{'─' * 30}\n"
        f"📈 *Overall Stats*\n"
        f"  • Total rounds tracked: {stats['total']}\n"
        f"  • Correct predictions: {stats['correct']}\n"
        f"  • Wrong predictions: {stats['incorrect']}\n"
        f"  • No trade signals: {stats['no_trade']}\n"
        f"  • Win rate (all): {win_rate}\n"
        f"{'─' * 30}\n"
        f"⚡ *High Conviction Only*\n"
        f"  • Total signals: {stats['high_conv_total']}\n"
        f"  • Correct: {stats['high_conv_correct']}\n"
        f"  • Win rate: {hc_win_rate}\n"
        f"{'─' * 30}\n"
        f"📋 *Recent Trades (last 20)*\n"
        f"`{log_text}`\n"
        f"{'─' * 30}\n"
        f"⚡ = High conviction | ⚠️ = Low conviction\n"
        f"✅ = Correct | ❌ = Wrong"
    )

# ── Main bot loop ─────────────────────────────────────────────
async def run_bot():
    global last_event_id, last_signal_result, consecutive_losses, last_report_time

    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT,
        text=(
            "🤖 *Bayse BTC Bot is live!*\n"
            "Monitoring Bitcoin Up or Down — 15 minutes.\n"
            f"Confidence threshold: {CONFIDENCE:.0%}\n"
            f"Reports every {REPORT_INTERVAL_HOURS} hours."
        ),
        parse_mode=ParseMode.MARKDOWN
    )
    log.info("Bot started")

    seed_candle_window()
    poll_interval = 30

    while True:
        try:
            update_candle_window()
            market = fetch_btc_bayse_market()

            if not market:
                log.info("No BTC market found — waiting...")
                await asyncio.sleep(poll_interval)
                continue

            current_event_id = market["event_id"]

            # ── Check if it's time to send a report ──────────
            now = datetime.now(timezone.utc)
            hours_since_report = (now - last_report_time).total_seconds() / 3600
            if hours_since_report >= REPORT_INTERVAL_HOURS and stats["total"] > 0:
                report_msg   = format_report_message(REPORT_INTERVAL_HOURS)
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT,
                    text=report_msg,
                    parse_mode=ParseMode.MARKDOWN
                )
                last_report_time = now
                log.info(f"Report sent — {stats['total']} rounds tracked")

            # ── New round detected ────────────────────────────
            if current_event_id != last_event_id:
                log.info(f"New round: {current_event_id}")

                # Check result of previous round
                if last_signal_result and last_event_id:
                    await asyncio.sleep(10)  # wait for resolution
                    prev = fetch_btc_bayse_market()
                    resolved = prev.get("resolved_outcome", "") if prev else ""

                    if resolved:
                        final_price  = CANDLE_WINDOW[-1]["close"]
                        predicted_up = last_signal_result["direction"].startswith("BUY YES")
                        actual_up    = resolved.lower() in ["up", "yes"]
                        correct      = predicted_up == actual_up
                        high_conv    = last_signal_result["signal"] == "✅ TRADE"

                        stats["total"] += 1
                        if correct:
                            stats["correct"] += 1
                            consecutive_losses = 0
                        else:
                            stats["incorrect"] += 1
                            consecutive_losses += 1

                        if high_conv:
                            stats["high_conv_total"] += 1
                            if correct:
                                stats["high_conv_correct"] += 1
                        else:
                            stats["no_trade"] += 1

                        # Add to trade log
                        trade_log.append({
                            "time"            : datetime.now(timezone.utc).strftime("%H:%M"),
                            "direction_short" : "UP" if predicted_up else "DOWN",
                            "confidence"      : last_signal_result["confidence"],
                            "correct"         : correct,
                            "high_conviction" : high_conv,
                            "actual"          : resolved,
                            "final_price"     : final_price,
                            "price_target"    : last_signal_result["price_target"]
                        })

                        result_msg = format_result_message(last_signal_result, resolved, final_price)
                        await bot.send_message(
                            chat_id=TELEGRAM_CHAT,
                            text=result_msg,
                            parse_mode=ParseMode.MARKDOWN
                        )
                        log.info(f"Result: {'✅' if correct else '❌'} | Stats: {stats}")

                # Generate signal for new round
                feat_vals, vol_regime = compute_features()
                if feat_vals is not None:
                    result             = get_signal(market, feat_vals, vol_regime)
                    last_signal_result = result
                    last_event_id      = current_event_id

                    last_trade = trade_log[-1] if trade_log else None
                    msg = format_signal_message(result, last_trade)
                    await bot.send_message(
                        chat_id=TELEGRAM_CHAT,
                        text=msg,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    log.info(f"Signal: {result['signal']} | {result['direction']} | {result['confidence']:.1%}")
                else:
                    log.warning("Not enough candles")

            await asyncio.sleep(poll_interval)

        except Exception as e:
            log.error(f"Bot error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run_bot())
