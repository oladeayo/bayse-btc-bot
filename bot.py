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
CANDLE_WINDOW = deque(maxlen=100)

current_round = {
    "event_id"         : None,
    "start_time"       : None,
    "round_start_price": None,
    "signal_fired"     : False,
    "pending_signal"   : None,
}

# ── Log-based result tracking ─────────────────────────────────
# Instead of relying on Bayse API resolvedOutcome (which is unreliable),
# we track results by comparing:
#   - round_start_price (the Bayse target)
#   - btc_price_at_signal (BTC when we signalled)
#   - btc_price_at_end (BTC when next round starts = approx round end)
# When a new round opens, we close out the previous round using current BTC price

pending_result = {
    "event_id"         : None,
    "round_start_price": None,
    "btc_at_signal"    : None,
    "bot_direction"    : None,   # "UP" or "DOWN"
    "confidence"       : None,
    "high_conviction"  : False,
    "signal_time"      : None,
    "yes_price_at_signal": None,
    "no_price_at_signal" : None,
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
            log.error(f"TG error: {r.status_code} {r.text[:150]}")
    except Exception as e:
        log.error(f"TG exception: {e}")

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
        elif text.startswith("/log")  : tg_send(build_log_message(), chat_id)
        elif text.startswith("/start"): tg_send(start_message(), chat_id)

def start_message():
    return (
        "🤖 *Bayse BTC Bot v2*\n\n"
        "Commands:\n"
        "  /price — BTC vs target + Bayse odds\n"
        "  /trade — current Yes/No % on Bayse\n"
        "  /stats — performance summary\n"
        "  /log   — last 10 trade results\n\n"
        f"Model confidence: {CONFIDENCE:.0%} | Fee: {FEE:.0%}"
    )

def cmd_price(chat_id):
    """BTC vs target AND current Yes/No % — combined."""
    btc    = get_btc_price()
    target = current_round.get("round_start_price")
    m      = current_market_cache
    now    = datetime.now(timezone.utc)

    if btc and target and m:
        diff     = btc - target
        diff_pct = (diff / target) * 100
        arrow    = "📈 ABOVE" if diff > 0 else "📉 BELOW"
        yes      = m.get("yes_price", 0)
        no       = m.get("no_price", 0)
        crowd    = "YES favoured" if yes > 0.5 else "NO favoured" if no > 0.5 else "Even"

        tg_send(
            f"💰 *BTC Snapshot*\n"
            f"{'─'*28}\n"
            f"  Current BTC : ${btc:,.2f}\n"
            f"  Round Target: ${target:,.2f}\n"
            f"  Gap         : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
            f"  Status      : {arrow} target\n"
            f"{'─'*28}\n"
            f"  YES (UP)    : {yes:.0%}\n"
            f"  NO (DOWN)   : {no:.0%}\n"
            f"  Crowd says  : {crowd}\n"
            f"{'─'*28}\n"
            f"  Orders      : {m.get('total_orders', 0)}\n"
            f"  Liquidity   : ${m.get('liquidity', 0):,.2f}\n"
            f"  Time        : {now.strftime('%H:%M:%S UTC')}",
            chat_id
        )
    else:
        tg_send("⚠️ No active round data yet.", chat_id)

def cmd_trade(chat_id):
    """Same as /price — redirect."""
    cmd_price(chat_id)

# ── Trade log ─────────────────────────────────────────────────
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "round_start_price", "btc_at_signal",
                "btc_at_end", "bot_direction", "actual_direction",
                "correct", "confidence", "conviction",
                "yes_price", "no_price", "price_diff_pct"
            ])

def write_log(entry):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            entry.get("timestamp", ""),
            entry.get("round_start_price", ""),
            entry.get("btc_at_signal", ""),
            entry.get("btc_at_end", ""),
            entry.get("bot_direction", ""),
            entry.get("actual_direction", ""),
            entry.get("correct", ""),
            entry.get("confidence", ""),
            "HIGH" if entry.get("high_conviction") else "LOW",
            entry.get("yes_price", ""),
            entry.get("no_price", ""),
            entry.get("price_diff_pct", ""),
        ])

def build_log_message():
    """Last 10 trades as readable report."""
    if not trade_log:
        return "📋 *Trade Log*\nNo completed trades yet."

    lines = ["📋 *Last 10 Trades*\n" + "─"*30]
    for t in trade_log[-10:]:
        e      = "✅" if t["correct"] else "❌"
        conv   = "⚡" if t["high_conviction"] else "⚠️"
        target = t.get("round_start_price", 0)
        end    = t.get("btc_at_end", 0)
        diff   = end - target if (end and target) else 0
        lines.append(
            f"{e}{conv} *{t['time']}*\n"
            f"  Bot: {t['bot_direction']} ({t['confidence']:.0%}) | "
            f"Actual: {t['actual_direction']}\n"
            f"  Target: ${target:,.2f} → End: ${end:,.2f} ({diff:+.2f})\n"
            f"  YES {t.get('yes_price', 0):.0%} | NO {t.get('no_price', 0):.0%}"
        )

    wr = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    lines.append(f"─"*30)
    lines.append(f"📊 {stats['total']} rounds | {wr} WR | ✅{stats['correct']} ❌{stats['incorrect']}")
    return "\n".join(lines)

def build_stats_message():
    now  = datetime.now(timezone.utc)
    wr   = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    hcwr = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
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
        c = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(
            f"{e}{c} {t['time']} | {t['bot_direction']} | "
            f"{t['confidence']:.0%} | {t['actual_direction']}"
        )
    table = "\n".join(lines) if lines else "No trades yet"

    return (
        f"📊 *BOT STATS*\n"
        f"{'─'*30}\n"
        f"🕐 {now.strftime('%H:%M UTC')}\n"
        f"{'─'*30}\n"
        f"📈 All time  : {stats['total']} rounds | {wr} WR\n"
        f"   ✅ {stats['correct']} correct | ❌ {stats['incorrect']} wrong\n"
        f"⏱ Last hour : {len(last_1h)} rounds | {l1hr} WR\n"
        f"🔟 Last 10   : {l10r} WR\n"
        f"⚡ High conv : {stats['high_conv_total']} | {hcwr} WR\n"
        f"{'─'*30}\n"
        f"`{table}`"
    )

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
                            params={"symbol": "BTC-USDT", "type": "1min", "pageSize": 3},
                            timeout=5)
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
                        "event_id"   : detail["id"],
                        "yes_price"  : m.get("outcome1Price", 0),
                        "no_price"   : m.get("outcome2Price", 0),
                        "total_orders": m.get("totalOrders", 0),
                        "liquidity"  : detail.get("liquidity", 0),
                        "created_at" : detail.get("createdAt", ""),
                        "rules"      : m.get("rules", ""),
                        "status"     : m.get("status", ""),
                    }
    except Exception as e:
        log.error(f"Bayse fetch error: {e}")
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
        atr_val = float(latest["atr"]) if latest["atr"] > 0 else 1e-8

        price_vs_target  = (latest["close"] - round_start_price) / round_start_price
        price_gap_usd    = latest["close"] - round_start_price
        gap_vs_atr       = price_gap_usd / atr_val
        gap_pct_abs      = abs(price_vs_target)
        early_momentum   = price_vs_target
        rsi_vs_neutral   = float(latest["rsi_14"]) - 50

        extra = {
            "price_vs_target": price_vs_target,
            "price_gap_usd"  : price_gap_usd,
            "gap_vs_atr"     : gap_vs_atr,
            "gap_pct_abs"    : gap_pct_abs,
            "early_momentum" : early_momentum,
            "rsi_vs_neutral" : rsi_vs_neutral,
        }
        for k, v in extra.items():
            latest[k] = v

        return latest[features].values, float(latest["rolling_std_60"])

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

    price_diff     = (btc - round_start_price) if btc else 0
    price_diff_pct = (price_diff / round_start_price * 100) if round_start_price else 0

    feat_vals, vol_regime = compute_features(round_start_price)
    if feat_vals is None:
        return None

    feat_df = pd.DataFrame([feat_vals], columns=features)
    proba   = model.predict_proba(feat_df)[0][1]
    yes_p   = market["yes_price"]
    no_p    = market["no_price"]
    edge    = abs(proba - yes_p)

    direction     = "BUY YES (UP) 📈" if proba > 0.5 else "BUY NO (DOWN) 📉"
    direction_s   = "UP" if proba > 0.5 else "DOWN"
    high_conv     = proba > CONFIDENCE or proba < (1 - CONFIDENCE)
    conviction    = "⚡ HIGH" if high_conv else "⚠️ LOW"

    if vol_regime and vol_regime > 0.003:
        signal = "⛔ NO TRADE"
        reason = "Volatility too high"
    elif 0 < secs_remaining < 120:
        signal = "⛔ NO TRADE"
        reason = f"Only {secs_remaining:.0f}s left"
    elif high_conv:
        signal = "✅ TRADE"
        reasons = [
            f"Model: {proba:.1%} {'UP' if proba > 0.5 else 'DOWN'}",
            f"BTC ${price_diff:+.2f} ({price_diff_pct:+.3f}%) vs target",
            f"Crowd: YES {yes_p:.0%} | NO {no_p:.0%}",
            f"Model edge over crowd: {edge:.1%}"
        ]
        reason = " | ".join(reasons)
    else:
        signal = "⛔ NO TRADE"
        reason = f"Confidence {proba:.1%} below {CONFIDENCE:.0%}"

    return {
        "signal"           : signal,
        "direction"        : direction,
        "direction_short"  : direction_s,
        "conviction"       : conviction,
        "confidence"       : round(proba, 4),
        "edge"             : round(edge, 4),
        "btc_price"        : btc,
        "round_start_price": round_start_price,
        "price_diff"       : round(price_diff, 2),
        "price_diff_pct"   : round(price_diff_pct, 4),
        "mins_remaining"   : mins_remaining,
        "yes_price"        : yes_p,
        "no_price"         : no_p,
        "total_orders"     : market.get("total_orders", 0),
        "liquidity"        : market.get("liquidity", 0),
        "reason"           : reason,
        "timestamp"        : now.isoformat(),
        "high_conviction"  : high_conv,
    }

# ── Messages ──────────────────────────────────────────────────
def msg_open_alert(market, btc, target, now, mins_remaining):
    diff     = btc - target
    diff_pct = (diff / target * 100) if target else 0

    if last_completed["resolved"]:
        lc      = last_completed
        l_diff  = (lc["btc_at_end"] - lc["round_start_price"]) if lc["btc_at_end"] else 0
        l_dir   = "UP 📈" if l_diff >= 0 else "DOWN 📉"
        won_txt = "✅ WON" if lc["correct"] else "❌ LOST"
        wr      = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
        last_block = (
            f"{'─'*30}\n"
            f"📋 *Last Round*\n"
            f"  Target : ${lc['round_start_price']:,.2f}\n"
            f"  Ended  : ${lc['btc_at_end']:,.2f} → {l_dir}\n"
            f"  Bot    : {lc['bot_direction']} → {won_txt}\n"
            f"  Session: {stats['total']} rounds | {wr} WR\n"
        )
    else:
        last_block = f"{'─'*30}\n📋 *Last Round:* No data yet\n"

    return (
        f"🔔 *NEW BTC ROUND*\n"
        f"{'─'*30}\n"
        f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ ~{mins_remaining} mins\n"
        f"{'─'*30}\n"
        f"💰 BTC Now   : ${btc:,.2f}\n"
        f"🎯 Target    : ${target:,.2f}\n"
        f"💹 Gap       : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
        f"📊 Bayse     : YES {market['yes_price']:.0%} | NO {market['no_price']:.0%}\n"
        f"{'─'*30}\n"
        + last_block +
        f"{'─'*30}\n"
        f"⏳ Signal in ~{SIGNAL_DELAY_MINS} mins..."
    )

def msg_signal(result, now):
    return (
        f"🤖 *TRADE SIGNAL — {now.strftime('%H:%M UTC')}*\n"
        f"{'─'*30}\n"
        f"💰 BTC     : ${result['btc_price']:,.2f}\n"
        f"🎯 Target  : ${result['round_start_price']:,.2f}\n"
        f"💹 Gap     : ${result['price_diff']:+.2f} ({result['price_diff_pct']:+.3f}%)\n"
        f"⏱ Left    : {result['mins_remaining']} mins\n"
        f"{'─'*30}\n"
        f"📊 *Bayse Market*\n"
        f"   YES (UP)  : {result['yes_price']:.0%}\n"
        f"   NO (DOWN) : {result['no_price']:.0%}\n"
        f"   Orders    : {result['total_orders']}\n"
        f"{'─'*30}\n"
        f"🔔 *Signal    : {result['signal']}*\n"
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
    for t in trade_log[-20:]:
        e = "✅" if t["correct"] else "❌"
        c = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(
            f"{e}{c} {t['time']} | {t['bot_direction']} | "
            f"{t['confidence']:.0%} | {t['actual_direction']}"
        )
    log_text = "\n".join(lines) if lines else "No trades yet"
    return (
        f"📊 *{REPORT_INTERVAL_HOURS}H REPORT*\n"
        f"{'─'*30}\n"
        f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"  All      : {stats['total']} rounds | {wr} WR\n"
        f"  High conv: {stats['high_conv_total']} | {hcwr} WR\n"
        f"{'─'*30}\n"
        f"`{log_text}`"
    )

# ── Main loop ─────────────────────────────────────────────────
def main():
    global current_market_cache, last_report_time

    init_log()
    seed_candles()
    tg_send(start_message())
    log.info("Bot v2 started")

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
            event_id  = market["event_id"]
            btc       = get_btc_price()
            rules     = market.get("rules", "")
            target    = parse_target(rules)
            end_dt    = parse_end_dt(rules, now)
            mins_left = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None

            # ── 12h report ────────────────────────────────────
            if (now - last_report_time).total_seconds() / 3600 >= REPORT_INTERVAL_HOURS and stats["total"] > 0:
                tg_send(msg_report())
                last_report_time = now

            # ── New round detected ────────────────────────────
            if event_id != current_round["event_id"]:
                log.info(f"New round: {event_id}")

                # ── Close out previous round using current BTC price ──
                # When a new round opens, the current BTC price IS the
                # final price of the previous round (Bayse resolves at
                # the boundary, and the new round starts right after)
                prev_signal = current_round.get("pending_signal")
                prev_target = current_round.get("round_start_price")

                if prev_signal and prev_target and btc:
                    final_btc    = btc
                    actual_up    = final_btc >= prev_target   # Bayse: UP if >= target
                    predicted_up = prev_signal["direction_short"] == "UP"
                    correct      = predicted_up == actual_up
                    high_conv    = prev_signal["high_conviction"]
                    actual_dir   = "UP" if actual_up else "DOWN"

                    # Update stats
                    stats["total"] += 1
                    stats["correct" if correct else "incorrect"] += 1
                    if high_conv:
                        stats["high_conv_total"] += 1
                        if correct:
                            stats["high_conv_correct"] += 1

                    # Update last_completed for open alert display
                    last_completed.update({
                        "round_start_price": prev_target,
                        "btc_at_end"       : final_btc,
                        "actual_direction" : actual_dir,
                        "bot_direction"    : prev_signal["direction_short"],
                        "correct"          : correct,
                        "resolved"         : True,
                    })

                    # Write to log
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
                        "high_conviction"  : high_conv,
                        "yes_price"        : prev_signal.get("yes_price", ""),
                        "no_price"         : prev_signal.get("no_price", ""),
                        "price_diff_pct"   : prev_signal.get("price_diff_pct", ""),
                    }
                    trade_log.append(entry)
                    write_log(entry)
                    log.info(f"Round closed: {'✅' if correct else '❌'} | "
                             f"Target: {prev_target:.2f} | Final: {final_btc:.2f} | "
                             f"Actual: {actual_dir} | Bot: {prev_signal['direction_short']} | "
                             f"Stats: {stats}")

                elif current_round["event_id"] and not prev_signal:
                    log.info("Previous round had no signal fired — skipping result")

                # ── Reset for new round ───────────────────────
                current_round.update({
                    "event_id"         : event_id,
                    "start_time"       : now,
                    "round_start_price": target,
                    "signal_fired"     : False,
                    "pending_signal"   : None,
                })

                # Send open alert
                if target and btc and mins_left:
                    tg_send(msg_open_alert(market, btc, target, now, mins_left))
                    log.info(f"Open alert sent | Target: {target} | BTC: {btc}")

            # ── Fire signal 5 mins in ─────────────────────────
            start_time = current_round.get("start_time")
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
                        log.info(
                            f"Signal fired at {mins_in:.1f} mins: "
                            f"{result['signal']} {result['direction_short']} "
                            f"{result['confidence']:.1%}"
                        )
                    else:
                        log.warning("Not enough candles for features")

            time.sleep(30)

        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
