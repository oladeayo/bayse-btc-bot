import logging
import requests
import joblib
import json
import re
import os
import csv
import sys
import time
import io
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone, timedelta
from collections import deque

logging.basicConfig(format="%(asctime)s — %(levelname)s — %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
TELEGRAM_TOKEN        = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT         = os.environ.get("TELEGRAM_CHAT", "")
BAYSE_KEY             = os.environ.get("BAYSE_KEY", "")
BASE_URL              = "https://relay.bayse.markets/v1"
BAYSE_HEADERS         = {"X-Public-Key": BAYSE_KEY}
TELEGRAM_API          = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# ── Strategy config ───────────────────────────────────────────
CONFIDENCE            = 0.70    # minimum model confidence
FEE                   = 0.05    # Bayse fee
SIGNAL_DELAY_MINS     = 5       # fire signal N mins into round
MIN_GAP_PCT           = 0.02    # minimum BTC gap from target (%)
MIN_LIQUIDITY         = 50.0    # minimum Bayse liquidity ($)
MIN_ORDERS            = 10      # minimum orders on Bayse
# Odds filter: only trade when the relevant outcome price is in this range
# e.g. BUY YES only when YES price is between 0.20 and 0.65
# BUY NO only when NO price (= 1 - YES) is between 0.20 and 0.65
ODDS_MIN              = 0.20    # min price of the outcome you're buying
ODDS_MAX              = 0.65    # max price of the outcome you're buying
REPORT_INTERVAL_HOURS = 12
LOG_FILE              = "trade_log.csv"

# ── Startup validation ────────────────────────────────────────
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT or not BAYSE_KEY:
    print("ERROR: Missing env vars", flush=True)
    sys.exit(1)
print(f"✅ Token: {TELEGRAM_TOKEN[:10]}...", flush=True)
print(f"✅ Chat: {TELEGRAM_CHAT}", flush=True)

model    = joblib.load("btc_bayse_model_v2.joblib")
features = json.load(open("features_v2.json"))
print(f"✅ Model loaded | {len(features)} features", flush=True)

# ── State ─────────────────────────────────────────────────────
CANDLE_WINDOW = deque(maxlen=100)

bot_paused    = False
session_active = True
session_start  = datetime.now(timezone.utc)
session_num    = 1

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

stats = {
    "total": 0, "correct": 0, "incorrect": 0,
    "high_conv_total": 0, "high_conv_correct": 0,
    "skipped_odds": 0, "skipped_gap": 0, "skipped_liquidity": 0,
}
trade_log     = []
hour_stats    = {}   # {hour_utc: {"total": N, "correct": N}}

# ── Telegram ──────────────────────────────────────────────────
def tg_send(text, chat_id=None, parse_mode="Markdown"):
    try:
        r = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id or TELEGRAM_CHAT, "text": text, "parse_mode": parse_mode},
            timeout=10
        )
        if not r.ok:
            log.error(f"TG error: {r.status_code} {r.text[:100]}")
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
            log.error(f"TG doc error: {r.status_code}")
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

def handle_commands():
    global bot_paused, session_active, session_start, session_num
    global stats, trade_log, hour_stats, last_completed, current_round

    for update in tg_get_updates():
        msg     = update.get("message", {})
        text    = msg.get("text", "").strip().lower()
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if text.startswith("/pause"):
            bot_paused = True
            tg_send("⏸ *Bot paused.* Monitoring continues but no signals will fire.\nUse /play to resume.", chat_id)

        elif text.startswith("/play"):
            bot_paused = False
            tg_send("▶️ *Bot resumed.* Signals will fire again.", chat_id)

        elif text.startswith("/start_session"):
            session_num  += 1
            session_start = datetime.now(timezone.utc)
            session_active = True
            bot_paused    = False
            stats  = {"total": 0, "correct": 0, "incorrect": 0,
                      "high_conv_total": 0, "high_conv_correct": 0,
                      "skipped_odds": 0, "skipped_gap": 0, "skipped_liquidity": 0}
            trade_log  = []
            hour_stats = {}
            last_completed = {k: None for k in last_completed}
            last_completed["resolved"] = False
            current_round  = {k: (None if k != "signal_fired" else False)
                              for k in current_round}
            current_round["signal_fired"] = False
            tg_send(
                f"🆕 *Session {session_num} started*\n"
                f"Started: {session_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"All stats and trade log cleared.",
                chat_id
            )

        elif text.startswith("/stop_session"):
            session_active = False
            bot_paused     = True
            tg_send(
                build_final_report(),
                chat_id
            )
            tg_send("🛑 *Session stopped.* Bot is paused. Use /start_session to begin a new session.", chat_id)

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

def cmd_price(chat_id):
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
        # Odds filter check
        in_range_yes = ODDS_MIN <= yes <= ODDS_MAX
        in_range_no  = ODDS_MIN <= (1 - yes) <= ODDS_MAX
        odds_status  = "✅ tradeable" if (in_range_yes or in_range_no) else "⛔ outside odds range"
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
            f"  Odds status : {odds_status}\n"
            f"  Orders      : {m.get('total_orders', 0)} | Liquidity: ${m.get('liquidity', 0):,.2f}\n"
            f"  Time        : {now.strftime('%H:%M:%S UTC')}\n"
            f"{'─'*28}\n"
            f"  Paused      : {'Yes ⏸' if bot_paused else 'No ▶️'}\n"
            f"  Session     : #{session_num}",
            chat_id
        )
    else:
        tg_send("⚠️ No active round data yet.", chat_id)

def cmd_export(chat_id):
    if not trade_log:
        tg_send("No trades to export yet.", chat_id)
        return
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "timestamp", "time", "round_start_price", "btc_at_signal",
        "btc_at_end", "bot_direction", "actual_direction", "correct",
        "confidence", "conviction", "yes_price", "no_price",
        "gap_pct", "odds_filter_pass", "liquidity", "orders"
    ])
    writer.writeheader()
    for t in trade_log:
        writer.writerow({k: t.get(k, "") for k in writer.fieldnames})
    filename = f"session_{session_num}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"
    tg_send_document(
        filename,
        buf.getvalue().encode("utf-8"),
        f"Session #{session_num} trade log — {len(trade_log)} trades",
        chat_id
    )

# ── Log ───────────────────────────────────────────────────────
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "time", "round_start_price", "btc_at_signal",
                "btc_at_end", "bot_direction", "actual_direction", "correct",
                "confidence", "conviction", "yes_price", "no_price",
                "gap_pct", "odds_filter_pass", "liquidity", "orders", "session"
            ])

def write_log(entry):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([entry.get(k, "") for k in [
            "timestamp", "time", "round_start_price", "btc_at_signal",
            "btc_at_end", "bot_direction", "actual_direction", "correct",
            "confidence", "conviction", "yes_price", "no_price",
            "gap_pct", "odds_filter_pass", "liquidity", "orders", "session"
        ]])

# ── KuCoin ────────────────────────────────────────────────────
def seed_candles():
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
                         params={"symbol": "BTC-USDT", "type": "1min"}, timeout=10)
        for c in reversed(r.json().get("data", [])):
            CANDLE_WINDOW.append({
                "open_time": pd.Timestamp(int(c[0]), unit="s", tz="UTC"),
                "open": float(c[1]), "close": float(c[2]),
                "high": float(c[3]), "low": float(c[4]), "volume": float(c[5])
            })
        log.info(f"Seeded {len(CANDLE_WINDOW)} candles | ${CANDLE_WINDOW[-1]['close']:,.2f}")
    except Exception as e:
        log.error(f"Seed: {e}")

def update_candles():
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/candles",
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
        log.error(f"Update candles: {e}")

def get_btc_price():
    return CANDLE_WINDOW[-1]["close"] if CANDLE_WINDOW else None

# ── Bayse ─────────────────────────────────────────────────────
def fetch_btc_market():
    try:
        r = requests.get(f"{BASE_URL}/pm/events", headers=BAYSE_HEADERS,
                         params={"limit": 50}, timeout=10)
        for event in r.json().get("events", []):
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
                    }
    except Exception as e:
        log.error(f"Bayse: {e}")
    return None

def parse_target(rules):
    m = re.findall(r'\$([\d,]+\.?\d*)', rules)
    return float(m[0].replace(",", "")) if m else None

def parse_end_dt(rules, now):
    m = re.search(r'(\d+:\d+:\d+\s?[AP]M)\s?GMT', rules)
    if m:
        try:
            end = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {m.group(1).strip()}",
                                    "%Y-%m-%d %I:%M:%S %p").replace(tzinfo=timezone.utc)
            if (end - now).total_seconds() < -3600:
                end += timedelta(days=1)
            return end
        except Exception:
            pass
    return None

# ── Features ──────────────────────────────────────────────────
def compute_features(round_start_price):
    if len(CANDLE_WINDOW) < 60 or not round_start_price:
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

        latest  = df.iloc[-1].copy()
        atr_val = max(float(latest["atr"]), 1e-8)

        pvt = (latest["close"] - round_start_price) / round_start_price
        pgd = latest["close"] - round_start_price
        gva = pgd / atr_val

        latest["price_vs_target"] = pvt
        latest["price_gap_usd"]   = pgd
        latest["gap_vs_atr"]      = gva
        latest["gap_pct_abs"]     = abs(pvt)
        latest["early_momentum"]  = pvt
        latest["rsi_vs_neutral"]  = float(latest["rsi_14"]) - 50

        return latest[features].values, float(latest["rolling_std_60"])
    except Exception as e:
        log.error(f"Features: {e}")
        return None, None

# ── Odds & Kelly ──────────────────────────────────────────────
def check_odds_filter(direction_up, yes_price):
    """
    Only trade when the outcome you're buying is priced between ODDS_MIN and ODDS_MAX.
    BUY YES → check yes_price is in range
    BUY NO  → check no_price (= 1 - yes_price) is in range
    """
    if direction_up:
        outcome_price = yes_price
    else:
        outcome_price = 1.0 - yes_price
    return ODDS_MIN <= outcome_price <= ODDS_MAX, outcome_price

def kelly_fraction(win_prob, outcome_price):
    """
    Kelly criterion: fraction of bankroll to bet.
    outcome_price is the decimal probability of the outcome (e.g. 0.45 for YES at 45%).
    Payout = (1/outcome_price - 1) * (1-fee) per unit staked.
    Kelly = (win_prob * payout - loss_prob) / payout
    Capped at 0.25 to avoid overbetting.
    """
    if outcome_price <= 0 or outcome_price >= 1:
        return 0.05
    payout      = (1.0 / outcome_price - 1) * (1 - FEE)
    win_p       = win_prob
    loss_p      = 1 - win_p
    kelly       = (win_p * payout - loss_p) / payout
    kelly       = max(0.01, min(kelly, 0.25))  # cap at 25% of bankroll
    return round(kelly, 3)

# ── Signal ────────────────────────────────────────────────────
def get_signal(market, round_start_price, now):
    rules          = market.get("rules", "")
    end_dt         = parse_end_dt(rules, now)
    btc            = get_btc_price()
    mins_remaining = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None
    secs_remaining = (end_dt - now).total_seconds() if end_dt else 999

    price_diff     = (btc - round_start_price) if btc else 0
    price_diff_pct = (price_diff / round_start_price * 100) if round_start_price else 0

    yes_p   = market["yes_price"]
    no_p    = market["no_price"]
    liq     = market.get("liquidity", 0)
    orders  = market.get("total_orders", 0)
    abs_gap = abs(price_diff_pct)

    feat_vals, vol_regime = compute_features(round_start_price)
    if feat_vals is None:
        return None

    feat_df     = pd.DataFrame([feat_vals], columns=features)
    proba       = model.predict_proba(feat_df)[0][1]
    direction_up = proba > 0.5
    direction    = "BUY YES (UP) 📈" if direction_up else "BUY NO (DOWN) 📉"
    direction_s  = "UP" if direction_up else "DOWN"

    # Directional confidence (always show from the predicted direction)
    dir_conf     = proba if direction_up else (1 - proba)
    high_conv    = proba > CONFIDENCE or proba < (1 - CONFIDENCE)
    conviction   = "⚡ HIGH" if high_conv else "⚠️ LOW"

    # Odds filter
    odds_pass, outcome_price = check_odds_filter(direction_up, yes_p)

    # Kelly sizing
    kelly = kelly_fraction(dir_conf, outcome_price) if odds_pass else 0

    # Expected value display
    if outcome_price > 0:
        payout = (1.0 / outcome_price - 1) * (1 - FEE)
        ev     = dir_conf * payout - (1 - dir_conf)
    else:
        payout = 0
        ev     = 0

    # Determine signal
    skip_reasons = []
    if not high_conv:
        skip_reasons.append(f"Confidence {dir_conf:.1%} below {CONFIDENCE:.0%}")
    if vol_regime and vol_regime > 0.003:
        skip_reasons.append("Volatility too high")
    if 0 < secs_remaining < 180:
        skip_reasons.append(f"Only {secs_remaining:.0f}s left")
    if abs_gap < MIN_GAP_PCT:
        skip_reasons.append(f"Gap {abs_gap:.3f}% below {MIN_GAP_PCT:.2f}% minimum")
        stats["skipped_gap"] += 1
    if not odds_pass:
        skip_reasons.append(f"Outcome price {outcome_price:.0%} outside {ODDS_MIN:.0%}–{ODDS_MAX:.0%} range")
        stats["skipped_odds"] += 1
    if liq < MIN_LIQUIDITY:
        skip_reasons.append(f"Liquidity ${liq:.0f} below ${MIN_LIQUIDITY:.0f} minimum")
        stats["skipped_liquidity"] += 1
    if orders < MIN_ORDERS:
        skip_reasons.append(f"Only {orders} orders (min {MIN_ORDERS})")

    if skip_reasons:
        signal = "⛔ NO TRADE"
        reason = " | ".join(skip_reasons)
    else:
        signal = "✅ TRADE"
        reason = (
            f"Model: {dir_conf:.1%} {direction_s} | "
            f"Gap: ${price_diff:+.2f} ({price_diff_pct:+.3f}%) | "
            f"Outcome odds: {outcome_price:.0%} → payout {payout:.2f}x | "
            f"EV: {ev:+.3f} | Kelly: {kelly:.1%}"
        )

    return {
        "signal"           : signal,
        "direction"        : direction,
        "direction_short"  : direction_s,
        "conviction"       : conviction,
        "confidence"       : round(dir_conf, 4),   # directional confidence
        "raw_proba"        : round(proba, 4),
        "edge"             : round(abs(proba - yes_p), 4),
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
        "total_orders"     : orders,
        "liquidity"        : liq,
        "odds_filter_pass" : odds_pass,
        "reason"           : reason,
        "timestamp"        : now.isoformat(),
        "high_conviction"  : high_conv,
        "hour_utc"         : now.hour,
    }

# ── Messages ──────────────────────────────────────────────────
def start_message():
    return (
        "🤖 *Bayse BTC Bot v3*\n\n"
        "Commands:\n"
        "  /price        — BTC vs target + Bayse odds\n"
        "  /stats        — performance summary\n"
        "  /log          — last 10 trade results\n"
        "  /hours        — accuracy by hour of day\n"
        "  /config       — current strategy settings\n"
        "  /export       — download trade log CSV\n"
        "  /pause        — pause signals\n"
        "  /play         — resume signals\n"
        "  /start_session — new session (clears stats)\n"
        "  /stop_session  — end session + final report\n\n"
        f"Confidence: {CONFIDENCE:.0%} | Gap min: {MIN_GAP_PCT:.2f}%\n"
        f"Odds range: {ODDS_MIN:.0%}–{ODDS_MAX:.0%} | Liquidity min: ${MIN_LIQUIDITY:.0f}"
    )

def build_config_message():
    return (
        f"⚙️ *Current Config*\n"
        f"{'─'*28}\n"
        f"  Model confidence : {CONFIDENCE:.0%}\n"
        f"  Min gap          : {MIN_GAP_PCT:.2f}%\n"
        f"  Odds range       : {ODDS_MIN:.0%} – {ODDS_MAX:.0%}\n"
        f"  Min liquidity    : ${MIN_LIQUIDITY:.0f}\n"
        f"  Min orders       : {MIN_ORDERS}\n"
        f"  Signal delay     : {SIGNAL_DELAY_MINS} mins\n"
        f"  Fee              : {FEE:.0%}\n"
        f"{'─'*28}\n"
        f"  Paused           : {'Yes ⏸' if bot_paused else 'No ▶️'}\n"
        f"  Session          : #{session_num}\n"
        f"  Session start    : {session_start.strftime('%Y-%m-%d %H:%M UTC')}"
    )

def msg_open_alert(market, btc, target, now, mins_remaining):
    diff     = btc - target
    diff_pct = (diff / target * 100) if target else 0
    yes      = market.get("yes_price", 0)
    no       = market.get("no_price", 0)

    # Odds check for preview
    in_range_up = ODDS_MIN <= yes <= ODDS_MAX
    in_range_dn = ODDS_MIN <= (1 - yes) <= ODDS_MAX
    odds_note   = "✅ odds in range" if (in_range_up or in_range_dn) else "⛔ odds outside range — likely no signal"

    if last_completed["resolved"] and last_completed["round_start_price"]:
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
            f"  Session #{session_num}: {stats['total']} rounds | {wr} WR\n"
        )
    else:
        last_block = f"{'─'*30}\n📋 *Last Round:* No data yet\n"

    paused_note = "\n⏸ *Bot is paused — no signal will fire*" if bot_paused else ""

    return (
        f"🔔 *NEW BTC ROUND — Session #{session_num}*\n"
        f"{'─'*30}\n"
        f"⏰ {now.strftime('%H:%M:%S UTC')} | ⏱ ~{mins_remaining} mins\n"
        f"{'─'*30}\n"
        f"💰 BTC Now   : ${btc:,.2f}\n"
        f"🎯 Target    : ${target:,.2f}\n"
        f"💹 Gap       : ${diff:+.2f} ({diff_pct:+.3f}%)\n"
        f"📊 Bayse     : YES {yes:.0%} | NO {no:.0%}\n"
        f"🎲 {odds_note}\n"
        f"{'─'*30}\n"
        + last_block +
        f"{'─'*30}\n"
        f"⏳ Signal in ~{SIGNAL_DELAY_MINS} mins..."
        + paused_note
    )

def msg_signal(result, now):
    # Format reasons cleanly
    reason_lines = "\n".join([f"  • {x}" for x in result["reason"].split(" | ")])

    # Kelly suggestion
    if result["signal"] == "✅ TRADE":
        kelly_note = (
            f"\n{'─'*30}\n"
            f"💡 *Kelly sizing*\n"
            f"  Bet {result['kelly']:.1%} of bankroll\n"
            f"  e.g. $10 → bet ${10*result['kelly']:.2f}\n"
            f"  Expected value: {result['ev']:+.3f} per $1"
        )
    else:
        kelly_note = ""

    return (
        f"🤖 *SIGNAL — {now.strftime('%H:%M UTC')}*\n"
        f"{'─'*30}\n"
        f"💰 BTC    : ${result['btc_price']:,.2f}\n"
        f"🎯 Target : ${result['round_start_price']:,.2f}\n"
        f"💹 Gap    : ${result['price_diff']:+.2f} ({result['price_diff_pct']:+.3f}%)\n"
        f"⏱ Left   : {result['mins_remaining']} mins\n"
        f"{'─'*30}\n"
        f"📊 *Bayse Market*\n"
        f"   YES (UP)  : {result['yes_price']:.0%}\n"
        f"   NO (DOWN) : {result['no_price']:.0%}\n"
        f"   Outcome   : {result['outcome_price']:.0%} → payout {result['payout']:.2f}x\n"
        f"   Orders    : {result['total_orders']} | Liq: ${result['liquidity']:,.2f}\n"
        f"{'─'*30}\n"
        f"🔔 *Signal    : {result['signal']}*\n"
        f"📈 Direction  : {result['direction']}\n"
        f"⚡ Conviction : {result['conviction']} ({result['confidence']:.1%})\n"
        f"📉 Edge       : {result['edge']:.1%}\n"
        f"{'─'*30}\n"
        f"💡 *Why:*\n{reason_lines}"
        + kelly_note
    )

def build_log_message():
    if not trade_log:
        return "📋 *Trade Log*\nNo completed trades yet."
    lines = [f"📋 *Last 10 Trades — Session #{session_num}*\n" + "─"*30]
    for t in trade_log[-10:]:
        e    = "✅" if t["correct"] else "❌"
        conv = "⚡" if t["high_conviction"] else "⚠️"
        op   = t.get("outcome_price", 0)
        pyt  = t.get("payout", 0)
        lines.append(
            f"{e}{conv} *{t['time']}*\n"
            f"  Bot: {t['bot_direction']} ({t['confidence']:.0%}) | Actual: {t['actual_direction']}\n"
            f"  Target: ${t.get('round_start_price',0):,.2f} → End: ${t.get('btc_at_end',0):,.2f}\n"
            f"  Odds: {op:.0%} | Payout: {pyt:.2f}x | YES {t.get('yes_price',0):.0%}"
        )
    wr = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    lines.append(f"─"*30)
    lines.append(f"📊 {stats['total']} rounds | {wr} WR | ✅{stats['correct']} ❌{stats['incorrect']}")
    return "\n".join(lines)

def build_hour_stats():
    if not hour_stats:
        return "📊 *Hourly stats*\nNot enough data yet. Needs at least a few rounds per hour."
    lines = [f"🕐 *Accuracy by Hour (UTC) — Session #{session_num}*\n" + "─"*30]
    for h in sorted(hour_stats.keys()):
        d  = hour_stats[h]
        wr = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        bar = "█" * int(wr / 10)
        lines.append(f"  {h:02d}:00 | {bar:<10} {wr:.0f}% ({d['correct']}/{d['total']})")
    lines.append(f"─"*30)
    lines.append("Min 52.6% WR needed to beat 5% fee")
    return "\n".join(lines)

def build_stats_message():
    now   = datetime.now(timezone.utc)
    wr    = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    hcwr  = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
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
        c = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(f"{e}{c} {t['time']} | {t['bot_direction']} | {t['confidence']:.0%} | {t['actual_direction']}")
    table = "\n".join(lines) if lines else "No trades yet"
    return (
        f"📊 *BOT STATS — Session #{session_num}*\n"
        f"{'─'*30}\n"
        f"🕐 {now.strftime('%H:%M UTC')}\n"
        f"📅 Session since: {session_start.strftime('%m-%d %H:%M UTC')}\n"
        f"{'─'*30}\n"
        f"📈 All time  : {stats['total']} rounds | {wr} WR\n"
        f"   ✅ {stats['correct']} | ❌ {stats['incorrect']}\n"
        f"⏱ Last hour : {len(last_1h)} rounds | {l1hr} WR\n"
        f"🔟 Last 10   : {l10r} WR\n"
        f"⚡ High conv : {stats['high_conv_total']} | {hcwr} WR\n"
        f"{'─'*30}\n"
        f"🚫 Skipped — odds: {stats['skipped_odds']} | gap: {stats['skipped_gap']} | liq: {stats['skipped_liquidity']}\n"
        f"{'─'*30}\n"
        f"⏸ Paused: {'Yes' if bot_paused else 'No'}\n"
        f"{'─'*30}\n"
        f"`{table}`"
    )

def build_final_report():
    wr   = f"{stats['correct']/stats['total']*100:.1f}%" if stats["total"] > 0 else "N/A"
    hcwr = (f"{stats['high_conv_correct']/stats['high_conv_total']*100:.1f}%"
            if stats["high_conv_total"] > 0 else "N/A")
    duration = (datetime.now(timezone.utc) - session_start)
    hours    = int(duration.total_seconds() / 3600)
    mins     = int((duration.total_seconds() % 3600) / 60)
    lines = []
    for t in trade_log[-30:]:
        e = "✅" if t["correct"] else "❌"
        c = "⚡" if t["high_conviction"] else "⚠️"
        lines.append(f"{e}{c} {t['time']} | {t['bot_direction']} | {t['confidence']:.0%} | {t['actual_direction']}")
    log_text = "\n".join(lines) if lines else "No trades"
    return (
        f"📊 *SESSION #{session_num} FINAL REPORT*\n"
        f"{'─'*30}\n"
        f"📅 {session_start.strftime('%Y-%m-%d %H:%M')} → {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
        f"⏱ Duration: {hours}h {mins}m\n"
        f"{'─'*30}\n"
        f"  Total rounds : {stats['total']}\n"
        f"  Win rate     : {wr}\n"
        f"  ✅ {stats['correct']} correct | ❌ {stats['incorrect']} wrong\n"
        f"  High conv    : {stats['high_conv_total']} signals | {hcwr} WR\n"
        f"  Skipped      : odds {stats['skipped_odds']} | gap {stats['skipped_gap']} | liq {stats['skipped_liquidity']}\n"
        f"{'─'*30}\n"
        f"📋 *Last 30 trades*\n`{log_text}`"
    )

# ── Main loop ─────────────────────────────────────────────────
def main():
    global current_market_cache, last_report_time

    init_log()
    seed_candles()
    tg_send(start_message())
    tg_send(
        f"🟢 *Session #{session_num} started*\n"
        f"{session_start.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"Strategy:\n"
        f"  Confidence ≥ {CONFIDENCE:.0%}\n"
        f"  Gap ≥ {MIN_GAP_PCT:.2f}%\n"
        f"  Odds {ODDS_MIN:.0%}–{ODDS_MAX:.0%} (sweet spot for payout)\n"
        f"  Liquidity ≥ ${MIN_LIQUIDITY:.0f}\n"
        f"  Signal fires {SIGNAL_DELAY_MINS} mins into round"
    )
    log.info("Bot v3 started")

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
                time.sleep(30)
                continue

            current_market_cache = market
            event_id  = market["event_id"]
            btc       = get_btc_price()
            rules     = market.get("rules", "")
            target    = parse_target(rules)
            end_dt    = parse_end_dt(rules, now)
            mins_left = round((end_dt - now).total_seconds() / 60, 1) if end_dt else None

            # 12h report
            if (now - last_report_time).total_seconds() / 3600 >= REPORT_INTERVAL_HOURS and stats["total"] > 0:
                tg_send(build_final_report())
                last_report_time = now

            # New round
            if event_id != current_round["event_id"]:
                log.info(f"New round: {event_id}")

                # Close previous round
                prev_signal = current_round.get("pending_signal")
                prev_target = current_round.get("round_start_price")

                if prev_signal and prev_target and btc:
                    final_btc    = btc
                    actual_up    = final_btc >= prev_target
                    predicted_up = prev_signal["direction_short"] == "UP"
                    correct      = predicted_up == actual_up
                    high_conv    = prev_signal["high_conviction"]
                    actual_dir   = "UP" if actual_up else "DOWN"
                    hour_h       = prev_signal.get("hour_utc", now.hour)

                    stats["total"] += 1
                    stats["correct" if correct else "incorrect"] += 1
                    if high_conv:
                        stats["high_conv_total"] += 1
                        if correct:
                            stats["high_conv_correct"] += 1

                    # Hour stats
                    if hour_h not in hour_stats:
                        hour_stats[hour_h] = {"total": 0, "correct": 0}
                    hour_stats[hour_h]["total"] += 1
                    if correct:
                        hour_stats[hour_h]["correct"] += 1

                    last_completed.update({
                        "round_start_price": prev_target,
                        "btc_at_end"       : final_btc,
                        "actual_direction" : actual_dir,
                        "bot_direction"    : prev_signal["direction_short"],
                        "correct"          : correct,
                        "resolved"         : True,
                    })

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
                        "outcome_price"    : prev_signal.get("outcome_price", ""),
                        "payout"           : prev_signal.get("payout", ""),
                        "high_conviction"  : high_conv,
                    }
                    trade_log.append(entry)
                    write_log(entry)
                    log.info(f"Closed: {'✅' if correct else '❌'} | {stats}")

                # Reset
                current_round.update({
                    "event_id"         : event_id,
                    "start_time"       : now,
                    "round_start_price": target,
                    "signal_fired"     : False,
                    "pending_signal"   : None,
                })

                if target and btc and mins_left:
                    tg_send(msg_open_alert(market, btc, target, now, mins_left))

            # Fire signal 5 mins in
            start_time = current_round.get("start_time")
            if (not current_round["signal_fired"] and not bot_paused and
                    start_time and current_round["round_start_price"]):

                mins_in = (now - start_time).total_seconds() / 60
                if mins_in >= SIGNAL_DELAY_MINS:
                    result = get_signal(market, current_round["round_start_price"], now)
                    if result:
                        current_round["pending_signal"] = result
                        current_round["signal_fired"]   = True
                        tg_send(msg_signal(result, now))
                        log.info(
                            f"Signal: {result['signal']} {result['direction_short']} "
                            f"{result['confidence']:.1%} | odds {result['outcome_price']:.0%} | "
                            f"EV {result['ev']:+.3f} | Kelly {result['kelly']:.1%}"
                        )
                    else:
                        log.warning("Features not ready")

            time.sleep(30)

        except Exception as e:
            log.error(f"Loop: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
