import sys, io, os, time, csv, requests, itertools, concurrent.futures
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
import numpy as np

# --- (Windows) ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# --- Config, api key hardcoded for the easier demo use and testing ---
CMC_API_KEY = "api_key"
DEEPSEEK_API_KEY = "api_key"

INTERVAL = 180
TOP_MARKET_CAP = 5
TOP_VOLATILE = 5
ANALYZE_TOTAL = 8
HISTORY_LEN = 50  # more infi count of the volatility
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

DETAILED_OUTPUT = True

# --- Risk Management ---
RISK_PER_TRADE = 0.02    # 2% risk on the deal
MAX_DRAWDOWN  = 0.10     # stop trade if loss 10%
PORTFOLIO_VALUE = 10000  # portfolio cost hardcoded for the demo and testing  (USD)

VOLUME_SPIKE_THRESHOLD = 1.5  # 150% average volume spike

MIN_CONFIDENCE_THRESHOLD = 0.65 # Only trade with 65%+ confidence


# --- Persistent State ---
STATE_FILE = "trader_state.json"

# --- Цвета (ANSI) ---
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# --- Хранилища ---
price_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
last_decisions = {}
stats = defaultdict(Counter)
accuracy = defaultdict(lambda: {"success": 0, "total": 0})
volume_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
prediction_history = defaultdict(list)  # Saves new predictions for later accuracy evaluation


# === Rate Limiter для API (limit safe) ===
class RateLimiter:
    def __init__(self, calls_per_minute=30):
        self.calls_per_minute = calls_per_minute
        self.call_times = deque()
    def wait_if_needed(self):
        now = datetime.now()
        while self.call_times and now - self.call_times[0] > timedelta(minutes=1):
            self.call_times.popleft()
        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.call_times[0]).total_seconds()
            if sleep_time > 0:
                print(f"{YELLOW}⏳ Достигнут лимит API, ждём {sleep_time:.1f} сек...{RESET}")
                time.sleep(sleep_time)
        self.call_times.append(now)


class PerformanceMonitor:
    def __init__(self):
        self.cycle_times = deque(maxlen=10)
        self.api_times = deque(maxlen=50)
        self.start_time = time.time()
        self.cycle_start = None

    def start_cycle(self):
        self.cycle_start = time.time()

    def end_cycle(self):
        if self.cycle_start:
            self.cycle_times.append(time.time() - self.cycle_start)
            self.cycle_start = None

    def record_api_time(self, duration):
        self.api_times.append(duration)

    def get_stats(self):
        avg_cycle = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
        avg_api = sum(self.api_times) / len(self.api_times) if self.api_times else 0
        uptime = (time.time() - self.start_time) / 3600
        return {
            "avg_cycle_time": avg_cycle,
            "avg_api_time": avg_api,
            "uptime_hours": uptime,
            "cycles_completed": len(self.cycle_times)
        }

cmc_limiter = RateLimiter(calls_per_minute=25)
deepseek_limiter = RateLimiter(calls_per_minute=50)
monitor = PerformanceMonitor()  # ✅ FIX: global monitor instance


def save_state():
    import json, glob
    state = {
        "cycle_count": getattr(main, "cycle_count", 0),
        "active_symbols": getattr(main, "active_symbols", []),
        "accuracy": {k: dict(v) for k, v in accuracy.items()},
        "stats": {k: dict(v) for k, v in stats.items()},
        "price_history": {k: list(v) for k, v in price_history.items()},
        "saved_at": datetime.now().isoformat(),
        "version": "v4.4"
    }

    # 🧹 Delete old backups, keep last 3
    backups = sorted(glob.glob("trader_state_backup_*.json"), key=os.path.getmtime, reverse=True)
    for old in backups[3:]:
        try:
            os.remove(old)
            print(f"🧹 Удалён старый бэкап: {old}")
        except Exception:
            pass

    # New Back up
    if os.path.exists(STATE_FILE):
        backup = STATE_FILE.replace(".json", f"_backup_{int(time.time())}.json")
        try:
            os.rename(STATE_FILE, backup)
        except Exception:
            pass

    # 💾 Save current state
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)



def load_state():
    import json, os
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        main.cycle_count = state.get("cycle_count", 0)
        main.active_symbols = state.get("active_symbols", [])
        for sym, acc in state.get("accuracy", {}).items():
            accuracy[sym] = acc
        for sym, st in state.get("stats", {}).items():
            stats[sym] = Counter(st)
        print(f"♻️ Состояние восстановлено: цикл №{main.cycle_count}, {len(main.active_symbols)} активных монет.\n")
        for sym, prices in state.get("price_history", {}).items():
            price_history[sym] = deque(prices, maxlen=HISTORY_LEN)


# --- Animation of the status ---
def thinking(status_text, duration=1.2):
    spinner = itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])
    start = time.time()
    while time.time() - start < duration:
        sys.stdout.write(f"\r{next(spinner)} {status_text}   ")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f"\r✅ {status_text}\n")

# Risk Management: position sizing
def calculate_position_size(portfolio_value, confidence_score):
    """Calculate position size based on confidence and risk per trade."""
    position = portfolio_value * RISK_PER_TRADE * confidence_score
    return round(position, 2)

# Technical Indicators (additional)
def calculate_technical_indicators(prices):
    """Calculate basic technical indicators"""
    if not prices:
        return {'sma': 0.0, 'trend': 'unknown', 'resistance': 0.0, 'support': 0.0, 'above_sma': False}
    sma = sum(prices) / len(prices)
    if len(prices) >= 3:
        if prices[-1] > prices[-2] > prices[-3]:
            trend = "up"
        elif prices[-1] < prices[-2] < prices[-3]:
            trend = "down"
        else:
            trend = "sideways"
    else:
        trend = "unknown"
    resistance = max(prices)
    support = min(prices)
    return {
        'sma': round(sma, 6),
        'trend': trend,
        'resistance': resistance,
        'support': support,
        'above_sma': prices[-1] > sma
    }


def calculate_enhanced_features(symbol, prices, volume_data):
    """Calculate more meaningful features"""
    if len(prices) < 5:
        return {}

    # Price features
    recent_prices = prices[-5:]
    price_change_1h = ((recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]) * 100 if len(
        recent_prices) >= 2 else 0
    price_change_4h = ((recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]) * 100 if len(
        recent_prices) >= 5 else 0

    # Volume features
    avg_volume = sum(volume_data) / len(volume_data)
    current_volume = volume_data[-1] if volume_data else 0
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

    # Volatility features
    price_std = np.std(prices) if len(prices) > 1 else 0
    avg_price = sum(prices) / len(prices)
    volatility = (price_std / avg_price) * 100 if avg_price > 0 else 0

    # Momentum
    if len(prices) >= 3:
        short_trend = "up" if prices[-1] > prices[-3] else "down"
    else:
        short_trend = "neutral"

    return {
        'price_change_1h': round(price_change_1h, 2),
        'price_change_4h': round(price_change_4h, 2),
        'volume_ratio': round(volume_ratio, 2),
        'volatility': round(volatility, 2),
        'short_trend': short_trend,
        'signal_strength': min(1.0, abs(price_change_4h) / 10)  # Normalized signal strength
    }


def enhanced_decision_logic(symbol, ai_decision, features, market_context):
    """Add rule-based validation to AI decisions"""

    # Rule-based overrides
    if features.get('volatility', 0) > 25:
        # Too volatile - avoid trading
        return "HOLD", "High volatility protection"

    if features.get('volume_ratio', 1) < 0.5:
        # Low volume - unreliable signals
        return "HOLD", "Low volume - unreliable signals"

    # Market context alignment
    if market_context == "bearish" and ai_decision == "BUY":
        # Be cautious buying in bear markets
        if features.get('signal_strength', 0) < 0.7:
            return "HOLD", "Weak signal in bear market"

    if market_context == "bullish" and ai_decision == "SELL":
        # Be cautious selling in bull markets
        if features.get('signal_strength', 0) < 0.8:
            return "HOLD", "Weak sell signal in bull market"

    return ai_decision, "AI decision validated"



# Market Context Analysis
def get_market_trend(coins):
    """Оцениваем контекст рынка: бычий/медвежий/флэт по BTC и средним 24h изменениям."""
    try:
        btc = next((c for c in coins if c["symbol"] == "BTC"), None)
        avg24 = sum(c["quote"]["USD"].get("percent_change_24h", 0) for c in coins[:50]) / max(1, len(coins[:50]))
        btc24 = btc["quote"]["USD"].get("percent_change_24h", 0) if btc else 0
        score = 0.6*btc24 + 0.4*avg24
        if score > 0.7:
            return "bullish"
        elif score < -0.7:
            return "bearish"
        return "sideways"
    except Exception:
        return "sideways"


def calculate_market_regime(coins_data):
    """Determine current market regime using multiple indicators"""
    prices = [c['quote']['USD']['price'] for c in coins_data[:50]]
    changes_24h = [c['quote']['USD']['percent_change_24h'] for c in coins_data[:50]]

    # Calculate market metrics
    avg_change = np.mean(changes_24h)
    volatility = np.std(changes_24h)
    advancing = len([c for c in changes_24h if c > 0])
    declining = len([c for c in changes_24h if c < 0])
    advance_decline_ratio = advancing / max(declining, 1)

    # Determine regime
    if avg_change > 1.0 and advance_decline_ratio > 1.5:
        return "strong_bull"
    elif avg_change > 0.5 and advance_decline_ratio > 1.2:
        return "bull"
    elif avg_change < -1.0 and advance_decline_ratio < 0.67:
        return "strong_bear"
    elif avg_change < -0.5 and advance_decline_ratio < 0.8:
        return "bear"
    else:
        return "sideways"


def regime_aware_confidence(symbol, decision, market_regime, confidence):
    """Adjust confidence based on market regime"""
    regime_multipliers = {
        "strong_bull": {"BUY": 1.3, "SELL": 0.7, "HOLD": 1.0},
        "bull": {"BUY": 1.1, "SELL": 0.9, "HOLD": 1.0},
        "sideways": {"BUY": 1.0, "SELL": 1.0, "HOLD": 1.2},
        "bear": {"BUY": 0.9, "SELL": 1.1, "HOLD": 1.0},
        "strong_bear": {"BUY": 0.7, "SELL": 1.3, "HOLD": 1.0}
    }

    multiplier = regime_multipliers[market_regime].get(decision, 1.0)
    return min(0.95, confidence * multiplier)


def calculate_smart_confidence(symbol, decision, features, accuracy_ratio, market_context, technicals):
    """Much more sophisticated confidence scoring"""

    base_confidence = 0.25 + (accuracy_ratio / 100.0) * 0.75

    adjustments = []
    reasons = []

    # 1. Technical alignment score (0-100%)
    tech_score = 0
    if technicals['above_sma'] and technicals['trend'] == 'up':
        tech_score += 0.4
    if features.get('short_trend') == 'up':
        tech_score += 0.3
    if features.get('volume_ratio', 1) > 1.2:
        tech_score += 0.3

    if tech_score > 0.7:
        adjustments.append(1.3)
        reasons.append("strong technicals")
    elif tech_score > 0.5:
        adjustments.append(1.1)
        reasons.append("good technicals")
    else:
        adjustments.append(0.7)
        reasons.append("weak technicals")

    # 2. Market context alignment
    if (market_context == "bullish" and decision == "BUY") or \
            (market_context == "bearish" and decision == "SELL"):
        adjustments.append(1.25)
        reasons.append("market aligned")
    else:
        adjustments.append(0.8)
        reasons.append("against market")

    # 3. Volatility adjustment
    vol = features.get('volatility', 0)
    if 2 < vol < 10:
        adjustments.append(1.15)
        reasons.append("optimal volatility")
    elif vol > 25:
        adjustments.append(0.6)
        reasons.append("high volatility")

    # 4. Historical performance
    if accuracy_ratio > 70:
        adjustments.append(1.2)
        reasons.append("high accuracy history")
    elif accuracy_ratio < 40:
        adjustments.append(0.8)
        reasons.append("low accuracy history")

    # Calculate final confidence
    if adjustments:
        confidence = base_confidence * (sum(adjustments) / len(adjustments))
    else:
        confidence = base_confidence

    final_confidence = max(0.1, min(0.95, confidence))

    # Debug output
    if final_confidence > 0.6:
        print(f"   🎯 Confidence factors: {', '.join(reasons)}")

    return final_confidence


def calculate_unified_confidence(symbol, decision, features, accuracy_ratio, market_context, technicals):
    """Single, reliable confidence calculation"""

    # Base confidence from historical accuracy (30-100% range)
    base_confidence = 0.3 + (accuracy_ratio / 100.0) * 0.7

    # Technical alignment (most important - 40% weight)
    tech_score = 0
    if technicals['above_sma']: tech_score += 0.4
    if technicals['trend'] == 'up': tech_score += 0.3
    if features.get('short_trend') == 'up': tech_score += 0.2
    if features.get('volume_ratio', 1) > 1.2: tech_score += 0.1

    tech_multiplier = 0.6 + (tech_score * 0.8)  # 0.6 to 1.4 range

    # Market alignment (30% weight)
    market_multiplier = 1.0
    if (market_context == "bullish" and decision == "BUY") or \
            (market_context == "bearish" and decision == "SELL"):
        market_multiplier = 1.3
    elif (market_context == "bearish" and decision == "BUY") or \
            (market_context == "bullish" and decision == "SELL"):
        market_multiplier = 0.7

    # Volatility adjustment (20% weight)
    vol = features.get('volatility', 0)
    vol_multiplier = 1.0
    if 1 < vol < 8:
        vol_multiplier = 1.2  # Optimal volatility
    elif vol > 20:
        vol_multiplier = 0.6  # Too volatile
    elif vol < 0.5:
        vol_multiplier = 0.8  # Too stagnant

    # Recent performance boost (10% weight)
    perf_multiplier = 1.0
    if accuracy_ratio > 70:
        perf_multiplier = 1.2
    elif accuracy_ratio < 30:
        perf_multiplier = 0.8

    # Calculate weighted final confidence
    confidence = base_confidence * tech_multiplier * market_multiplier * vol_multiplier * perf_multiplier

    # Debug info for low confidence
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        print(
            f"   {YELLOW}⚠️ Low confidence breakdown: base={base_confidence:.2f}, tech={tech_multiplier:.2f}, market={market_multiplier:.2f}, vol={vol_multiplier:.2f}{RESET}")

    return max(0.1, min(0.95, confidence))



# CoinMarketCap API Request for the Data
def cmc_request(path, params=None):
    cmc_limiter.wait_if_needed()
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    url = f"https://pro-api.coinmarketcap.com{path}"
    try:
        print(f"{CYAN}🌐 Получаю данные с CoinMarketCap...{RESET}")
        start = time.time()  # ⏱️ START TIMER
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        elapsed = time.time() - start
        monitor.record_api_time(elapsed)  # ✅ FIX: record API time
        print(f"{GREEN}✅ Данные CMC получены ({elapsed:.2f}s)!{RESET}")
        return r.json()
    except Exception as e:
        print(f"{RED}❌ Ошибка CoinMarketCap: {e}{RESET}")
        return {"data": []}


def robust_deepseek_decision(prompt, symbol, retries=2):
    """Enhanced DeepSeek with retries"""
    for attempt in range(retries + 1):
        try:
            return deepseek_decision(prompt, symbol)
        except Exception as e:
            if attempt < retries:
                print(f"{YELLOW}⚠️  Повтор запроса {symbol} (попытка {attempt + 1})...{RESET}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
    return "DECISION: HOLD — max retries exceeded"

def deepseek_decision(prompt, symbol):
    deepseek_limiter.wait_if_needed()
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": (
                "You are a disciplined swing trading AI. "
                "Analyze short-term (1h–24h) crypto behavior and follow clear tactical rules: "
                "- BUY if momentum and trend rise, or volume spike with low volatility. "
                "- SELL if momentum falls after a rise or high volatility in bearish markets. "
                "- HOLD if movement is flat or mixed. "
                "Respond strictly: DECISION: BUY/SELL/HOLD — reason (<150 chars)."
            )},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 90
    }
    try:
        print(f"🤖 DeepSeek анализирует {symbol}...")
        start = time.time()
        r = requests.post(url, json=data, headers=headers, timeout=30)
        r.raise_for_status()
        elapsed = time.time() - start
        resp = r.json()
        answer = (resp.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
        if not answer:
            answer = "DECISION: HOLD — no valid response"
        print(f"🧩 Ответ {symbol} ({elapsed:.2f}s): {answer}")
        return answer
    except Exception as e:
        print(f"{RED}❌ Ошибка DeepSeek: {e}{RESET}")
        return "DECISION: HOLD — request error"

# Logging and Past Performance Evaluation
def log_coin(symbol, price, decision, reason):
    """Log trading decisions"""
    fn = os.path.join(LOG_DIR, f"{symbol}.csv")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    newfile = not os.path.exists(fn)
    with open(fn, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["Time", "Symbol", "Price", "Decision", "Comment"])
        w.writerow([ts, symbol, price, decision, reason])


def evaluate_past(symbol, current_price):
    """Улучшенная оценка прошлых сигналов с 1–24h окном"""
    fn = os.path.join(LOG_DIR, f"{symbol}.csv")
    if not os.path.exists(fn):
        return
    try:
        with open(fn, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        current_time = datetime.now()
        new_success, new_total = 0, 0

        for row in rows:
            if not row["Decision"] in ["BUY", "SELL"]:
                continue
            t = datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S")
            dt = (current_time - t).total_seconds()
            if 3600 < dt < 24 * 3600:  # 1h–24h окно
                old_price = float(row["Price"])
                change = ((current_price - old_price) / old_price) * 100
                new_total += 1
                if (row["Decision"] == "BUY" and change > 1.5) or (row["Decision"] == "SELL" and change < -1.5):
                    new_success += 1

        if new_total > 0:
            accuracy[symbol]["total"] += new_total
            accuracy[symbol]["success"] += new_success
    except Exception as e:
        print(f"⚠ Ошибка анализа истории {symbol}: {e}")


def update_recent_accuracy(symbol, decision, price):
    """Update recent prediction history for accuracy tracking"""
    prediction_history[symbol].append({
        'time': datetime.now(),
        'decision': decision,
        'price': price
    })



def create_enhanced_prompt(symbol, coin_data, technicals, accuracy_ratio, market_context):
    """Simplified but more effective prompt"""

    prices = list(price_history[symbol])[-5:]
    momentum = "rising" if len(prices) >= 3 and prices[-1] > prices[-3] else "falling" if len(prices) >= 3 and prices[
        -1] < prices[-3] else "neutral"

    prompt = f"""
SYMBOL: {symbol} | PRICE: ${coin_data['price']:.6f} | 24h: {coin_data['change_24h']}%
TREND: {technicals['trend']} | MOMENTUM: {momentum} | POSITION: {'ABOVE SMA' if technicals['above_sma'] else 'BELOW SMA'}
VOLATILITY: {coin_data['volatility']:.2f}% | MARKET: {market_context} | ACCURACY: {accuracy_ratio}%

STRICT RULES:
BUY IF: Price above SMA + Trend up + Momentum rising + Market not bearish
SELL IF: Price below SMA + Trend down + Momentum falling + Market not bullish  
HOLD IF: Any condition not met

DECISION: [BUY/SELL/HOLD] - [BRIEF REASON]
"""
    return prompt





# --- Paralle Analys ---
def analyze_single_coin(coin, market_context):

    """Complete analysis for one coin"""
    symbol = coin["symbol"]
    q = coin["quote"]["USD"]
    price = round(float(q["price"]), 6)
    change_24h = round(float(q.get("percent_change_24h", 0)), 2)
    change_7d = round(float(q.get("percent_change_7d", 0)), 2)
    market_cap = round(float(q.get("market_cap", 0)) / 1e9, 2)
    volume_24h = round(float(q.get("volume_24h", 0)) / 1e6, 2)

    # Local copy of the history with the new price
    prices_copy = list(price_history[symbol]) + [price]
    if prices_copy:
        # Restrict to HISTORY_LEN
        if len(prices_copy) > HISTORY_LEN:
            prices_copy = prices_copy[-HISTORY_LEN:]
    volatility = round((max(prices_copy) - min(prices_copy)) / max(1e-9, min(prices_copy)) * 100, 2) if len(prices_copy) > 1 else 0.0
    technicals = calculate_technical_indicators(prices_copy)

    # Past performance evaluation
    acc = accuracy[symbol]
    ratio = round(acc["success"] / acc["total"] * 100, 1) if acc["total"] else 0

    # # Market context
    # market_context = get_market_trend([])  # Simplified, pass coins if needed

    coin_data = {
        "price": price,
        "change_24h": change_24h,
        "change_7d": change_7d,
        "market_cap": market_cap,
        "volume_24h": volume_24h,
        "volatility": volatility
    }

    # Checking the volume spike
    # --- Volume Spike Detection ---
    volume_history[symbol].append(volume_24h)
    avg_vol = sum(volume_history[symbol]) / max(1, len(volume_history[symbol]))
    if avg_vol > 0 and volume_24h > VOLUME_SPIKE_THRESHOLD * avg_vol:
        print(f"{YELLOW}⚡ Обнаружен всплеск объёма у {symbol}! Добавляем в промпт.{RESET}")
        prompt_note = "\n⚡ VOLUME SPIKE detected — possible breakout."
    else:
        prompt_note = ""

    prompt = create_enhanced_prompt(symbol, coin_data, technicals, ratio, market_context) + prompt_note
    answer = robust_deepseek_decision(prompt, symbol)

    # Normalize decision
    decision = "HOLD"
    for w in ["BUY", "SELL", "HOLD"]:
        if w in answer.upper():
            decision = w
            break

    # Better decision logic
    features = calculate_enhanced_features(symbol, prices_copy, list(volume_history[symbol]))
    confidence = calculate_unified_confidence(symbol, decision, features, ratio, market_context, technicals)


    position_usd = calculate_position_size(PORTFOLIO_VALUE, confidence)

    # Add logging and history update
    return {
        "symbol": symbol,
        "price": price,
        "change_24h": change_24h,
        "decision": decision,
        "answer": answer,
        "volatility": volatility,
        "confidence": confidence,
        "position_usd": position_usd,
        "ratio": ratio,
        "prices_append": price  # what to add in history
    }

# Formatting Output and Reports
def print_coin_analysis(idx, total, res):
    color = GREEN if res["decision"] == "BUY" else RED if res["decision"] == "SELL" else YELLOW
    print(f"[{idx}/{total}] Анализ {res['symbol']}...")
    print(f"✅ AI обрабатывает рыночные метрики...")
    print(f"{color}💰 {res['symbol']}: ${res['price']:.6f} ({res['change_24h']:+.2f}%) → {res['answer']}{RESET}")
    if res["decision"] == "BUY":
        print(f"📈 Планируемая позиция: {res['position_usd']}$ "
              f"(уверенность {res['confidence']*100:.1f}%, волатильность {res['volatility']:.2f}%)\n")
    else:
        print(f"🤔 Без действия: {res['decision']} "
              f"(уверенность {res['confidence']*100:.1f}%, волатильность {res['volatility']:.2f}%)\n")



# IQ Report
def generate_enhanced_report(accuracy_map, stats_map):
    print("\n" + "="*60)
    print("🧠 === ENHANCED TRADER IQ REPORT ===")
    print("="*60)

    # Accuracy leaderboard
    report = []
    for sym, acc in accuracy_map.items():
        if acc["total"] > 10:
            success_rate = round(acc["success"] / acc["total"] * 100, 1)
            report.append((sym, success_rate, acc["total"]))
    if report:
        report.sort(key=lambda x: x[1], reverse=True)
        print("\n📊 PREDICTION ACCURACY LEADERBOARD:")
        for rank, (sym, rate, total) in enumerate(report[:7], 1):
            if rate >= 80: emoji = "🟢"
            elif rate >= 65: emoji = "🟡"
            elif rate >= 50: emoji = "🟠"
            else: emoji = "🔴"
            print(f"   {rank:2d}. {sym:8} {rate:5.1f}% ({total:3d} trades) {emoji}")
        avg = round(sum(r[1] for r in report) / len(report), 1)
        print(f"\n📈 Average model accuracy (eligible): {avg}%")
    else:
        print("\n(No sufficient trade history for accuracy leaderboard)")

    # Trading activity
    total_decisions = sum(sum(coin_stats.values()) for coin_stats in stats_map.values())
    buy_decisions = sum(coin_stats.get("BUY", 0) for coin_stats in stats_map.values())
    sell_decisions = sum(coin_stats.get("SELL", 0) for coin_stats in stats_map.values())
    print(f"\n📈 TRADING ACTIVITY:")
    print(f"   Total Decisions: {total_decisions}")
    if total_decisions:
        print(f"   BUY: {buy_decisions} ({buy_decisions/total_decisions*100:.1f}%)")
        print(f"   SELL: {sell_decisions} ({sell_decisions/total_decisions*100:.1f}%)")
        print(f"   HOLD: {total_decisions - buy_decisions - sell_decisions}")
    print()


def calculate_portfolio_performance():
    """Calculate actual portfolio performance"""
    total_trades = sum(acc["total"] for acc in accuracy.values())
    successful_trades = sum(acc["success"] for acc in accuracy.values())

    if total_trades > 0:
        overall_accuracy = (successful_trades / total_trades) * 100
        print(f"\n🎯 OVERALL PORTFOLIO PERFORMANCE:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Successful: {successful_trades}")
        print(f"   Accuracy: {overall_accuracy:.1f}%")

        # Trend analysis
        if overall_accuracy > 60:
            print(f"   {GREEN}📈 PERFORMANCE: EXCELLENT (>60%){RESET}")
        elif overall_accuracy > 50:
            print(f"   {YELLOW}📊 PERFORMANCE: GOOD (50-60%){RESET}")
        else:
            print(f"   {RED}⚠️ PERFORMANCE: NEEDS IMPROVEMENT (<50%){RESET}")
# def track_prediction_accuracy():
#     """Track accuracy in real-time during cycles"""
#     print(f"\n📈 REAL-TIME ACCURACY TRACKING:")
#     active_with_history = [(sym, acc) for sym, acc in accuracy.items()
#                            if acc["total"] > 3 and sym in getattr(main, "active_symbols", [])]
#
#     if active_with_history:
#         for sym, acc in sorted(active_with_history,
#                                key=lambda x: x[1]["success"] / x[1]["total"],
#                                reverse=True)[:5]:
#             rate = (acc["success"] / acc["total"]) * 100
#             color = GREEN if rate > 60 else YELLOW if rate > 45 else RED
#             print(f"   {color}{sym}: {rate:.1f}% ({acc['total']} trades){RESET}")
#     else:
#         print(f"   {YELLOW}Insufficient trade history yet{RESET}")


def track_prediction_outcomes(market_context):
    """Enhanced prediction outcome tracking"""
    print(f"\n📊 PREDICTION OUTCOME TRACKING:")

    current_time = datetime.now()
    evaluated_this_cycle = 0

    for symbol, predictions in list(prediction_history.items()):
        valid_predictions = []

        for pred in predictions:
            hours_old = (current_time - pred['time']).total_seconds() / 3600

            # Evaluate predictions that are 2-8 hours old
            if 2 <= hours_old <= 8:
                current_price = price_history[symbol][-1] if price_history[symbol] else 0

                if current_price > 0 and pred['decision'] in ["BUY", "SELL"]:
                    change = ((current_price - pred['price']) / pred['price']) * 100

                    # IMPROVED Success criteria with dynamic thresholds
                    success = False
                    if pred['decision'] == "BUY":
                        if change > 2.0:
                            success = True
                        elif change > 1.0 and market_context == "bullish":
                            success = True
                    elif pred['decision'] == "SELL":
                        if change < -2.0:
                            success = True
                        elif change < -1.0 and market_context == "bearish":
                            success = True

                    # Update accuracy (with duplicate protection)
                    pred_id = f"{symbol}_{pred['time'].timestamp():.0f}"
                    if not hasattr(track_prediction_outcomes, 'evaluated_ids'):
                        track_prediction_outcomes.evaluated_ids = set()

                    if pred_id not in track_prediction_outcomes.evaluated_ids:
                        track_prediction_outcomes.evaluated_ids.add(pred_id)
                        accuracy[symbol]["total"] += 1
                        if success:
                            accuracy[symbol]["success"] += 1

                        evaluated_this_cycle += 1
                        status = "✅ SUCCESS" if success else "❌ FAIL"
                        print(f"   {symbol}: {pred['decision']} at ${pred['price']:.6f} → ${current_price:.6f} ({change:+.2f}%) {status}")

                    continue

            if hours_old < 24:
                valid_predictions.append(pred)

        prediction_history[symbol] = valid_predictions

    if evaluated_this_cycle == 0:
        print("   No predictions ready for evaluation (2-8 hour window needed)")
    else:
        print(f"   Evaluated {evaluated_this_cycle} predictions this cycle")




def debug_confidence_calculation():
    """Debug why confidence is low"""
    print(f"\n🔍 CONFIDENCE DEBUG:")
    for symbol in getattr(main, "active_symbols", [])[:3]:  # Check first 3 symbols
        if symbol in accuracy:
            acc = accuracy[symbol]
            ratio = (acc["success"] / acc["total"] * 100) if acc["total"] > 0 else 0
            recent_trend = stats[symbol]["BUY"] + stats[symbol]["SELL"]
            print(f"   {symbol}: Accuracy={ratio:.1f}%, Recent signals={recent_trend}")


def analyze_failed_predictions(threshold=0.4, min_trades=10):
    """Analyze and exclude consistently failing coins"""
    bad = []
    for sym, acc in accuracy.items():
        if acc["total"] >= min_trades:
            ratio = acc["success"] / acc["total"]
            if sym in ("BTC", "ETH"):
                continue
            if ratio < threshold:
                bad.append((sym, ratio))

    if bad:
        print(f"\n{RED}🚨 Слабые монеты (<{threshold*100:.0f}% точности):{RESET}")
        for sym, r in bad:
            print(f"   {sym}: {r*100:.1f}% — исключаем")
        active_before = set(getattr(main, "active_symbols", []))
        main.active_symbols = [s for s in active_before if s not in [x[0] for x in bad]]
        print(f"✅ Новый активный набор: {', '.join(main.active_symbols) if main.active_symbols else '—'}")
    else:
        print(f"{GREEN}✅ Все активные монеты стабильны{RESET}")


def improve_trading_strategy():
    """Analyze and suggest strategy improvements"""
    print(f"\n🎯 STRATEGY IMPROVEMENT ANALYSIS:")

    # Analyze which conditions lead to success
    successful_coins = []
    for sym, acc in accuracy.items():
        if acc["total"] >= 5:
            success_rate = acc["success"] / acc["total"]
            if success_rate > 0.6:
                successful_coins.append((sym, success_rate))

    if successful_coins:
        print(f"   {GREEN}High-performing coins (>60% accuracy):{RESET}")
        for sym, rate in sorted(successful_coins, key=lambda x: x[1], reverse=True)[:3]:
            print(f"   📈 {sym}: {rate * 100:.1f}% success rate")
    else:
        print(f"   {YELLOW}No high-performing coins yet - need more data{RESET}")




# def track_prediction_outcomes():
#     """Track prediction outcomes in real-time"""
#     print(f"\n📊 PREDICTION OUTCOME TRACKING:")
#
#     current_time = datetime.now()
#     outcomes = []
#
#     for symbol, predictions in prediction_history.items():
#         for pred in predictions:
#             # Only evaluate predictions 2-6 hours old (optimal evaluation window)
#             hours_old = (current_time - pred['time']).total_seconds() / 3600
#             if 2 <= hours_old <= 6:
#                 current_price = price_history[symbol][-1] if price_history[symbol] else 0
#                 if current_price > 0:
#                     change = ((current_price - pred['price']) / pred['price']) * 100
#
#                     # Determine if prediction was successful
#                     success = False
#                     if pred['decision'] == "BUY" and change > 1.5:  # 1.5% gain for BUY
#                         success = True
#                     elif pred['decision'] == "SELL" and change < -1.5:  # 1.5% drop for SELL
#                         success = True
#
#                     if pred['decision'] in ["BUY", "SELL"]:
#                         outcomes.append((symbol, pred['decision'], success, change))
#
#                         # Update accuracy (avoid double-counting)
#                         pred_id = f"{symbol}_{pred['time'].timestamp()}"
#                         if pred_id not in getattr(track_prediction_outcomes, 'evaluated_ids', set()):
#                             if not hasattr(track_prediction_outcomes, 'evaluated_ids'):
#                                 track_prediction_outcomes.evaluated_ids = set()
#
#                             track_prediction_outcomes.evaluated_ids.add(pred_id)
#                             accuracy[symbol]["total"] += 1
#                             if success:
#                                 accuracy[symbol]["success"] += 1
#
#                             status = "✅ SUCCESS" if success else "❌ FAIL"
#                             print(f"   {symbol}: {pred['decision']} → {change:+.2f}% {status}")
#
#     # Clean old predictions (older than 24h)
#     for symbol in prediction_history:
#         prediction_history[symbol] = [p for p in prediction_history[symbol]
#                                       if (current_time - p['time']).total_seconds() < 24 * 3600]
#
#     if not outcomes:
#         print("   No predictions ready for evaluation (need 2-6 hour window)")



def update_active_coins():
    """Dynamic coin re-selection with BTC/ETH protection"""
    print(f"\n🔄 Пересборка активных монет (цикл №{main.cycle_count})...")

    candidates = []
    for sym, rec in stats.items():
        if rec["BUY"] + rec["SELL"] > 0:
            acc_data = accuracy[sym]
            acc_rate = (acc_data["success"] / acc_data["total"] * 100) if acc_data["total"] > 0 else 0
            score = (rec["BUY"] + rec["SELL"]) * 0.6 + acc_rate * 0.4
            candidates.append((sym, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in candidates[:ANALYZE_TOTAL]]

        # Ensure BTC and ETH are always included
        for core_sym in ("BTC", "ETH"):
            if core_sym not in top_symbols:
                top_symbols.insert(0, core_sym)

        #  Remove duplicates while preserving order
        main.active_symbols = list(dict.fromkeys(top_symbols))[:ANALYZE_TOTAL]
        print(f"⚙️  Новый набор монет: {', '.join(main.active_symbols)}")


# # --- Main Loop ---
def main():

    if not hasattr(main, 'cycle_count'):
        main.cycle_count = 0
    if not hasattr(main, 'active_symbols'):
        main.active_symbols = []


    print(f"{YELLOW}🚀 Quantum Trader v4.0 запущен...{RESET}\n")

    while True:
        try:
            # === Data Collection ===
            data = cmc_request("/v1/cryptocurrency/listings/latest", {"limit": 200})
            coins = data.get("data", [])
            if not coins:
                print(f"{RED}⚠ No data, will run again in 60s{RESET}")
                time.sleep(60)
                continue

            # === Coin Selection (improved) ===
            active = getattr(main, "active_symbols", None)

            if active:
                # If it has a custom set
                selected = [c for c in coins if c["symbol"] in active]
                print(f"🔍 uses a custom set: {len(selected)} монет")
            else:
                # If no custom set, build one
                core = [c for c in coins if c["symbol"] in ("BTC", "ETH")]

                # Take top by market cap
                top_cap = sorted(coins, key=lambda x: x["cmc_rank"])[:TOP_MARKET_CAP]

                # Take top volatile
                top_vol = sorted(coins, key=lambda x: abs(x["quote"]["USD"].get("percent_change_24h", 0)),
                                 reverse=True)[:TOP_VOLATILE]

                # Combine and deduplicate
                selected, seen = [], set()
                for c in core + top_cap + top_vol:
                    sym = c["symbol"]
                    if sym not in seen:
                        selected.append(c)
                        seen.add(sym)
                    if len(selected) >= ANALYZE_TOTAL:
                        break

                # Set active symbols for next cycles
                main.active_symbols = [c["symbol"] for c in selected]
                print(f"⚙️  Новый набор активных монет: {', '.join(main.active_symbols)}")


            print(f"\n=== Цикл #{getattr(main, 'cycle_count', 0) + 1} ({datetime.now().strftime('%H:%M:%S')}) — {len(selected)} монет ===\n")
            if DETAILED_OUTPUT:
                print(f"🚀 Активных монет для анализа: {len(selected)} / {ANALYZE_TOTAL}\n")

            # === High-Efficiency Filter ===
            HIGH_EFFICIENCY = True
            PRICE_DELTA_THRESHOLD = 0.8  # a bit more sensitive
            if HIGH_EFFICIENCY:
                active_set, skipped = [], []
                for coin in selected:
                    symbol = coin["symbol"]
                    price = coin["quote"]["USD"]["price"]
                    last_prices = price_history[symbol]
                    if len(last_prices) >= 1:
                        prev_price = last_prices[-1]
                        delta = abs(price - prev_price) / prev_price * 100
                        if delta >= PRICE_DELTA_THRESHOLD:
                            active_set.append(coin)
                        else:
                            skipped.append(symbol)
                    else:
                        active_set.append(coin)
                if skipped:
                    print(f"{YELLOW}⚙️  High-Efficiency: пропущено {len(skipped)} монет{RESET}")
                selected = active_set

            if not selected:
                print(f"{YELLOW}⏭️ No active coins on the cycle {RESET}")
                time.sleep(INTERVAL)
                continue

            # === Parallel Analysis ===
            market_context = get_market_trend(coins)
            market_regime = calculate_market_regime(coins)  # ✅ NEW
            print(f"{CYAN}📊 Market regime: {market_regime.upper()}{RESET}")

            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futures = [ex.submit(analyze_single_coin, coin, market_context) for coin in selected]

                for fut in concurrent.futures.as_completed(futures):
                    try:
                        results.append(fut.result(timeout=45))
                    except Exception as e:
                        print(f"{RED}Ошибка анализа: {e}{RESET}")

            # === Process Results ===
            # for res in results:
            #     symbol = res["symbol"]
            #     # Update history and stats
            #     price_history[symbol].append(res["prices_append"])
            #     stats[symbol][res["decision"]] += 1
            #     last_decisions[symbol] = res["decision"]
            #
            #     # Display results
            #     color = GREEN if res["decision"] == "BUY" else RED if res["decision"] == "SELL" else YELLOW
            #     print(f"{color}💰 {symbol}: ${res['price']} ({res['change_24h']:+}%) → {res['answer']}{RESET}")
            #     print(f"   Уверенность: {res['confidence']*100:.1f}%, Volatile: {res['volatility']:.2f}%")
            #
            #     # Log decision
            #     log_coin(symbol, res["price"], res["decision"], res["answer"])
            #     time.sleep(0.2)

            # === Process Results ===
            # === Process Results ===
            # ✅ IMPROVED FILTER LOGIC
            total = len(results)
            filtered_count = 0

            for i, res in enumerate(results, 1):
                symbol = res["symbol"]

                # 🚦 STRICT Confidence Filter (ACTUALLY FILTER)
                if res["confidence"] < MIN_CONFIDENCE_THRESHOLD:
                    res["decision"] = "HOLD"
                    res["answer"] += f" (filtered: {res['confidence']:.2f} confidence)"
                    filtered_count += 1
                    print(
                        f"{YELLOW}🚫 {symbol}: CONFIDENCE FILTERED - {res['confidence']:.2f} < {MIN_CONFIDENCE_THRESHOLD}{RESET}")

                # Always update history for price tracking
                price_history[symbol].append(res["prices_append"])

                # Only count non-HOLD decisions in stats
                if res["decision"] != "HOLD":
                    stats[symbol][res["decision"]] += 1
                    last_decisions[symbol] = res["decision"]

                # Display and log ALL decisions (including filtered ones)
                if DETAILED_OUTPUT:
                    print_coin_analysis(i, total, res)

                log_coin(symbol, res["price"], res["decision"], res["answer"])
                update_recent_accuracy(symbol, res["decision"], res["price"])

                time.sleep(0.2)

            if filtered_count > 0:
                print(f"{YELLOW}📊 Confidence filter blocked {filtered_count}/{total} signals{RESET}")

            low_conf = [r for r in results if r.get("confidence", 1.0) < MIN_CONFIDENCE_THRESHOLD]

            if low_conf:
                print(
                    f"{YELLOW}⚠ Low-confidence signals detected ({len(low_conf)}) — running confidence debug...{RESET}")
                debug_confidence_calculation()

            if DETAILED_OUTPUT:
                print("✅Cycle finished decisions processing.")
                print("-" * 60)
                print("\n🧠 === Trader IQ Report ===")
                top5 = sorted(accuracy.items(),
                              key=lambda kv: kv[1]['success'] / max(1, kv[1]['total']) if kv[1]['total'] > 0 else 0,
                              reverse=True)[:5]
                print("📊 Top 5 coins with highest accuracy:")
                for i, (sym, acc) in enumerate(top5, 1):
                    rate = round(acc['success'] / acc['total'] * 100, 1) if acc['total'] else 0
                    emoji = "🟢" if rate > 60 else "🔴"
                    print(f" {i}. {sym}: {rate}% accuracy through {acc['total']} trades {emoji}")
                print(
                    f"\n📈 Average model accuracy: {sum((a['success'] / a['total']) * 100 for s, a in accuracy.items() if a['total'] > 0) / max(1, len([a for a in accuracy.values() if a['total'] > 0])):.1f}%")

            # === Cycle Completion ===
            main.cycle_count = getattr(main, "cycle_count", 0) + 1

            # # === Risk Control: Drawdown check ===
            # loss_streak = sum(1 for s, a in accuracy.items() if a["total"] > 5 and a["success"] / a["total"] < 0.4)
            # if loss_streak / max(1, len(accuracy)) > MAX_DRAWDOWN:
            #     print(f"{RED}🚨 Просадка превышает лимит! Пауза 10 минут для стабилизации.{RESET}")
            #     save_state()
            #     time.sleep(600)
            #     continue

            bad_ratio = sum(1 for s, a in accuracy.items() if a["total"] > 5 and a["success"] / a["total"] < 0.4)
            total_tracked = sum(1 for a in accuracy.values() if a["total"] > 5)
            if total_tracked and bad_ratio / total_tracked > MAX_DRAWDOWN:
                print(f"{RED}🚨 Down time exceeds limit! Pausing for 10 minutes to stabilize.{RESET}")
                save_state()
                time.sleep(600)
                continue

            # === CYCLE COMPLETION ACTIVITIES ===
            # # 1. Always do after each cycle
            # track_prediction_accuracy()
            # # 2. Periodic activities
            # if main.cycle_count % 10 == 0:  # Report
            #     generate_enhanced_report(accuracy, stats)
            # if main.cycle_count % 15 == 0:  # Rebalance
            #     update_active_coins()
            #     analyze_failed_predictions()
            # # 3. Maintenance (keep your existing)
            # monitor.end_cycle()
            # save_state()

            # === CYCLE COMPLETION ACTIVITIES ===

            # 1. Track prediction outcomes (primary accuracy system)
            track_prediction_outcomes(market_context)

            # 2. Performance analytics
            if main.cycle_count % 5 == 0:
                calculate_portfolio_performance()

            # 3. Strategy improvement analysis
            if main.cycle_count % 20 == 0:  # Every 20 cycles
                improve_trading_strategy()

            # 4. Debug low confidence if needed
            low_conf_count = len([r for r in results if r.get("confidence", 1) < MIN_CONFIDENCE_THRESHOLD])
            if low_conf_count > len(results) * 0.6:  # If >60% low confidence
                print(f"{YELLOW}🔍 High rate of low-confidence signals ({low_conf_count}/{len(results)}){RESET}")
                debug_confidence_calculation()

            # 5. Periodic activities
            if main.cycle_count % 10 == 0:
                generate_enhanced_report(accuracy, stats)

            if main.cycle_count % 15 == 0:
                update_active_coins()
                analyze_failed_predictions()

            # 6. Always save state and monitor
            monitor.end_cycle()
            save_state()


            # stats_data = monitor.get_stats()
            # print(f"{CYAN}⏱ Среднее время цикла: {stats_data['avg_cycle_time']:.2f} сек, "
            #       f"Uptime: {stats_data['uptime_hours']:.1f} ч.{RESET}")


            # Countdown to next cycle
            print(f"\n⏳  {INTERVAL}с...")
            for i in range(INTERVAL, 0, -1):
                print(f"\r🕒 Следующий цикл через: {i:03d}с", end="", flush=True)
                time.sleep(1)
            print("\n")

        except Exception as e:
            print(f"{RED}❌ Ошибка в основном цикле: {e}{RESET}")
            print(f"{YELLOW}⚠️ Попытка восстановить состояние...{RESET}")
            save_state()
            time.sleep(10)
            continue

        stats_data = monitor.get_stats()
        print(f"{CYAN}⏱ Среднее время цикла: {stats_data['avg_cycle_time']:.2f} сек, "
              f"Среднее API-время: {stats_data['avg_api_time']:.2f} сек, "
              f"Uptime: {stats_data['uptime_hours']:.1f} ч.{RESET}")

        # Автосохранение каждые 5 циклов
        if main.cycle_count % 5 == 0:
            print(f"{YELLOW}💾 Автосохранение состояния (цикл {main.cycle_count})...{RESET}")
            save_state()














if __name__ == "__main__":
    print(f"{CYAN}========================================{RESET}")
    print(f"{GREEN}🚀 Quantum Trader v4.4 — ONLINE{RESET}")
    print(f"{YELLOW}⏱ Start time: {datetime.now().strftime('%H:%M:%S')}{RESET}")

    # Restore state BEFORE printing stats
    load_state()

    print(f"{CYAN}Last saved state: {getattr(main, 'cycle_count', 0)} cycles, "
          f"{len(getattr(main, 'active_symbols', []))} active coins{RESET}")
    print(f"{CYAN}========================================\n{RESET}")

    monitor.start_cycle()

    # Initialize accuracy from historical CSV logs
    def initialize_accuracy_from_history():
        """One-time initialization of accuracy from historical data"""
        print(f"{CYAN}📊 Initializing accuracy from historical data...{RESET}")
        for fn in os.listdir(LOG_DIR):
            if fn.endswith(".csv"):
                sym = fn.replace(".csv", "")
                try:
                    with open(os.path.join(LOG_DIR, fn), "r", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    if len(rows) > 1:
                        current_price = float(rows[-1]["Price"])
                        evaluate_past(sym, current_price)
                except Exception:
                    continue
        print(f"{GREEN}✅ Accuracy initialization complete{RESET}\n")

    initialize_accuracy_from_history()

    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}🛑 Stop by user request.{RESET}")
        save_state()
        print(f"{GREEN}💾 State save complete. Exiting.{RESET}")
    except Exception as e:
        print(f"{RED}❌ Accidental crash: {e}{RESET}")
        save_state()
        print(f"{YELLOW}⚠️State saved before exit.{RESET}")







