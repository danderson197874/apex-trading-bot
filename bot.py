#!/usr/bin/env python3
"""
APEX CRYPTO STRATEGIST v2 — SELF-IMPROVING TRADING ENGINE
5-Layer Signal Architecture: Momentum + Arbitrage + ML + Sentiment + OrderFlow + Meta-Learner
Auto-credentials, auto-pair validation, auto-retry, self-learning
"""

import sys, os, time, json, glob, math
from datetime import datetime, timedelta

# ── IMMEDIATE BANNER ──
print()
print("=" * 65)
print("  APEX CRYPTO STRATEGIST v2  |  5-Layer Signal Engine")
print("=" * 65)
sys.stdout.flush()

# ── DEPENDENCY CHECK ──
missing = []
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    missing.append('python-dotenv')

try:
    import pandas as pd
    import numpy as np
except ImportError:
    missing.append('pandas')

try:
    from coinbase.rest import RESTClient
except ImportError:
    missing.append('coinbase-advanced-py')

try:
    import requests as req_lib
except ImportError:
    missing.append('requests')

if missing:
    print(f"\n  ERROR: Missing packages: {', '.join(missing)}")
    print(f"  Fix:   pip install {' '.join(missing)}")
    sys.exit(1)

# ── ML (graceful degradation) ──
SKLEARN_OK = False
try:
    from sklearn.ensemble import GradientBoostingClassifier
    import joblib
    SKLEARN_OK = True
except ImportError:
    pass

import logging, uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('apex.log', mode='a')
    ]
)
log = logging.getLogger('APEX')

# ── CREDENTIALS ──
def load_credentials():
    home = os.path.expanduser('~')
    candidates = [
        os.path.join(home, 'cdp_api_key.json'),
        os.path.join(home, 'trading-agent', 'cdp_api_key.json'),
        '/root/cdp_api_key.json',
        '/tmp/cdp_api_key.json',
        os.path.join(home, 'Downloads', 'cdp_api_key.json'),
    ]
    for found in glob.glob(os.path.join(home, '**', 'cdp_api_key.json'), recursive=True):
        candidates.append(found)
    seen = set()
    for path in candidates:
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            data = json.loads(open(path).read())
        except Exception as e:
            print(f"  WARNING: Could not read {path}: {e}")
            continue
        key_id = data.get('name') or data.get('id') or ''
        priv   = data.get('privateKey') or data.get('private_key') or ''
        if key_id and priv:
            print(f"  Credentials: {path}")
            print(f"  Key id field: {'name' if data.get('name') else 'id'}")
            sys.stdout.flush()
            return key_id, priv
    k = os.getenv('COINBASE_API_KEY', '')
    s = os.getenv('COINBASE_API_SECRET', '')
    if k and s:
        print("  Credentials: .env file")
        sys.stdout.flush()
        return k, s
    print("\n  ERROR: cdp_api_key.json not found!")
    print("  Upload it to your home directory and restart.")
    sys.exit(1)

print("  Searching for credentials...")
sys.stdout.flush()
API_KEY, API_SECRET = load_credentials()
print("  Credentials OK")
sys.stdout.flush()


def sdk_get(obj, key, default=None):
    """Extract a value from a Coinbase SDK response (works with both dict and object)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── CONFIG ──
DRY_RUN             = os.getenv('DRY_RUN', 'true').lower() == 'true'
MAX_COIN_DRAWDOWN   = 0.05
MAX_GLOBAL_DRAWDOWN = 0.15
MIN_RISK_REWARD     = 2.0
POSITION_SIZE_PCT   = {1: 0.12, 2: 0.08, 3: 0.05}
# PAPER (practice) mode trades far more actively so the bot collects learning data.
# LIVE mode keeps the original strict settings — nothing below changes live behaviour.
CYCLE_SECS          = 900 if DRY_RUN else 3600        # 15-min cycles in paper
MIN_TRADE_USDC      = 5.0
LOSS_PAUSE_HOURS    = 1 if DRY_RUN else 3
ENTRY_THRESHOLD     = 0.15 if DRY_RUN else 0.40      # fused score needed to BUY
EXIT_THRESHOLD      = -0.15 if DRY_RUN else -0.35    # fused score that closes an open position
TARGET_SCALE        = 0.5 if DRY_RUN else 1.0        # paper: half-size stop/target => faster round trips
MAX_HOLD_HOURS      = 6 if DRY_RUN else 72           # close stale positions and learn from the result
MIN_HOLD_HOURS      = 1.0                            # signal exits wait this long (avoids 15-min noise flips)
ML_LABEL_HOURS      = 6                              # ML predicts: up >= ML_LABEL_GAIN within 6h?
ML_LABEL_GAIN       = 0.015
ML_RETRAIN_HOURS    = 6
DASHBOARD_URL       = os.getenv('DASHBOARD_URL', '')
INGEST_TOKEN        = os.getenv('INGEST_TOKEN', '')

SYMBOL_REMAP = {'MATIC': 'POL', 'FREYSA': 'FAI'}

DESIRED_COINS = {
    'XRP':    {'tier':1,'stop_pct':0.03,'tp_pct':0.08},
    'DOGE':   {'tier':1,'stop_pct':0.03,'tp_pct':0.08},
    'SHIB':   {'tier':1,'stop_pct':0.03,'tp_pct':0.08},
    'MATIC':  {'tier':1,'stop_pct':0.03,'tp_pct':0.08},
    'CRO':    {'tier':1,'stop_pct':0.03,'tp_pct':0.08},
    'HBAR':   {'tier':2,'stop_pct':0.05,'tp_pct':0.12},
    'XLM':    {'tier':2,'stop_pct':0.05,'tp_pct':0.12},
    'KAVA':   {'tier':2,'stop_pct':0.05,'tp_pct':0.12},
    'FORT':   {'tier':2,'stop_pct':0.05,'tp_pct':0.12},
    'XYO':    {'tier':3,'stop_pct':0.08,'tp_pct':0.20},
    'FREYSA': {'tier':3,'stop_pct':0.08,'tp_pct':0.20},
    'VARA':   {'tier':3,'stop_pct':0.08,'tp_pct':0.20},
    'ONYX':   {'tier':3,'stop_pct':0.08,'tp_pct':0.20},
}

# Arbitrage pairs
ARBI_PAIRS = [('XRP','XLM'), ('DOGE','SHIB'), ('HBAR','XLM'), ('MATIC','CRO')]

# CoinGecko id map for on-chain / market-data enrichment (best-effort; unmapped coins degrade to neutral)
COINGECKO_IDS = {
    'XRP': 'ripple', 'DOGE': 'dogecoin', 'SHIB': 'shiba-inu', 'MATIC': 'polygon-ecosystem-token',  # MATIC rebranded to POL; CoinGecko ID updated
    'CRO': 'crypto-com-chain', 'HBAR': 'hedera-hashgraph', 'XLM': 'stellar', 'KAVA': 'kava',
    'XYO': 'xyo-network', 'VARA': 'vara-network',
}

HTTP_UA = 'APEX-Strategist/2.0 (+https://apex-crypto.abacusai.app)'


# ── MARKET INTELLIGENCE (external knowledge base) ──
class MarketIntel:
    """
    Fuses four FREE, keyless external data sources into the bot's knowledge base:
      1. Fear & Greed Index (alternative.me)   -> market-wide macro sentiment
      2. CoinGecko markets                     -> on-chain / market momentum & volume
      3. CoinGecko community votes             -> social sentiment per coin
      4. Coinbase order book                   -> live order-flow (bid/ask imbalance)
    Every network call is wrapped and degrades to a neutral 0.0 on any failure.
    Global sources are fetched once per cycle; per-coin sources are cached within a cycle.
    """
    def __init__(self):
        self.fng = 50.0                 # neutral fear & greed
        self.cg = {}                    # symbol -> {'d1': pct, 'd7': pct, 'vol_score': x}
        self._social_cache = {}         # symbol -> (score, fetched_at) ; persists across cycles
        self._last_cg_call = 0.0        # throttle CoinGecko free tier (~5 req burst limit)
        self._social_calls = 0          # new social lookups made this cycle (max 3)
        self._book_cache = {}           # symbol -> score

    # ---- called once at the start of every cycle ----
    def refresh_global(self):
        self._book_cache = {}
        self._social_calls = 0
        self._fetch_fear_greed()
        self._fetch_coingecko()

    def _fetch_fear_greed(self):
        try:
            r = req_lib.get('https://api.alternative.me/fng/?limit=1',
                            headers={'User-Agent': HTTP_UA}, timeout=10)
            if r.status_code == 200:
                self.fng = float(r.json()['data'][0]['value'])
                log.info(f"  [INTEL] Fear&Greed index: {self.fng:.0f}/100")
        except Exception as e:
            log.warning(f"  [INTEL] Fear&Greed fetch failed: {e}")

    def _fetch_coingecko(self):
        try:
            ids = ','.join(sorted(set(COINGECKO_IDS.values())))
            if not ids:
                return
            r = req_lib.get(
                'https://api.coingecko.com/api/v3/coins/markets',
                params={'vs_currency': 'usd', 'ids': ids,
                        'price_change_percentage': '24h,7d'},
                headers={'User-Agent': HTTP_UA}, timeout=15)
            if r.status_code != 200:
                log.warning(f"  [INTEL] CoinGecko HTTP {r.status_code}")
                return
            id_to_sym = {v: k for k, v in COINGECKO_IDS.items()}
            for row in r.json():
                sym = id_to_sym.get(row.get('id'))
                if not sym:
                    continue
                d1 = row.get('price_change_percentage_24h') or 0.0
                d7 = row.get('price_change_percentage_7d_in_currency') or 0.0
                # volume relative to market cap = liquidity/interest proxy
                mcap = row.get('market_cap') or 0.0
                vol = row.get('total_volume') or 0.0
                vol_ratio = (vol / mcap) if mcap > 0 else 0.0
                self.cg[sym] = {'d1': float(d1), 'd7': float(d7),
                                'vol_score': max(-1.0, min(1.0, vol_ratio * 4 - 0.3))}
            log.info(f"  [INTEL] CoinGecko market data: {len(self.cg)} coins")
        except Exception as e:
            log.warning(f"  [INTEL] CoinGecko fetch failed: {e}")

    def _cg_social(self, symbol):
        """Per-coin community sentiment from CoinGecko public API (keyless).
        Returns a score in [-1,1], or None when unavailable (caller falls back to macro)."""
        cached = self._social_cache.get(symbol)
        # Community votes change slowly -> reuse a good value for 6 hours
        if cached and cached[0] is not None and time.time() - cached[1] < 6 * 3600:
            return cached[0]
        cid = COINGECKO_IDS.get(symbol)
        # Gentle on the free API: at most 3 new lookups per hourly cycle (coins rotate in)
        if not cid or self._social_calls >= 3:
            return cached[0] if cached else None
        self._social_calls += 1
        score = None
        try:
            wait = 6.0 - (time.time() - self._last_cg_call)
            if wait > 0:
                time.sleep(wait)
            self._last_cg_call = time.time()
            r = req_lib.get(
                f'https://api.coingecko.com/api/v3/coins/{cid}',
                params={'localization': 'false', 'tickers': 'false',
                        'market_data': 'false', 'community_data': 'false',
                        'developer_data': 'false', 'sparkline': 'false'},
                headers={'User-Agent': HTTP_UA}, timeout=10)
            if r.status_code == 200:
                up = r.json().get('sentiment_votes_up_percentage')
                if up is not None:
                    # 50% up = neutral -> 0 ; 100% up -> +1 ; 0% up -> -1
                    score = max(-1.0, min(1.0, (float(up) - 50.0) / 50.0))
            elif r.status_code == 429:
                log.warning(f"  [INTEL] CoinGecko social {symbol} rate-limited")
        except Exception as e:
            log.warning(f"  [INTEL] CoinGecko social {symbol} failed: {e}")
        if score is None:
            # Failed/rate-limited: keep last known value (if any) rather than dropping to n/a
            return cached[0] if cached else None
        self._social_cache[symbol] = (score, time.time())
        return score

    def _orderbook_imbalance(self, symbol, cb, pair):
        if symbol in self._book_cache:
            return self._book_cache[symbol]
        score = 0.0
        try:
            book = cb.get_product_book(product_id=pair, limit=50)
            pb = sdk_get(book, 'pricebook', {}) or sdk_get(book, 'product_book', {}) or {}
            bids = sdk_get(pb, 'bids', []) or []
            asks = sdk_get(pb, 'asks', []) or []
            bid_vol = sum(float(sdk_get(b, 'size', 0) or 0) for b in bids)
            ask_vol = sum(float(sdk_get(a, 'size', 0) or 0) for a in asks)
            if bid_vol + ask_vol > 0:
                score = max(-1.0, min(1.0, (bid_vol - ask_vol) / (bid_vol + ask_vol)))
        except Exception as e:
            log.warning(f"  [INTEL] Order book {symbol} failed: {e}")
        self._book_cache[symbol] = score
        return score

    # ---- layer scores exposed to the engine ----
    def sentiment_score(self, symbol):
        """Macro Fear&Greed (contrarian) blended with per-coin community sentiment."""
        try:
            # Contrarian: extreme fear (0) -> +1 buy, extreme greed (100) -> -1 caution
            fng_score = max(-1.0, min(1.0, (50.0 - self.fng) / 50.0))
            social = self._cg_social(symbol)
            if social is None:
                # No per-coin social data -> run full-strength on macro sentiment
                return max(-1.0, min(1.0, fng_score)), f'fng={self.fng:.0f} social=n/a'
            score = 0.5 * fng_score + 0.5 * social
            return max(-1.0, min(1.0, score)), f'fng={self.fng:.0f} social={social:+.2f}'
        except Exception:
            return 0.0, 'sentiment_err'

    def orderflow_score(self, symbol, cb, pair):
        """Live order-book imbalance blended with CoinGecko on-chain momentum/volume."""
        try:
            book = self._orderbook_imbalance(symbol, cb, pair)
            cg = self.cg.get(symbol, {})
            d1, d7 = cg.get('d1', 0.0), cg.get('d7', 0.0)
            cg_mom = max(-1.0, min(1.0, (d1 / 10.0 + d7 / 20.0) / 2.0))
            vol = cg.get('vol_score', 0.0)
            score = 0.55 * book + 0.30 * cg_mom + 0.15 * vol
            return max(-1.0, min(1.0, score)), f'book={book:+.2f} cg={cg_mom:+.2f}'
        except Exception:
            return 0.0, 'orderflow_err'


# ── META-LEARNER ──
class MetaLearner:
    def __init__(self, dashboard_url, ingest_token):
        self.weights = {'momentum': 0.28, 'arbitrage': 0.15, 'ml': 0.24,
                        'sentiment': 0.16, 'orderflow': 0.17}
        self._fetch_weights(dashboard_url)

    def _fetch_weights(self, url):
        if not url:
            return
        try:
            r = req_lib.get(f"{url}/api/weights", timeout=10)
            if r.status_code == 200:
                data = r.json()
                for k in ['momentum', 'arbitrage', 'ml', 'sentiment', 'orderflow']:
                    if k in data and isinstance(data[k], (int, float)):
                        self.weights[k] = float(data[k])
                log.info(f"  Weights loaded: {self.weights}")
        except Exception as e:
            log.warning(f"  Could not fetch weights: {e} — using defaults")

    def fuse(self, scores):
        total = 0.0
        max_contrib = 0.0
        dominant = 'momentum'
        for layer, score in scores.items():
            w = self.weights.get(layer, 0.33)
            contrib = w * score
            total += contrib
            if abs(contrib) > max_contrib:
                max_contrib = abs(contrib)
                dominant = layer
        return total, dominant

    def record_outcome(self, symbol, entry_scores, outcome, return_pct, pnl_usdc, dashboard_url, token):
        try:
            outcome_sign = 1 if outcome == 'WIN' else -1
            for layer in ['momentum', 'arbitrage', 'ml', 'sentiment', 'orderflow']:
                score = entry_scores.get(layer, 0)
                contribution = 1.0 if (math.copysign(1, score) if score != 0 else 0) == outcome_sign else 0.0
                old_w = self.weights.get(layer, 0.33)
                self.weights[layer] = 0.85 * old_w + 0.15 * contribution
            # Normalize
            total = sum(self.weights.values()) or 1
            for k in self.weights:
                self.weights[k] /= total
        except Exception as e:
            log.warning(f"  Meta-learner update error: {e}")

        # Alerts reach the dashboard via the [ALERT WIN]/[ALERT LOSS] log lines (shipped by
        # log_shipper), so no direct POST here — that used to create duplicate alerts/emails.


# ── ENGINE ──
class ApexEngine:
    def __init__(self):
        print("  Connecting to Coinbase...")
        sys.stdout.flush()
        self.client      = self._connect()
        print("  Connected")
        sys.stdout.flush()
        self.coins       = self._validate_pairs()
        self.daily_start = 0.0
        self.positions   = {}
        self.losses      = {}
        self.paused      = {}
        self.trade_log   = []
        self.win_rate    = {}
        self.cycle       = 0
        self.errors      = {}
        self.consecutive_api_failures = 0
        self.meta        = MetaLearner(DASHBOARD_URL, INGEST_TOKEN)
        self.intel       = MarketIntel()
        self._load_positions()

        # ML state
        self.ml_models = {}
        self.ml_trained_at = {}
        self.ml_acc = {}
        self.ml_warned = False
        self._load_ml_models()

    def _connect(self, retries=5):
        for attempt in range(1, retries + 1):
            try:
                client = RESTClient(api_key=API_KEY, api_secret=API_SECRET)
                client.get_accounts()
                return client
            except Exception as e:
                log.warning(f"Connection attempt {attempt}/{retries}: {e}")
                if attempt == retries:
                    log.error("Cannot connect. Check API key and permissions.")
                    sys.exit(1)
                time.sleep(5 * attempt)

    def _validate_pairs(self):
        print("  Validating trading pairs (this takes ~10s)...")
        sys.stdout.flush()
        valid = {}
        for symbol, cfg in DESIRED_COINS.items():
            base = SYMBOL_REMAP.get(symbol, symbol)
            found = False
            for suffix in ['USDC', 'USD']:
                pair = f"{base}-{suffix}"
                try:
                    resp = self.client.get_candles(
                        product_id=pair,
                        start=str(int(time.time()) - 7200),
                        end=str(int(time.time())),
                        granularity='ONE_HOUR')
                    if sdk_get(resp, 'candles', []):
                        valid[symbol] = {**cfg, 'pair': pair}
                        print(f"    {symbol:6s} -> {pair}")
                        sys.stdout.flush()
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                print(f"    {symbol:6s} -> skipped (not on Coinbase)")
                sys.stdout.flush()
            time.sleep(0.3)
        print(f"  Trading {len(valid)}/{len(DESIRED_COINS)} coins")
        sys.stdout.flush()
        return valid

    # ── ML Model Management ──
    def _load_ml_models(self):
        if not SKLEARN_OK:
            if not self.ml_warned:
                log.warning("sklearn not installed — ML layer disabled (returns 0.0)")
                self.ml_warned = True
            return
        model_dir = os.path.expanduser('~/trading-agent')
        for symbol in DESIRED_COINS:
            path = os.path.join(model_dir, f'model_{symbol}.pkl')
            if os.path.isfile(path):
                try:
                    obj = joblib.load(path)
                    if isinstance(obj, dict) and 'model' in obj:
                        self.ml_models[symbol] = obj
                        self.ml_trained_at[symbol] = obj.get('t', 0)
                        log.info(f"  ML model loaded: {symbol}")
                except Exception:
                    pass

    # ── Open positions survive restarts (so updates don't throw away learning trades) ──
    POSITIONS_FILE = os.path.expanduser('~/trading-agent/positions.json')

    def _load_positions(self):
        try:
            if os.path.isfile(self.POSITIONS_FILE):
                data = json.loads(open(self.POSITIONS_FILE).read())
                if isinstance(data, dict) and data.get('mode') == ('PAPER' if DRY_RUN else 'LIVE'):
                    self.positions = {s: p for s, p in data.get('positions', {}).items() if s in self.coins}
                    if self.positions:
                        log.info(f"  Restored {len(self.positions)} open positions: {', '.join(self.positions)}")
        except Exception as e:
            log.warning(f"  Could not restore positions: {e}")

    def _save_positions(self):
        try:
            os.makedirs(os.path.dirname(self.POSITIONS_FILE), exist_ok=True)
            tmp = self.POSITIONS_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'mode': 'PAPER' if DRY_RUN else 'LIVE', 'positions': self.positions}, f, default=float)
            os.replace(tmp, self.POSITIONS_FILE)
        except Exception as e:
            log.warning(f"  Could not save positions: {e}")

    def _save_ml_model(self, symbol, model):
        try:
            model_dir = os.path.expanduser('~/trading-agent')
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model, os.path.join(model_dir, f'model_{symbol}.pkl'))
        except Exception:
            pass

    # ── Portfolio ──
    def portfolio_value(self):
        try:
            total = 0.0
            for acct in sdk_get(self.client.get_accounts(), 'accounts', []):
                ab = sdk_get(acct, 'available_balance', {})
                bal = float(sdk_get(ab, 'value', 0))
                if bal <= 0:
                    continue
                cur = sdk_get(acct, 'currency', '')
                if cur in ('USDC', 'USD'):
                    total += bal
                else:
                    try:
                        book = self.client.get_best_bid_ask(product_ids=[f"{cur}-USDC"])
                        pbs = sdk_get(book, 'pricebooks', [])
                        bids = sdk_get(pbs[0], 'bids', []) if pbs else []
                        price = float(sdk_get(bids[0], 'price', 0)) if bids else 0.0
                        total += bal * price
                    except Exception:
                        pass
            self.consecutive_api_failures = 0
            return total
        except Exception as e:
            log.error(f"Portfolio error: {e}")
            self.consecutive_api_failures += 1
            return 0.0

    # ── Candles & Indicators ──
    def candles(self, pair, gran='ONE_HOUR', limit=60, retries=3):
        secs = {'ONE_HOUR': 3600, 'ONE_DAY': 86400}.get(gran, 3600)
        end = int(time.time())
        start = end - limit * secs
        for attempt in range(1, retries + 1):
            try:
                resp = self.client.get_candles(
                    product_id=pair, start=str(start), end=str(end), granularity=gran)
                c = sdk_get(resp, 'candles', [])
                if not c:
                    return None
                c = [vars(x) if hasattr(x, '__dict__') and not isinstance(x, dict) else x for x in c]
                df = pd.DataFrame(c)
                for col in ('close', 'high', 'low', 'open', 'volume', 'start'):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df.sort_values('start').reset_index(drop=True)
            except Exception:
                if attempt == retries:
                    return None
                time.sleep(2 ** attempt)

    @staticmethod
    def ema(s, n):
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def rsi(s, n=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(n).mean()
        l = (-d.clip(upper=0)).rolling(n).mean()
        return 100 - 100 / (1 + g / l.replace(0, 1e-9))

    @staticmethod
    def macd(s, fast=12, slow=26, signal=9):
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger(s, n=20, k=2):
        sma = s.rolling(n).mean()
        std = s.rolling(n).std()
        upper = sma + k * std
        lower = sma - k * std
        return upper, lower, sma

    def at_bottom(self, df):
        if df is None or len(df) < 20:
            return False
        price = df['close'].iloc[-1]
        low20 = df['low'].iloc[-20:].min()
        near_bot = price <= low20 * 1.05
        r = self.rsi(df['close'])
        rsi_ok = r.iloc[-3] <= 35 and r.iloc[-1] > r.iloc[-2]
        green = df['close'].iloc[-1] > df['open'].iloc[-1]
        avg_vol = df['volume'].iloc[-20:-1].mean()
        vol_ok = df['volume'].iloc[-1] > avg_vol * 1.2
        return near_bot and rsi_ok and green and vol_ok

    # ── LAYER 1: Momentum/Trend ──
    def score_momentum(self, df, tier, symbol):
        if df is None or len(df) < 50:
            return 0.0, 'insufficient_data'
        close = df['close']
        e9 = self.ema(close, 9)
        e21 = self.ema(close, 21)
        e50 = self.ema(close, 50)
        r = self.rsi(close)
        macd_line, signal_line, histogram = self.macd(close)
        bb_upper, bb_lower, bb_sma = self.bollinger(close)

        death = e9.iloc[-2] >= e21.iloc[-2] and e9.iloc[-1] < e21.iloc[-1]
        if death or r.iloc[-1] >= 70:
            return -0.8, 'death_cross' if death else 'overbought'

        golden = e9.iloc[-2] <= e21.iloc[-2] and e9.iloc[-1] > e21.iloc[-1]
        above50 = close.iloc[-1] > e50.iloc[-1]
        rsi_now, rsi_prev = r.iloc[-1], r.iloc[-2]
        rsi_bounce = r.iloc[-3] <= 35 and rsi_now > rsi_prev
        vol = df['volume'].iloc[-1] > df['volume'].iloc[-20:-1].mean() * 1.5
        mom = close.iloc[-1] > df['high'].iloc[-20:-1].max() * 1.02

        # 1) Full-confirmation setups (original strict patterns) score highest.
        score, reason = 0.0, ''
        if self.at_bottom(df):
            if tier == 1 and golden and above50 and rsi_bounce:
                score, reason = 0.9, 'golden_cross+above_ema50+rsi_bounce'
            elif tier == 2 and (golden or rsi_bounce) and vol:
                score, reason = 0.7, f"{'ema_cross' if golden else 'rsi_bounce'}+volume"
            elif tier == 3 and (mom or rsi_bounce):
                score, reason = 0.6, 'momentum_breakout' if mom else 'rsi_bounce'

        # 2) Graded evidence: partial credit for each bullish / bearish sign,
        #    so the layer produces a usable signal instead of 0 most of the time.
        if score == 0.0:
            parts = []
            def add(v, tag):
                nonlocal score
                score += v
                parts.append(tag)
            if e9.iloc[-1] > e21.iloc[-1]:
                add(0.15, 'ema9>21')
            else:
                add(-0.10, 'ema9<21')
            add(0.10 if above50 else -0.10, 'above_ema50' if above50 else 'below_ema50')
            if rsi_now < 40 and rsi_now > rsi_prev:
                add(0.20, 'rsi_low_rising')
            elif rsi_now > 60 and rsi_now < rsi_prev:
                add(-0.15, 'rsi_high_falling')
            if histogram.iloc[-1] > histogram.iloc[-2]:
                add(0.10, 'macd_rising')
            else:
                add(-0.05, 'macd_falling')
            if close.iloc[-1] > df['open'].iloc[-1]:
                add(0.05, 'green')
            if vol:
                add(0.10 if close.iloc[-1] > df['open'].iloc[-1] else -0.10, 'volume')
            if mom:
                add(0.25, 'breakout')
            reason = '+'.join(parts) or 'neutral'

        # 3) Bonuses
        if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
            score += 0.1
            reason += '+macd_turn'
        if close.iloc[-1] <= bb_lower.iloc[-1] * 1.01 and rsi_now > rsi_prev:
            score += 0.1
            reason += '+bb_lower'

        return max(-1.0, min(1.0, score)), reason

    # ── LAYER 2: Statistical Arbitrage ──
    def score_arbitrage(self, df, symbol):
        scores = []
        for a, b in ARBI_PAIRS:
            if symbol not in (a, b):
                continue
            partner = b if symbol == a else a
            if partner not in self.coins:
                continue
            try:
                partner_df = self.candles(self.coins[partner]['pair'], limit=24)
                if partner_df is None or len(partner_df) < 10:
                    continue
                if df is None or len(df) < 10:
                    continue
                # Align lengths
                min_len = min(len(df), len(partner_df))
                my_close = df['close'].iloc[-min_len:].values
                p_close = partner_df['close'].iloc[-min_len:].values
                # Log price ratio
                ratio = np.log(my_close / np.maximum(p_close, 1e-12))
                mean_r = np.mean(ratio)
                std_r = np.std(ratio)
                if std_r < 1e-9:
                    continue
                z = (ratio[-1] - mean_r) / std_r
                if z > 2.0:
                    scores.append(0.7)  # coin is cheap relative to partner
                elif z < -2.0:
                    scores.append(-0.7)  # coin is expensive
                else:
                    scores.append(0.0)
            except Exception:
                continue
        if not scores:
            return 0.0, 'no_arbi_pairs'
        avg = sum(scores) / len(scores)
        if avg > 0:
            return avg, 'underpriced_vs_pair'
        elif avg < 0:
            return avg, 'overpriced_vs_pair'
        return 0.0, 'arbi_neutral'

    # ── LAYER 3: ML Pattern Recognition ──
    def _ml_features(self, df):
        close = df['close']
        e9 = self.ema(close, 9).iloc[-1]
        e21 = self.ema(close, 21).iloc[-1]
        e50 = self.ema(close, 50).iloc[-1]
        r = self.rsi(close).iloc[-1] / 100.0
        macd_line, signal_line, _ = self.macd(close)
        price = close.iloc[-1]
        macd_norm = macd_line.iloc[-1] / max(price, 1e-9)
        signal_norm = signal_line.iloc[-1] / max(price, 1e-9)
        bb_upper, bb_lower, _ = self.bollinger(close)
        bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / max(price, 1e-9)
        vol_avg = df['volume'].iloc[-20:].mean()
        vol_ratio = df['volume'].iloc[-1] / max(vol_avg, 1e-9)
        ret_1h = (close.iloc[-1] - close.iloc[-2]) / max(close.iloc[-2], 1e-9) if len(close) >= 2 else 0
        ret_6h = (close.iloc[-1] - close.iloc[-7]) / max(close.iloc[-7], 1e-9) if len(close) >= 7 else 0
        return [
            e9 / max(price, 1e-9),
            e21 / max(price, 1e-9),
            e50 / max(price, 1e-9),
            r,
            macd_norm,
            signal_norm,
            bb_width,
            vol_ratio,
            ret_1h,
            ret_6h,
        ]

    def _ml_frame(self, df):
        """Vectorised feature matrix: one row per candle (same features as _ml_features)."""
        close = df['close']
        price = close.replace(0, np.nan)
        macd_line, signal_line, _ = self.macd(close)
        bb_upper, bb_lower, _ = self.bollinger(close)
        vol_avg = df['volume'].rolling(20).mean().replace(0, np.nan)
        return pd.DataFrame({
            'e9': self.ema(close, 9) / price,
            'e21': self.ema(close, 21) / price,
            'e50': self.ema(close, 50) / price,
            'rsi': self.rsi(close) / 100.0,
            'macd': macd_line / price,
            'signal': signal_line / price,
            'bbw': (bb_upper - bb_lower) / price,
            'volr': df['volume'] / vol_avg,
            'r1': close.pct_change(1),
            'r6': close.pct_change(6),
        }).replace([np.inf, -np.inf], np.nan)

    def _train_ml(self, df, symbol):
        """Train on real price history: did price rise ML_LABEL_GAIN within ML_LABEL_HOURS?"""
        X = self._ml_frame(df)
        close = df['close']
        fut = pd.concat([close.shift(-k) for k in range(1, ML_LABEL_HOURS + 1)], axis=1).max(axis=1)
        y = (fut >= close * (1 + ML_LABEL_GAIN)).astype(int)
        valid = X.notna().all(axis=1) & close.shift(-ML_LABEL_HOURS).notna()
        valid.iloc[:50] = False
        Xv, yv = X[valid].values, y[valid].values
        if len(yv) < 80 or yv.min() == yv.max():
            return None
        # Honest accuracy check: train on older 80%, test on newest 20%
        cut = int(len(yv) * 0.8)
        acc = None
        if 0 < yv[:cut].sum() < cut:
            m = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
            m.fit(Xv[:cut], yv[:cut])
            acc = float((m.predict(Xv[cut:]) == yv[cut:]).mean())
        model = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
        model.fit(Xv, yv)
        obj = {'model': model, 'base': float(yv.mean()), 'acc': acc, 'n': int(len(yv)), 't': time.time()}
        self.ml_models[symbol] = obj
        self.ml_trained_at[symbol] = obj['t']
        self._save_ml_model(symbol, obj)
        log.info(f"  ML trained {symbol}: {obj['n']} samples, base_rate={obj['base']:.2f}, "
                 f"holdout_acc={acc if acc is None else round(acc, 2)}")
        return obj

    def score_ml(self, df, symbol):
        if not SKLEARN_OK:
            if not self.ml_warned:
                log.warning("sklearn not available — ML layer returns 0.0")
                self.ml_warned = True
            return 0.0, 'sklearn_missing'
        try:
            if df is None or len(df) < 60:
                return 0.0, 'insufficient_data'
            obj = self.ml_models.get(symbol)
            if obj is None or time.time() - self.ml_trained_at.get(symbol, 0) > ML_RETRAIN_HOURS * 3600:
                hist = None
                cfg = self.coins.get(symbol)
                if cfg:
                    hist = self.candles(cfg['pair'], limit=300)  # ~12 days of hourly candles for training
                if hist is None or len(hist) < 150:
                    hist = df
                obj = self._train_ml(hist, symbol) or obj
                if obj is None:
                    self.ml_trained_at[symbol] = time.time()  # don't retry every cycle
            if obj is None:
                return 0.0, 'model_not_trained'
            row = self._ml_frame(df).iloc[[-1]]
            if row.isna().any(axis=1).iloc[0]:
                return 0.0, 'ml_features_nan'
            prob = obj['model'].predict_proba(row.values)[0]
            up_prob = prob[1] if len(prob) > 1 else obj['base']
            # Score relative to how often this coin normally rises (base rate)
            score = max(-1.0, min(1.0, (up_prob - obj['base']) * 2.5))
            return score, f"ml_p={up_prob:.2f} base={obj['base']:.2f}"
        except Exception as e:
            log.warning(f"ML error for {symbol}: {e}")
            return 0.0, 'ml_error'

    # ── Position & Order Management ──
    def is_paused(self, symbol):
        if symbol in self.paused:
            if datetime.utcnow() < self.paused[symbol]:
                return True
            del self.paused[symbol]
            self.losses[symbol] = 0
        return False

    def record(self, symbol, win):
        wr = self.win_rate.setdefault(symbol, {'wins': 0, 'total': 0})
        wr['total'] += 1
        if win:
            self.losses[symbol] = 0
            wr['wins'] += 1
        else:
            count = self.losses.get(symbol, 0) + 1
            self.losses[symbol] = count
            if count >= 3:
                self.paused[symbol] = datetime.utcnow() + timedelta(hours=LOSS_PAUSE_HOURS)
                log.warning(f"  {symbol} paused {LOSS_PAUSE_HOURS}h after 3 losses")

    def check_position(self, df, symbol, cfg, price, fused=None):
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pnl = (price - pos['entry']) / pos['entry']
        coin_loss = (pos['entry'] - price) / pos['entry']
        held_h = (time.time() - pos.get('opened', time.time())) / 3600
        value = pos['size'] * price / max(pos['entry'], 1e-12)  # current USDC value of the position

        sell_reason = None
        outcome = None

        if coin_loss >= MAX_COIN_DRAWDOWN:
            log.warning(f"  {symbol} 5% drawdown breaker — exiting")
            self.place_order(cfg['pair'], 'SELL', value)
            self.paused[symbol] = datetime.utcnow() + timedelta(hours=24)
            self.record(symbol, False)
            sell_reason = 'drawdown_breaker'
            outcome = 'LOSS'
        elif price <= pos['stop']:
            log.warning(f"  STOP HIT: {symbol} | {pnl:.2%}")
            self.place_order(cfg['pair'], 'SELL', value)
            self.record(symbol, False)
            sell_reason = 'stop_hit'
            outcome = 'LOSS'
        elif price >= pos['target']:
            log.info(f"  TARGET HIT: {symbol} | {pnl:.2%}")
            self.place_order(cfg['pair'], 'SELL', value)
            self.record(symbol, True)
            sell_reason = 'target_hit'
            outcome = 'WIN'
        elif fused is not None and fused <= EXIT_THRESHOLD and held_h >= MIN_HOLD_HOURS:
            log.info(f"  SIGNAL EXIT: {symbol} | fusion={fused:.2f} | {pnl:.2%}")
            self.place_order(cfg['pair'], 'SELL', value)
            outcome = 'WIN' if pnl > 0 else 'LOSS'
            self.record(symbol, outcome == 'WIN')
            sell_reason = 'signal_exit'
        elif held_h >= MAX_HOLD_HOURS:
            log.info(f"  TIME EXIT: {symbol} held {held_h:.1f}h | {pnl:.2%}")
            self.place_order(cfg['pair'], 'SELL', value)
            outcome = 'WIN' if pnl > 0 else 'LOSS'
            self.record(symbol, outcome == 'WIN')
            sell_reason = 'time_exit'

        if outcome:
            return_pct = (price - pos['entry']) / pos['entry'] * 100
            pnl_usdc = pos['size'] * return_pct / 100
            entry_scores = pos.get('entry_scores', {'momentum': 0, 'arbitrage': 0, 'ml': 0})
            dominant = pos.get('dominant', 'momentum')

            log.info(f"[PERF] symbol={symbol} outcome={outcome} return_pct={return_pct:.2f} "
                     f"pnl_usdc={pnl_usdc:.2f} momentum={entry_scores.get('momentum',0):.3f} "
                     f"arbitrage={entry_scores.get('arbitrage',0):.3f} ml={entry_scores.get('ml',0):.3f} "
                     f"sentiment={entry_scores.get('sentiment',0):.3f} orderflow={entry_scores.get('orderflow',0):.3f}")

            if outcome == 'WIN':
                log.info(f"[ALERT WIN] symbol={symbol} return_pct={abs(return_pct):.2f} "
                         f"pnl_usdc={abs(pnl_usdc):.2f} layer={dominant}")
            else:
                log.info(f"[ALERT LOSS] symbol={symbol} return_pct={abs(return_pct):.2f} "
                         f"pnl_usdc={abs(pnl_usdc):.2f} reason={sell_reason} layer={dominant}")

            self.meta.record_outcome(symbol, entry_scores, outcome, return_pct, pnl_usdc, DASHBOARD_URL, INGEST_TOKEN)
            del self.positions[symbol]
            self._save_positions()

    def place_order(self, pair, side, usdc, retries=3):
        if DRY_RUN:
            log.info(f"[PAPER] {side:4s} ${usdc:.2f} {pair}")
            self.trade_log.append({
                't': datetime.utcnow().isoformat(),
                'pair': pair, 'side': side, 'usdc': usdc
            })
            return True
        for attempt in range(1, retries + 1):
            try:
                oid = str(uuid.uuid4())
                if side == 'BUY':
                    self.client.market_order_buy(
                        client_order_id=oid, product_id=pair,
                        quote_size=str(round(usdc, 2)))
                else:
                    book = self.client.get_best_bid_ask(product_ids=[pair])
                    pbs = sdk_get(book, 'pricebooks', [])
                    bids = sdk_get(pbs[0], 'bids', []) if pbs else []
                    price = float(sdk_get(bids[0], 'price', 0)) if bids else 0.0
                    if price <= 0:
                        log.warning(f"  No bid price for {pair}")
                        return False
                    self.client.market_order_sell(
                        client_order_id=oid, product_id=pair,
                        base_size=str(round(usdc / price, 8)))
                log.info(f"LIVE ORDER: {side} {pair} ${usdc:.2f}")
                return True
            except Exception as e:
                log.warning(f"Order attempt {attempt}: {e}")
                if attempt < retries:
                    time.sleep(3 * attempt)
        log.error(f"Order FAILED: {pair} {side}")
        return False

    # ── Main Cycle ──
    def run_cycle(self, pv):
        self.cycle += 1
        usdc_cash = pv - sum(p.get('size', 0) for p in self.positions.values())

        log.info("=" * 65)
        log.info(f"CYCLE #{self.cycle}  |  {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
        log.info(f"Portfolio: ${pv:,.2f}  |  {'PAPER' if DRY_RUN else 'LIVE'}  |  Positions: {len(self.positions)}")
        log.info("=" * 65)

        # Refresh external knowledge base once per cycle (Fear&Greed + CoinGecko)
        self.intel.refresh_global()

        for symbol, cfg in self.coins.items():
            try:
                if self.is_paused(symbol):
                    log.info(f"  {symbol:6s} PAUSED")
                    continue

                df = self.candles(cfg['pair'])
                if df is None:
                    self.errors[symbol] = self.errors.get(symbol, 0) + 1
                    if self.errors.get(symbol, 0) >= 5:
                        log.warning(f"  {symbol} auto-removed after 5 failures")
                        self.paused[symbol] = datetime.utcnow() + timedelta(hours=6)
                    continue
                self.errors[symbol] = 0

                price = df['close'].iloc[-1]
                rsi_val = self.rsi(df['close']).iloc[-1]

                # Get layer scores (5-layer knowledge base)
                s1, r1 = self.score_momentum(df, cfg['tier'], symbol)
                s2, r2 = self.score_arbitrage(df, symbol)
                s3, r3 = self.score_ml(df, symbol)
                s4, r4 = self.intel.sentiment_score(symbol)
                s5, r5 = self.intel.orderflow_score(symbol, self.client, cfg['pair'])

                scores = {'momentum': s1, 'arbitrage': s2, 'ml': s3,
                          'sentiment': s4, 'orderflow': s5}
                fused, dominant = self.meta.fuse(scores)

                if symbol in self.positions:
                    self.check_position(df, symbol, cfg, price, fused)
                    if symbol in self.positions:  # still open
                        # Fix: include full signal breakdown [sentiment, orderflow] so WATCHING coins
                        # show the same detail as WAIT/BUY coins in the dashboard reason field.
                        log.info(f"  {symbol:6s} WATCHING  ${price:.6f}  RSI:{rsi_val:.0f}  |  fusion={fused:.2f}  [{r4}, {r5}]")
                    time.sleep(0.3)
                    continue

                # Decision thresholds (paper mode uses a lower bar to generate learning trades)
                action = None
                reason = f"WAIT (fusion={fused:.2f})"
                amt = pv * POSITION_SIZE_PCT[cfg['tier']]
                if DRY_RUN:
                    amt = max(amt, MIN_TRADE_USDC)  # paper money: let small-tier coins trade too
                if fused > ENTRY_THRESHOLD and s1 > 0:
                    if amt < MIN_TRADE_USDC:
                        reason = f"WAIT (signal {fused:.2f} but size ${amt:.2f} < ${MIN_TRADE_USDC:.0f} min)"
                    elif cfg['tp_pct'] / cfg['stop_pct'] < MIN_RISK_REWARD:
                        reason = f"WAIT (signal {fused:.2f} but risk/reward too low)"
                    else:
                        action = 'BUY'
                        reason = f"Layer fusion: {fused:.2f} (dominant: {dominant})"
                elif fused < EXIT_THRESHOLD:
                    reason = f"WAIT (bearish, fusion={fused:.2f} — not holding)"

                reason = f"{reason}  [{r4}, {r5}]"
                log.info(f"  {symbol:6s} {action or 'WAIT':8s} ${price:.6f}  RSI:{rsi_val:.0f}  |  {reason}")

                if action == 'BUY':
                    if self.place_order(cfg['pair'], 'BUY', amt):
                        self.positions[symbol] = {
                            'entry': price,
                            'stop': price * (1 - cfg['stop_pct'] * TARGET_SCALE),
                            'target': price * (1 + cfg['tp_pct'] * TARGET_SCALE),
                            'size': amt,
                            'opened': time.time(),
                            'entry_scores': scores,
                            'dominant': dominant
                        }
                        self._save_positions()
                time.sleep(0.4)
            except Exception as e:
                log.error(f"  {symbol} error: {e}")
                self.errors[symbol] = self.errors.get(symbol, 0) + 1

        # Log holdings
        holdings_json = json.dumps({
            sym: {'qty': round(p['size'] / max(p['entry'], 1e-9), 6), 'entry': p['entry'], 'current': price}
            for sym, p in self.positions.items()
        })
        log.info(f"[HOLDINGS] usdc={usdc_cash:.2f} total={pv:.2f} positions={holdings_json}")

        # Win rate summary
        wins = sum(v['wins'] for v in self.win_rate.values())
        total = sum(v['total'] for v in self.win_rate.values())
        wr_str = f"{wins}/{total} ({wins/total:.0%})" if total else "no trades yet"
        log.info(f"Cycle #{self.cycle} done  |  Win rate: {wr_str}  |  "
                 f"Weights: M={self.meta.weights.get('momentum',0):.2f} "
                 f"A={self.meta.weights.get('arbitrage',0):.2f} "
                 f"ML={self.meta.weights.get('ml',0):.2f} "
                 f"S={self.meta.weights.get('sentiment',0):.2f} "
                 f"O={self.meta.weights.get('orderflow',0):.2f}  |  "
                 f"Sleeping {CYCLE_SECS // 60}min\n")

    # ── Main Loop ──
    def run(self):
        log.info("=" * 65)
        log.info("APEX CRYPTO STRATEGIST v2 — ENGINE ONLINE")
        log.info(f"Coins: {len(self.coins)}  |  Mode: {'PAPER (DRY RUN)' if DRY_RUN else 'LIVE TRADING'}")
        log.info(f"Layers: Momentum({self.meta.weights.get('momentum',0.40):.0%}) + "
                 f"Arbitrage({self.meta.weights.get('arbitrage',0.25):.0%}) + "
                 f"ML({self.meta.weights.get('ml',0.35):.0%})")
        log.info(f"Entry: confirmed bottom only (RSI bounce + green candle + volume)")
        log.info(f"Breakers: {MAX_COIN_DRAWDOWN:.0%}/coin | {MAX_GLOBAL_DRAWDOWN:.0%} global | {LOSS_PAUSE_HOURS}h after 3 losses")
        log.info("=" * 65)

        last_reset = datetime.utcnow().date()
        while True:
            try:
                if datetime.utcnow().date() != last_reset:
                    self.daily_start = self.portfolio_value()
                    last_reset = datetime.utcnow().date()
                    log.info(f"Day reset  |  Baseline: ${self.daily_start:,.2f}")

                pv = self.portfolio_value()
                if pv == 0.0 and self.consecutive_api_failures >= 2:
                    log.warning(f"[ALERT OFFLINE] cycles_missed={self.consecutive_api_failures}")

                if self.daily_start == 0:
                    self.daily_start = pv
                    log.info(f"Baseline set: ${pv:,.2f}")

                if self.daily_start > 0:
                    loss = (self.daily_start - pv) / self.daily_start
                    if loss >= MAX_GLOBAL_DRAWDOWN:
                        log.warning(f"GLOBAL HALT: down {loss:.1%} — waiting 1h")
                        time.sleep(CYCLE_SECS)
                        continue

                self.run_cycle(pv)
                time.sleep(CYCLE_SECS)
            except KeyboardInterrupt:
                log.info(f"Stopped.  Cycles: {self.cycle}  |  Trades: {len(self.trade_log)}")
                break
            except Exception as e:
                log.error(f"Main loop error: {e} — restarting in 5min")
                time.sleep(300)


if __name__ == '__main__':
    ApexEngine().run()
