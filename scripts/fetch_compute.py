#!/usr/bin/env python3

"""
Skeleton data builder for Stock Dashboard (free-tier, no-server)
- Reads tickers from scripts/tickers.txt (one per line; lines starting with # are ignored)
- For each ticker, writes a minimal JSON stub to /data/<TICKER>.json so the web UI can render
- Writes /data/meta.json containing { generated_at, tickers }
Next steps (replace stubs with real data):
- Quotes & indicators via yfinance + pandas/pandas_ta
- Sentiment via RSS (feedparser) + VADER
- Fundamentals via yfinance.Ticker.info or alternative sources
- Map metrics to points per README and compute total score
"""

from __future__ import annotations
import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import time
import math
try:
    import certifi as _certifi
except Exception:
    _certifi = None
from typing import Any, Callable, Optional, TypeVar, Tuple

# Ensure TLS CA bundle for curl_cffi/requests on Windows/OneDrive paths
# Copy certifi CA to an ASCII-only path to avoid libcurl issues with non-ASCII user profiles

import os as _os_for_ca
def _ensure_ascii_ca_bundle() -> None:
    try:
        if _certifi is None:
            return
        _src = _certifi.where()
        _ca_path = _src
        if _os_for_ca.name == 'nt':
            _target_dir = r"C:\Users\Public\stockdash-cert"
            _target_file = _target_dir + r"\cacert.pem"
            try:
                _os_for_ca.makedirs(_target_dir, exist_ok=True)
                import shutil as _sh
                _sh.copyfile(_src, _target_file)
                _ca_path = _target_file
            except Exception:
                _ca_path = _src
        # Set env for requests, curl_cffi, and OpenSSL consumers within this process
        _os_for_ca.environ["SSL_CERT_FILE"] = _ca_path
        _os_for_ca.environ["CURL_CA_BUNDLE"] = _ca_path
        _os_for_ca.environ["REQUESTS_CA_BUNDLE"] = _ca_path
    except Exception:
        # Best-effort; let downstream libs decide defaults
        pass

_ensure_ascii_ca_bundle()

# --- Optional Polygon.io integration (free-tier EOD) ---
USE_POLYGON = os.getenv("USE_POLYGON", "").strip() == "1"
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or ""

# Respect 5 calls/min (12s) on free tier
_POLY_LAST_CALL = 0.0

def _poly_sleep_if_needed():
    global _POLY_LAST_CALL
    now = time.time()
    delta = now - _POLY_LAST_CALL
    min_gap = 12.2
    if delta < min_gap:
        time.sleep(min_gap - delta)
    _POLY_LAST_CALL = time.time()

_US_EX_SUFFIXES = {".CO",".TO",".L",".DE",".PA",".SW",".HK",".SI",".AX",".NZ",".SA",".MX",".VI",".HE",".ST",".OL",".MI",".BR",".MC",".IS"}

def _poly_us_symbol(symbol: str) -> bool:
    s = symbol.upper().strip()
    if not s or any(c.isspace() for c in s):
        return False
    # Non-US usually carry country suffix like ".CO"; allow class dots like BRK.B
    if any(s.endswith(suf) for suf in _US_EX_SUFFIXES):
        return False
    return True

def _poly_fetch_hist_df(symbol: str, days: int = 730):
    if not (USE_POLYGON and POLYGON_API_KEY and _poly_us_symbol(symbol)):
        return None
    try:
        from datetime import date, timedelta
        import urllib.request, urllib.parse, json as _json
        end = date.today()
        start = end - timedelta(days=days)
        base = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        qs = urllib.parse.urlencode({"adjusted":"true","sort":"asc","limit":50000,"apiKey":POLYGON_API_KEY})
        url = f"{base}?{qs}"
        _poly_sleep_if_needed()
        req = urllib.request.Request(url, headers={"Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, dict) or (data.get("resultsCount",0) <= 0):
            return None
        rows = data.get("results") or []
        # Build pandas DataFrame with Yahoo-like columns
        assert pd is not None
        import pandas as _pd
        ts = [datetime.fromtimestamp(int(r.get("t",0))/1000.0).date() for r in rows]
        close = [float(r.get("c")) for r in rows]
        high = [float(r.get("h")) for r in rows]
        low = [float(r.get("l")) for r in rows]
        vol = [float(r.get("v")) for r in rows]
        df = _pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": vol}, index=_pd.to_datetime(ts))
        return df
    except Exception as e:
        # Fallback handled by caller
        print(f"WARN: polygon fetch failed for {symbol}: {e}")
        return None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TICKERS_FILE = ROOT / "scripts" / "tickers.txt"

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def read_tickers(path: Path) -> list[str]:
    tickers: list[str] = []
    if not path.exists():
        return tickers
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(line)
    return tickers

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), indent=None), encoding="utf-8")
    tmp.replace(path)
T = TypeVar("T")

def run_with_retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 5.0, label: Optional[str] = None) -> T:
    last_err: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt >= attempts:
                break
            wait = base_delay * (2 ** (attempt - 1))
            name = label or getattr(fn, "__name__", "call")
            print(f"Retry {name} attempt {attempt} failed: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)
    assert last_err is not None
    raise last_err

def use_cached_payload(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    flags = payload.get("flags")
    if isinstance(flags, list):
        if "stale_data" not in flags:
            flags.append("stale_data")
    else:
        flags = ["stale_data"]
    payload["flags"] = flags
    payload["stale_at"] = utcnow_iso()
    try:
        write_json(path, payload)
        print(f"INFO: using cached data for {path.stem}")
        return True
    except Exception as exc:
        print(f"WARN: failed to persist cached data for {path.stem}: {exc}")
        return False

def stub_stock_json(ticker: str) -> dict:
    """Return a minimum viable JSON so the UI can render before real data is wired."""
    t = ticker.upper().strip()
    return {
        "ticker": t,
        "score": 0,
        "fund_points": 0,
        "tech_points": 0,
        "sent_points": 0,
        "price": None,
        "sma50": None,
        "sma200": None,
        "updated_at": utcnow_iso(),
        "flags": ["stub_data"],
    }

# --- v0.2: full data pipeline helpers ---
try:
    import yfinance as yf
    import pandas as pd
    import feedparser
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    # Defer import errors to runtime per-ticker, so the script file can exist without deps.
    yf = None  # type: ignore
    pd = None  # type: ignore
    feedparser = None  # type: ignore
    SentimentIntensityAnalyzer = None  # type: ignore

# Safe NA checker that works even if pandas is unavailable at runtime
from typing import Any as _Any

def safe_pd_isna(x: _Any) -> bool:
    try:
        if 'pd' in globals() and pd is not None:  # type: ignore[name-defined]
            return bool(pd.isna(x))  # type: ignore[union-attr]
    except Exception:
        pass
    # Fallback: treat None and float('nan') as NA
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return x is None


def _get_info_safe(ticker: Any) -> dict:
    info = {}
    for attr in ("get_info", "info"):
        try:
            if hasattr(ticker, attr):
                v = getattr(ticker, attr)()
                if isinstance(v, dict) and v:
                    info = v
                    break
        except Exception:
            continue
    return info or {}

def compute_indicators(symbol: str) -> Tuple[dict, list[str]]:
    flags: list[str] = []
    if pd is None:
        raise RuntimeError("pandas not installed; run: python -m pip install pandas")
    # Try Polygon first if enabled and symbol is US-like
    hist = None
    if USE_POLYGON and POLYGON_API_KEY and _poly_us_symbol(symbol):
        hist = _poly_fetch_hist_df(symbol, days=730)
        if hist is None:
            flags.append("poly_fallback")
    # Fallback to yfinance
    if hist is None:
        if yf is None:
            raise RuntimeError("yfinance not installed; run: python -m pip install yfinance pandas feedparser vaderSentiment")
        t = yf.Ticker(symbol)
        try:
            hist = t.history(period="1y", interval="1d", auto_adjust=False)
        except Exception as e:
            raise RuntimeError(f"history failed for {symbol}: {e}")

    if hist is None or len(hist) == 0 or "Close" not in hist:
        flags.append("no_price_data")
        return {
            "price": None,
            "sma50": None,
            "sma200": None,
            "rsi": None,
            "atr_pct": None,
            "vol20_rising": False,
            "price_gt_ma20": False,
            "series": {"dates": [], "close": [], "sma50": [], "sma200": [], "rsi": [], "atr_pct": []},
        }, flags

    close = hist["Close"].astype(float)
    high = hist.get("High", close)
    low = hist.get("Low", close)
    vol = hist.get("Volume")

    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()
    # RSI(14)
    try:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
    except Exception:
        rsi = pd.Series([None] * len(close), index=close.index)
        flags.append("rsi_fail")

    # ATR(14) and ATR%
    try:
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_pct = (atr / close) * 100.0

    except Exception:

        atr_pct = pd.Series([None] * len(close), index=close.index)
        flags.append("atr_fail")
    price = float(close.iloc[-1]) if len(close) else None
    sma50_last = float(sma50.iloc[-1]) if len(sma50) else None
    sma200_last = float(sma200.iloc[-1]) if len(sma200) else None
    rsi_last = float(rsi.iloc[-1]) if len(rsi) and pd.notna(rsi.iloc[-1]) else None
    atrpct_last = float(atr_pct.iloc[-1]) if len(atr_pct) and pd.notna(atr_pct.iloc[-1]) else None

    vol20_rising = False
    price_gt_ma20 = False
    if vol is not None and len(vol) >= 25:
        vol20 = vol.rolling(20, min_periods=20).mean()
        if vol20.notna().sum() >= 10:
            recent = vol20.tail(5).mean()
            prior = vol20.tail(10).head(5).mean()
            vol20_rising = bool(recent > prior)
    ma20 = close.rolling(20, min_periods=20).mean()
    if len(ma20) and pd.notna(ma20.iloc[-1]):
        price_gt_ma20 = bool(close.iloc[-1] > ma20.iloc[-1])

    # Build compact series for charts (last ~120 points)
    try:
        series_tail = 120
        df = close.to_frame("close").join([
            sma50.rename("sma50"),
            sma200.rename("sma200"),
            rsi.rename("rsi"),
            atr_pct.rename("atr_pct"),
        ])
        df = df.tail(series_tail)
        dates = [idx.strftime("%Y-%m-%d") for idx in df.index]
        def to_list(col):
            vals = []
            for x in df[col].tolist():
                try:
                    vals.append(float(x) if x is not None and not safe_pd_isna(x) else None)
                except Exception:
                    vals.append(None)
            return vals
        series = {
            "dates": dates,
            "close": to_list("close"),
            "sma50": to_list("sma50"),
            "sma200": to_list("sma200"),
            "rsi": to_list("rsi"),
            "atr_pct": to_list("atr_pct"),
        }

    except Exception:

        series = {"dates": [], "close": [], "sma50": [], "sma200": [], "rsi": [], "atr_pct": []}
    # Relative strength vs SPY (optional)
    rs_ratio = None
    rs_rising = False
    try:
        sc = None
        # Try Polygon first if enabled
        try:
            df_spy = _poly_fetch_hist_df("SPY", days=400)
        except Exception:
            df_spy = None
        if df_spy is not None and "Close" in df_spy:
            spy_close = df_spy["Close"].astype(float)
            sc = spy_close.reindex(close.index)
        elif yf is not None:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y", interval="1d", auto_adjust=False)
            if spy_hist is not None and len(spy_hist) and "Close" in spy_hist:
                spy_close = spy_hist["Close"].astype(float)
                sc = spy_close.reindex(close.index)
        if sc is not None:
            try:
                sc = sc.ffill().bfill()
            except Exception:
                pass
            rs = close / sc
            rs50 = rs.rolling(50, min_periods=20).mean()
            rs_rising = bool(rs50.tail(5).mean() > rs50.tail(10).head(5).mean())
            rs_ratio = float(rs.iloc[-1])
    except Exception:
        pass
    return {
        "price": price,
        "sma50": sma50_last,
        "sma200": sma200_last,
        "rsi": rsi_last,
        "atr_pct": atrpct_last,
        "vol20_rising": vol20_rising,
        "price_gt_ma20": price_gt_ma20,
        "series": series,
        "rs_ratio": rs_ratio,
        "rs_rising": rs_rising,
    }, flags

def fetch_news_sentiment(label: str, symbol: str) -> dict:
    """Fetch news headlines for the last 7/30 days and compute a simple sentiment score.
    Be robust to missing fields and never raise – return a neutral payload on any error.
    """
    if feedparser is None or SentimentIntensityAnalyzer is None:
        # Dependencies missing – return neutral payload (caller will still score it)
        return {
            "points": 0,
            "mean7": None,
            "count7": 0,
            "count30": 0,
            "flow": "low",
            "signal_terms": False,
            "buckets": {"sent": "none"},
        }

    try:
        # Query: prefer symbol, fallback to label. Build encoded Google News query
        from urllib.parse import urlencode
        params = {
            "q": f"{symbol} stock OR {label} stock",
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        source = "google"
        url = "https://news.google.com/rss/search?" + urlencode(params)

        # Fetch with explicit User-Agent to avoid bot filtering
        try:
            from urllib.request import Request, urlopen  # type: ignore
            req = Request(url, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 StockDashBot/1.0",
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                "Accept-Language": "en-US,en;q=0.7",
            })
            with urlopen(req, timeout=20) as resp:
                content = resp.read()
            d = feedparser.parse(content)
        except Exception:
            # Fallback to feedparser direct URL fetch
            d = feedparser.parse(url)

        from time import mktime
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        d7 = now - dt.timedelta(days=7)
        d30 = now - dt.timedelta(days=30)

        titles_7: list[str] = []
        titles_30: list[str] = []

        entries = getattr(d, "entries", None) or []
        # Fallback: if Google returns no entries, try Bing News RSS
        if not entries:
            try:
                from urllib.parse import quote_plus
                q = quote_plus(f"{symbol} stock OR {label} stock")
                b_url = f"https://www.bing.com/news/search?q={q}&format=RSS&cc=US&setlang=en-US"
                source = "bing"
                try:
                    from urllib.request import Request, urlopen  # type: ignore
                    req2 = Request(b_url, headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36 StockDashBot/1.0",
                        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.7",
                    })
                    with urlopen(req2, timeout=20) as resp2:
                        content2 = resp2.read()
                    d2 = feedparser.parse(content2)
                except Exception:
                    d2 = feedparser.parse(b_url)
                entries = getattr(d2, "entries", None) or []
            except Exception:
                entries = entries
        for e in entries:
            # title can be str or list fragments
            title_raw = e.get("title", "") if hasattr(e, "get") else getattr(e, "title", "")
            if isinstance(title_raw, list):
                title = " ".join(str(t) for t in title_raw).strip()
            else:
                title = str(title_raw or "").strip()
            if not title:
                continue

            # published_parsed can be None/missing or different name; try a few
            published = None
            if hasattr(e, "get"):
                published = e.get("published_parsed") or e.get("updated_parsed") or e.get("created_parsed")
            else:
                published = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None) or getattr(e, "created_parsed", None)

            # Try to construct a timezone-aware datetime (UTC)
            pdt = None
            if published and isinstance(published, (tuple, time.struct_time)):
                try:
                    # mktime returns seconds since epoch in local time; interpret and convert explicitly to UTC
                    ts = mktime(published)
                    pdt = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
                except Exception:
                    pdt = None

            if pdt is None:
                # Fallback: parse string-based published dates (RFC2822) if available
                try:
                    if hasattr(e, "get"):
                        published_str = e.get("published") or e.get("updated") or e.get("created")
                    else:
                        published_str = getattr(e, "published", None) or getattr(e, "updated", None) or getattr(e, "created", None)
                    if published_str:
                        from email.utils import parsedate_to_datetime
                        _tmp = parsedate_to_datetime(str(published_str))
                        if _tmp is not None:
                            pdt = _tmp.astimezone(dt.timezone.utc) if _tmp.tzinfo else _tmp.replace(tzinfo=dt.timezone.utc)
                except Exception:
                    pdt = None

            if pdt is None:
                # If we still cannot determine a date, skip this entry conservatively
                continue

            if pdt >= d30:
                titles_30.append(title)
            if pdt >= d7:
                titles_7.append(title)

        analyzer = SentimentIntensityAnalyzer()

        def mean_comp(titles: list[str]) -> Optional[float]:
            if not titles:
                return None
            vals = []
            for t in titles:
                try:
                    vals.append(float(analyzer.polarity_scores(t).get("compound", 0.0)))
                except Exception:
                    continue
            return (sum(vals) / len(vals)) if vals else None

        sent7 = mean_comp(titles_7)
        count7 = len(titles_7)
        count30 = len(titles_30)
        avg7 = (count30 / 30.0) * 7.0 if count30 else 0.0

        # flow intensity
        if avg7 <= 0.0:
            flow_points = 0
            flow = "low"
        else:
            ratio = count7 / avg7
            if ratio >= 1.4:
                flow_points = 6
                flow = "high"
            elif ratio >= 0.7:
                flow_points = 3
                flow = "neutral"
            else:
                flow_points = 0
                flow = "low"

        # headline sentiment points
        if sent7 is None:
            sent_points = 0
            sent_bucket = "none"
        elif sent7 > 0.2:
            sent_points = 15
            sent_bucket = ">0.2"
        elif sent7 >= 0.0:
            sent_points = 8
            sent_bucket = "0-0.2"
        else:
            sent_points = 0
            sent_bucket = "<0"

        # signal terms
        terms = ["contract award", "guidance raise", "insider buying"]
        has_signal = any(any(term in t.lower() for term in terms) for t in titles_30)
        signal_points = 4 if has_signal else 0

        total_points = sent_points + flow_points + signal_points

        return {
            "points": total_points,
            "mean7": sent7,
            "count7": count7,
            "count30": count30,
            "flow": flow,
            "signal_terms": has_signal,
            "buckets": {"sent": sent_bucket},
            "source": source,
            "entries": int(count30),
        }
    except Exception as _e:
        # On any unexpected error, return a neutral payload instead of raising
        return {
            "points": 0,
            "mean7": None,
            "count7": 0,
            "count30": 0,
            "flow": "low",
            "signal_terms": False,
            "buckets": {"sent": "none"},
            "source": "none",
            "entries": 0,
        }





def fetch_fundamentals(symbol: str) -> dict:

    if yf is None:

        raise RuntimeError("yfinance not installed")

    t = yf.Ticker(symbol)

    info = _get_info_safe(t)

    # Normalize numeric fields

    def num(key: str) -> Optional[float]:

        v = info.get(key)

        try:

            return float(v) if v is not None else None

        except Exception:

            return None



    fcf = num("freeCashflow")

    mcap = num("marketCap")

    total_debt = num("totalDebt")

    cash = num("totalCash")

    ebitda = num("ebitda")

    gross_margins = num("grossMargins")  # ratio 0-1

    revenue_growth = num("revenueGrowth")  # ratio 0-1

    insiders = num("heldPercentInsiders")  # ratio 0-1



    fcf_yield = (fcf / mcap) if fcf is not None and mcap and mcap > 0 else None

    net_debt = (total_debt or 0.0) - (cash or 0.0) if (total_debt is not None or cash is not None) else None

    nd_to_ebitda = (net_debt / ebitda) if (net_debt is not None and ebitda and ebitda > 0) else None



    return {

        "fcf_yield": fcf_yield,

        "nd_to_ebitda": nd_to_ebitda,

        "gross_margin": gross_margins,

        "revenue_growth": revenue_growth,

        "insider_own": insiders,

    }





def score_fundamentals(f: dict) -> Tuple[int, dict, list[str]]:

    flags: list[str] = []

    # Individual metric points

    pts = []

    max_pts = []



    # FCF Yield

    v = f.get("fcf_yield")

    if v is not None:

        if v > 0.08:

            pts.append(12)

        elif v >= 0.04:

            pts.append(8)

        elif v >= 0.00:

            pts.append(4)

        else:

            pts.append(0)

        max_pts.append(12)

    # Net Debt / EBITDA

    v = f.get("nd_to_ebitda")

    if v is not None:

        if v < 1.0:

            pts.append(8)

        elif v < 2.0:

            pts.append(5)

        elif v < 3.0:

            pts.append(2)

        else:

            pts.append(0)

        max_pts.append(8)

    # Gross Margin

    v = f.get("gross_margin")

    if v is not None:

        if v > 0.45:

            pts.append(8)

        elif v >= 0.30:

            pts.append(4)

        else:

            pts.append(0)

        max_pts.append(8)

    # Revenue Growth

    v = f.get("revenue_growth")

    if v is not None:

        if v > 0.15:

            pts.append(6)

        elif v >= 0.05:

            pts.append(3)

        elif v >= 0.0:

            pts.append(1)

        else:

            pts.append(0)

        max_pts.append(6)

    # Insider Ownership

    v = f.get("insider_own")

    if v is not None:

        if v >= 0.10:

            pts.append(6)

        elif v >= 0.03:

            pts.append(3)

        else:

            pts.append(0)

        max_pts.append(6)



    if not max_pts:

        flags.append("fundamentals_missing")

        return 0, {"available": 0, "max": 40}, flags



    raw = sum(pts)

    avail = sum(max_pts)

    scaled = int(round((raw / avail) * 40)) if avail > 0 else 0

    if len(max_pts) < 3:

        flags.append("low_data")



    return scaled, {"raw": raw, "available": avail, "max": 40}, flags





def score_technicals(t: dict) -> Tuple[int, dict]:

    points = 0

    details = {}



    price = t.get("price")

    sma50 = t.get("sma50")

    sma200 = t.get("sma200")

    rsi = t.get("rsi")

    atr_pct = t.get("atr_pct")

    vol20_rising = t.get("vol20_rising")

    price_gt_ma20 = t.get("price_gt_ma20")



    # Trend

    if price is not None and sma200 is not None and price > sma200:

        points += 8

    if sma50 is not None and sma200 is not None and sma50 > sma200:

        points += 4



    # Momentum (RSI 14)

    if rsi is not None:

        if 60 <= rsi < 70:

            points += 10

        elif 45 <= rsi < 60 or rsi > 80:

            points += 6



    # Volume confirmation

    if vol20_rising and price_gt_ma20:

        points += 7

    elif vol20_rising:

        points += 3



    # ATR quality

    if atr_pct is not None:

        if 2.0 <= atr_pct <= 6.0:

            points += 6

        elif 1.0 <= atr_pct <= 8.0:

            points += 4

        else:

            points += 1



    details = {

        "trend": {"close_gt_sma200": bool(price is not None and sma200 is not None and price > sma200),

                   "sma50_gt_sma200": bool(sma50 is not None and sma200 is not None and sma50 > sma200)},

        "rsi": rsi,

        "atr_pct": atr_pct,

        "vol_confirm": {"rising20": bool(vol20_rising), "price_gt_ma20": bool(price_gt_ma20)},

        "series": t.get("series"),

    }



    return int(points), details





def score_sentiment(s: dict) -> Tuple[int, dict]:

    pts = s.get("points", 0)

    details = {

        "mean7": s.get("mean7"),

        "count7": s.get("count7"),

        "count30": s.get("count30"),

        "flow": s.get("flow"),

        "signal_terms": s.get("signal_terms"),

    }

    return int(pts), details





def process_ticker(label: str) -> Tuple[dict, list[str]]:

    # Use provided label as symbol by default. If it contains spaces, try it verbatim first;

    # if data unavailable, we'll flag and continue.

    symbol = label

    flags: list[str] = []



    # Indicators and price

    try:

        tech, tflags = run_with_retry(lambda: compute_indicators(symbol), attempts=3, base_delay=5.0, label=f"indicators:{symbol}")

        flags.extend(tflags)

    except Exception as e:

        flags.append(f"indicators_fail:{e.__class__.__name__}")

        tech = {"price": None, "sma50": None, "sma200": None, "rsi": None, "atr_pct": None,

                "vol20_rising": False, "price_gt_ma20": False,

                "series": {"dates": [], "close": [], "sma50": [], "sma200": [], "rsi": [], "atr_pct": []}

               }



    # Sentiment

    try:

        sent = run_with_retry(lambda: fetch_news_sentiment(label=label, symbol=symbol), attempts=2, base_delay=5.0, label=f"sentiment:{symbol}")

    except Exception as e:

        flags.append(f"sent_fail:{e.__class__.__name__}")

        sent = {"points": 0, "mean7": None, "count7": 0, "count30": 0, "flow": "low", "signal_terms": False}



    # Fundamentals

    try:

        f = run_with_retry(lambda: fetch_fundamentals(symbol), attempts=3, base_delay=5.0, label=f"fundamentals:{symbol}")

        fund_pts, _, fflags = score_fundamentals(f)

        flags.extend(fflags)


    except Exception as e:

        flags.append(f"fund_fail:{e.__class__.__name__}")

        f = {}

        fund_pts, _ = 0, {"available": 0, "max": 40}



    # Technicals + Sentiment scores

    tech_pts, tech_meta = score_technicals(tech)

    sent_pts, sent_meta = score_sentiment(sent)



    # Compute simple buy zones (ATR/SMA pullbacks + breakout retest)
    zones: list[dict] = []
    try:
        price = tech.get("price")
        sma50 = tech.get("sma50")
        atr_pct = tech.get("atr_pct")
        tm = tech_meta or {}
        trend = (tm.get("trend") or {}) if isinstance(tm, dict) else {}
        uptrend = bool(trend.get("close_gt_sma200") and trend.get("sma50_gt_sma200"))
        atr_abs = (price * (atr_pct / 100.0)) if (isinstance(price,(int,float)) and isinstance(atr_pct,(int,float))) else None
        series = (tm.get("series") or {}) if isinstance(tm, dict) else {}
        closes = [float(c) for c in (series.get("close") or []) if isinstance(c,(int,float))]
        # SMA20 from last 20 closes
        sma20_last = (sum(closes[-20:]) / 20.0) if len(closes) >= 20 else None
        # Config overrides for buy zones
        k50 = 1.0; k20 = 0.8; k_retest_lo = 1.0; k_retest_hi = 0.2
        try:
            conf = json.loads((ROOT / "config" / "buyzones.json").read_text(encoding="utf-8"))
            k50 = float(conf.get("sma50_k_atr", k50))
            k20 = float(conf.get("sma20_k_atr", k20))
            k_retest_lo = float(conf.get("retest_k_lo", k_retest_lo))
            k_retest_hi = float(conf.get("retest_k_hi", k_retest_hi))
        except Exception:
            pass

        if uptrend and sma50 is not None and atr_abs is not None:
            low = max(0.0, float(sma50) - k50 * atr_abs)
            high = float(sma50)
            zones.append({"type":"sma_pullback","ma":"SMA50","price_low":round(low,2),"price_high":round(high,2),"confidence":0.7,"rationale":f"Uptrend; pullback to SMA50 ±{k50} ATR"})
        if uptrend and sma20_last is not None and atr_abs is not None:
            low = max(0.0, float(sma20_last) - k20 * atr_abs)
            high = float(sma20_last)
            zones.append({"type":"sma_pullback","ma":"SMA20","price_low":round(low,2),"price_high":round(high,2),"confidence":0.6,"rationale":f"Uptrend; pullback to SMA20 ±{k20} ATR"})
        # Breakout retest: prior recent high (exclude last 5 days)
        recent_high = None
        if len(closes) >= 30:
            look = min(60, len(closes)-5)
            prior = closes[-(look+5):-5]
            if prior:
                recent_high = max(prior)
        if uptrend and recent_high is not None and atr_abs is not None and isinstance(price,(int,float)) and price > recent_high:
            pl = max(0.0, recent_high - k_retest_lo * atr_abs)
            ph = max(0.0, recent_high - k_retest_hi * atr_abs)
            zones.append({"type":"breakout_retest","level":round(float(recent_high),2),"price_low":round(pl,2),"price_high":round(ph,2),"confidence":0.6,"rationale":"Retest of recent resistance"})
    except Exception:
        zones = []
    # Exit levels and portfolio health (lightweight EOD heuristics)
    exit_levels: dict | None = None
    position_health: dict | None = None
    try:
        price = tech.get("price")
        sma50 = tech.get("sma50")
        sma200 = tech.get("sma200")
        atr_pct = tech.get("atr_pct")
        trend = (tech_meta.get("trend") or {}) if isinstance(tech_meta, dict) else {}
        volc = (tech_meta.get("vol_confirm") or {}) if isinstance(tech_meta, dict) else {}
        in_zone = False
        if isinstance(price, (int, float)):
            for z in zones:
                try:
                    if float(z.get("price_low", -1)) <= price <= float(z.get("price_high", -1)):
                        in_zone = True
                        break
                except Exception:
                    pass
        atr_abs = (price * (atr_pct / 100.0)) if (isinstance(price,(int,float)) and isinstance(atr_pct,(int,float))) else None
        # Stop at SMA50 minus 1 ATR; targets at +1.5R and +2.5R from current price
        stop_suggest = None
        if atr_abs is not None and isinstance(sma50,(int,float)) and isinstance(price,(int,float)):
            stop_suggest = max(0.0, min(price, float(sma50) - 1.0 * atr_abs))
        targets: list[float] = []
        if atr_abs is not None and isinstance(price,(int,float)):
            targets = [round(price + 1.5 * atr_abs, 2), round(price + 2.5 * atr_abs, 2)]
        if stop_suggest is not None or targets:
            exit_levels = {
                "stop_suggest": None if stop_suggest is None else round(float(stop_suggest), 2),
                "targets": targets,
                "rationale": "Stop≈SMA50−1·ATR; Targets≈+1.5R/+2.5R from last"
            }
        # Entry readiness and exit risk scores (0-100)
        trend_ok = bool(trend.get("close_gt_sma200") and trend.get("sma50_gt_sma200"))
        vol_ok = bool(volc.get("rising20") and volc.get("price_gt_ma20"))
        above_sma50 = isinstance(price,(int,float)) and isinstance(sma50,(int,float)) and price >= sma50
        entry_score = 0
        entry_score += 30 if trend_ok else 0
        entry_score += 40 if in_zone else 0
        entry_score += 15 if vol_ok else 0
        entry_score += 15 if above_sma50 else 0
        entry_score = max(0, min(100, entry_score))
        exit_risk = 0
        if isinstance(price,(int,float)) and isinstance(sma50,(int,float)) and price < sma50:
            exit_risk += 50
        if isinstance(price,(int,float)) and isinstance(sma200,(int,float)) and price < sma200:
            exit_risk += 80
        if not vol_ok:
            exit_risk += 10
        if in_zone:
            exit_risk -= 10
        exit_risk = max(0, min(100, exit_risk))
        dist_to_stop_pct = None
        dist_to_t1_pct = None
        if isinstance(price,(int,float)) and isinstance(stop_suggest,(int,float)) and price > 0:
            dist_to_stop_pct = (price - float(stop_suggest)) / price * 100.0
        if isinstance(price,(int,float)) and targets and price > 0:
            dist_to_t1_pct = (targets[0] - price) / price * 100.0
        position_health = {
            "entry_readiness": entry_score,
            "exit_risk": exit_risk,
            "in_buy_zone": in_zone,
            "dist_to_stop_pct": None if dist_to_stop_pct is None else round(dist_to_stop_pct, 2),
            "dist_to_t1_pct": None if dist_to_t1_pct is None else round(dist_to_t1_pct, 2),
        }
    except Exception:
        pass


    # Weighted total (configurable via config/weights.json)
    _w = {"fundamentals": 0.40, "technicals": 0.35, "sentiment": 0.25}
    try:
        _w_conf = json.loads((ROOT / "config" / "weights.json").read_text(encoding="utf-8"))
        for k in ("fundamentals","technicals","sentiment"):
            if k in _w_conf:
                _w[k] = float(_w_conf[k])
    except Exception:
        pass
    _sum = sum([_w.get("fundamentals",0.0), _w.get("technicals",0.0), _w.get("sentiment",0.0)])
    if _sum <= 0:
        _sum = 1.0
    wf = _w.get("fundamentals",0.0) / _sum
    wt = _w.get("technicals",0.0) / _sum
    ws = _w.get("sentiment",0.0) / _sum
    total = round(wf * fund_pts + wt * tech_pts + ws * sent_pts)



    payload = {

        "ticker": label.upper().strip(),

        "score": int(total),

        "fund_points": int(fund_pts),

        "tech_points": int(tech_pts),

        "sent_points": int(sent_pts),

        "fundamentals": f,

        "technicals": {**tech_meta},

        "sentiment": {**sent_meta},
        "buy_zones": zones,
        "exit_levels": exit_levels,
        "position_health": position_health,

        "price": tech.get("price"),

        "sma50": tech.get("sma50"),

        "sma200": tech.get("sma200"),

        "updated_at": utcnow_iso(),

        "flags": flags,

    }

    return payload, flags







def _sma(arr: list[float], window: int) -> list[float]:
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for x in arr:
        q.append(x); s += x
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out



def _backtest_buyzones_from_series(series: dict, conf: dict | None = None) -> dict:
    # Inputs
    dates = series.get("dates") or []
    closes_raw = series.get("close") or []
    sma50_raw = series.get("sma50") or []
    sma200_raw = series.get("sma200") or []
    atr_pct_raw = series.get("atr_pct") or []
    # Cast and clean
    closes: list[float] = [float(c) for c in closes_raw if c is not None]
    # Align lengths by index
    n = min(len(dates), len(closes_raw), len(sma50_raw), len(sma200_raw), len(atr_pct_raw))
    if n < 60:
        return {"trades": 0, "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0}
    closes = [float(closes_raw[i]) if closes_raw[i] is not None else None for i in range(n)]  # type: ignore
    sma50 = [float(sma50_raw[i]) if sma50_raw[i] is not None else None for i in range(n)]  # type: ignore
    sma200 = [float(sma200_raw[i]) if sma200_raw[i] is not None else None for i in range(n)]  # type: ignore
    atrp = [float(atr_pct_raw[i]) if atr_pct_raw[i] is not None else None for i in range(n)]  # type: ignore
    # SMA20 from closes
    closes_num: list[float] = [c for c in closes if isinstance(c, (int, float))]
    if not closes_num:
        return {"trades": 0, "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0}
    sma20 = _sma([float(c) if c is not None else closes_num[-1] for c in closes], 20)
    # Config
    k50 = 1.0; k20 = 0.8; k_retest_lo = 1.0; k_retest_hi = 0.2
    if conf:
        k50 = float(conf.get("sma50_k_atr", k50))
        k20 = float(conf.get("sma20_k_atr", k20))
        k_retest_lo = float(conf.get("retest_k_lo", k_retest_lo))
        k_retest_hi = float(conf.get("retest_k_hi", k_retest_hi))
    # Walk and simulate
    horizon = 10
    rets: list[float] = []
    for i in range(50, n - horizon - 1):
        c = closes[i]
        if c is None: continue
        s50 = sma50[i]; s200 = sma200[i]; ap = atrp[i]
        if s50 is None or s200 is None or ap is None: continue
        uptrend = (c > s200) and (s50 > s200)
        if not uptrend: continue
        atr_abs = c * (ap / 100.0)
        if not atr_abs or atr_abs <= 0: continue
        # Zones for this bar
        zones: list[tuple[float, float]] = []
        zones.append((max(0.0, s50 - k50 * atr_abs), s50))
        s20 = sma20[i]
        if s20:
            zones.append((max(0.0, s20 - k20 * atr_abs), s20))
        # Retest based on prior high (simple heuristic)
        prior = [cl for cl in closes[max(0, i-60):i-5] if isinstance(cl, (int, float))]
        if prior:
            recent_high = max(prior)
            if c > recent_high:
                pl = max(0.0, recent_high - k_retest_lo * atr_abs)
                ph = max(0.0, recent_high - k_retest_hi * atr_abs)
                zones.append((pl, ph))
        # Look forward for touch and outcome
        for (lo, hi) in zones:
            entry_idx = None
            entry = None
            for j in range(i + 1, min(i + horizon + 1, n)):
                cj = closes[j]
                if cj is None: continue
                if lo <= cj <= hi:
                    entry = cj
                    entry_idx = j
                    break
            if entry is None or entry_idx is None: continue
            exit_idx = min(entry_idx + horizon, n - 1)
            exit_px = closes[exit_idx]
            if exit_px is None: continue
            ret = (exit_px - entry) / entry
            rets.append(ret)
    if not rets:
        return {"trades": 0, "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0}
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    hit = (len(wins) / len(rets)) if rets else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = hit * avg_win + (1 - hit) * avg_loss
    return {"trades": len(rets), "hit_rate": hit, "avg_win": avg_win, "avg_loss": avg_loss, "expectancy": expectancy}


def run_backtest(tickers: list[str]) -> int:
    print("Running mini-backtest across", len(tickers), "tickers...")
    try:
        conf = json.loads((ROOT / "config" / "buyzones.json").read_text(encoding="utf-8"))
    except Exception:
        conf = {}
    total = {"trades": 0, "hit_rate_sum": 0.0, "avg_win_sum": 0.0, "avg_loss_sum": 0.0, "expectancy_sum": 0.0, "n": 0}
    for label in tickers:
        try:
            tech, _ = run_with_retry(lambda: compute_indicators(label), attempts=2, base_delay=3.0, label=f"indicators:{label}")
            series = (tech.get("series") or {}) if isinstance(tech, dict) else {}
            if not series:
                print(f"{label:>8}: no series")
                continue
            res = _backtest_buyzones_from_series(series, conf)
            if res["trades"] == 0:
                print(f"{label:>8}: trades=0")
                continue
            total["trades"] += res["trades"]
            total["hit_rate_sum"] += res["hit_rate"]
            total["avg_win_sum"] += res["avg_win"]
            total["avg_loss_sum"] += res["avg_loss"]
            total["expectancy_sum"] += res["expectancy"]
            total["n"] += 1
            print(f"{label:>8}: trades={res['trades']:3d} hit={res['hit_rate']*100:5.1f}% avgW={res['avg_win']*100:5.1f}% avgL={res['avg_loss']*100:5.1f}% exp={res['expectancy']*100:5.1f}%")
        except Exception as e:
            print(f"{label:>8}: ERROR {e.__class__.__name__}")
            continue
    if total["n"]:
        agg = {
            "trades": total["trades"],
            "hit_rate": total["hit_rate_sum"] / total["n"],
            "avg_win": total["avg_win_sum"] / total["n"],
            "avg_loss": total["avg_loss_sum"] / total["n"],
            "expectancy": total["expectancy_sum"] / total["n"],
        }
        print("\nSummary:")
        print(f" tickers={total['n']} trades={agg['trades']} hit={agg['hit_rate']*100:5.1f}% avgW={agg['avg_win']*100:5.1f}% avgL={agg['avg_loss']*100:5.1f}% exp={agg['expectancy']*100:5.1f}%")
    else:
        print("No trades across universe.")
    return 0

def main() -> int:

    DATA_DIR.mkdir(parents=True, exist_ok=True)



    tickers = read_tickers(TICKERS_FILE)

    # Mini-backtest mode
    args = set(sys.argv[1:])
    if "--backtest" in args or "-B" in args:
        if not tickers:
            print(f"No tickers found in {TICKERS_FILE}.")
            return 0
        return run_backtest(tickers)

    if not tickers:
        print(f"No tickers found in {TICKERS_FILE}. Create the file with one ticker per line.")
        return 0



    ok: list[str] = []

    for label in tickers:

        file_key = label.replace(" ", "_")

        path = DATA_DIR / f"{file_key}.json"

        try:

            payload, flags = process_ticker(label)

            if "no_price_data" in flags:

                print(f"WARN: no price data for {label}; writing minimal artifact")

            write_json(path, payload)

            ok.append(label)

        except Exception as e:

            print(f"WARN: failed {label}: {e}")

            if use_cached_payload(path):

                ok.append(label)

                continue

            try:

                payload = stub_stock_json(label)

                payload.setdefault("flags", []).append(f"build_fail:{e.__class__.__name__}")

                payload["updated_at"] = utcnow_iso()

                write_json(path, payload)

                ok.append(label)

                print(f"INFO: wrote stub for {label}")

            except Exception as inner:

                print(f"ERROR: unable to write stub for {label}: {inner}")

            continue



    meta = {"generated_at": utcnow_iso(), "tickers": ok}

    write_json(DATA_DIR / "meta.json", meta)

    print(f"Built {len(ok)} tickers to {DATA_DIR} and meta.json")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())



