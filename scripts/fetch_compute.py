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

This script is intentionally robust and idempotent: it skips bad tickers,
creates the /data directory if needed, and continues on errors.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import certifi

# Ensure TLS CA bundle for curl_cffi/requests on Windows/OneDrive paths
# Copy certifi CA to an ASCII-only path to avoid libcurl issues with non-ASCII user profiles
import os as _os_for_ca

def _ensure_ascii_ca_bundle() -> None:
    try:
        _src = certifi.where()
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
from typing import Optional, Tuple

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


def _get_info_safe(ticker: "yf.Ticker") -> dict:
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
    if yf is None:
        raise RuntimeError("yfinance not installed; run: python -m pip install yfinance pandas feedparser vaderSentiment")
    if pd is None:
        raise RuntimeError("pandas not installed; run: python -m pip install pandas")

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
                    vals.append(float(x) if x is not None and not pd.isna(x) else None)
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

    return {
        "price": price,
        "sma50": sma50_last,
        "sma200": sma200_last,
        "rsi": rsi_last,
        "atr_pct": atrpct_last,
        "vol20_rising": vol20_rising,
        "price_gt_ma20": price_gt_ma20,
        "series": series,
    }, flags


def fetch_news_sentiment(label: str, symbol: str) -> dict:
    if feedparser is None or SentimentIntensityAnalyzer is None:
        raise RuntimeError("feedparser/vaderSentiment not installed; run: pip install feedparser vaderSentiment")

    # Query: prefer symbol, fallback to label.
    # Build a safe, encoded Google News query
    from urllib.parse import urlencode
    q = f"{symbol} stock OR {label} stock"
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = "https://news.google.com/rss/search?" + urlencode(params)
    d = feedparser.parse(url)

    from time import mktime
    import datetime as dt

    now = dt.datetime.utcnow()
    d7 = now - dt.timedelta(days=7)
    d30 = now - dt.timedelta(days=30)

    titles_7: list[str] = []
    titles_30: list[str] = []

    for e in d.entries or []:
        title = (e.title or "").strip()
        if not title:
            continue
        try:
            published = e.get("published_parsed")
            if not published:
                continue
            pdt = dt.datetime.fromtimestamp(mktime(published))
        except Exception:
            continue
        if pdt >= d30:
            titles_30.append(title)
        if pdt >= d7:
            titles_7.append(title)

    analyzer = SentimentIntensityAnalyzer()
    def mean_comp(titles: list[str]) -> Optional[float]:
        if not titles:
            return None
        vals = [analyzer.polarity_scores(t)["compound"] for t in titles]
        return sum(vals) / len(vals) if vals else None

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
        tech, tflags = compute_indicators(symbol)
        flags.extend(tflags)
    except Exception as e:
        flags.append(f"indicators_fail:{e.__class__.__name__}")
        tech = {"price": None, "sma50": None, "sma200": None, "rsi": None, "atr_pct": None,
                "vol20_rising": False, "price_gt_ma20": False,
                "series": {"dates": [], "close": [], "sma50": [], "sma200": [], "rsi": [], "atr_pct": []}
               }

    # Sentiment
    try:
        sent = fetch_news_sentiment(label=label, symbol=symbol)
    except Exception as e:
        flags.append(f"sent_fail:{e.__class__.__name__}")
        sent = {"points": 0, "mean7": None, "count7": 0, "count30": 0, "flow": "low", "signal_terms": False}

    # Fundamentals
    try:
        f = fetch_fundamentals(symbol)
        fund_pts, fund_meta, fflags = score_fundamentals(f)
        flags.extend(fflags)
    except Exception as e:
        flags.append(f"fund_fail:{e.__class__.__name__}")
        f = {}
        fund_pts, fund_meta = 0, {"available": 0, "max": 40}

    # Technicals + Sentiment scores
    tech_pts, tech_meta = score_technicals(tech)
    sent_pts, sent_meta = score_sentiment(sent)

    # Weighted total
    total = round(0.40 * fund_pts + 0.35 * tech_pts + 0.25 * sent_pts)

    payload = {
        "ticker": label.upper().strip(),
        "score": int(total),
        "fund_points": int(fund_pts),
        "tech_points": int(tech_pts),
        "sent_points": int(sent_pts),
        "fundamentals": f,
        "technicals": {**tech_meta},
        "sentiment": {**sent_meta},
        "price": tech.get("price"),
        "sma50": tech.get("sma50"),
        "sma200": tech.get("sma200"),
        "updated_at": utcnow_iso(),
        "flags": flags,
    }
    return payload, flags


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers(TICKERS_FILE)
    if not tickers:
        print(f"No tickers found in {TICKERS_FILE}. Create the file with one ticker per line.")
        return 0

    ok: list[str] = []
    for label in tickers:
        try:
            file_key = label.replace(" ", "_")
            path = DATA_DIR / f"{file_key}.json"
            payload, flags = process_ticker(label)
            if "no_price_data" in flags:
                print(f"WARN: no price data for {label}; writing minimal artifact")
            write_json(path, payload)
            ok.append(label)
        except Exception as e:
            print(f"WARN: failed {label}: {e}")
            continue

    meta = {"generated_at": utcnow_iso(), "tickers": ok}
    write_json(DATA_DIR / "meta.json", meta)
    print(f"Built {len(ok)} tickers to {DATA_DIR} and meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

