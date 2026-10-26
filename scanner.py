"""
UK Market Scanner
-----------------
Scans the FTSE 350 for stocks doing something statistically unusual today
(big moves vs their own normal, heavy volume, gaps, RSI extremes, breakouts)
and writes a mobile-friendly dashboard to site/index.html.

Data: Yahoo Finance via yfinance (free, ~15 min delayed, no API key).
Universe: FTSE 100 + FTSE 250 constituents from Wikipedia.
"""

import json
import math
import os
import sys
from datetime import datetime, time
from html import escape
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings you might want to tweak
# ----------------------------------------------------------------------------
TOP_N = 15                        # how many stocks to show as cards
MIN_SCORE = 3.0                   # below this a stock isn't "unusual" enough to show
MIN_VALUE_TRADED_GBP = 250_000    # skip illiquid names (median daily £ traded, 20 days)
EXCLUDE_SECTORS = {"investment trusts"}   # trusts mostly track their holdings
UNIVERSE_CACHE = "universe.csv"   # fallback list if Wikipedia can't be reached
OUT_DIR = "site"
HISTORY_FILE = "history.csv"      # one row per stock per day: the forward-test log (kept in the repo)
HORIZONS = (5, 10, 20)            # trading days we look ahead when scoring past signals
COST_RT = 0.010                   # assumed round-trip cost on a buy: 0.5% stamp duty + ~0.5% spread
BACKFILL_DAYS = 60                # on the first run, replay this many past days as an indicative check
MIN_N = 30                        # fewer matured signals than this = too early to conclude
# --- "your rule" simulator (top 2 by score, buy ~09:00, +3% take profit, -3% stop) ---
MARKET_TZ = "Europe/London"
ENTRY_MIN = 9 * 60                # minutes after midnight when you buy (09:00)
RULE_K, RULE_TP, RULE_SL = 2, 0.03, 0.03
STOP_SLIP = 0.003                 # extra slippage assumed when a stop is hit
N_RANDOM = 40                     # random-pick runs used as the control
START_CASH = 240

LONDON = ZoneInfo("Europe/London")
SESSION_OPEN = time(8, 0)
SESSION_CLOSE = time(16, 30)
DATA_DELAY_MIN = 15               # Yahoo's UK delay

WIKI_PAGES = [
    "https://en.wikipedia.org/wiki/FTSE_100_Index",
    "https://en.wikipedia.org/wiki/FTSE_250_Index",
]


# ----------------------------------------------------------------------------
# 1. Universe
# ----------------------------------------------------------------------------
def to_yahoo(tidm: str) -> str:
    """LSE code -> Yahoo ticker. 'BT.A' -> 'BT-A.L', 'BP.' -> 'BP.L'."""
    t = str(tidm).strip().upper().rstrip(".")
    return t.replace(".", "-") + ".L"


def fetch_universe() -> pd.DataFrame:
    import requests

    frames = []
    headers = {"User-Agent": "uk-scanner/1.0 (personal stock screener)"}
    for url in WIKI_PAGES:
        html = requests.get(url, headers=headers, timeout=30).text
        for t in pd.read_html(StringIO(html)):
            cols = {str(c).lower(): c for c in t.columns}
            tick = next((cols[c] for c in cols if "ticker" in c or "epic" in c), None)
            comp = next((cols[c] for c in cols if "company" in c), None)
            sect = next((cols[c] for c in cols if "sector" in c or "industry" in c), None)
            if tick is not None and comp is not None and len(t) >= 50:
                df = pd.DataFrame({
                    "tidm": t[tick].astype(str).str.strip(),
                    "name": t[comp].astype(str).str.strip(),
                    "sector": t[sect].astype(str).str.strip() if sect is not None else "Other",
                })
                frames.append(df)
                break
    uni = pd.concat(frames, ignore_index=True)
    uni = uni[uni["tidm"].str.len().between(1, 6)]
    uni["sector"] = uni["sector"].str.strip().str.capitalize()
    uni["symbol"] = uni["tidm"].map(to_yahoo)
    uni = uni.drop_duplicates("symbol")
    if len(uni) < 200:
        raise RuntimeError(f"Only found {len(uni)} stocks on Wikipedia")
    return uni


def get_universe() -> pd.DataFrame:
    try:
        uni = fetch_universe()
        uni.to_csv(UNIVERSE_CACHE, index=False)
        print(f"Universe: {len(uni)} stocks from Wikipedia")
    except Exception as e:
        if not os.path.exists(UNIVERSE_CACHE):
            raise
        print(f"Wikipedia failed ({e}); using cached universe")
        uni = pd.read_csv(UNIVERSE_CACHE)
    return uni[~uni["sector"].str.lower().isin(EXCLUDE_SECTORS)].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Prices
# ----------------------------------------------------------------------------
def download_prices(symbols):
    import yfinance as yf

    data = yf.download(
        symbols, period="400d", interval="1d", group_by="ticker",
        auto_adjust=True, threads=True, progress=False,
    )
    out = {}
    for s in symbols:
        try:
            df = data[s].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(df) >= 200:
            out[s] = df
    print(f"Prices: {len(out)}/{len(symbols)} stocks downloaded")
    return out


# ----------------------------------------------------------------------------
# 3. Session timing (so we compare today's half-day volume fairly)
# ----------------------------------------------------------------------------
def session_state(now: datetime, last_bar_date) -> dict:
    """Share of a normal day's volume we'd expect to have seen by now."""
    today = now.date()
    t = now.time()
    if now.weekday() >= 5 or t >= time(16, 45):
        return {"frac": 1.0, "status": "Closed", "partial": False}
    if t < SESSION_OPEN:
        return {"frac": 1.0, "status": "Pre-market", "partial": False}
    if last_bar_date != today:   # holiday, or Yahoo hasn't printed today's bar yet
        return {"frac": 1.0, "status": "Open (no data yet today)", "partial": False}
    open_dt = datetime.combine(today, SESSION_OPEN, LONDON)
    total = (datetime.combine(today, SESSION_CLOSE, LONDON) - open_dt).total_seconds() / 60
    elapsed = (now - open_dt).total_seconds() / 60 - DATA_DELAY_MIN
    p = min(max(elapsed / total, 0.05), 1.0)   # floor avoids silly volume ratios just after the open
    # ~22% of UK volume prints in the 16:35 closing auction; mornings are busier
    frac = 0.78 * p ** 0.85 if t < SESSION_CLOSE else 0.80
    return {"frac": frac, "status": "Open" if t < SESSION_CLOSE else "Closing auction", "partial": True}


# ----------------------------------------------------------------------------
# 4. Scoring
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def cap(x, hi):
    return float(max(-hi, min(hi, x)))


def find_gap_setup(o, h, l, c, atr):
    """Looks for the unfilled-gap-up pullback setup (the one rule from my backtests with a positive result):
    in the last 10 bars a day whose LOW was 2-25% above the previous day's HIGH (a real gap, no overlap).
    Entry idea = middle of that gap, stop = previous high minus 0.25 x ATR, target = 2x the risk above entry.
    Returns None, or a dict with state 'waiting' (price still above the entry), 'buyzone' (price has pulled back
    to the entry zone and is still above the stop) or 'bounced' (it already pulled back and moved away)."""
    n = len(c)
    for g in range(n - 1, max(n - 11, 20) - 1, -1):          # most recent gap first
        a = atr.iloc[g]
        if not np.isfinite(a) or not (l.iloc[g] > h.iloc[g - 1]):
            continue
        size = l.iloc[g] / h.iloc[g - 1] - 1
        if not (0.02 <= size <= 0.25):
            continue
        lo, hi = float(h.iloc[g - 1]), float(l.iloc[g])
        mid, stop = (lo + hi) / 2, lo - 0.25 * float(a)
        after = l.iloc[g + 1:]
        if len(after) and after.min() <= stop:
            continue                                          # stop level traded: setup failed
        touched = bool(len(after) and after.min() <= mid)
        close = float(c.iloc[-1])
        risk = mid - stop
        if close <= mid * 1.01:
            state = "buyzone"
        elif touched:
            state = "bounced"
        else:
            state = "waiting"
        return {"state": state, "gap_pct": size, "age": n - 1 - g, "entry": mid, "stop": stop,
                "target": mid + 2 * risk, "risk_pct": risk / mid}
    return None


def gap_down_active(h, l, c):
    """An unfilled gap DOWN in the last 10 bars (high below the previous low) that price has not recovered."""
    n = len(c)
    for g in range(n - 1, max(n - 11, 20) - 1, -1):
        if h.iloc[g] < l.iloc[g - 1] and (l.iloc[g - 1] / h.iloc[g] - 1) >= 0.02:
            return bool(c.iloc[-1] < l.iloc[g - 1])
    return False


def classify(m: dict):
    """BUY / MONITOR / AVOID. A checklist, not a prediction (see the 'How to read' box)."""
    gs = m.get("gap_setup")
    down = m["price"] < m["sma200"] and m["sma50"] < m["sma200"]
    up = m["price"] > m["sma200"] and m["sma50"] > m["sma200"]
    if gs and gs["state"] == "buyzone":
        if gs["risk_pct"] < 0.02:
            return "MONITOR", "Gap pullback, but the stop is under 2% away: UK stamp duty (0.5%) would eat too much of it"
        return "BUY", f"Pulled back into the entry zone of a {gs['gap_pct'] * 100:.1f}% gap-up from {gs['age']} day(s) ago, stop still intact"
    if m["new_low"] or (m["break_down"] and m["rvol"] >= 1.3):
        return "AVOID", "Breaking down to new lows on volume"
    if m.get("gap_down"):
        return "AVOID", "Unfilled gap down, not recovered"
    if m["death"]:
        return "AVOID", "Death cross (50-day fell under the 200-day)"
    if down:
        return "AVOID", "Downtrend: below the 200-day and the 50-day is under the 200-day"
    if m["rsi"] >= 85:
        return "AVOID", f"Blow-off: RSI {m['rsi']:.0f}, chasing risk"
    if gs and gs["state"] == "waiting":
        return "MONITOR", f"Fresh {gs['gap_pct'] * 100:.1f}% gap-up: wait for a pullback to the entry level"
    if gs and gs["state"] == "bounced":
        return "MONITOR", "Gap-up already pulled back and bounced: the entry has gone, don't chase"
    if up:
        return "MONITOR", "Uptrend, but no tested entry trigger today"
    return "MONITOR", "Mixed signals, no clear trend"


def score_stock(df: pd.DataFrame, vol_frac: float):
    df = df.tail(260)
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    r = c.pct_change()
    sd = r.iloc[-61:-1].std()
    if not sd or np.isnan(sd) or sd == 0:
        return None

    value_traded = (c * v / 100).iloc[-21:-1].median()   # prices are in pence
    if np.isnan(value_traded) or value_traded < MIN_VALUE_TRADED_GBP:
        return None

    ret = r.iloc[-1]
    ret_z = ret / sd
    gap = o.iloc[-1] / c.iloc[-2] - 1
    gap_z = gap / sd
    mom5 = c.iloc[-1] / c.iloc[-6] - 1
    mom5_z = mom5 / (sd * math.sqrt(5))

    avg_vol = v.iloc[-21:-1].mean()
    rvol = float(v.iloc[-1] / (avg_vol * vol_frac)) if avg_vol > 0 else 1.0

    rsi_series = rsi(c)
    rsi_now = float(rsi_series.iloc[-1])

    hi52 = h.iloc[-253:-1].max()
    lo52 = l.iloc[-253:-1].min()
    hi20 = h.iloc[-21:-1].max()
    lo20 = l.iloc[-21:-1].min()

    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    diff = (sma50 - sma200).iloc[-2:]
    golden = diff.iloc[0] <= 0 < diff.iloc[1]
    death = diff.iloc[0] >= 0 > diff.iloc[1]

    bw = (4 * c.rolling(20).std() / c.rolling(20).mean()).dropna()
    bw_pct = bw.rank(pct=True)
    squeeze_recent = bw_pct.iloc[-6:-1].min() <= 0.05 if len(bw_pct) > 60 else False
    squeeze_now = bw_pct.iloc[-1] <= 0.05 if len(bw_pct) > 60 else False

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    gap_setup = find_gap_setup(o, h, l, c, atr)
    s50, s200 = sma50.iloc[-1], sma200.iloc[-1]
    if not (np.isfinite(s50) and np.isfinite(s200)):
        return None

    return {
        "sma50": float(s50), "sma200": float(s200),
        "gap_setup": gap_setup, "gap_down": gap_down_active(h, l, c),
        "price": float(c.iloc[-1]),
        "ret": float(ret), "ret_z": float(ret_z),
        "gap": float(gap), "gap_z": float(gap_z),
        "mom5": float(mom5), "mom5_z": float(mom5_z),
        "rvol": rvol, "rsi": rsi_now,
        "new_high": bool(c.iloc[-1] >= hi52), "new_low": bool(c.iloc[-1] <= lo52),
        "break_up": bool(c.iloc[-1] > hi20), "break_down": bool(c.iloc[-1] < lo20),
        "golden": bool(golden), "death": bool(death),
        "squeeze_release": bool(squeeze_recent and abs(ret_z) >= 1.5),
        "squeeze_now": bool(squeeze_now),
        "spark": [round(float(x), 2) for x in c.iloc[-60:]],
    }


def finalise(m: dict) -> dict:
    """Turn raw metrics into a composite score + plain-English reasons."""
    rel = m["rel_z"]
    s = 1.0 * abs(cap(rel, 6))
    s += 1.2 * min(max(math.log2(max(m["rvol"], 1e-6)), 0), 4)
    s += 0.6 * abs(cap(m["gap_z"], 5))
    s += 0.5 * abs(cap(m["mom5_z"], 5))
    s += min(max(abs(m["rsi"] - 50) - 20, 0) / 5, 3)
    s += 1.5 * (m["new_high"] or m["new_low"])
    s += 0.75 * ((m["break_up"] or m["break_down"]) and not (m["new_high"] or m["new_low"]))
    s += 1.0 * (m["golden"] or m["death"])
    s += 1.0 * m["squeeze_release"]

    reasons = []
    if abs(m["ret_z"]) >= 2:
        reasons.append(f"{m['ret_z']:+.1f}σ move today")
    if abs(rel) >= 2 and abs(rel - m["ret_z"]) >= 0.5:
        reasons.append(f"{rel:+.1f}σ vs sector")
    if m["rvol"] >= 2:
        reasons.append(f"{m['rvol']:.1f}× normal volume")
    if abs(m["gap_z"]) >= 2:
        reasons.append(f"Gapped {m['gap'] * 100:+.1f}%")
    if abs(m["mom5_z"]) >= 2:
        reasons.append(f"5-day {m['mom5'] * 100:+.1f}% ({m['mom5_z']:+.1f}σ)")
    if m["rsi"] >= 75:
        reasons.append(f"RSI {m['rsi']:.0f} (overbought)")
    elif m["rsi"] <= 25:
        reasons.append(f"RSI {m['rsi']:.0f} (oversold)")
    if m["new_high"]:
        reasons.append("52-week high")
    elif m["new_low"]:
        reasons.append("52-week low")
    elif m["break_up"]:
        reasons.append("20-day breakout")
    elif m["break_down"]:
        reasons.append("20-day breakdown")
    if m["golden"]:
        reasons.append("Golden cross (50/200)")
    if m["death"]:
        reasons.append("Death cross (50/200)")
    if m["squeeze_release"]:
        reasons.append("Big move after a squeeze")
    elif m["squeeze_now"]:
        reasons.append("Volatility squeeze (coiling)")

    m["score"] = round(s, 2)
    m["reasons"] = reasons
    m["signal"], m["signal_why"] = classify(m)
    m["direction"] = "up" if (m["ret"] if abs(m["ret_z"]) >= 0.5 else m["mom5"]) >= 0 else "down"
    return m


# ----------------------------------------------------------------------------
# 5b. Track record: logs every day's signals and checks what happened next
# ----------------------------------------------------------------------------
HIST_COLS = ["date", "symbol", "tidm", "signal", "score", "rank", "shown", "source"]


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        try:
            h = pd.read_csv(HISTORY_FILE, dtype={"date": str})
            return h[HIST_COLS]
        except Exception as e:
            print(f"  history unreadable ({e}), starting fresh")
    return pd.DataFrame(columns=HIST_COLS)


def save_history(h: pd.DataFrame):
    h = h.copy()
    h["score"] = h["score"].astype(float).round(2)
    h["rank"] = h["rank"].astype(int)
    h["shown"] = h["shown"].astype(int)
    h = h.sort_values(["date", "rank", "symbol"]).reset_index(drop=True)
    h.to_csv(HISTORY_FILE, index=False, float_format="%.2f")


def snapshot(results, shown_syms, date_str, source) -> pd.DataFrame:
    rows = [dict(date=date_str, symbol=m["symbol"], tidm=m["tidm"], signal=m["signal"], score=round(m["score"], 2),
                 rank=i + 1, shown=int(m["symbol"] in shown_syms), source=source)
            for i, m in enumerate(results)]
    return pd.DataFrame(rows, columns=HIST_COLS)


def _index_dates(df):
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def backfill(prices, uni, have_dates, last_complete):
    """Replay the labels on past days using only the data available on each day."""
    ref = max(prices.values(), key=len)
    dates = [d for d in _index_dates(ref) if d.date() <= last_complete]
    dates = [d for d in dates[-BACKFILL_DAYS:] if d.strftime("%Y-%m-%d") not in have_dates]
    idx_cache = {s: _index_dates(df) for s, df in prices.items()}
    out = []
    for d in dates:
        cut = {}
        for s, df in prices.items():
            n = int(idx_cache[s].searchsorted(d, side="right"))
            if n >= 200 and idx_cache[s][n - 1] == d:
                cut[s] = df.iloc[:n]
        if len(cut) < 50:
            continue
        res, _ = score_all(cut, uni, 1.0)
        shown = {m["symbol"] for m in [m for m in res if m["score"] >= MIN_SCORE and m["reasons"]][:TOP_N]}
        out.append(snapshot(res, shown, d.strftime("%Y-%m-%d"), "replay"))
    print(f"Backfilled {len(out)} past days")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=HIST_COLS)


def attach_returns(h, prices):
    for H in HORIZONS:
        h[f"r{H}"] = np.nan
    for sym, g in h.groupby("symbol"):
        if sym not in prices:
            continue
        c = prices[sym]["Close"].values
        pos = _index_dates(prices[sym]).get_indexer(pd.to_datetime(g["date"]))
        for H in HORIZONS:
            p2 = pos + H
            ok = (pos >= 0) & (p2 < len(c))
            r = np.full(len(g), np.nan)
            r[ok] = c[p2[ok]] / c[pos[ok]] - 1
            h.loc[g.index, f"r{H}"] = r
    for H in HORIZONS:    # compare with the average stock on the same day, so a rising/falling market cancels out
        h[f"x{H}"] = h[f"r{H}"] - h.groupby("date")[f"r{H}"].transform("mean")
    return h


def tier(row):
    if row["shown"] and row["rank"] <= 5:
        return "Top 5 by score"
    return "Shown on page (6-15)" if row["shown"] else "Not shown"


def new_entries(h, key):
    """Count a stock once when it ENTERS a group, not every day it stays there."""
    h = h.sort_values(["symbol", "date"])
    prev = h.groupby("symbol")[key].shift(1)
    return h[(prev != h[key]) | prev.isna()]


def _fmt(x):
    return f"{x * 100:+.1f}%"


def stats_table(h, key, groups, costed, title):
    e = new_entries(h, key)
    rows = []
    for gname in groups:
        sub = e[e[key] == gname]
        cost = COST_RT if gname in costed else 0.0
        cells = ""
        for H in HORIZONS:
            d = sub[sub[f"r{H}"].notna()]
            n = len(d)
            if n == 0:
                cells += '<td class="num dim">–</td>'
                continue
            r, x = d[f"r{H}"] - cost, d[f"x{H}"] - cost
            hit = (r > 0).mean() * 100
            cls = "dim" if n < MIN_N else ("up" if r.mean() > 0 else "down")
            cells += (f'<td class="num {cls}"><b>{_fmt(r.mean())}</b>'
                      f'<div class="sub">vs avg {_fmt(x.mean())} · {hit:.0f}% win · n={n}</div></td>')
        rows.append(f"<tr><td>{escape(gname)}{'' if gname in costed else ' <span class=sub>(no costs)</span>'}</td>{cells}</tr>")
    head = "".join(f'<th class="num">{H} days</th>' for H in HORIZONS)
    return (f'<table class="trk"><thead><tr><th>{escape(title)}</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def verdict(h):
    e = new_entries(h, "signal")
    out = []
    for H in (10,):
        b = e[(e["signal"] == "BUY") & e[f"x{H}"].notna()]
        a = e[(e["signal"] == "AVOID") & e[f"x{H}"].notna()]
        if len(b) < MIN_N:
            out.append(f"Too early to judge: only {len(b)} BUY signals have had {H} days to play out (about {MIN_N} are needed).")
            continue
        xb = b[f"x{H}"] - COST_RT
        t = xb.mean() / (xb.std() / math.sqrt(len(xb))) if xb.std() > 0 else 0
        word = "ahead of" if xb.mean() > 0 else "behind"
        sure = "a gap that is probably real" if abs(t) >= 2 else "a gap small enough to be luck"
        out.append(f"BUY signals ended {abs(xb.mean()) * 100:.1f}% {word} the average stock after {H} days and after costs "
                   f"({len(b)} signals, {sure}).")
        if len(a) >= MIN_N:
            out.append(f"AVOID signals: {_fmt(a[f'x{H}'].mean())} vs the average stock.")
    return " ".join(out)


def track_html(h, prices, label):
    if h.empty:
        return ""
    h = attach_returns(h.copy(), prices)
    h["tier"] = h.apply(tier, axis=1)
    first, last = h["date"].min(), h["date"].max()
    parts = [f'<div class="meta">{label}: {h["date"].nunique()} day{"" if h["date"].nunique() == 1 else "s"} logged ({first} to {last}). '
             f'Each stock is counted once, on the day it first gets that label. Returns are close to close, '
             f'compared with the average stock scanned that day.</div>',
             f'<div class="meta"><b>{escape(verdict(h))}</b></div>',
             stats_table(h, "signal", ["BUY", "MONITOR", "AVOID"], {"BUY", "MONITOR"}, "By label"),
             stats_table(h, "tier", ["Top 5 by score", "Shown on page (6-15)", "Not shown"],
                         {"Top 5 by score", "Shown on page (6-15)", "Not shown"}, "By score rank")]
    return "".join(parts)


def download_hourly(symbols):
    """Hourly bars for the last ~6 months, used to replay a 09:00 entry with take-profit and stop."""
    import yfinance as yf
    data = yf.download(symbols, period="6mo", interval="1h", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    out = {}
    for s in symbols:
        try:
            df = data[s].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(df) > 20:
            out[s] = _hourly_pack(df)
    print(f"Hourly bars: {len(out)}/{len(symbols)} stocks")
    return out


def _hourly_pack(df):
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    idx = idx.tz_convert(MARKET_TZ)
    dates = idx.normalize().tz_localize(None)
    hm = (idx.hour * 60 + idx.minute).values
    b = dict(d=dates.values, hm=hm, o=df["Open"].values, h=df["High"].values,
             l=df["Low"].values, c=df["Close"].values)
    b["ent"] = {pd.Timestamp(d): i for i, (d, m) in enumerate(zip(dates, hm)) if m == ENTRY_MIN}
    return b


def _trade_path(b, i0):
    """Buy at the entry bar's open; walk forward until +TP or -SL (stop wins ties). Returns trade dict."""
    entry = b["o"][i0]
    tp, sl = entry * (1 + RULE_TP), entry * (1 - RULE_SL)
    n = len(b["o"])
    for j in range(i0, n):
        if b["l"][j] <= sl or (j > i0 and b["o"][j] <= sl):
            px = min(b["o"][j], sl) if j > i0 else sl
            return dict(j=j, kind="stop", ret=px * (1 - STOP_SLIP) / entry - 1)
        if b["h"][j] >= tp or (j > i0 and b["o"][j] >= tp):
            px = max(b["o"][j], tp) if j > i0 else tp
            return dict(j=j, kind="target", ret=px / entry - 1)
    return dict(j=n - 1, kind="open", ret=b["c"][n - 1] / entry - 1)


def run_rule(order_by_day, cal, hourly, k=RULE_K):
    open_pos, trades = [], []
    for nd in cal:
        key_in = (nd.toordinal(), ENTRY_MIN)
        open_pos = [p for p in open_pos if p["exit_key"] is None or p["exit_key"] > key_in]
        held = {p["sym"] for p in open_pos}
        for sym in order_by_day.get(nd, []):
            if len(open_pos) >= k:
                break
            b = hourly.get(sym)
            if sym in held or b is None or nd not in b["ent"]:
                continue
            i0 = b["ent"][nd]
            r = _trade_path(b, i0)
            ed = pd.Timestamp(b["d"][r["j"]])
            p = dict(sym=sym, entry_key=key_in, kind=r["kind"], gross=r["ret"], net=r["ret"] - COST_RT,
                     exit_key=None if r["kind"] == "open" else (ed.toordinal(), int(b["hm"][r["j"]]) + 60),
                     days=(ed - nd).days)
            open_pos.append(p)
            held.add(sym)
            trades.append(p)
    return trades


def summarise(trades, k=RULE_K):
    closed = [t for t in trades if t["kind"] != "open"]
    ev = [(t["entry_key"], 1, t) for t in trades] + [(t["exit_key"], 0, t) for t in closed]
    ev.sort(key=lambda e: (e[0], e[1]))
    E, peak, dd = float(START_CASH), float(START_CASH), 0.0
    for _, is_entry, t in ev:
        if is_entry:
            t["size"] = E / k
        else:
            E += t["size"] * t["net"]
            peak = max(peak, E)
            dd = min(dd, E / peak - 1)
    final = E + sum(t["size"] * t["net"] for t in trades if t["kind"] == "open")
    wins = sum(1 for t in closed if t["net"] > 0)
    return dict(n=len(closed), n_open=len(trades) - len(closed), win=wins / len(closed) if closed else float("nan"),
                avg=float(np.mean([t["net"] for t in closed])) if closed else float("nan"),
                days=float(np.mean([t["days"] for t in closed])) if closed else float("nan"),
                final=final, dd=dd)


def rule_html(h, hourly):
    base = h[h["source"].isin(["live", "replay"])]
    snap = base.sort_values(["date", "rank"]).groupby("date")["symbol"].apply(list).to_dict()
    am = h[h["source"] == "am"].sort_values(["date", "rank"]).groupby("date")["symbol"].apply(list).to_dict()
    cal_all = sorted({pd.Timestamp(d) for b in hourly.values() for d in b["ent"]})
    sdates = sorted(snap)
    if not cal_all or not sdates:
        return ""
    cal, prev_of = [], {}
    for nd in cal_all:
        ps = [d for d in sdates if pd.Timestamp(d) < nd]
        if ps:
            cal.append(nd)
            prev_of[nd] = ps[-1]
    if len(cal) < 3:
        return ""
    order = {}
    for nd in cal:
        prev = snap[prev_of[nd]]
        first = am.get(nd.strftime("%Y-%m-%d"), [])
        order[nd] = first + [s for s in prev if s not in first]
    mine = summarise(run_rule(order, cal, hourly))
    rng = np.random.default_rng(7)
    rand = []
    for _ in range(N_RANDOM):
        ro = {nd: list(rng.permutation(snap[prev_of[nd]])) for nd in cal}
        rand.append(summarise(run_rule(ro, cal, hourly)))
    ravg = np.array([r["avg"] for r in rand])
    pct = float(np.mean(ravg <= mine["avg"])) if mine["n"] else float("nan")
    rm = lambda f: float(np.nanmean([r[f] for r in rand]))
    if mine["n"] < MIN_N:
        say = (f"Too few finished trades ({mine['n']}) to say whether the ranking adds anything. "
               f"Compare the two rows once there are 30+ each.")
    elif pct >= 0.95:
        say = "Your picks beat 95%+ of random picks run through the same rule, so the ranking looks like it adds something."
    elif pct <= 0.05:
        say = "Your picks did worse than 95%+ of random picks, so the ranking is not helping."
    else:
        say = (f"Your picks beat {pct * 100:.0f}% of the random runs: inside the normal range, so the ranking is not clearly "
               f"adding anything and the +3%/-3% rule in this market may be doing the work.")
    def row(name, r, avg_n=None):
        n = f"{r['n']:.0f}" if avg_n is None else f"{avg_n:.0f}"
        return (f"<tr><td>{name}</td><td class='num'>{n}</td><td class='num'>{r['win'] * 100:.0f}%</td>"
                f"<td class='num'>{_fmt(r['avg'])}</td><td class='num'>{r['days']:.1f}</td>"
                f"<td class='num'><b>£{r['final']:.0f}</b><div class='sub'>worst dip {r['dd'] * 100:.0f}%</div></td></tr>")
    rr = dict(n=rm("n"), win=rm("win"), avg=rm("avg"), days=rm("days"), final=rm("final"), dd=rm("dd"))
    tbl = ("<table class='trk'><thead><tr><th></th><th class='num'>Done</th><th class='num'>Win</th>"
           "<th class='num'>Avg/trade</th><th class='num'>Days</th><th class='num'>£" + str(START_CASH) + " becomes</th></tr></thead><tbody>"
           + row("Top 2 by score", mine) + row(f"Random 2 (avg of {N_RANDOM})", rr) + "</tbody></table>")
    ds = f"{min(cal).date()} to {max(cal).date()}"
    return ("<h3>Your rule, replayed</h3><div class='meta'>Hold 2 stocks. Buy at 09:00 from the previous evening's ranking "
            "(or the morning list once the scanner has logged it), +3% take profit, -3% stop, "
            f"after 1% costs and extra slippage on stops. {ds}. Open positions are counted at their latest price.</div>"
            f"<div class='meta'><b>{escape(say)}</b></div>" + tbl +
            "<div class='meta'>The real 09:00 ranking also uses the first hour's moves, which can't be rebuilt for past days, "
            "so this is close but not exact.</div>")


def record_am(h, top, now, last_bar):
    """Log the morning top-10 once a day, at about 09:00-09:45 UK, so the 9am ranking can be tested later."""
    t = now.time()
    if now.weekday() >= 5 or not (time(9, 0) <= t <= time(9, 45)) or last_bar != now.date():
        return h
    d = now.strftime("%Y-%m-%d")
    if ((h["source"] == "am") & (h["date"] == d)).any():
        return h
    rows = [dict(date=d, symbol=m["symbol"], tidm=m["tidm"], signal=m["signal"], score=round(m["score"], 2),
                 rank=i + 1, shown=1, source="am") for i, m in enumerate(top[:10])]
    return pd.concat([h, pd.DataFrame(rows, columns=HIST_COLS)], ignore_index=True)


def track_section(prices, uni, results, top, last_bar, now):
    h = load_history()
    n_before = len(h)
    complete = last_bar < now.date() or (last_bar == now.date() and now.time() >= time(16, 45))
    last_complete = last_bar if complete else max(
        (d.date() for d in _index_dates(max(prices.values(), key=len)) if d.date() < last_bar), default=last_bar)
    if not (h["source"] == "replay").any():
        have = set(h["date"])
        h = pd.concat([h, backfill(prices, uni, have, last_complete)], ignore_index=True)
    if complete:
        d = last_bar.strftime("%Y-%m-%d")
        h = h[~((h["date"] == d) & (h["source"] == "live"))]
        h = pd.concat([h, snapshot(results, {m["symbol"] for m in top}, d, "live")], ignore_index=True)
    h = record_am(h, top, now, last_bar)
    if len(h) != n_before or complete:
        save_history(h)
    live, replay = h[h["source"] == "live"], h[h["source"] == "replay"]
    html = ""
    if live.empty:
        html += ('<div class="meta"><b>Live record:</b> starts after the next full trading day closes. '
                 'Every day the scanner logs each stock and its label, then checks what happened 5, 10 and 20 days later.</div>')
    else:
        html += "<h3>Live signals</h3>" + track_html(live, prices, "Live")
    if not replay.empty:
        html += ("<h3>Replay of recent weeks (indicative)</h3>"
                 "<div class='meta'>Same rules run on past days using only the data available on each day, "
                 "on today's FTSE 100/250 list (so stocks that dropped out are missing). Useful, not proof.</div>"
                 + track_html(replay, prices, "Replay"))
    try:
        hourly = download_hourly(sorted(set(h["symbol"])))
        html += rule_html(h, hourly)
    except Exception as e:
        import traceback
        traceback.print_exc()
        html += f'<div class="meta">Your-rule replay unavailable this run ({escape(str(e))}).</div>'
    return html


# ----------------------------------------------------------------------------
# 5. Dashboard
# ----------------------------------------------------------------------------
def sparkline(values, direction):
    w, h, pad = 120, 36, 2
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1
    pts = " ".join(
        f"{pad + i * (w - 2 * pad) / (len(values) - 1):.1f},{pad + (hi - val) * (h - 2 * pad) / rng:.1f}"
        for i, val in enumerate(values)
    )
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
            f'aria-label="60-day price, low {lo:g}p high {hi:g}p">'
            f'<polyline points="{pts}" class="spark-{direction}"/></svg>')


def fmt_price(p):
    return f"{p:,.0f}p" if p >= 100 else f"{p:,.2f}p"


BADGE_ICON = {"BUY": "✓", "MONITOR": "●", "AVOID": "✕"}


def badge(sig):
    return f'<span class="badge b-{sig.lower()}">{BADGE_ICON[sig]} {sig}</span>'


def levels_html(m):
    gs = m.get("gap_setup")
    if not gs or gs["state"] not in ("buyzone", "waiting"):
        return ""
    head = "Entry zone" if gs["state"] == "buyzone" else "Watch level (buy on a pullback to)"
    return (f'<div class="levels"><b>{head}:</b> {fmt_price(gs["entry"])} · '
            f'<b>Stop:</b> {fmt_price(gs["stop"])} ({gs["risk_pct"] * 100:.1f}% away) · '
            f'<b>Target:</b> {fmt_price(gs["target"])} (2× the risk)</div>')


def card(m):
    arrow = "▲" if m["direction"] == "up" else "▼"
    chips = "".join(f'<span class="chip">{escape(r)}</span>' for r in m["reasons"])
    tidm = escape(m["tidm"])
    return f"""
<article class="card">
  <div class="row1">
    <div>
      <div class="tick">{tidm} {badge(m['signal'])}</div>
      <div class="name">{escape(m['name'])}</div>
      <div class="sector">{escape(m['sector'])}</div>
    </div>
    <div class="right">
      <div class="price">{fmt_price(m['price'])}</div>
      <div class="chg {m['direction']}">{arrow} {m['ret'] * 100:+.2f}%</div>
      <div class="score" title="Unusualness score">Score {m['score']:.1f}</div>
    </div>
  </div>
  <div class="why"><b>{escape(m['signal'].title())}:</b> {escape(m['signal_why'])}</div>
  {levels_html(m)}
  {sparkline(m['spark'], m['direction'])}
  <div class="chips">{chips}</div>
  <div class="links">
    <a href="https://www.investegate.co.uk/company/{tidm.rstrip('.')}" target="_blank" rel="noopener">News (RNS)</a>
    <a href="https://uk.finance.yahoo.com/quote/{escape(m['symbol'])}" target="_blank" rel="noopener">Chart</a>
  </div>
</article>"""


def heat_tile(name, ret, n):
    # diverging blue (up) / red (down), grey midpoint; text always in ink colours
    mag = min(abs(ret) / 0.03, 1.0)
    side = "up" if ret >= 0 else "down"
    return (f'<div class="tile {side}" style="--mag:{mag:.2f}">'
            f'<span class="tname">{escape(name)}</span>'
            f'<span class="tval">{"▲" if ret >= 0 else "▼"} {ret * 100:+.1f}%</span>'
            f'<span class="tn">{n} stocks</span></div>')


def build_html(top, everything, sectors, meta, track=""):
    cards = "".join(card(m) for m in top) or '<p class="empty">Nothing unusual right now. Quiet market.</p>'
    buys = [m for m in everything if m["signal"] == "BUY"]
    buy_cards = "".join(card(m) for m in buys) or '<p class="empty">No stock is in a tested buy zone right now. That is normal: the setup is rare.</p>'
    counts = {k: sum(1 for m in everything if m["signal"] == k) for k in ("BUY", "MONITOR", "AVOID")}
    tiles = "".join(heat_tile(s["sector"], s["ret"], s["n"]) for s in sectors)
    order = {"BUY": 0, "MONITOR": 1, "AVOID": 2}
    rows = "".join(
        f"<tr data-sig='{m['signal']}'><td>{escape(m['tidm'])}<div class='sub'>{escape(m['name'])}</div></td>"
        f"<td>{badge(m['signal'])}<div class='sub'>{escape(m['signal_why'])}</div></td>"
        f"<td class='num {m['direction']}'>{m['ret'] * 100:+.1f}%</td>"
        f"<td class='num'>{m['rvol']:.1f}×</td><td class='num'>{m['rsi']:.0f}</td>"
        f"<td class='num'>{m['score']:.1f}</td></tr>"
        for m in sorted(everything, key=lambda m: (order[m["signal"]], -m["score"]))
    )
    return HTML_TEMPLATE.format(
        updated=escape(meta["updated"]), status=escape(meta["status"]),
        n=meta["n_scanned"], cards=cards, tiles=tiles, rows=rows, buy_cards=buy_cards,
        track=track, n_buy=counts["BUY"], n_mon=counts["MONITOR"], n_avoid=counts["AVOID"],
        partial_note=("Volume is compared with what's normal by this time of day."
                      if meta["partial"] else "Showing the last full session."),
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>UK Scanner</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --line: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --up: #2a78d6; --down: #d03b3b; --mid: #f0efec; --chip: #eeede8;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --line: #2c2c2a; --ring: rgba(255,255,255,0.10);
    --up: #3987e5; --down: #e66767; --mid: #383835; --chip: #262624;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --line: #2c2c2a; --ring: rgba(255,255,255,0.10);
  --up: #3987e5; --down: #e66767; --mid: #383835; --chip: #262624;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 760px; margin: 0 auto; padding: 16px; }}
header h1 {{ font-size: 22px; margin: 0 0 4px; }}
.meta {{ color: var(--ink2); font-size: 13px; }}
.meta b {{ color: var(--ink); }}
h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink2); margin: 24px 0 10px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 2px; }}
.tile {{ padding: 8px 10px; border-radius: 4px; display: flex; flex-direction: column;
  background: color-mix(in oklab, var(--mid), var(--side) calc(var(--mag) * 55%)); }}
.tile.up {{ --side: var(--up); }} .tile.down {{ --side: var(--down); }}
.tname {{ font-size: 12px; color: var(--ink); line-height: 1.25; }}
.tval {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.tn {{ font-size: 11px; color: var(--ink2); }}
.cards {{ display: grid; gap: 10px; }}
.card {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 12px 14px; }}
.row1 {{ display: flex; justify-content: space-between; gap: 12px; }}
.tick {{ font-weight: 700; font-size: 18px; }}
.name {{ font-size: 13px; color: var(--ink); }}
.sector {{ font-size: 12px; color: var(--muted); }}
.right {{ text-align: right; font-variant-numeric: tabular-nums; }}
.price {{ font-weight: 600; }}
.chg {{ font-weight: 600; }}
.chg.up, .num.up {{ color: var(--up); }} .chg.down, .num.down {{ color: var(--down); }}
.score {{ font-size: 12px; color: var(--ink2); }}
.spark {{ width: 100%; height: 36px; margin: 8px 0 4px; display: block; }}
.spark polyline {{ fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; stroke-linejoin: round; }}
.spark-up {{ stroke: var(--up); }} .spark-down {{ stroke: var(--down); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ background: var(--chip); border-radius: 999px; padding: 2px 9px; font-size: 12.5px; }}
.links {{ display: flex; gap: 16px; margin-top: 10px; font-size: 14px; }}
.links a {{ color: var(--up); text-decoration: none; font-weight: 600; padding: 4px 0; }}
details {{ margin-top: 20px; background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 10px 14px; }}
summary {{ font-weight: 600; cursor: pointer; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13.5px; }}
th, td {{ padding: 6px 4px; border-bottom: 1px solid var(--line); text-align: left; }}
th {{ color: var(--muted); font-weight: 500; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.help p {{ margin: 8px 0; color: var(--ink2); font-size: 14px; }}
.help b {{ color: var(--ink); }}
.empty {{ color: var(--ink2); }}
.badge {{ display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .04em; padding: 1px 8px;
  border-radius: 999px; border: 1.5px solid var(--ink2); color: var(--ink); vertical-align: middle; margin-left: 6px; }}
.b-buy {{ border-color: var(--up); background: color-mix(in oklab, var(--up) 18%, transparent); }}
.b-avoid {{ border-color: var(--down); background: color-mix(in oklab, var(--down) 18%, transparent); }}
.b-monitor {{ border-style: dashed; }}
.why {{ font-size: 13px; color: var(--ink2); margin-top: 8px; }} .why b {{ color: var(--ink); }}
.levels {{ font-size: 13px; background: var(--chip); border-radius: 8px; padding: 6px 10px; margin-top: 8px; }}
.sub {{ font-size: 11.5px; color: var(--muted); font-weight: 400; }}
.counts {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 0; }}
.filters {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0 4px; }}
.filters button {{ font: inherit; font-size: 13px; padding: 4px 12px; border-radius: 999px; border: 1.5px solid var(--ink2);
  background: transparent; color: var(--ink); cursor: pointer; }}
.filters button[aria-pressed="true"] {{ background: var(--ink); color: var(--page); }}
h3 {{ font-size: 14px; margin: 16px 0 4px; }}
table.trk {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
table.trk th, table.trk td {{ padding: 6px 4px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
table.trk td.num, table.trk th.num {{ text-align: right; }}
table.trk .dim {{ color: var(--muted); }}
table.trk .sub {{ font-size: 11px; color: var(--muted); font-weight: 400; }}
footer {{ color: var(--muted); font-size: 12px; margin: 24px 0 8px; }}
</style></head>
<body><main>
<header>
  <h1>UK Scanner</h1>
  <div class="meta">Updated <b>{updated}</b> · Market: <b>{status}</b> · {n} stocks scanned</div>
  <div class="meta">Prices ~15 min delayed. {partial_note}</div>
</header>

<div class="counts"><span class="badge b-buy">✓ BUY {n_buy}</span><span class="badge b-monitor">● MONITOR {n_mon}</span><span class="badge b-avoid">✕ AVOID {n_avoid}</span></div>
<div class="meta" style="margin-top:6px">Signals are rule-based checklists, not recommendations. See "How to read this".</div>

<h2>Buy list (tested setup only)</h2>
<div class="cards">{buy_cards}</div>

<h2>Sectors today</h2>
<div class="tiles">{tiles}</div>

<h2>Worth a look</h2>
<div class="cards">{cards}</div>

<details><summary>All stocks with their signal</summary>
<div class="filters" id="flt">
  <button type="button" data-f="ALL" aria-pressed="true">All</button>
  <button type="button" data-f="BUY" aria-pressed="false">Buy</button>
  <button type="button" data-f="MONITOR" aria-pressed="false">Monitor</button>
  <button type="button" data-f="AVOID" aria-pressed="false">Avoid</button>
</div>
<table id="tbl"><thead><tr><th>Stock</th><th>Signal</th><th class="num">Today</th><th class="num">Volume</th><th class="num">RSI</th><th class="num">Score</th></tr></thead>
<tbody>{rows}</tbody></table></details>
<script>
document.getElementById("flt").addEventListener("click", function (e) {{
  var b = e.target.closest("button"); if (!b) return;
  var f = b.getAttribute("data-f");
  this.querySelectorAll("button").forEach(function (x) {{ x.setAttribute("aria-pressed", x === b ? "true" : "false"); }});
  document.querySelectorAll("#tbl tbody tr").forEach(function (r) {{
    r.style.display = (f === "ALL" || r.getAttribute("data-sig") === f) ? "" : "none";
  }});
}});
</script>

<h2>Track record</h2>
{track}

<details class="help"><summary>How to read this</summary>
<p><b>σ (sigma)</b>: how big today's move is compared with this stock's normal daily move over the last 60 days. 1σ is an ordinary day; 3σ or more is rare.</p>
<p><b>vs sector</b>: the move after taking out what its sector did, so you can tell a stock-specific story from the whole sector moving.</p>
<p><b>× normal volume</b>: shares traded so far compared with a typical day <i>at this time of day</i>. Heavy volume means real money is behind the move.</p>
<p><b>Gapped</b>: opened well away from yesterday's close, usually on news. Tap News (RNS) to see why.</p>
<p><b>RSI</b>: above 75 is stretched to the upside, below 25 stretched to the downside.</p>
<p><b>Breakout / 52-week high or low</b>: price has left its recent range.</p>
<p><b>Squeeze</b>: price has been unusually quiet; big moves often follow.</p>
<p><b>Score</b>: everything combined. It ranks how <i>unusual</i> a stock is, not whether to buy it.</p>
<p><b>✓ Buy</b>: the ONLY setup here that came out positive in my backtests (US stocks and ETFs, not UK): a stock gapped up 2-25% in the last 10 days with no overlap, has since pulled back to the middle of that gap, and has not hit its stop level. The card gives the entry, stop (just under the gap) and a target 2× the risk. Backtest: roughly +0.3 of the risk per trade on average, winning under half of trades, with orders assumed to fill exactly at the entry price. UK stamp duty (0.5%) and wider spreads cut that, so a Buy needs a stop at least 2% away.</p>
<p><b>● Monitor</b>: no tested trigger today. This includes fresh gap-ups waiting for a pullback (the card shows the level to watch) and anything else without a clear signal.</p>
<p><b>✕ Avoid</b>: risk flags, not a forecast. Downtrend, new lows on volume, an unfilled gap down, a death cross or RSI above 85. In my tests stocks in downtrends did <i>not</i> do worse than average over the next 20 days, so Avoid means "none of my tested setups apply and the picture is weak", not "this will fall".</p>
<p><b>Limits</b>: a single day's gap and a 10-day window is a small signal; the daily data here is delayed ~15 min and today's bar is incomplete during the session, so a level can change before the close.</p>
</details>

<footer>FTSE 100 + 250 (investment trusts excluded). Signals are rule-based checklists, not financial advice or recommendations. Refreshes every 30 min in market hours.</footer>
</main></body></html>"""


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def score_all(prices: dict, uni: pd.DataFrame, frac: float):
    """Score every stock and label it. Returns (results sorted by score, sector table)."""
    meta_by_sym = uni.set_index("symbol").to_dict("index")
    results = []
    for sym, df in prices.items():
        try:
            m = score_stock(df, frac)
        except Exception as e:
            print(f"  skip {sym}: {e}")
            continue
        if m is None:
            continue
        info = meta_by_sym.get(sym, {})
        m.update(symbol=sym, tidm=info.get("tidm", sym[:-2]), name=info.get("name", sym),
                 sector=info.get("sector", "Other"))
        results.append(m)
    if not results:
        return [], None
    res = pd.DataFrame(results)
    sect = res.groupby("sector").agg(med_z=("ret_z", "median"), ret=("ret", "median"), n=("ret", "size"))
    for m in results:
        row = sect.loc[m["sector"]]
        m["rel_z"] = m["ret_z"] - row["med_z"] if row["n"] >= 3 else m["ret_z"]
        finalise(m)
    results.sort(key=lambda m: m["score"], reverse=True)
    return results, sect


def run(prices: dict, uni: pd.DataFrame, now: datetime):
    last_dates = [df.index[-1].date() for df in prices.values()]
    last_bar = max(set(last_dates), key=last_dates.count) if last_dates else now.date()
    sess = session_state(now, last_bar)

    results, sect = score_all(prices, uni, sess["frac"])
    top = [m for m in results if m["score"] >= MIN_SCORE and m["reasons"]][:TOP_N]
    sectors = [{"sector": s, "ret": float(r["ret"]), "n": int(r["n"])}
               for s, r in sect[sect["n"] >= 3].sort_values("ret", ascending=False).iterrows()]

    meta = {"updated": now.strftime("%a %d %b, %H:%M"), "status": sess["status"],
            "partial": sess["partial"], "n_scanned": len(results)}
    try:
        track = track_section(prices, uni, results, top, last_bar, now)
    except Exception as e:           # the forward test must never break the dashboard
        import traceback
        traceback.print_exc()
        track = f'<p class="empty">Track record unavailable this run ({escape(str(e))}).</p>'
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(build_html(top, results, sectors, meta, track))
    slim = [{k: v for k, v in m.items() if k != "spark"} for m in results]
    with open(os.path.join(OUT_DIR, "data.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "top": [m["symbol"] for m in top], "stocks": slim}, f, default=float)
    print(f"Done: {len(top)} flagged, {len(results)} scored, market {sess['status']}")
    return top, results


def main():
    now = datetime.now(LONDON)
    uni = get_universe()
    prices = download_prices(uni["symbol"].tolist())
    if not prices:
        sys.exit("No price data came back from Yahoo. Try again later.")
    run(prices, uni, now)


if __name__ == "__main__":
    main()
