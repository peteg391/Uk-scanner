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

    return {
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
    m["direction"] = "up" if (m["ret"] if abs(m["ret_z"]) >= 0.5 else m["mom5"]) >= 0 else "down"
    return m


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


def card(m):
    arrow = "▲" if m["direction"] == "up" else "▼"
    chips = "".join(f'<span class="chip">{escape(r)}</span>' for r in m["reasons"])
    tidm = escape(m["tidm"])
    return f"""
<article class="card">
  <div class="row1">
    <div>
      <div class="tick">{tidm}</div>
      <div class="name">{escape(m['name'])}</div>
      <div class="sector">{escape(m['sector'])}</div>
    </div>
    <div class="right">
      <div class="price">{fmt_price(m['price'])}</div>
      <div class="chg {m['direction']}">{arrow} {m['ret'] * 100:+.2f}%</div>
      <div class="score" title="Unusualness score">Score {m['score']:.1f}</div>
    </div>
  </div>
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


def build_html(top, everything, sectors, meta):
    cards = "".join(card(m) for m in top) or '<p class="empty">Nothing unusual right now. Quiet market.</p>'
    tiles = "".join(heat_tile(s["sector"], s["ret"], s["n"]) for s in sectors)
    rows = "".join(
        f"<tr><td>{escape(m['tidm'])}</td><td class='num {m['direction']}'>{m['ret'] * 100:+.1f}%</td>"
        f"<td class='num'>{m['rvol']:.1f}×</td><td class='num'>{m['rsi']:.0f}</td>"
        f"<td class='num'>{m['score']:.1f}</td></tr>"
        for m in everything[:60]
    )
    return HTML_TEMPLATE.format(
        updated=escape(meta["updated"]), status=escape(meta["status"]),
        n=meta["n_scanned"], cards=cards, tiles=tiles, rows=rows,
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
footer {{ color: var(--muted); font-size: 12px; margin: 24px 0 8px; }}
</style></head>
<body><main>
<header>
  <h1>UK Scanner</h1>
  <div class="meta">Updated <b>{updated}</b> · Market: <b>{status}</b> · {n} stocks scanned</div>
  <div class="meta">Prices ~15 min delayed. {partial_note}</div>
</header>

<h2>Sectors today</h2>
<div class="tiles">{tiles}</div>

<h2>Worth a look</h2>
<div class="cards">{cards}</div>

<details><summary>All scored stocks (top 60)</summary>
<table><thead><tr><th>Stock</th><th class="num">Today</th><th class="num">Volume</th><th class="num">RSI</th><th class="num">Score</th></tr></thead>
<tbody>{rows}</tbody></table></details>

<details class="help"><summary>How to read this</summary>
<p><b>σ (sigma)</b>: how big today's move is compared with this stock's normal daily move over the last 60 days. 1σ is an ordinary day; 3σ or more is rare.</p>
<p><b>vs sector</b>: the move after taking out what its sector did, so you can tell a stock-specific story from the whole sector moving.</p>
<p><b>× normal volume</b>: shares traded so far compared with a typical day <i>at this time of day</i>. Heavy volume means real money is behind the move.</p>
<p><b>Gapped</b>: opened well away from yesterday's close, usually on news. Tap News (RNS) to see why.</p>
<p><b>RSI</b>: above 75 is stretched to the upside, below 25 stretched to the downside.</p>
<p><b>Breakout / 52-week high or low</b>: price has left its recent range.</p>
<p><b>Squeeze</b>: price has been unusually quiet; big moves often follow.</p>
<p><b>Score</b>: everything combined. It ranks how <i>unusual</i> a stock is, not whether to buy it.</p>
</details>

<footer>FTSE 100 + 250 (investment trusts excluded). Not financial advice. Refreshes every 30 min in market hours.</footer>
</main></body></html>"""


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------
def run(prices: dict, uni: pd.DataFrame, now: datetime):
    meta_by_sym = uni.set_index("symbol").to_dict("index")
    last_dates = [df.index[-1].date() for df in prices.values()]
    last_bar = max(set(last_dates), key=last_dates.count) if last_dates else now.date()
    sess = session_state(now, last_bar)

    results = []
    for sym, df in prices.items():
        try:
            m = score_stock(df, sess["frac"])
        except Exception as e:
            print(f"  skip {sym}: {e}")
            continue
        if m is None:
            continue
        info = meta_by_sym.get(sym, {})
        m.update(symbol=sym, tidm=info.get("tidm", sym[:-2]), name=info.get("name", sym),
                 sector=info.get("sector", "Other"))
        results.append(m)

    # sector adjustment
    res = pd.DataFrame(results)
    sect = res.groupby("sector").agg(med_z=("ret_z", "median"), ret=("ret", "median"), n=("ret", "size"))
    for m in results:
        row = sect.loc[m["sector"]]
        m["rel_z"] = m["ret_z"] - row["med_z"] if row["n"] >= 3 else m["ret_z"]
        finalise(m)

    results.sort(key=lambda m: m["score"], reverse=True)
    top = [m for m in results if m["score"] >= MIN_SCORE and m["reasons"]][:TOP_N]
    sectors = [{"sector": s, "ret": float(r["ret"]), "n": int(r["n"])}
               for s, r in sect[sect["n"] >= 3].sort_values("ret", ascending=False).iterrows()]

    meta = {"updated": now.strftime("%a %d %b, %H:%M"), "status": sess["status"],
            "partial": sess["partial"], "n_scanned": len(results)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(build_html(top, results, sectors, meta))
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
