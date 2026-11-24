# Σ Sigma — Self-learning Trading Analysis Robot

Sigma analyses forex, gold/silver, crypto, indices and Deriv synthetic indices. You can chat with it, it gives
risk-defined trade signals, and it learns from every paper trade it takes.

**It never places real trades.** Every analysis is journalled and paper-traded, so the robot can grade itself.

## What it does

| Layer | Sources (all free) | What it produces |
|---|---|---|
| Prices | **Deriv** public WebSocket (forex, metals, crypto, synthetics, no key), **Binance** (crypto, no key), **Twelve Data** (fallback, stocks/indices) | Clean OHLC candles, M1 to D1 |
| Technical | Built in: EMA 20/50/200, SuperTrend, RSI, MACD, ATR, ADX/DI, Bollinger, Stochastic, swing structure, support/resistance zones, pivots, Fibonacci, candle patterns, RSI divergence, regime detection, higher-timeframe trend | 8 component scores from −1 to +1 |
| Fundamental | **FRED** (policy & 3-month rates, 10-year yields, real yields, dollar index, VIX, yield curve), **CFTC COT** (speculative positioning), Crypto Fear & Greed | Macro bias per instrument |
| News | RSS (Investing.com, FXStreet, Fed, ECB, Cointelegraph), **Finnhub**, **Alpha Vantage** scored news | Currency-aware hawkish/dovish sentiment |
| Event risk | ForexFactory weekly calendar | Blocks new signals 30 min around high-impact events |
| Chat | Claude with 14 tools + web search, reads chart screenshots | Explains, compares timeframes, scans, sizes positions, reports what it learned |

## How it learns

1. **Every analysis is recorded** (scheduled scans at each candle close + every chat request) with all component scores.
2. **Every analysis is paper-traded.** Qualifying ones are *signals*; the rest are *shadow trades*, which give the model unbiased data (it learns from what it rejected too).
3. **Trades are resolved against real prices** (M1 candles, stop assumed hit first if both levels are inside one candle, spread/cost charged).
4. **The outcome model is refitted** after every resolution: a regularised logistic regression that learns how much each component (and each component *in trending vs ranging markets*) actually predicts winners. It starts from classic TA beliefs and moves away from them as evidence arrives.
5. **Calibration** — the model's past predictions are compared with results. If its "60%" calls only won 45%, today's 60% is shown as ~45%. It starts sceptical (assumes 60% of claimed edge is real) and adjusts.
6. **Setup gating** — any setup (market × timeframe × regime × direction) with 20+ signals and negative expectancy (with statistical shrinkage) is switched off automatically.
7. **Lessons** — after each live signal closes, Claude writes a one-to-two sentence post-mortem, which the chat can recall.
8. **Walk-forward backtests** bootstrap all of this from history without look-ahead, so you don't wait weeks for data.

Tested behaviour (see `tests/`, 25 tests): on pure random-walk prices the robot learns there is no edge and nearly stops signalling;
on data with a real momentum edge its signals turned positive (+0.30R per trade) and improved as it learned.

## Talk to it

Open the **Chat** and write naturally, for example:

- "Analyse EURUSD on H1" · "Full multi-timeframe view of gold with news" · "Scan the watchlist on H4"
- **"Review my trade: buy XAUUSD at 2650, stop 2640, target 2675"** → Sigma checks your trade against live
  analysis and gives a verdict (GOOD / CAUTION / AVOID), your win probability vs the break-even win rate,
  expected R after costs, and the problems it sees (stop inside the last swing, levels in the way, news soon,
  robot disagrees). It then tracks your trade on paper (as a pending order if price isn't there yet) and grades it.
- Attach a **chart screenshot** (📎 or paste) and ask "what do you see here, should I buy?"
- "How are my trades doing?" · "What have you learned?"

Your reviewed trades appear in the **Your trades** tab and are scored separately, so they never distort the robot's own learning.

## Install it as an app & sync across devices

Sigma is an installable web app (PWA). Host it once (below), then on each device open its link and sign in with your
`APP_PASSWORD`:

- **Android (Chrome):** menu ⋮ → *Install app* (or the *Install app* button in Sigma's header).
- **iPhone/iPad (Safari):** Share → *Add to Home Screen*.
- **Windows/Mac (Chrome or Edge):** the install icon in the address bar, or the *Install app* button.

Every device talks to the same robot and the same database, so the chat, signals, paper trades and learning are identical
everywhere. A message sent from your phone appears on your laptop within a few seconds.

## Deploy online (free: GitHub + Render + Supabase)

1. **Database (keeps your data and syncs devices):** create a free project at https://supabase.com → *Connect* →
   copy the **Session pooler** connection string (it works over IPv4) and put your database password in it.
2. **Host:** at https://render.com sign in with GitHub → *New* → *Blueprint* → pick this repository. Render reads
   `render.yaml`. Fill in `APP_PASSWORD`, `DATABASE_URL` (from step 1), `ANTHROPIC_API_KEY`, `FRED_API_KEY` and any other keys.
3. **Keep it awake:** free Render services sleep after ~15 minutes idle, which would pause scanning. In GitHub → this repo →
   *Settings → Secrets and variables → Actions* add `SIGMA_URL` = your Render URL. The `keep-sigma-awake` workflow pings it every 10 minutes.
   (A paid Render instance or any always-on VPS avoids this.)
4. Open the Render URL, sign in, install the app on your devices, and run a few backtests from the **Backtest** tab.

Without `DATABASE_URL` the robot uses a local SQLite file - fine on one computer, but a free host wipes it on every restart.

## Run it on your computer

Needs Python 3.11+.

```bash
cd sigma
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your keys into .env
python -m sigma             # open http://localhost:8000
```

Then, so learning starts from history right away, run one backtest per market from the **Backtest** tab or:

```bash
python -m sigma backtest EURUSD H1 3000
python -m sigma backtest XAUUSD H1 3000
```

Other commands: `python -m sigma analyze GBPJPY H4`, `pytest -q` (runs the test suite).

## Free keys (all optional except Anthropic for chat)

- **Anthropic** (chat + lessons): https://console.anthropic.com
- **FRED** (macro): https://fred.stlouisfed.org/docs/api/api_key.html
- **Twelve Data** (fallback prices, stocks): https://twelvedata.com/register — free plan is 8 requests/min, 800/day
- **Finnhub** (news): https://finnhub.io/register
- **Alpha Vantage** (scored news, 25/day): https://www.alphavantage.co/support/#api-key
- **Deriv app id**: 1089 is Deriv's public test id; register your own at https://developers.deriv.com for long-term use.

## Run it 24/7 (so it keeps learning while your PC is off)

The robot must stay running to scan and resolve trades. Any host that runs a Docker container with a persistent disk works
(Render, Railway, Fly.io, a small VPS). Mount a disk at `/data` (the database lives at `/data/sigma.db`) and set your keys
as environment variables. Put it behind a password (most hosts offer basic auth) because the API has no login of its own.

## Settings (`.env`)

`WATCHLIST`, `SCAN_TIMEFRAMES` (default M15,H1,H4), `MIN_PROBABILITY` (0.45), `MIN_EXPECTANCY_R` (0.10 R after costs),
`REWARD_R` (take profit = 2 × risk), `SL_ATR_MULT` (1.5), `MAX_HOLD_BARS` (30), `NEWS_BLACKOUT_MIN` (30), `DISPLAY_TZ` (Africa/Johannesburg).

## Crowd positioning (optional, free)

Sigma can read where retail traders are positioned and use it as a contrarian factor (crowd mostly
long → leans bearish), weighted by what its own results show it is worth.

- **Myfxbook Community Outlook** - free account at myfxbook.com; set `MYFXBOOK_EMAIL` and `MYFXBOOK_PASSWORD`.
- **IG Client Sentiment** - API key from My IG → Settings → API keys; set `IG_API_KEY`, `IG_USERNAME`,
  `IG_PASSWORD` and `IG_ACCOUNT_TYPE` (`live` or `demo`, matching the login). Read only: Sigma never trades on IG.

Either one is enough; with both, the readings are averaged. Synthetic indices have no crowd data.

## Auto bot (🤖 page)

An automatic trader that runs on its own paper account (default) or a live Deriv account.

- **Two modes, picked per market:** *short trades* when a confirmed 15-minute signal appears in a trending
  market (2 positions: one closes at 1R, the other at 2R and moves to breakeven once the first closes);
  *scalps* on strong 1-minute setups that agree with the 5-minute trend (several small positions, each
  closing at a dollar profit: 50c under $300, $1 under $1,000, 0.1% of equity above; more positions on bigger accounts).
- **Smart exits:** take profit, stop loss, *protect profit* (a position that reached 60% of its target and
  gives most of it back is closed), and time exits for scalps (15 min in profit / 30 min max).
- **Prop-firm rules:** daily loss limit (3%) closes everything for the day; max drawdown (8%) stops the bot
  until you reset it; total open risk capped at 3%; 30-minute pause after 3 losses in a row; no trading
  near high-impact news; after half the daily target it trades smaller and pickier, at the full target it stops.
- **Self-learning:** every bot trade is graded; (mode, market) pairs with negative expectancy after 20+
  trades are paused; scalp targets shrink when trades keep reversing after being in profit; daily self-review notification.
- **Brokers:** paper and Deriv (API, Multipliers with take profit / stop loss in money). Set
  `DERIV_API_TOKEN` in Render → Environment; going live needs a typed confirmation on the Bot page.

## Honest notes

- Paper results are not live results: real fills, slippage and spreads during news differ. Treat any market as unproven
  until it has at least 50–100 live paper signals with positive expectancy.
- Deriv synthetic indices come from a random number generator; no fundamentals or news apply and lasting technical edges
  are unlikely. The robot will still learn on them and will switch setups off if they lose.
- Investing.com has no public API; Sigma only reads its public RSS feeds.
- Fundamental and news weights can only be learned from live paper trades (no historical data for them in backtests).

## Project layout

```
sigma/
  data/        market.py (Deriv/Binance/Twelve Data), macro.py (FRED/COT/F&G), news.py (RSS/Finnhub/AV/calendar)
  analysis/    indicators.py, technical.py, fundamental.py, sentiment.py
  engine.py    combines everything through the learned model into a trade idea
  learning.py  outcome model, calibration, setup stats, gating, insights
  paper.py     paper broker + trade monitor
  backtest.py  walk-forward backtester
  agent.py     Claude chat with tools
  scheduler.py candle-close scanner + monitor loops
  server.py    API + web dashboard (static/index.html)
  review.py    grades trades you propose in the chat and tracks them
  static/      dashboard, login page, app manifest, service worker, icons
tests/         25 tests: indicator maths, R/expectancy, learning, paper fills, trade review, login, sync, full pipeline
render.yaml    one-click Render deploy · .github/workflows: tests + keep-alive
```
