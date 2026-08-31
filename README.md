# Crypto Quant Research System

A small, reproducible research system for answering one question carefully: does a
trading hypothesis survive real data, explicit costs, out-of-sample testing, and
risk review?

## Current Scope

- Binance public spot OHLCV data stored in local SQLite.
- K-line ingestion rejects non-finite values, bad OHLC ranges, misaligned bars,
  inconsistent close times, and inconsistent taker-volume fields.
- Fractional taker-buy base volume is preserved exactly; base/quote pairs that
  imply a trade price outside the bar range are isolated as a taker-flow warning
  without blocking price-only OHLCV research.
- Predeclared, non-optimized strategy definitions.
- Event-style backtesting with next-bar-open execution.
- Explicit commission and slippage.
- Risk-adjusted performance, drawdown, volatility, monthly-return distribution,
  consecutive-loss streaks, turnover, trade, and liquidity diagnostics.
- Full-period and fixed out-of-sample reports with an append-only experiment ledger.
- USD-M funding ingestion and a delta-neutral carry study gated by funding coverage.
- USD-M public derivatives-metrics ingestion for paired open interest and
  long/short positioning ratios, with SHA-256 archive registration and conflict
  rejection.
- Raw funding archives are registered by SHA-256; cached files are verified before reuse.
- Native-cadence coverage gates use the declared 5-minute or funding schedule,
  not the observed median gap, so uniformly missing observations cannot redefine
  themselves as complete history.
- Maintenance/listing transitions may produce short partial bars and gaps;
  both are retained and reported rather than filled.

This is research software, not investment advice or an execution system.

## Repository and local data

The Git repository contains source code, tests, research rules, lightweight
ledgers, and reproducible data-acquisition definitions. Market databases are
local artifacts and are deliberately not committed: `market_data/` can be
rebuilt by the public-data pipelines, while coverage, source URLs, row counts,
and hashes remain auditable through the project metadata. Large generated
backtests, download state, logs, and exchange-session state under `experiments/`
are also local-only.

Copy `.env.example` to `.env.demo` before using Binance Demo, then fill the two
values locally. Populated `.env*` files are excluded from version control.

## On-demand public data requests

The research data layer uses Binance USDT Spot public data.  It keeps all
requests and Codex Goal resume checkpoints under `experiments/`; it downloads
only missing ranges.  Historical `1h` is the base layer, `4h`/`1d` can be
derived locally, and a candidate may request `5m`, `1m`, or `aggTrades` as
needed.  Order-book requests are recorded as `collecting_forward` until a
forward capture exists; no historical L2 is fabricated.  Low-volume and
delisted symbols remain eligible for historical research.

Typical Goal handoff:

```bash
python -m crypto_quant.cli data-request-create --spec-json request.json
python -m crypto_quant.cli research-checkpoint-create --strategy-version-id STRATEGY --request-ids REQUEST_ID --completed-steps community_research,hypothesis
AUTHORIZED_PUBLIC_DOWNLOAD=1 python -m crypto_quant.cli data-request-fulfill --request-id REQUEST_ID --authorize-public-download
python -m crypto_quant.cli research-resume-status --checkpoint-id CHECKPOINT_ID
```

The public-download flag is an explicit operator gate.  Tests and complete
local requests never touch the network.  Once the request is `ready`, the Goal
continues at the checkpoint's `next_action` and may mark the request consumed.
`data-inventory-plan` writes a no-download plan for the full universe and
rolling historical-liquidity top-N work; it does not start a backfill.

## Unified local market-data interface

`MarketDataStore` is the read-only research entry point for the local database.
It exposes spot trade bars; USD-M perpetual trade, mark, index, and premium-index
bars; native funding events; and native 5-minute open-interest/positioning
metrics without requiring a strategy to know the underlying SQLite tables.
It does not download or mutate data.

```python
from pathlib import Path
from crypto_quant.market_data import MarketDataStore, USD_M_PERPETUAL

data = MarketDataStore(Path("market_data/crypto_quant.sqlite"))
bars = data.load_bars(USD_M_PERPETUAL, "BTCUSDT", interval="4h")
features = data.load_feature_frame(
    USD_M_PERPETUAL,
    "BTCUSDT",
    interval="4h",
    include_funding=True,
    include_metrics=True,
)
```

Multi-hour and daily bars use the common local 1-hour base whenever it is
available, so symbols follow one aggregation rule even when a legacy exact bar
table also exists. Buckets with a missing source hour are omitted rather than
filled; `derive=False` can explicitly request a stored exact interval. Auxiliary
events use a backward as-of join, retain their original observation timestamps,
and therefore cannot leak a later observation into an earlier bar.

Inspect the whole local catalog or exact coverage for selected symbols:

```bash
make market-data-catalog
MARKET_DATA_SYMBOLS=BTCUSDT,ETHUSDT make market-data-catalog
```

## Causal derived factors

`FactorEngine` turns the unified source data into a stable factor frame for one
symbol. The first catalog includes basis, mark/index dislocation, funding carry,
open-interest change, long/short positioning, taker flow, liquidity, relative
returns, and realized volatility. Missing optional inputs produce `NaN` only for
their dependent factors and are listed in `input_availability`; they do not
silently remove the symbol.

```python
from pathlib import Path
from crypto_quant.factors import load_factor_frame

factors = load_factor_frame(
    Path("market_data/crypto_quant.sqlite"),
    "BTCUSDT",
    interval="4h",
    start="2023-01-01",
)
```

Every row is timestamped by the base bar open and includes `available_at`, the
actual bar close. Price sources must be closed by `available_at`; funding and
metrics use a backward as-of match to that time. The factor row may therefore be
used only for the next bar or later, never for execution inside its own bar.

```bash
make market-factor-catalog
FACTOR_SYMBOL=BTCUSDT FACTOR_INTERVAL=4h make market-factor-sample
```

Run the exploratory cross-sectional evaluator with the declared major-symbol
universe:

```bash
make factor-evaluate
```

For every factor, the training segment fixes whether high or low values are the
active direction. The test segment then reports cross-sectional rank IC, ICIR,
quarterly and cross-symbol stability, and an equal-weight long/short portfolio
using next-bar open-to-close returns with turnover costs. The resulting ranking
is a research-lead generator: because the test segment is used to rank factors,
it cannot also serve as untouched confirmation for a strategy built from it.
Raw base-unit open interest remains available for within-symbol time-series
research but is excluded from cross-sectional ranking because contract units are
not comparable across assets; data-age fields are diagnostics and are excluded
for the same evaluation purpose.

To discover current and historical USDT Spot symbols and materialize resumable
1h batches (still without downloading the batches):

```bash
python -m crypto_quant.cli data-inventory-plan --discover --batch-size 50 --materialize
python -m crypto_quant.cli data-inventory-resume --plan-id PLAN_ID
python -m crypto_quant.cli data-top50-plan --inventory-plan experiments/data_requests/PLAN_ID.json
```

Discovery uses Binance `exchangeInfo` plus the official `data.binance.vision`
archive listing, retains historical/delisted symbols, and records the first
and last valid 1h archive bar.  The Top-N plan ranks each historical lookback
window by local 1h `quote_volume` and requests the following 5m holding window,
so the ranking does not use future data.

## Realtime shadow trading (Phase 1)

The independent shadow session consumes only Binance Spot public `4h` klines
and `bookTicker` events. It maintains a local virtual account and virtual
fills; it has no API-key path, account stream, private endpoint, or exchange
order router. The frozen `donchian_96_48` BTC/ETH 50/50 long/flat rule is
recorded in a separate `experiments/shadow_sessions/` schema and does not alter
schema-3 forward-paper sessions or the experiment ledger.

Initialize and inspect a session offline:

```bash
make shadow-init
make shadow-status
make shadow-replay SHADOW_EVENTS=/path/to/local/public-events.jsonl
make shadow-kill
```

`shadow-replay` accepts local JSONL (or a JSON array) only. A public WebSocket
requires two explicit operator actions: `AUTHORIZED_PUBLIC_STREAM=1` in the
Makefile invocation and `--authorize-public-stream` (the Make target supplies
the latter). For example, after human approval:

```bash
AUTHORIZED_PUBLIC_STREAM=1 make shadow-run
```

Official Spot `bookTicker` messages may omit `e` and `E`; live mode supplies a
local receive timestamp. Offline replay of such messages must supply an
explicit deterministic `--received-at` value and never invents wall-clock time.
`shadow-init` freezes a validated common BTC/ETH 4h history prefix, signal
prefix digest, database digest, parameters, and source manifest. `shadow-kill`
persists the kill switch through recovery.

The stream gate rejects non-UTC-4h or incomplete strategy candles, duplicate
conflicts, time/order gaps, stale data, and inconsistent persisted state. Risk
is long-only, unlevered, capped at 100% gross exposure and 50% per BTC/ETH
sleeve; uncertain data can only hold or reduce risk. All tests use local fake
events and never access the network.

## Two-track execution simulation

The research/shadow layer remains the local virtual OMS driven by production
public market data. Binance Spot Demo Mode is the primary exchange-side
realistic execution simulation; neither its PnL nor its fills are research
evidence. Testnet is retained as a lower-level integration and reset-prone
compatibility path.

## Local strategy platform (Phase 1)

`crypto_quant.strategy_platform` is a local, deterministic slot and sub-ledger
framework. It provides eight incumbent slots and two challenger slots, each
starting with an isolated virtual 500 USDT ledger. It does not connect to an
exchange, does not auto-fund, and allows product/long-short behavior only as a
virtual accounting result; losses, insufficient funds, and negative positions
remain visible outcomes. Its V1 score is `H=30P+25R+30B+15X` and
`G=0.35H+0.40F+0.25C`, with hard challenger/final thresholds encoded in code.
Each registered version may also record one `primary_family`, up to three
`secondary_tags`, and short `profit_mechanism`, `expected_regimes`, and
`failure_regimes` descriptions.  The eight families describe research
coverage only and do not restrict same-family strategies.  The composite
contribution score is
`C=0.35N(delta_sharpe)+0.25N(delta_cagr)+0.20N(mdd_improvement)+0.20V`,
where `V=100*(1-(0.60*max(0,return_correlation)+0.25*position_overlap+0.15*trade_overlap))`.
`V` rewards behaviorally different candidates without changing the existing
incumbent/challenger/queue flow or the ten-slot capacity; it cannot bypass any
existing hard gate.  Historical score inputs without overlap fields use the
known correlation as a conservative proxy; a partially missing overlap is
treated as full overlap.
Checkpoints are generated JSON/Markdown artifacts whose `research_evidence`,
`demo_research_evidence`, `demo_execution_evidence_only`, and
`shadow_forward_evidence` fields come from structured platform inputs.
The official non-production environment matrix is in
[`docs/BINANCE_ENVIRONMENT_MATRIX.md`](/Users/stellan/量化投资/docs/BINANCE_ENVIRONMENT_MATRIX.md).

## Binance Spot Demo Mode simulation

The separate `demo-*` commands use only the hard-coded Demo Mode origin
`https://demo-api.binance.com/api` and session directory
`experiments/demo_sessions/`. Demo warnings deliberately say `DEMO ONLY`,
`REALISTIC IS NOT REAL`, and `NOT RESEARCH EVIDENCE`. Demo balances can be
reset through the Binance UI; Demo market data and order-book behavior are
similar to live but are not proof of live performance. See Binance's [official
Demo Mode documentation](https://developers.binance.com/en/docs/products/spot/demo-mode/general-info).
Offline status keeps `maintenance_status=unknown` and
`operator_must_check_demo_changelog=true`; no changelog is fetched automatically.

Offline setup and status require no credentials or network:

```bash
make demo-init
make demo-status
make demo-kill
```

`demo-quote` reads only the public best bid/ask and requires
`AUTHORIZED_BINANCE_DEMO=1` plus `--authorize-demo-market-data`; it does not
read credentials. `demo-cancel` in an authorized workflow only
accepts a local non-terminal `dm_` order from the current Demo session; it
cannot cancel an arbitrary venue order or symbol.

Put the two Demo HMAC values in the local `.env.demo` file. The loader accepts
only `BINANCE_DEMO_API_KEY` and `BINANCE_DEMO_SECRET_KEY`, rejects broader
authorization fields, symlinks, and group/world-readable files, and never
persists their values in session state or events. Explicit environment
variables take precedence when present. The file is ignored by Git; network
authorization remains a separate, per-command decision.

Authorized Demo reconciliation or simulation requires both
`AUTHORIZED_BINANCE_DEMO=1` and `--authorize-demo-orders`, and reads only
`BINANCE_DEMO_API_KEY` / `BINANCE_DEMO_SECRET_KEY`. This task does not run
those network commands. BUY is limited to 10 USDT per order and 25 USDT total;
SELL is only for reconciled free base balances. Demo client IDs use `dm_` and
Demo sessions cannot be opened by the Testnet client.

## Strategy-platform Demo bridge (Phase 2)

The `platform-demo-*` commands are a finite, externally schedulable one-cycle
bridge from active platform slots to Binance Spot Demo. They do not start a
daemon and they never call Testnet, derivatives, or a production host. The
bridge uses the hard-coded `demo-api.binance.com` profile, the two currently
supported BTCUSDT/ETHUSDT Spot symbols, and the last complete local 4h bar.

The Demo account is one shared physical account, not ten Binance subaccounts.
The bridge keeps an append-only internal strategy attribution ledger and
enforces 500 USDT per active strategy, with a session cap of
`500 * active_slots`. Unknown strategies and non-Spot strategy contracts are
rejected. Sells are submitted before buys; exchange `stepSize`, `minNotional`,
and `tickSize` rules are loaded every cycle. Re-running a cycle is idempotent
for the same completed bar or while an owned order is unresolved.

After `make demo-init` and platform initialization, enable once with the
persistent authorization flag, inspect status, and invoke one cycle from an
external scheduler:

```bash
AUTHORIZED_BINANCE_DEMO=1 make platform-demo-enable
make platform-demo-status
AUTHORIZED_BINANCE_DEMO=1 make platform-demo-run-cycle
make platform-demo-disable
make platform-demo-kill
```

The cycle requires `AUTHORIZED_BINANCE_DEMO=1` and the two credentials in the
private `.env.demo` file. The bridge does not print or persist credentials.
All tests use an injected fake transport; these commands are not run as part
of the test suite.

## Binance Spot Testnet execution rehearsal (compatibility)

The separate `testnet-*` commands validate engineering order flow against the
Binance Spot Testnet only. They are explicitly `execution_rehearsal_only` and
never produce research or strategy-performance evidence. The REST origin is
hard-coded to `https://testnet.binance.vision/api`; production hosts and
`/sapi` paths are rejected. Testnet assets cannot be transferred in or out and
the environment may reset balances and orders without notice.
[Binance's Spot Testnet documentation](https://developers.binance.com/en/docs/products/spot/testnet/general-info)
is the authoritative environment contract used by this adapter.

Offline setup and status require no credentials or network:

```bash
make testnet-init
make testnet-status
make testnet-kill
```

Offline status starts as `awaiting_authorized_reconciliation` with
`network_connected=false`; it never claims that a testnet kill has reached the
venue. After an authorized reconciliation, kill state is either
`kill_cancel_pending` or `killed_confirmed`. Reconciliation first synchronizes
server time, refreshes local order IDs (including UNKNOWN and partial orders),
discovers open venue orders, and only then performs kill cancellations.

Reconciliation or a rehearsal order requires both
`AUTHORIZED_BINANCE_TESTNET=1` and `--authorize-testnet-orders`, and reads only
`BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_SECRET_KEY`. This task does not
run those network commands. Requests use deterministic `clientOrderId` values;
timeouts, 5xx responses, and Binance `-1007` responses are treated as UNKNOWN
and queried for status rather than retried blindly. Credentials are never
written to session state, event logs, or errors.
This first rehearsal implementation uses bounded REST reconciliation only; it
does not open a private user-data WebSocket stream.
Each initialized rehearsal freezes the current project source manifest. After
code or governance changes, initialize a new session instead of continuing an
old one.

## Quick Start

From this directory:

```bash
PYTHONPATH=src python3 -m crypto_quant.cli update-data \
  --authorize-public-download \
  --symbols BTCUSDT,ETHUSDT \
  --interval 4h \
  --start 2019-01-01

PYTHONPATH=src python3 -m crypto_quant.cli first-study \
  --symbols BTCUSDT,ETHUSDT \
  --interval 4h \
  --test-start 2023-01-01
```

The authorization flag is an intentional operator action. It does not grant API
account access or approve live trading; it only authorizes the named public-data
download after a human decision.

`make status` reports `attention_required` whenever a required or registered
research source is stale, incomplete, or degraded; the existence of a database
alone is not treated as readiness.

For a reproducible local installation, use a virtual environment and install
the tested package versions first:

```bash
python3 -m pip install -r requirements-lock.txt
python3 -m pip install -e .
```

This also exposes the shorter `crypto-quant` command.

The public endpoint is `https://data-api.binance.vision`. It does not require an
API key and does not permit private account access. Network fetching is a human
decision: do not refresh market data without explicit authorization.
Open-interest metrics use `https://data.binance.vision` under the same rule.

## Layout

- `src/crypto_quant/data.py`: download, persist, validate, and load market data.
- `src/crypto_quant/data_quality.py`: offline OHLCV, funding, derivatives,
  and raw-cache integrity snapshots.
- `src/crypto_quant/data_sources.py`: deterministic inventory of research inputs,
  acquisition constraints, local coverage, and human authorization gates.
- `src/crypto_quant/futures.py`: funding ingestion, validation, and coverage gates.
- `src/crypto_quant/open_interest.py`: public derivatives-metrics ingestion,
  open-interest/positioning validation, coverage gates, and raw-archive
  integrity.
- `src/crypto_quant/open_interest_study.py`: preregistered open-interest
  confirmation diagnostic with causal-prefix, coverage, stress, and DSR gates.
- `src/crypto_quant/positioning_study.py`: preregistered top-trader positioning
  extremes event study with causal weekly origins and bootstrap intervals.
- `src/crypto_quant/order_book_replay.py`: deterministic local L2 event replay
  for depth-aware fills, slippage, and partial-execution diagnostics.
- `src/crypto_quant/carry.py`: causal delta-neutral carry signal and paired-leg L0 backtest.
- `src/crypto_quant/indicators.py`: causal indicators used by strategy rules.
- `src/crypto_quant/strategies.py`: predeclared hypotheses and target weights.
- `src/crypto_quant/backtest.py`: next-bar-open event backtest and cost model.
- `src/crypto_quant/metrics.py`: performance, risk, trade, and execution metrics.
- `src/crypto_quant/validation.py`: chronological splits and risk gates.
- `src/crypto_quant/agents.py`: local data/evidence/risk/research agent briefing.
- `src/crypto_quant/strategy_library.py`: persistent lifecycle snapshot for explored strategies.
- `src/crypto_quant/strategy_platform.py`: local isolated strategy slots, virtual sub-ledgers, V1 promotion scoring, and checkpoints.
- `src/crypto_quant/audit.py`: ledger, artifact, and paper-session integrity audits.
- `src/crypto_quant/integrity.py`: SHA-256 artifact manifests and append-only ledger checks.
- `src/crypto_quant/diagnostics.py`: return-source, regime, drawdown, and trade autopsy.
- `src/crypto_quant/portfolio.py`: multiasset sleeve portfolio backtesting.
- `src/crypto_quant/provenance.py`: deterministic source-code and governance manifest.
- `src/crypto_quant/research_contracts.py`: machine-readable preregistrations for
  future diagnostics, including fixed gates and prohibited actions.
- `src/crypto_quant/portfolio_robustness.py`: joint portfolio parameter-neighborhood testing.
- `src/crypto_quant/capacity_replay.py`: dynamic partial-fill execution replay under liquidity caps.
- `src/crypto_quant/capacity.py`: ex ante liquidity-participation and first-order capacity analysis.
- `src/crypto_quant/walk_forward.py`: causal calendar-fold stability validation.
- `src/crypto_quant/portfolio_regimes.py`: causal market-state stress for the portfolio.
- `src/crypto_quant/portfolio_drawdowns.py`: basket-drawdown episode attribution.
- `src/crypto_quant/portfolio_diversification.py`: rolling correlation and sleeve-mix sensitivity.
- `src/crypto_quant/uncertainty.py`: deterministic paired block-bootstrap analysis.
- `src/crypto_quant/multiple_testing.py`: pinned Deflated Sharpe selection-burden audit.
- `src/crypto_quant/forward_review.py`: predeclared read-only forward-paper review gates.
- `src/crypto_quant/paper.py`: append-only forward-paper sessions with hash-chain audits.
- `AGENTS.md`: agent roles, decision taxonomy, and audit rules.
- `experiments/`: generated runs and the experiment ledger.

## Research Rules

1. A signal computed at bar close may only execute at the next bar open.
2. Strategy parameters are declared before evaluation; this repository does not
   search parameters to improve the reported result.
3. Every result includes fees and slippage.
4. Full-sample results never replace out-of-sample evidence.
5. Failed or fragile strategies remain in the ledger as evidence.
6. Execution feasibility is reviewed with dollar-volume and turnover context.
7. New experiment records include a deterministic source manifest when their
   module supports it; older records are retained with that limitation visible.
8. Future metric-based studies are preregistered in
   `crypto_quant.research_contracts`; parameters and failure rules are fixed
   before the local history is downloaded.
9. Period metrics include the causal account value immediately before the
   evaluation boundary, so the first selected bar's return, cost, and drawdown
   are not dropped.
10. Forward-paper sessions freeze both parameters and the complete historical
    signal prefix; a rule change requires a new session rather than rewriting an
    existing one.

Run the local multi-agent review at any time:

```bash
make research-brief
```

Before downloading metrics for `RC-20260822-01`, freeze its prior trial set:

```bash
make export-rc01-trials
```

After an authorized metrics refresh, run the contract unchanged:

```bash
make oi-donchian-study
```

Run the separately registered positioning diagnostic with:

```bash
make positioning-event-study
```

Replay locally captured order-book events against hypothetical requests with:

```bash
make replay-order-book \
  ORDER_BOOK_EVENTS=market_data/raw/order_book/events.csv \
  ORDER_BOOK_REQUESTS=market_data/raw/order_book/requests.csv
```

Build a machine-readable strategy library from the current evidence graph:

```bash
make strategy-library
```

Run an offline local-data quality audit:

```bash
make data-quality
```

Audit the local evidence graph without network access:

```bash
make system-audit
```

Every system-audit directory is SHA-256 sealed immediately after its report and
machine-readable results are written, so the audit evidence is tamper-evident too.

Create SHA-256 manifests for existing research artifacts:

```bash
make seal-artifacts
```

Inspect the surviving rule's return sources without changing its parameters:

```bash
make strategy-autopsy
```

## Current First Study

The corrected first study with cross-symbol risk status is in
`experiments/runs/20260821T171737Z_first_spot_study/report.md`. A later run may
become authoritative as the system gains futures, walk-forward, and stress
checks. The first same-name run is invalidated in its directory because of a
sale-accounting bug.

The current robustness diagnostic is
`experiments/runs/20260821T175617Z_spot_robustness_study/report.md`. It reports
every evaluated parameter neighbor rather than promoting a best curve. Two
earlier incomplete/misleading robustness directories retain `INVALIDATION.md`
audit notes.

The current execution-stress diagnostic is
`experiments/runs/20260821T181002Z_spot_execution_stress/report.md`. It reports
all 60 settings per rule after correcting a cost-table aggregation issue.

The current Donchian return-source diagnosis is
`experiments/runs/20260821T183542Z_donchian_96_48_autopsy/report.md`.

The current Donchian BTC/ETH equal-sleeve portfolio study is
`experiments/runs/20260821T190008Z_donchian_btc_eth_portfolio/report.md`.

The current portfolio parameter-neighborhood diagnostic is
`experiments/runs/20260821T213636Z_donchian_portfolio_neighborhood/report.md`.
It reuses the previously fixed Donchian neighbors and reports every joint
BTC/ETH variant without selecting a replacement rule.

The current portfolio execution-stress diagnostic is
`experiments/runs/20260821T191133Z_donchian_portfolio_stress/report.md`.

The current portfolio liquidity-capacity diagnostic is
`experiments/runs/20260821T201508Z_donchian_portfolio_capacity/report.md`. It
uses a shifted 30-bar quote-dollar-volume median and reports every participation
cap rather than choosing a favorable threshold.

The current dynamic capacity-execution replay is
`experiments/runs/20260821T235627Z_donchian_portfolio_capacity_replay/report.md`.
It tests every account-size/participation-cap cell with partial fills, reports
tracking versus the unconstrained path, and estimates the commission that drives
out-of-sample return to zero.

The current portfolio calendar-fold diagnostic is
`experiments/runs/20260821T202757Z_donchian_portfolio_walk_forward/report.md`.
It keeps the rule fixed, reports every year, audits expanding-prefix signal
reproduction, and explicitly does not claim fresh out-of-sample confirmation.

The current portfolio market-state diagnostic is
`experiments/runs/20260821T211126Z_donchian_portfolio_regimes/report.md`. It
uses prior-bar trend, volatility, and benchmark-drawdown states and reports all
sparse or unfavorable cells.

The current portfolio bootstrap diagnostic is
`experiments/runs/20260821T214817Z_donchian_portfolio_bootstrap/report.md`. It
uses deterministic paired block resampling and reports both one-week and
three-week block scenarios without selecting the more favorable curve.

The current portfolio drawdown-attribution diagnostic is
`experiments/runs/20260821T223047Z_donchian_portfolio_drawdowns/report.md`.
It reports every basket-drawdown episode of at least 20%, including failed
short V-shaped episodes and the still-open 2025–2026 episode.

The current portfolio diversification diagnostic is
`experiments/runs/20260821T225616Z_donchian_portfolio_diversification/report.md`.
It reports the declared 50/50 mix, four symmetric allocation neighbors, and
rolling BTC/ETH correlation diagnostics without selecting a better-looking mix.

Compare the surviving rule across BTC and ETH sleeves:

```bash
make portfolio-study
```

Estimate first-order liquidity capacity without changing the portfolio rule:

```bash
make portfolio-capacity
```

Run the fixed-rule calendar-fold check:

```bash
make portfolio-walk-forward
```

Evaluate the already-fixed portfolio neighborhood:

```bash
make portfolio-neighborhood
```

Diagnose causal market states without changing the rule:

```bash
make portfolio-regimes
```

Quantify out-of-sample sampling uncertainty:

```bash
make portfolio-bootstrap
```

Apply the pinned multiple-testing correction:

```bash
make portfolio-multiple-testing
```

Attribute all declared basket-drawdown episodes:

```bash
make portfolio-drawdowns
```

Replay constrained execution across account sizes and participation caps:

```bash
make portfolio-capacity-replay
```

Run hypothetical cross-margin and liquidation stress without changing the
declared spot rule:

```bash
make portfolio-margin-stress
```

Test rolling diversification and declared sleeve-mix neighbors:

```bash
make portfolio-diversification
```

The latest integrity audit is written under `experiments/audits/`. It verifies
all sealed artifact manifests, the paper hash chain, and the append-only ledger
prefix. Missing funding history remains a high-severity research gap.

Initialize an append-only forward-paper session from local data:

```bash
make paper-init
```

Convert the genesis session to append-safe artifact auditing:

```bash
make paper-append-safe
```

After an authorized market-data refresh adds future complete bars, process them
without replaying history:

```bash
make paper-update
```

Review the immutable session against the predeclared forward gates:

```bash
make paper-review
```

The current session is
`experiments/paper_sessions/20260821T205201Z_donchian_portfolio_paper`. It is
flat at `$10,000`; both 50% sleeve signals are pending for the next complete
bar. No forward-performance conclusion is possible until bars accumulate.
Its manifest now anchors the genesis event prefix: later updates may rotate the
manifest, but rewriting or truncating processed events remains a hard failure.
