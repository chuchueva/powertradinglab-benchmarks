# Method — the rules and the benchmarks

`MAP.md` shows where things live. This file states the rules the code obeys and
what each published number means. Each rule is guarded by a test in `tests/`;
the test docstrings say why the rule exists.

## The nine rules

1. **UTC only, tz-aware, everywhere.** Market-local time exists in one file,
   `src/period.py`, and even there what leaves is a UTC instant. A market day
   runs from local midnight to local midnight: 23, 24 or 25 hours.
2. **Two time entities.** `delivery_ts` is the 15-minute period (PTU) a value is
   about; `knownby_ts` is when the value became known. The as-of cursor
   `current_ts` is not a third kind of time: in our own outputs it IS the
   `knownby_ts`. A price the source publishes per hour (or half-hour) is
   written to each quarter-hour it covers, with the same value: the same
   product, and the same sum over 1 MW.
3. **`as_of` is two steps:** keep rows with `knownby_ts < current_ts`, then keep
   the newest `knownby_ts` per key. Forgetting the second step is invisible
   until a value has two versions, and then it silently corrupts.
4. **`as_of` or `shift(2)`, never both.** They express the same causality;
   stacked, they lag the signal twice.
5. **The store keeps versions; collapsing happens at read time.** That is what
   lets the code ask what was known at any past instant.
6. **Only changes are written.** `knownby_ts` therefore reads as "the first time
   we saw this value", and as-known / as-final are the first and the last row.
7. **Currency is read from the document, never assumed.** EUR is a unit, not a
   rate: 1.0, never looked up. Other currencies are converted with the central
   bank reference rate of the day (ECB; NBU for UAH). Every rate carries its
   own `knownby_ts`, so a past decision can be replayed with the rates it had.
8. **A delivery day is graded exactly twice:** as-known, when it first settles
   complete; as-final, after the 20th of the following month.
9. **Never invent data.** No half-published hour used as whole, no decision
   defaulted when the signal window has a hole (`<NA>` instead), no exchange
   rate carried past a date whose own rate was due. One stated exception: in
   the live decision, today's already-delivered hours are valued at the last
   rate known at that moment — the rate a trader actually has at 09:40 UTC,
   before the ECB publishes today's. A gap is information.

Two rules for what leaves the machine:

- **A published file is frozen.** It is written once and never rewritten; a
  correction is a new file beside it. Every file and every row carries the
  `schema_version` (shape) and `methodology_version` (numbers) in force when it
  was computed, and history is never re-rendered under a newer version.
- **Derived values only.** No day-ahead or imbalance price is republished, in
  the archive or in the summary files.

## The benchmarks

All values are EUR per 1 MW, per PTU (DA−IM price difference × 0.25 h), summed
over the delivery day. Position: bought or sold on the day-ahead auction (DA),
settled in imbalance (IM). In dual-pricing zones `buy = max(pos, neg)` and
`sell = min(pos, neg)` of the two imbalance prices.

| published name | internal column | definition |
|---|---|---|
| `passive_da_buy_value` | `always_buy` | buy on DA every PTU, settle in IM; no foresight |
| `passive_da_sell_value` | `always_sell` | sell on DA every PTU, settle in IM; no foresight |
| `extractable_value` | `ideal` | the better of the two passive positions per PTU, with perfect hindsight; can be negative in dual-pricing zones |
| `adverse_value` | `devil` | the worse of the two per PTU; the exact mirror of `extractable_value` in single-pricing zones |
| `trailing_bias_value` | `action_pnl` | the side chosen per UTC hour from the mean of the two passive values over the trailing window (below), using only data known before the DA gate; empty where no decision was issued |

Per PTU, `extractable + adverse = passive_buy + passive_sell` exactly;
`tests/test_summary.py` checks it.

## The decision and the gate

The trailing-bias decision for delivery day D is made at 09:40 UTC on D−1, before
the 12:00 (Europe/Brussels) day-ahead gate.

**The window is tied to timestamps**, per delivery hour h, with `n_days`
observations (default 3):

- `d1` — the same UTC hour on the freshest market day where it is **whole**
  (all four quarter-hours) and known before the decision: today (D−1) if the
  zone has already published it, otherwise yesterday (D−2). Never older.
- `d2`, `d3` — exactly one and two days before `d1`, never "the next one
  available".
- the signal: buy on DA if mean(`passive_da_buy`) > mean(`passive_da_sell`)
  over `d1..dn`, else sell.

A hole anywhere in the window — a missing or partly published hour — gives
that hour no action. On the 23-hour spring day one UTC hour does not exist in
yesterday's market day at all; for that hour only, the day before is used.
An hour without action carries its reason (`skip_reason`): `hour_incomplete`
for some hours of the day, `input_incomplete` when no hour could be decided.
A run after the gate never issues a decision.

The backtest and the report use the same rule with `d1` always yesterday,
because the archive does not say when a value was first published. Live can
be fresher than the backtest; it is never staler.
