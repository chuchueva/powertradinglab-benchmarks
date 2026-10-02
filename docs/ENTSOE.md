# ENTSO-E Transparency Platform — everything about our one source

Companion to `METHOD.md`. The split:

- **METHOD** holds OUR rules — things we chose and can change.
- **This file** holds facts about THE SOURCE — how the platform behaves, what it
  publishes, where the gaps are, and what its operators have told us. We do not
  get to decide any of it; we can only observe it and write it down.

Kept apart because they age differently. A rule of ours is revised by argument; a
fact about ENTSO-E is revised by evidence, and usually arrives as an email.

## Licensing and attribution

- Source: ENTSO-E Transparency Platform only. API key in `.env` (never committed),
  masked everywhere — it travels as a URL query parameter, so it leaks through
  exception text unless masked at the point of the request.
- **Imbalance (A85)**: owner = TSO → covered by the ENTSO-E CC-BY 4.0 reuse list.
  Free to use with attribution.
- **Day-ahead (A44)**: owner = Power Exchange / NEMO → **NOT** in the CC-BY list.
  The single DA auction (Euphemia/SDAC) computes prices centrally and NEMOs
  publish the result; for NL the publisher seen in the data is Nord Pool
  (EIC `10X1001A1001A450`).
  We intend to inform the exchanges of the derived-benchmark use, cite the
  sources, and adjust if asked.
- We publish **derived results, never raw DA price series**. Attribution to
  ENTSO-E and to the exchanges appears in every published file.

## How the API behaves

Structure facts, most of them learned the hard way.

- **DA**: `documentType=A44`, `in_Domain` == `out_Domain`, response is BARE XML,
  root `Publication_MarketDocument`, price field `price.amount`.
- **IM**: `documentType=A85`, a single `controlArea_Domain`, response is a ZIP
  (detect the `PK` signature, unzip in memory), root `Balancing_MarketDocument`,
  price field `imbalance_Price.amount`. Dual price arrives as per-point
  `imbalance_Price.category`: **A04 = long/surplus, A05 = short/deficit**.
- **Period bounds `YYYYMMDDHHMM` are read as UTC, not market-local.** A market day
  is anchored to local midnight, so `period.py` converts local midnight to its UTC
  instant before formatting.
- **A44 and A85 disagree about the same window.** A85 returns exactly the
  requested UTC window; A44 snaps to whole CET delivery days overlapping it, so a
  UTC bound poking into the next delivery day used to return an extra day.
  Emitting the UTC instant of local midnight makes both return exactly one day.
- **`<createdDateTime>` is regenerated on every request** — the platform assembles
  the document on the fly. Two pulls minutes apart carry different values with
  identical prices. It tracks OUR fetch time, not a republication event, and is
  **not** a finality flag.
- **`curveType A03` is a variable-sized block**: a position is published only when
  the value changes, a skipped position repeats the previous one, and the last
  published point holds to the end of the grid. The parser must forward-fill by
  position — and for IM, **each category independently**, because they skip
  different positions.
- Never hardcode 96 points a day: derive the expected count from interval and
  resolution (DST days give 92 or 100).
- Sender role A32 = market operator, the real publisher. Receiver role A33 is
  header boilerplate and says nothing.
- **"No data" is HTTP 200, not an error.** Asked for a period it holds nothing
  for, the platform returns an `Acknowledgement_MarketDocument` with a `Reason`:

      code 999
      No matching data found for Data item IMBALANCE_PRICES_R3 [17.1.G]
      (10YCS-CG-TSO---S) and interval 2026-07-15T00:00:00Z/2026-07-16T00:00:00Z.

  It has no `<type>` element, so an unguarded parser reports `Expected
  documentType A85, got None` — which reads as OUR bug. It cost one wrong
  diagnosis. `src/entsoe_ack.py` now raises `NoDataAvailable` carrying the
  platform's own code and text.
- Known glitches: A44 sometimes needs `businessType=A62` to avoid HTTP 400
  (Jan 2026). The NL feed had a full outage in Feb 2026. Scheduled maintenance
  returns an 18 KB styled HTML page — "Service Temporarily Unavailable" — seen
  2026-08-06.

## How the platform publishes — timing

Measured, not looked up. **ENTSO-E sets no deadline for first publication**: the
wording is "as soon as possible; intermediate figures are published if final
prices are not available". So the delay is an empirical quantity.

- **Imbalance for a completed day is on the platform by ~05:30 UTC the next
  morning** — observed on five consecutive days, 6–10 August 2026. On 2026-08-05
  at 08:35 UTC the whole of the previous day was present and none of the day in
  progress. **NL does not publish imbalance intraday at all**; it arrives as a
  batch for the finished day. That is why one recompute a day is enough.
- **Revisions run until roughly the 15th of the month following delivery** (market
  practice, Irina). We wait until the 20th for the as-final pass — five days of
  margin cost nothing, catching a still-moving value costs a wrong number.
- The platform exposes **no intermediate/final flag**. The difference between
  as-known and as-final is found by re-fetching and comparing, never by reading a
  field.

## Resolution is NOT uniform across zones

Our own notes used to state that DA became natively 15-minute for all European zones on
2025-10-01 (the SDAC MTU change). **That is not true**, and the correction came
from a TSO directly (see the Albania reply below): some zones still publish
hourly day-ahead prices and have not implemented 15-minute products.

Our code never relied on the claim — resolution is read per Period from the
document and upsampled accordingly — so nothing broke. Only the documentation
was claiming more than the data supports. Do not reintroduce the assumption.

## Data gaps — the catalogue

Built from `modelling/coverage.py` over the File Library archive 2020-01-01 →
2026-07-25, then confirmed against the live API (which answers with the code-999
acknowledgement above, i.e. the gap is at the source, not in our copy).

**Day-ahead**: 360 missing timestamps across 2 areas (AL, MK). See `da_gaps.csv`.

**Imbalance**: 300 000+ missing timestamps across 32 areas. See `im_gaps.csv`.

| area | EIC | missing timestamps |
|---|---|---|
| ME | 10YCS-CG-TSO---S | 83 889 |
| GR | 10YGR-HTSO-----Y | 71 625 |
| LT | 10YLT-1001A0008Q | 63 565 |
| AL | 10YAL-KESH-----5 | 38 596 |
| DE_LU | 10Y1001A1001A82H | 9 608 |
| LV | 10YLV-1001A00074 | 8 |
| EE | 10Y1001A1001A39I | 7 780 |
| DK2 | 10YDK-2--------M | 4 156 |
| DK1 | 10YDK-1--------W | 4 155 |
| FR | 10YFR-RTE------C | 3 448 |
| CH | 10YCH-SWISSGRIDZ | 3 300 |
| SK | 10YSK-SEPS-----K | 2 780 |
| UA_IPS | 10Y1001C--000182 | 1 357 |
| SE1 | 10Y1001A1001A44P | 644 |
| SE2 | 10Y1001A1001A45N | 636 |
| SE3 | 10Y1001A1001A46L | 624 |
| SE4 | 10Y1001A1001A47J | 624 |
| NO1 | 10YNO-1--------2 | 620 |
| NO2 | 10YNO-2--------T | 616 |
| NO5 | 10Y1001A1001A48H | 614 |
| NO3 | 10YNO-3--------J | 612 |
| NO4 | 10YNO-4--------9 | 612 |
| MK | 10YMK-MEPSO----8 | 576 |
| FI | 10YFI-1--------U | 332 |
| IT-NORTH | 10Y1001A1001A73I | 324 |
| IT-CNORTH | 10Y1001A1001A70O | 228 |
| IT-CSOUTH | 10Y1001A1001A71M | 228 |
| IT-Sardinia | 10Y1001A1001A74G | 228 |
| IT-Sicily | 10Y1001A1001A75E | 228 |
| IT-SOUTH | 10Y1001A1001A788 | 228 |
| IT-Calabria | 10Y1001C--00096J | 228 |
| PT | 10YPT-REN------W | 192 |
| RO | 10YRO-TEL------P | 1 |

Notable shapes, because a count alone hides them:

- **ME** — complete through 2025-07, then nothing. Still missing as of 2026-07.
  An ONGOING gap, which is why it made the strongest case for a ticket.
- **AL** — complete through 2024-04, absent 2024-05 … 2025-03, two broken months
  (2 rows in 2025-04, 144 rows over 7 days in 2025-05), complete from 2025-06.
- **NO1–NO5, SE1–SE4, DK, FI, EE** — the Nordic zones only began delivering
  imbalance to ENTSO-E around 2021; NO1's first timestamp is 2021-10-31. Their
  "2021" is two months, not twelve.
- **A handful of rows is not presence.** Some months hold 2 or 4 rows. A check
  for "is the area code in the file?" passes them; counting intervals does not.
  This is why coverage is measured by interval count.

## Zero-filled price legs — Italy and Slovakia

Found 2026-08-28. `NegativeImbalancePrice` is published as a literal `0` for whole
months in two places. Not our parser: the raw CSV contains the string `0`, and
our only conversion (`to_numeric(errors="coerce")`) turns an empty field into
NaN, never into zero.

**Zero IS a legitimate price** — €0/MWh happens, and in NL it is real (both legs
are populated there, 0% zeros). What is not a price is one leg sitting at exactly
0.00 for every quarter-hour of a month while the other runs at 150–280 €/MWh.

| zone | leg | from | pattern |
|---|---|---|---|
| IT-NORTH, IT-CNORTH, IT-CSOUTH, IT-SOUTH, IT-Sicily, IT-Sardinia, IT-Calabria | Negative | **2022-04-01** | 8% zeros in 2022-03, then **100.00%** every month since — an abrupt, clean switch |
| SK | Negative | **2024-07** | gradual: 0% in 2024-06, 44% in 07, 88% by 11, 96% by 2025-02; medians of 0.95 and 1.00 in the transition months |

Everything else is clean: a scan of March and September of every year 2020-2026,
all zones, found no other zone with a leg above 90% zeros (BA 2021-03 and
CY 2026-03 are single months, and both zones are excluded anyway).

**Why it matters beyond the prices.** The benchmark takes `buy = max(pos, neg)`
and `sell = min(pos, neg)`. A zero leg makes `sell = 0`, i.e. "imbalance can be
sold at nothing", so `always_buy` is hugely negative every PTU and `ideal` goes
negative for whole years. That is the anomaly we first recorded as "dual zones
with ideal ≤ 0 for whole years: ... all Italy" — not a punishing market design,
an unused column.

It also corrupts the dual/single classification, which is derived from
`pos != neg`: a zero leg reads as a second price. **Our earlier "SK: single →
dual (2024)" is therefore suspect** — the dual signal begins in the same month the
zeros do.

Italy is unambiguous. Slovakia is not: a drift towards zero with intermediate
medians near 1 could equally be a genuine change in market design. Which is why
the question goes to the source rather than to reasoning.

### Ticket text to send

Two tickets, one per data provider.

> **Subject: Negative Imbalance Prices published as zero for IT zones**
>
> For all seven Italian bidding zones (IT-NORTH, IT-CNORTH, IT-CSOUTH, IT-SOUTH,
> IT-Sicily, IT-Sardinia, IT-Calabria), the field NegativeImbalancePrice in
> Imbalance Prices [17.1.G] is published as exactly 0 for 100% of MTUs from
> 2022-04-01 onwards, while PositiveImbalancePrice carries normal values
> (150–280 EUR/MWh). Before that date the field carried ordinary values (in
> 2022-03 only 8% of MTUs were zero, median 294 EUR/MWh).
>
> Could you confirm whether this is intended — for example because the Italian
> imbalance settlement uses a single price and the second field is not
> applicable — or whether the data is missing? If it is intended, which field
> should be read as the imbalance price for these zones?

> **Subject: Negative Imbalance Prices trending to zero for SK area**
>
> For SK (10YSK-SEPS-----K), NegativeImbalancePrice in Imbalance Prices [17.1.G]
> was populated normally until 2024-06 (0% zeros, median 63.58 EUR/MWh) and then
> moved towards zero: 44% of MTUs in 2024-07, 88% in 2024-11, 96% by 2025-02,
> with medians of about 1 EUR/MWh in the transition months.
>
> Could you confirm whether this reflects a change in the Slovak imbalance
> settlement design, or a change in how the field is populated?

## Correspondence log

**2026-08-18 — first contact.** One email describing both gap sets, with
`da_gaps.csv` and `im_gaps.csv.zip` attached. Ticket **#9466490**.

**2026-08-18 — ENTSO-E reply.** One data provider can be involved per ticket, so
a separate ticket per area is required. Suggested subject format:
`Missing DA Prices for XX area` / `Missing Imbalance Prices for XX Area`, sent to
`transparency@entsoe.eu`. Initial ticket closed.

**2026-08-19 — three DA tickets** raised (AL, MK and one more).

**2026-08-23 — nineteen IM tickets** raised. Zones sharing a TSO were merged into
one ticket each: NO1–NO5, SE1–SE4, and the Italian zones. **UA_IPS and LT were
not covered** and still need tickets.

**2026-08-25** — account registered on the Transparency Platform Helpdesk;
roughly twenty open tickets are easier to follow there than by email.

**2026-09-01 — NL, platform outage.** Ticket **#9468341**, one ticket for two
observations: the API answering HTTP 503 `Scheduled maintenance` continuously
since 08-31 09:30 UTC (first seen 08-30), and Imbalance Prices [17.1.G] for NL
absent from 08-31 on the web interface while imbalance VOLUMES were published
normally. Same-day reply: **both are acknowledged platform-wide incidents**,
each with its own announcement in TP News — a delay in data publication, and an
API service disruption. No ETA; a **migration to new infrastructure scheduled
02/09** is expected to help, with a fix attempted before it.

What that settles for us: the missing NL imbalance prices are the PLATFORM's,
not TenneT's — so the thing to wait for is the platform, and no per-TSO ticket
is needed. Worth remembering as a diagnostic: "volumes present, prices absent"
looked like a data-provider fault and was not.

The migration is itself a risk to watch. A change of infrastructure is exactly
when an API client breaks in a new way — host, path, certificate, token
handling, rate limits, or the shape of the response. And our input gate checks
that a day is COMPLETE, not that it is SANE: 96 plausible-looking PTU of wrong
content would pass it. Read the first successful poll after the migration by
eye, not by row count.

**2026-09-14 — the incident is declared closed.** TP News: the platform is
stabilised, the publication backlog and the API performance problems resolved,
temporary API limits to be lifted 15/09 ("1 year of data with a single request")
and website export limits 16/09. Subscription messages may still lag.

Ticket **#9468341** therefore needs no chasing, and no new ticket was raised
during the whole episode — deliberately, the queue of open ones is long enough.
What the episode cost us, recorded so the next one can be compared against it:

- a **four-day outage** (the migration of 09-02 was rolled back; the second
  attempt on 09-07 held),
- **five manual backfills**,
- **six refused forecasts** (09-02, 09-03, 09-09, 09-10, 09-15, 09-17) — the
  input gate working as designed, plus 09-16 issued under the older one-day rule
  and deliberately left as it is,
- **one permanently missing day**: DA for delivery day **2026-09-13** was never
  published, Acknowledgement 999. That day will never be scored. Not a gap in
  our pipeline and not repairable by re-fetching.

The diagnostic worth keeping: for five days "the source is quiet" and "we are
broken" looked identical from here, and the news feed was the only thing that
told them apart — read by hand, because nothing in the run reaches it. That is
still the case (queue item: the news feed into `status.xml`).

### Albania (DA) — answered

The TSO submitted the missing data for 01.04.2024, 02.04.2024 and 27.08.2025,
confirmed there were no other gaps, and stated that 15-minute products are not
yet implemented in its system.

ENTSO-E's gloss: Albanian day-ahead prices are published at 60-minute resolution
and the switch to 15 minutes has not happened yet.

Two things follow. The gap was real and is now filled — which means this exercise
works. And our earlier resolution assumption was wrong (see above).

### Germany/Luxembourg (IM) — backfilled

**2026-08-28** — notified that the missing data has been filled. 9 608 timestamps,
the second-largest IM gap after ME, GR, LT and AL. Not yet verified on our side.

### Policy: one round of tickets, then a cut-off

Tickets are filed once. Gaps that remain unfilled when the report is written stay
unfilled, and the report says so — refiling indefinitely would mean never
publishing. **Consequence to handle deliberately:** the report must state the
date at which the data was taken, because the same query a month later gives
different numbers as backfills land.

### Backfills make OUR archive stale

When a TSO fills a gap, the fix lands on the platform — not in
`data/archive/*.zip`, and not in `history_*.feather`. Both still hold the holes.

So before the report:

1. re-download the monthly extracts for the affected zones and months,
2. rebuild the history cache (`uv run python src/build_history.py all`),
3. re-run `modelling/coverage.py` and compare against the previous table,
4. note that `yearly_fixed.csv` stops being the reference for any zone whose data
   changed — it was frozen against the pre-backfill archive.

Step 4 is the one that bites quietly: the validation in `modelling/validate_as_of.py`
asserts the report path still reproduces `yearly_fixed.csv`, and it SHOULD start
failing for refreshed zones. That failure is correct, not a regression, and the
reference has to be re-frozen deliberately.

## Open questions

- **LT, imbalance missing since 2024-09 and never resumed.** 63 565 timestamps.
  Hypothesis worth checking rather than assuming: the Baltic balancing
  arrangements changed around the synchronisation with continental Europe, and
  the data may now be published under a different area or a joint Baltic one. A
  ticket has not been raised yet.
- **UA_IPS** — 1 357 missing timestamps, no ticket yet.
- Whether the backfills promised in the open tickets actually land, and how long
  they take. Worth recording per zone: it tells us which sources can be relied on
  when choosing which zones to add.
