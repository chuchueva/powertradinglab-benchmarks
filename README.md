# PowerTradingLab benchmarks

Daily trading benchmarks for European short-term power markets: was this market
worth trading, and is it still?

Every day the service takes day-ahead and imbalance prices from the ENTSO-E
Transparency Platform, publishes a decision for the next delivery day before the
day-ahead gate, grades every finished day against five benchmarks, and freezes
the result in a file that is never rewritten.

- Site and data: <https://powertradinglab.org>
- Report 2020–2026: see the site

Live zones: NL, RO. Values are EUR per 1 MW, summed per delivery day.

## Where to read

| file | what |
|---|---|
| `docs/METHOD.md` | the rules and the benchmark definitions |
| `docs/MAP.md` | the code on one page: the flow, who owns what, the tables |
| `docs/ENTSOE.md` | facts about the source: licensing, API behaviour, data gaps |
| `tests/` | each test docstring says WHY a rule exists |

## Running it

Python 3.13, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env            # put your ENTSO-E API token here
uv run pytest -q
```

`settings.json` holds every tunable number: zones, run times, scoring windows,
publication versions. The scheduled chain is `scripts/run.sh`; `scripts/make_systemd.py`
installs it as systemd timers.

## Licence and data

Code: MIT, see `LICENSE`.

Data: derived from the ENTSO-E Transparency Platform. Published files carry
derived values only, never day-ahead or imbalance prices; obtain prices from
ENTSO-E under its terms. Exchange rates: European Central Bank.
