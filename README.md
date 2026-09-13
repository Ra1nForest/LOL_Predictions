# LoL Match Prediction

A staged probability model for professional League of Legends matches (LPL / LCK / LEC / LCS,
2022–2026, 8,453 games from Oracle's Elixir), served as a calibrated live API.

**The interesting part is not the accuracy. It is the measurement apparatus.**
Fifteen hypotheses were put through a paired walk-forward gate; eleven were rejected,
including several that looked obviously correct beforehand. The model also spent its first
version producing probabilities that scored *worse than a constant predictor* on Brier score
— a failure that accuracy alone could not have detected.

---

## What it does

Four stages, each usable on its own, each with a different information set:

| Stage | Available when | Features | Accuracy | Baseline |
|---|---|---|---|---|
| 1 `pre_draft` | before the game | 79 | 59.9% | 53.3% |
| 2 `post_draft` | after champion select | 112 | 61.1% | 53.3% |
| 3 `debate` | optional | — | *advisory only, never changes the number* | |
| 4 `ingame` | during the game | 28 | 73.7% (80.5% at 25 min) | 56.3% |

Calibrated output ranges: `[0.18, 0.84]` pre-game, `[0.12, 0.87]` in-game.
Those bounds are a real limitation and the UI draws them explicitly — see
[Calibration](#calibration-was-the-single-largest-improvement).

---

## Quick start

```bash
pip install -r requirements.txt

python fetch_data.py       # pull the season CSVs into data/ — see the caveat below
python train.py            # Stage 1 + 2  -> artifacts/
python ingame_train.py     # Stage 4      -> artifacts/

uvicorn api:app --port 8000
```

Open `http://localhost:8000/` for the UI, `/docs` for the OpenAPI console.

For a persistent deployment see [Deployment](#deployment).

### Getting the data is the fragile step

Oracle's Elixir distributes **only** through a public Google Drive folder — the
downloads page says so in as many words — and updates the files once per day.
`fetch_data.py` lists that folder and pulls the seasons you ask for.

Anonymous downloads from that folder are subject to a quota that is, in
practice, frequently exhausted: on 2026-08-16 all four sampled files returned a
page titled *"Google Drive - Quota exceeded"*, including the 2014 file nobody
downloads — so the limit is on the owner's account, not on any file's
popularity. It normally clears within a day; `--retry N` backs off by the hour.

**That error page arrives as HTTP 200 with an HTML body.** Written naively into
`data/`, a 2 KB web page replaces a 58 MB season — and `feature_store.py` skips
CSVs it cannot read *silently*, so a "successful" update could quietly drop a
year from training. This is the same failure shape as the training-serving skew
bug below. So `fetch_data.py` streams to a temp file, checks that the first
bytes are the real column header and that the result is at least a megabyte,
and only then does an atomic replace. A failed fetch leaves the existing file
untouched and exits non-zero.

Anonymous fetching shares the *owner's* quota, so in practice you need your own
Google credentials:

```bash
# once: create a Google Cloud project, enable the Drive API, make a Desktop
# OAuth client, save the JSON as google_credentials.json, then
python fetch_data.py --auth
```

Scopes requested are `drive.readonly` and `drive.file` — read access, plus the
ability to touch only files this program itself creates. That second scope
exists for the fallback: an authenticated download of *someone else's* file
still has a quota, while a file you own has none, so on a quota error the
script copies the file into your Drive, downloads its own copy, and deletes it.
`google_credentials.json` and `google_token.json` are credentials, are listed
in `.gitignore`, and are never printed.

---

## Methodology

This is the part worth reading.

### The paired walk-forward gate

`research/harness.py` runs expanding-window time-series validation across multiple folds and seeds.
The design question it answers is: **how large does an improvement have to be before I am
allowed to believe it?**

Two independent sources of noise, which behave differently:

```
seed SD  ≈ 0.014   the model's own estimation variance   <- more training data helps
fold SD  ≈ 0.023   how hard each validation window is    <- only a larger window helps
```

The second one is the trap. It is a property of the *validation window*, not of the model,
so adding training data does not reduce it. Going from two years with a 7% window to five
years with a 15% window moved fold SD from 0.104 to 0.023 — the improvement came from the
window, not the data.

Comparisons are **paired**: subtract fold-by-fold rather than comparing means. This cancels
"how hard was this period" and is roughly 5× more sensitive at the same sample size.

```
t = Δ / (paired SD / √n)          accept only when |t| > 2.5
```

One early mistake worth recording: I initially used `ratio = |Δ| / SD` as the criterion.
That is an effect size — it does not improve with `n`, so it can never accumulate evidence.
It is the wrong tool for a decision rule.

### Gate record: 4 accepted, 11 rejected

**Accepted**

- Draft features — `t = −9.52` when removed, 8/8 folds worse, worth ~2.1pp. Re-run on
  2026-08-16 against 8,453 games; the previously recorded figure was `t = −4.50` at
  Δ = −0.031. The point estimate moved to −0.0444, further than 2.3% more data explains
  on its own — the validation windows also rolled forward (fold 8 now ends 2026-08-10),
  and Δ is a property of the window, not only of the model. Four configurations now:
  −0.0325, −0.0310, −0.031, −0.0444.
- Five years of data over two — `t = 2.66`, monotone across four configurations.
- In-game snapshots — the single largest jump, 64% → 73%.
- GD@15 over pure pre-game features — `t = 4.60`.

**Rejected**

- Time-decay weighting (three half-lives) and rolling training windows. Data volume beats
  recency, decisively.
- Patch-limited champion win rates (3 / 6 / 10 patches). The 6-patch variant was reliably
  *worse*.
- Series momentum — correlates 0.71 with `streak`; redundant.
- `blue_firstpick` decontamination (three approaches).
- Symmetrisation at prediction time.
- Hand-written champion-scaling interaction terms — tree models construct interactions
  themselves; the explicit products added nothing.
- The "comeback risk" hypothesis — 468 games, observed gap −0.032 ± 0.089. Inside the noise.
- Champion pair synergy — see below.

### Split-half reliability

Before asking whether a feature helps, ask whether the underlying quantity can be measured
repeatably at all. Split the data in half, compute the statistic on each, correlate.

```
single-champion win rate       r ≈ 0.25–0.36   <- the ceiling at this data volume
champion scaling index         r ≈ 0.05–0.11   <- the leaderboard looks sensible; it is noise
champion pair synergy          r ≈ 0.00        <- 695 pairs, unmeasurable
```

Pair synergy was killed here rather than by the gate, which is cheaper and more conclusive.

**But low individual reliability does not mean the feature is useless.** Draft win-rate
features are built on `r ≈ 0.25` estimates and are nonetheless load-bearing (`t = +3.45`
alone, `t = +5.61` on top of count features). The model consumes a five-champion mean and a
between-team difference — aggregation reduces the noise by √n. Reliability tells you whether
a single estimate is trustworthy; it does not directly tell you whether a derived feature
carries signal.

### Calibration was the single largest improvement

The first version's probabilities were, as probabilities, worse than useless:

```
                Brier
uncalibrated    0.2536
constant        0.2495   <- predicting the base rate every time beat the model
Platt           0.2276   ECE 0.119 -> 0.021
```

Isotonic regression lost — not enough calibration data, it overfit itself.

**Update 2026-08-24: that verdict was re-tested and now splits by stage.** The reason it lost was
never the method, it was the sample size — and the in-game model's calibration set has since grown
about twentyfold.

```
                       calibration set   Brier         verdict
Stage 4  in-game        ~35,000 snaps    t = +6.91     isotonic, shipped
Stage 1  pre_draft           1,716 games t = -4.74     Platt kept (6/6 folds worse)
Stage 2  post_draft          1,716 games t = -1.97     Platt kept
```

Same change, opposite conclusions, decided separately rather than generalised
(`research/gate_calibration.py`, `research/gate_calibration_pre.py`). On Stage 4: ECE 0.0351 →
0.0160, range [0.06, 0.94] → [0.01, 0.99], accuracy unchanged at 77.2% → 77.3% (a monotone
transform cannot reorder anything).

What made it worth re-opening was a diagnosis, not a hunch. Platt can only draw one fixed S-shape,
and measurement showed it bulging in the middle and collapsing at both ends:

```
gold lead     model said   actually    after isotonic
  < -6k          7.0%        2.7%          2.1%
  1k~2k         78.0%       74.6%         75.2%
  > +6k         93.3%       98.0%         97.8%
```

At 6k+ the model's **raw** output averaged 97.8% against a measured 98.0% — near perfect. Platt was
squashing a correct judgement down to 93.3%. Weighted mean absolute deviation across all buckets
fell 58%. The clipped-range defect described below is therefore largely fixed for Stage 4, though
the warning about it still fires when a probability genuinely reaches the cap.

Accuracy and calibration are independent properties. **Only the second determines whether
an output can be used to compute expected value.** A model can rank matchups correctly and
still emit numbers that are systematically wrong as probabilities.

The cost of Platt scaling is the clipped range. At 6k+ gold lead the observed win rate is
97–99%, but the model tops out at 0.89 — a systematic ~10pp underestimate in blowouts. This
is a real defect, not a rounding artifact, so the API returns it as a warning and the UI
renders the unreachable range as hatched dead zones on the scale.

---

## Three data problems that no statistical method would have found

### 1. A region renamed itself

In 2025 the LCS appears in the data as `LTA N`. Without a mapping you silently drop a full
year of North American matches. Nothing errors.

```python
raw["league"] = raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})
```

### 2. Zombie columns

| Column | Problem |
|---|---|
| `atakhans` | Real variable in 2025; identically zero in 2026 after the mechanic was removed. Column present, non-null, semantically dead. |
| `turretplates` | Ceiling moved from 15 to 45 when plating was extended to all towers. Same name, different denominator. |
| `blue_firstpick` | In 2025 blue side always had first pick; after the 2026 decoupling only 33.5% do. Pooled across years the column becomes a year label. |

None of these raise an error. `research/audit_columns.py` does a cross-year semantic
audit that catches the first two.

### 3. The LPL has no minute-level data at all

```
        games   T=10   T=15   T=20   T=25
LPL       453     0%     0%     0%     0%
LCK       349   100%   100%   100%    97%
LEC       246   100%   100%   100%    99%
LCS       157   100%   100%   100%    98%
```

The LPL is operated by TJ Sports and its data is scraped from `lpl.qq.com`, which does not
expose minute-level snapshots. Riot's pipeline does not cover it.

**Consequence: the Stage 4 training set contains zero LPL games** (4,585 of 8,390 games are
usable). In-game predictions for the LPL are extrapolated from other regions and *cannot be
validated, ever*. The API returns an explicit warning rather than quietly producing a number.

This is an external data constraint, not an implementation gap. Scraping gol.gg to recover
GD@15 for the LPL was measured and does add +1.5pp (`t = 4.60`), but it covers one time slice
only and still cannot validate LPL-specific behaviour, so it was not pursued. The commercial
route is closed: Bayes Esports went bankrupt in 2025 and GRID acquired the assets along with
exclusive Riot first-party rights, B2B only.

### Related: the 2026 draft rule change exposed a confounder

Blue side's raw advantage in the data is +8.7pp. But in a series, 77.6% of the time blue side
is the previous game's winner. Controlling for that, the real map-side advantage is
**+1.5 to +2pp**. Game 1 has no selection mechanism and independently gives +1.9pp.

---

## Empirical comeback rates

Win rate of the leading team, measured across five years. Useful as a sanity check on any
in-game model's output:

| Gold lead | T=10 | T=15 | T=20 | T=25 |
|---|---|---|---|---|
| 0–0.5k | 55.0% | 53.4% | 51.6% | 52.7% |
| 0.5–1k | 63.7% | 59.7% | 57.4% | 57.8% |
| 1–1.5k | 74.6% | 67.1% | 66.4% | 66.6% |
| 1.5–2k | 75.9% | 72.3% | 71.0% | 72.3% |
| 2–2.5k | 83.5% | 83.2% | 76.1% | 66.8% |
| 2.5–3.1k | 89.1% | 78.3% | 78.5% | 77.6% |
| 3.1–4k | 96.4% | 90.0% | 85.8% | 79.8% |
| 4–6k | — | 94.1% | 91.1% | 91.4% |
| 6k+ | — | 96.5% | 98.6% | 97.8% |

`research/check_comeback.py` regenerates this. The model's output at 2.5–3.1k / T=20 is 78.9% against
a measured 78.5% (n=410) — consistent. The 6k+ row is where the calibration clip bites.

---

## Stage 3: the agent layer, and why it is advisory-only

Four LLMs from four different lineages, selected by `research/screen_agents.py`:

```
verifier      deepseek-ai/deepseek-v4-pro
supplementer  openai/gpt-oss-120b
sceptic       minimaxai/minimax-m3
coordinator   google/gemma-4-31b-it
```

Selection used three probes run inside *real* two-round debate contexts rather than
simplified test items. The hard gate is runaway output: after stripping `<think>` blocks,
any model emitting meta-deliberation markers or exceeding 900 characters scores zero and is
disqualified. A second probe uses **invented team names**, where the correct answer is
"nothing to add" — anything mentioning transfers, coaches or patches is fabricating.

Writing that screener surfaced four bugs in the *scoring* code (a mistyped model name,
alphabetical sorting that displaced a good model, a minimum-length threshold that scored the
correct 4-character answer as zero, and two scorers stripping `<think>` inconsistently).
**The measuring instrument was harder to get right than the thing being measured, and had to
be tested against itself first.**

### Why the agents cannot touch the probability

The first version let the coordinator adjust by ±0.10. Result:

```
Stage 2   0.6305   calibrated, validated on held-out data
Stage 3   0.6005   agents revised down by 3pp
```

The direction was wrong — the reliability table showed that bucket was if anything slightly
*under*-confident. The sceptic's argument contained two factual errors (reading `playoffs=0`
as "the model makes no playoff adjustment", and treating ECE as a signed bias), and the
coordinator accepted it because it literally satisfied "identify a factor the model cannot
see". But that is a *class of blind spot*, not *a specific fact*.

The root cause is structural: three agents with no information source outside the model's own
brief can never have legitimate grounds to adjust it. Tightening the prompt does not fix a
badly posed task.

So the probability now always comes from Stage 2 or Stage 4. Stage 3 emits only classified
risk points — `survived` / `conceded` / `contested` / `refuted` — plus `external_fact`
entries, which are **required to carry a URL** and are automatically downgraded without one.

With retrieval attached (Tavily, `websearch.py`) the supplementer produced its first genuine
external fact: Fnatic signing top laner Soboro on 2026-07-15 — landing exactly in the blind
spot statistics cannot cover, since only 33 games in 2026 occur after a roster change and the
noise floor there widens to ±0.42.

---

## The service

```
GET  /                     single-page UI
GET  /health               model metrics + per-region data freshness
GET  /teams?league=LCK     known team names
POST /predict              Stage 1, or Stage 2 if a draft is supplied
POST /predict/ingame       Stage 4
GET  /esports/schedule     upcoming and past matches, from lolesports
GET  /esports/live         matches running right now
GET  /esports/match/{id}   per-game state + a ready-to-post ingame payload
POST /session/start        start tracking a live game
POST /session/{id}/tick    report the state every few minutes
GET  /session/{id}         the full win-probability trajectory
POST /debate               Stage 3 (needs NVIDIA_API_KEY)
GET  /api                  endpoint index
```

### Live match data, and the one field it cannot supply

`esports_feed.py` wraps the unofficial lolesports endpoints so the UI can list
matches and drive a prediction from a click rather than from typed team names.
Four things about that source cost real time to establish:

- `startingTime` must be floored to 10 seconds **and pushed into the past by the
  publish delay**. The feed publishes each 10-second window whole, 20–60 seconds after
  it starts (the delay changes from day to day, so it is measured live, not fixed);
  asking for a window that is not out yet returns an error or HTTP 204.
- **A game's `state` from the schedule API lags reality.** Observed live: the
  schedule reported a game `inProgress` while the frame feed reported
  `gameState: finished` with totals frozen. `frames[-1].gameState` is the only
  authoritative signal; trusting the other one means publishing a frozen
  probability as if it were live.
- Calling the window endpoint *without* `startingTime` returns the game's
  **first** frames, not its latest. Useful for finding the kickoff instant,
  useless as current state — and it looks like current state.
- The schedule's `startTime` is broadcast start, not game start. Measured gap:
  67 minutes.

Team names do not line up with Oracle's Elixir. Of 40 names across the four
leagues, **18 need mapping** — case (`GIANTX` / `GiantX`), sponsor suffixes
(`Team Liquid Alienware` / `Team Liquid`), city prefixes (`Suzhou LNG Esports`),
and one abbreviation no rule catches (`Beijing JDG Esports` → `JD Gaming`).
The table is explicit and fuzzy matching is deliberately absent: an early
`difflib` fallback mapped `Beijing JDG Esports` onto `LNG Esports` — a different
team, silently, with no error. A wrong team name does not fail; it just makes
that side's features missing.

**The live feed has no XP field**, so `xpdiff` cannot be sourced. Passing `0`
would be a fabrication, and `explain.py` duly turned it into the sentence
*"both sides are even on XP"* — a measured claim about data that does not exist.
So `ingame_train.py` trains a second 26-feature variant with `xpdiff` and its
interaction term `x_xp_scaling` removed, and `/predict/ingame` switches to it
when `state.xpdiff` is `null`. Cost, from the gate: significant only at T=20
(Δ −0.021, `t = −3.24`); at T=15 the smaller model is *better* on 7 of 8 folds.

| | full | live variant |
|---|---|---|
| features | 28 | 26 |
| accuracy | 73.7% | 72.9% |

### Responses are in plain language, not field names

`explain.py` translates every model field into a readable sentence, because
`diff_avg_towers = 3.2` means nothing to someone watching a game:

```
diff_avg_towers  = +3.2    ->  G2 takes 3.2 more towers per game
golddiff_norm    = 0.078   ->  the gold lead is 7.8% of all gold on the map
b_bot_champ_wr   = 0.722   ->  G2's ADC has a 72% historical win rate on this champion
```

All 148 field names the models can emit are covered; the 13 that are deliberately not
translated (region flags, meaningless differences like `diff_streak`) are dropped rather than
shown raw. Raw field names never appear in a response unless `raw_features: true` is passed,
which exists for debugging.

Each stat carries its own sentence template rather than sharing one generic pattern —
comparisons are phrased differently per metric, and a single template produces text that
reads like machine output.

### The UI shows probability as a position on a scale

Not a large number. The scale renders the calibrated range as hatched **dead zones** at both
ends, so the thing the model *cannot* express is visible rather than hidden. During a live
game the pre-game probability is drawn as a ghost marker with a bracket showing how far the
game has moved.

Every claim in the interface comes from the API's `reasons`, `warnings`, and model metrics —
the front end has no domain logic of its own.

---

## Deployment

**There is no server any more** (since 2026-09-13): the Oracle VM that used to run everything
could be reclaimed at any time, so it is no longer maintained. Two things run instead:

- **The public site**, <https://ra1nforest.github.io/LOL_Predictions/> — GitHub Pages, rebuilt
  by `.github/workflows/pages.yml` on every push that touches `frontend/`. It has no backend:
  the browser calls lolesports directly and runs Stages 1, 2 and 4 itself (`frontend/src/web/`),
  held to the Python implementation by `test:golden` and `test:diff`.
- **A Windows machine** running four scheduled tasks installed by `deploy/windows_install.ps1`:
  the API (`run_api.py`, 127.0.0.1:8000), live snapshot collection every 2 minutes, the
  twice-daily fetch → retrain → gate → swap (`daily_update.py --publish-web`, which then
  exports the new models and pushes them to the site), and the nightly frame backfill.

### If a server comes back

The systemd units in `deploy/` are kept for that.

```bash
# on the server, code at ~/lol/service, venv at ~/lol/venv
sudo bash deploy/install.sh
```

The script checks prerequisites, installs the environment file at `~/lol/service.env` with
mode 600, installs and enables the unit, kills any foreground uvicorn holding port 8000, and
polls `/health` until the service answers.

```bash
journalctl -u lol-predict -f          # logs
sudo systemctl restart lol-predict    # restart
```

Notes that cost time to learn:

- **Single worker only.** The feature store is 2–4 GB resident and each worker gets its own
  copy.
- **`TimeoutStartSec=300`.** Cold start loads five years of CSV and builds indices; the
  default 90s kills it mid-startup and systemd then restart-loops.
- **`KillSignal=SIGINT`.** uvicorn shuts down gracefully on SIGINT, not SIGTERM.
- **Environment via `EnvironmentFile`.** A process's environment is a snapshot taken at exec;
  editing `.bashrc` has no effect on a systemd-managed service, and systemd does not read
  your login shell at all.
- **systemd unit files do not support trailing comments.** A `#` after a value is parsed as
  part of the value. `systemd-analyze verify` catches this.
- **Oracle has three firewall layers** — subnet Security List, NIC-level NSG, and in-instance
  iptables. The Ubuntu image ships a catch-all `REJECT` rule, and `iptables -I INPUT 6`
  inserts *after* it, so the rule never takes effect. It must go before the REJECT.
- **VS Code auto-forwards ports**, which creates a convincing illusion that `localhost:8000`
  works while the public address is unreachable.

### Exposure

The API binds **127.0.0.1**, not a public port: `/esports/board` fans out to lolesports with
the shared public key, and getting that key rate-limited would kill the only live data source.
The public site does not need it — each visitor's browser talks to lolesports directly.
`/debate` costs four LLM calls plus retrieval per request and can additionally be gated by
setting `DEBATE_KEY`, after which it requires an `X-Debate-Key` header.

---

## A bug worth naming: training-serving skew

`api.py` once loaded two years of data while `train.py` trained on five. Nothing errored.
The only symptom was that player-champion familiarity statistics were systematically low.
After the fix, known player-champion combinations went from 5,841 to 11,587.

`feature_store.py` is shared by training and inference specifically to make this class of
bug structurally harder — but the loader was outside it, which was enough.

---

## Files

Every path is anchored to the file's own location, so scripts can be run from any
working directory.

**Runtime** (project root — this is what gets deployed)

```
api.py               FastAPI service
feature_store.py     feature definitions, shared by training and inference
train.py             Stage 1 + 2 training
ingame_model.py      Stage 4 core
ingame_train.py      Stage 4 training
ingame_service.py    Stage 4 inference + session tracking
esports_feed.py      lolesports schedule / live / frame feed + team-name map
fetch_data.py        pull fresh season CSVs from Oracle's Elixir
explain.py           field-name -> plain language
debate.py            multi-round debate protocol
agents.py            older advisory layer, still used by /predict?agent_review
websearch.py         retrieval backends (Tavily / Brave / Serper)
agent_config.json    agent assignment, written by research/screen_agents.py
static/index.html    single-page UI
artifacts/           trained models + calibrators
data/                the five Oracle's Elixir CSVs
deploy/              systemd unit, env template, install script
```

**Instruments** — `research/`

```
harness.py                 walk-forward validation + the paired t-gate
audit_columns.py           cross-year column semantics audit
multiyear_gate.py          multi-year loading + column audit; owns data_path()
reliability.py             split-half reliability
gate_draft_final.py        draft features, final gate
gate_draft_split.py        draft features, split by information source
gate_synergy.py            champion pair synergy (rejected)
gate_ingame_features.py    in-game feature gate
check_comeback.py          empirical comeback rates
check_golgg_value.py       value of partial data
screen_agents.py           agent screening (with the runaway gate)
```

`multiyear_gate.data_path(year)` is the single place that knows where the CSVs
live. The gate scripts share `load_year`, which returns `None` for a missing
file rather than raising — five scripts each building their own path is how a
year gets silently dropped.

**Ad-hoc checks** — `tools/`

```
check_draft.py    validate player / champion names before submitting a draft
check_nim.py      NIM connectivity self-check
list_models.py    list callable NIM model IDs
```

**A note on `harness.py`.** The copy recovered on 2026-08-16 was the
*pre-correction* version: it computed `t` but branched its verdict on
`ratio = |Δ| / SD`, the effect-size criterion this document describes as the
wrong tool. Its verdicts could not have produced the gate record above. It now
decides on `|t| > 2.5` as documented, with `ratio` demoted to a printed
reference.

Fixing it surfaced an edge case the old version had: when the per-fold deltas are
identical the paired SD is zero, and the old code set `ratio = 0` — classifying a
perfectly consistent *degradation* as noise. That branch now reports `t = ±inf`
and rejects on sign.

`attic/` holds superseded work kept only for reference: `discover_models.py` and
`finalize_agents.py` (v1 and v2 of the agent screener, replaced by
`research/screen_agents.py`), `test_kimi.py` (a one-off endpoint diagnostic), and
`SERVICE.md` (an earlier service doc, still describing three agents and the
±0.10 adjustment power that was later removed).

---

## Honest summary

Stages 1 and 2 work for all four regions. Stage 4 works for LCK / LEC / LCS and is
unvalidatable extrapolation for the LPL.

The model beats its baseline by a real but modest margin, and the walk-forward estimate of
that margin is +0.13 ± 0.10 — wide enough that single-match predictions carry substantial
uncertainty, which the API states rather than hides.

Not betting advice.
