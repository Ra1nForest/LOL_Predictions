# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` documents the *methodology* (the walk-forward gate, what was accepted/rejected, the
data problems). Read it before touching the models or the research scripts. This file covers the
operational and structural things that are spread across several files.

## Commands

Use the project venv: `.venv\Scripts\python.exe` (the scheduled tasks run it via `sys.executable`).
A bare `python` on this machine is a separate system Python 3.13 with a **different XGBoost**
(3.3.0 vs the venv's 3.4.1) — models trained or golden cases exported with it will not match what
the tasks produce. `frontend/scripts/diff-board.mjs` resolves `.venv` itself for the oracle and
falls back to bare `python` (with a warning) only when there is no venv.

```bash
pip install -r requirements.txt

python fetch_data.py --years 2025 2026     # refresh season CSVs from Oracle's Elixir (Drive)
python fetch_data.py --list                # list remote files without downloading
python fetch_data.py --auth                # one-time OAuth, needed because the anon quota is shared
python fetch_data.py --auth --force        # re-auth; the token dies every 7 days while the Cloud app is in Testing

python train.py                            # Stage 1 + 2  -> artifacts/model_{pre,post}_draft.json
python ingame_train.py                     # Stage 4      -> artifacts/model_ingame{,_live}.json
python xregion.py --build                  # cross-region model C -> artifacts/xregion.json (sanity-checked)
python xregion.py --check artifacts/xregion.json

uvicorn api:app --reload --port 8000       # UI at /, OpenAPI at /docs
```

Tests — stdlib `unittest`, no pytest, no extra dependency (deliberate). They run in ~9s and
need neither network nor API keys; the debate tests stub the LLM.

```bash
python -m unittest discover -s tests -v
```

Front end (Node is only needed to build — `api.py` just serves the built files):

```bash
npm --prefix frontend install
npm --prefix frontend run dev        # :5173, proxies API to 127.0.0.1:8000
npm --prefix frontend run build      # -> static/index.html + static/assets/
npm --prefix frontend run typecheck
```

Static GitHub Pages build — no backend at all; see `frontend/src/web/` under Architecture:

```bash
python tools/export_web_model.py          # models + team table -> frontend/public/web/, plus golden cases
npm --prefix frontend run test:golden     # browser models vs Python, bit-for-bit
npm --prefix frontend run test:diff       # browser board vs a fresh Python process, finished games
npm --prefix frontend run dev:pages       # :5174
npm --prefix frontend run build:pages     # -> dist-pages/
python tools/publish_web.py               # export + test:golden + commit/push frontend/public/web only
```

The site is https://ra1nforest.github.io/LOL_Predictions/ (repo `Ra1nForest/LOL_Predictions`,
public). `.github/workflows/pages.yml` rebuilds it on every push that touches `frontend/`. The local
`LoL-Update` scheduled task runs `daily_update.py --publish-web`, which calls `tools/publish_web.py`
after a successful model swap; the outcome lands in `update_status.json` → `web_publish` and in
`/health`. It only ever commits `frontend/public/web/`, and refuses to push if any other local
commit is unpushed or the remote is ahead — it is an unattended push to a public repo.

CI runs only `build:pages` (the CSS-variable check + `tsc`). **`test:golden` and `test:diff` never
run in CI** — the training data is not in the repo, so CI cannot export the golden cases. A push
that changes `frontend/src/web/` deploys without either check, so run both locally first.

One class or one test (test method names are Chinese, so `-k` on the class is usually easier):

```bash
python -m unittest discover -s tests -k SideOrdering -v
```

Research gates and ad-hoc tools are run from the project root, never from inside their directory:

```bash
python research/gate_league_mix.py --folds 5 --seeds 3
python research/harness.py
python tools/debate_dryrun.py --dump       # exercises the whole Stage 3 protocol with stubs, no key
node research/gate_smoothing.mjs           # gates for display-only changes import frontend/src/web/*.ts directly
```

Unattended pipeline (what the local Windows scheduled tasks run — see Deployment):

```bash
python daily_update.py --dry-run           # fetch + train into artifacts_staging/, no swap, no restart
python daily_update.py                     # + metric gate, atomic swap, restart, /health probe
python collect_live.py --status            # live late-game snapshot collector (scheduled every 2 min)
python prediction_log.py --resolve --score # backfill real results for logged predictions, then score them
```

## Architecture

Four stages, three of which produce a number. Everything below is in the project root, which is
also what gets deployed — `research/`, `tools/`, `attic/` are not needed at runtime.

- **`feature_store.py`** is the single definition of Stage 1/2 features, used by *both*
  `build_training_matrix()` (offline) and `FeatureStore.make_row()` (online). Any feature change
  belongs here and nowhere else. `select_features(mdf, stage)` is what decides which columns a
  stage sees; `pre_draft` is `post_draft` minus anything matching `DRAFT_PREFIXES`.
- **`train.py`** fits XGBoost + a **Platt** calibrator per stage and writes
  `artifacts/model_<stage>.json` plus `artifacts/calib_<stage>.json`. The calib file is the real
  contract: it carries `coef`, `intercept`, the exact `features` list, and `metrics`. `api.py`'s
  `Stage` class reads the feature list from there, so feature order is never re-derived at serving
  time.
- **`ingame_model.py` / `ingame_train.py` / `ingame_service.py`** are Stage 4. `ingame_model.build()`
  owns the snapshot table, the champion scaling index, and the interaction terms; `ingame_train.py`
  trains **two** variants from it — the full 28-feature model and a 26-feature `_live` variant with
  `xpdiff` and `x_xp_scaling` removed, because the live frame feed has no XP field.
  `/predict/ingame` picks the variant by whether `state.xpdiff` is null. Any new xp-derived column
  must be added to `XP_DERIVED` or the live variant leaks XP information back in.
  Stage 4 calibrates with **isotonic regression, not Platt** — its calib file carries
  `calibrator: "isotonic"`, `iso_x`/`iso_y` (the monotone lookup table) and `clip`, and keeps
  `coef`/`intercept` purely so an older service build can still load it. `IngameModel.calibrate()`
  falls back to Platt when `calibrator` is absent, so code and artifacts can be rolled out
  separately.
- **`api.py`** is the whole HTTP surface (~1500 lines, all routes in one file). Models, the feature
  store, the session store, and the esports feed are built once in the `lifespan` handler and live
  in the module-level `STATE` dict. Inference is `_predict_core` / `_ingame_core`; the routes are
  thin wrappers over those two.
- **`esports_feed.py`** wraps the unofficial lolesports endpoints and is the only place that knows
  about their quirks (lagged `startingTime`, the schedule's stale `state`, broadcast-vs-game start,
  team-name aliases). `/esports/board` in `api.py` is the big consumer. `LEAGUE_IDS` is the league
  whitelist for live detection and schedules: the four majors plus Worlds / MSI / First Stand / DCGI /
  Esports World Cup, keyed by the upstream `league.name.upper()` (`live()` filters on exactly that),
  while `Match.league` keeps the upstream casing (`"Worlds"`). `is_major()` is the single
  major/non-major test; `research/backfill_late.oe_league()` the single lolesports→OE league lookup
  (both case-insensitive).
- **`xregion.py`** is the cross-region pre-game model C ("分层 Elo"), the board's number for
  international matchups that Stage 1/2 cannot serve. It is the **single definition** of the
  all-league game table, the engine (`Engine`: sequential updates, same-timestamp batching), the
  walk-forward coefficients, decay, `predict()`, the exported state (`build_state()` →
  `artifacts/xregion.json`, written by `python xregion.py --build` and by the daily update) and the
  board's in-event replay (`board_predict()` and the pure helpers around it). `research/xregion_data.py`
  and `research/gate_cross_region.py` import it; it must not import `research/`. `api._xregion_board`
  only fetches (event schedule, final frames — `esports_feed`'s `event_tournament` /
  `completed_events` / `match_detail` / `match_ids` / `final_frame`) and hands the data to the pure
  functions. Other series' ids come from `match_ids` (read at `live_ttl`, memoised once complete):
  never re-read the shared `getEventDetails` URL with a long TTL — `_get` keeps whatever TTL the first
  writer chose, so that match's own board and its live-list score would go up to an hour stale.
- **`explain.py`** turns field names into sentences. Raw field names must never reach a response
  unless `raw_features: true`; a field with no template is dropped, not shown raw.
- **`debate.py`** (Stage 3) runs the 4-round protocol against NVIDIA NIM. `agents.py` is the older
  advisory layer, still reachable via `/predict?agent_review`. `websearch.py` provides retrieval
  (Tavily / Brave / Serper, probed in that order); with none configured the service still runs.
- **`frontend/`** is the UI source (Vite + React + TypeScript). `npm run build` emits into
  `static/`, which is what `api.py` mounts — a server (when there is one) has no Node, so builds
  happen locally and only the artefacts (`static/index.html` + `static/assets/*`) get copied over. `npm run dev`
  serves on 5173 and proxies the API prefixes to `127.0.0.1:8000` (an SSH tunnel to the box works).
  `static/legacy.html` is the previous single-file front end, kept reachable for comparison.
  The UI still has no domain logic: every claim it renders comes from the API's `reasons`,
  `warnings`, and model metrics.
  Two guards run as part of `npm run build`: `scripts/check-css-vars.mjs` (a `var(--typo)`
  fails silently — TypeScript cannot see it) and `scripts/clean-assets.mjs` (`emptyOutDir` is
  off to preserve `legacy.html`, so hashed bundles would otherwise pile up).
  **Response types are hand-written, so they only guarantee front-end-internal consistency.**
  Derive them from a real response, never from a guess: `/predict` nests its stages under
  `stages["1_pre_draft"]` / `["2_post_draft"]`, which is not what the obvious guess produces.
  The `/debate` client timeout must stay **above** `debate.py`'s own 300s budget, or the front
  end abandons a call that was still going to succeed.
- **`frontend/src/web/`** is a **second implementation of the serving path**, for the static
  GitHub Pages build (`vite --mode pages`, `VITE_STATIC=1`; `client.ts` switches on `IS_STATIC`
  and loads it lazily, so the server bundle does not contain it). `feed.ts` ports
  `esports_feed.py`, `board.ts` ports the `/esports/*` routes, `stage1.ts` / `ingame.ts` / `xgb.ts`
  run the Stage 1 and Stage 4 models in the browser, `explain.ts` ports `explain.py` (its text
  tables are *exported*, not copied). The browser calls lolesports directly — both hosts send
  `Access-Control-Allow-Origin: *` — so traffic is spread over visitors' own IPs. `stage2.ts`
  computes the board's post-draft line from two exported win/game tallies (champion×league,
  player×champion); Stage 3 is not in the static build. `xregion.ts` ports the board half of
  `xregion.py` (the engine's replay / decay / predict, final-frame verdicts, score-constrained
  decoding, OE de-dup, `boardPredict`) over `public/web/xregion.json` (a sanity-checked copy of
  `artifacts/xregion.json`); `board.ts`'s `xregionBoard` mirrors `api._xregion_board`, and
  `feed.ts` carries the same five schedule / final-frame helpers as `esports_feed.py`.
  Being a second implementation, **it is only trustworthy while two checks pass**: `test:golden`
  (features, probabilities, whole `api._ingame_core` and `/predict` responses, bit-for-bit) and
  `test:diff` (the whole board, against `tools/board_oracle.py`). The oracle is a **fresh**
  Python process on purpose: the long-running service carries live-time cache state that is
  never recomputed, and diffing against it measures that state, not the code. Any change to
  features, models, `explain.py` or `esports_feed.py` must be mirrored in `web/` and re-verified,
  and **after every retrain `tools/export_web_model.py` must be re-run** — the files in
  `frontend/public/web/` are a snapshot, and a stale one serves the old model without error.
  Node 24 runs the `.ts` files directly (type stripping), which is why `erasableSyntaxOnly` is on
  and `web/` imports carry `.ts` extensions.
- **`frontend/src/i18n.ts`** switches the UI to English unless the system language is Chinese
  (`?lang=en|zh` forces it). Hard-coded UI text goes through `T("中文原文")` (the Chinese string is
  the key). Generated sentences — reasons, warnings, notes, errors — are **not** translated at the
  source: `web/*.ts` and `api.py` must keep emitting the exact Chinese that `test:golden` compares
  against Python. Instead `client.ts` runs every response through `localize()`, which matches each
  sentence against Chinese templates (compiled to regexes) and fills the English one; anything it
  doesn't recognise stays Chinese (dev mode logs `[i18n] 没有英文`). Changing a sentence in
  `explain.py`/`web/` therefore needs the matching template in `i18n.ts` updated too. Rune names
  come from Data Dragon in the requested locale (`Feed({ runeLocale })`, default `zh_CN` for parity).

## Invariants that are easy to break silently

Nearly every bug this project has had produced a plausible wrong number rather than an exception.
`tests/test_regressions.py` exists specifically to pin these down — each test names the trap it
guards. If one goes red, work out whether that trap is back before changing the test.

- **Training and serving must read the same data.** `api.py` and `train.py` both build `DATA` from
  the same five-year list. Loading a different span does not error; it just makes
  player-champion familiarity systematically low.
- **`raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})`** must survive any refactor of the
  loaders. Without it a full year of NA matches silently disappears.
- **`EXCLUDED = {"blue_firstpick", "atakhans", "turretplates"}`** in `feature_store.py` — columns
  that are present, non-null, and semantically dead or year-dependent.
- **`frames[-1].gameState` is the authoritative liveness signal**, not the schedule's `state`.
  Likewise `frame_sides()` takes blue/red from the frame's `gameMetadata`, not from the persisted
  API — the two disagree about 8% of the time, and following the wrong one attributes one team's
  numbers to the other without any error.
- **No fuzzy team matching.** `TEAM_ALIASES` is explicit; an unmapped name returns `None` so the
  caller can fail loudly. A `difflib` fallback once mapped `Beijing JDG Esports` onto `LNG Esports`.
  Only `known=None` means "no validation": `map_team(x, [])` is `None`, and so is an alias whose
  target is not in `known` (it used to be `if known:`, so the always-empty
  `known_teams("Worlds")` let every international name through unchecked). Every call site gets
  its list from `api._known_for(league)` — the league's own known teams for a major, the union of
  all four (`store.known_teams(None)`, exported as `teams.json` → `known_all`) otherwise.
- **Stage 3 never changes the probability.** The number always comes from Stage 2 or Stage 4;
  the agents emit classified risk points only, and `external_fact` entries without a URL are
  downgraded.
- **Accept a modelling change only at `|t| > 2.5`** on the paired walk-forward gate
  (`research/harness.py`, `paired_t` / `gate`). Do not use `ratio = |Δ| / SD` — it is an effect
  size and never accumulates evidence. Single-holdout comparisons are noise at this data volume.
- **`research/multiyear_gate.data_path()` is the only place that knows where the CSVs are.** Gate
  scripts share `load_year`, which returns `None` for a missing file rather than raising, so a
  script that builds its own path is how a year gets dropped from a comparison.
- Stage 4 trains on **all leagues, not just the four majors** (`TARGET = None` in
  `ingame_model.py`, gated at `t = +5.19`), while `/predict`, `/predict/ingame` and `/debate` still
  only accept the four (the board also serves international events — see the next bullet).
  Narrowing the training pool back to the majors would undo a measured gain.
- **International events: Stage 1/2 only for same-home-league matchups, Stage 4 always.**
  `api.pregame_league(store, blue, red, league)` (mirrored by `web/` and pinned by
  `golden/intl_cases.json`) decides, and is exposed as `match.pregame_league`: a major returns its
  league unchanged (domestic boards do not change at all); an international match whose two teams
  share a home league (`FeatureStore.home_league` = league of the team's latest game) returns that
  home league, and every Stage 1/2 number on the board (too_early, post-draft line, blend anchor,
  curve) is computed **with the home league**; anything else (cross-region, or a team with no
  major-league history) returns `None` → no Stage 1/2 anywhere on the board, and Stage 4 runs with
  `include_pre=False` (no `diff_pre_*`, no silent fallback to `_pre_context`). On those boards the
  pre-game number comes from the cross-region model C instead (next bullet) when it can be computed;
  when it cannot, the board is exactly what it was before C existed (no number before minute 3, no
  blend, the "不显示" sentence). Stage 4 always gets the **event** league, so `is_*` stay all-zero exactly as
  international rows were encoded in training — passing the home league there would not error.
  Measured in `research/gate_international.py` with the live artifacts: on 523 historical
  cross-region games Stage 1/2 have no predictive value (accuracy ~50%, calibration slope ~0, yet
  as confident as domestically; significantly worse than a constant only in 2022-24, about equal to
  it out-of-sample in 2025/26); the 314 same-home-league international games behave like domestic
  ones; Stage 4 beats a constant on international games (t +2.7 to +9.6 per slice, cross-region
  t = +3.43, n = 102 — a degradation smaller than ~0.055 Brier is undetectable at that n), and
  nulling its `diff_pre_*` inputs changes nothing measurable (t = +0.15). A board with an unmapped
  team still shows Stage 4 (the lolesports name is only a label, never looked up) but must not
  call `log_prediction` — `prediction_log.resolve` could never match it. DCGI is deliberately not in
  `LEAGUE_TO_OE`: OE's `DCup` was the December LPL cup, and how OE will label DCGI is unknown.
  `test:diff` only replays final frames, so the first three minutes never reach it: the too_early
  payload and the reason sentences are pure functions (`_early_prediction`, `_pregame_reason`,
  `_board_warnings` and their `web/board.ts` twins) pinned by `golden/intl_cases.json` instead.
- **Cross-region boards use model C, and only in its in-event online form.** Shipped 2026-10-04
  (user approved) after a frozen DEV/HOLDOUT gate (`research/gate_cross_region.py`): HOLDOUT
  450 games, Brier 0.2231 vs constant 0.2478, series `t = +2.87` (a thin pass; it ties the
  "home-league identity only" baseline, `t = +0.40`), accuracy ~65%, over-confident (says ~85%,
  wins ~76%). The post-draft variant D **failed** (`t = +2.47`, BP term `t = −1.64`), so C boards
  have no post-draft line. **Only the version that updates after every finished game passed** —
  the OE-only "results count from the next day" version fails (`t = +2.02` to `+2.48`), and OE lags
  about a day. So `api._xregion_board` (twin: `web/` board) must replay the running event's finished
  games from lolesports on top of `artifacts/xregion.json` (`xregion.board_predict`); if the event
  schedule (`getEventDetails` tournament → `getCompletedEvents`) cannot be fetched, C is **not
  shown** — never fall back to the state alone. Replay rules (all in `xregion.py`, pinned by
  `golden/xregion_cases.json` and `XregionReplay`): series of the same tournament that started
  before the current one, plus the current series' games numbered below the current game; finished
  count = sum of the series score (not `games[].state`), and **if the current game's number − 1
  exceeds the current series' score (upstream registers the score ~6 min after the frames) C is
  withheld** (`withheld: "score_lag"`) rather than silently replaying one game short; games already
  in OE are skipped by `oe_lookup` (same game number, OE start within [startTime − 6 h, startTime
  + 18 h], one team name equal and the other equal **or a stale OE name that `TEAM_ALIASES` maps to
  that team** — OE logged 2026 EWC's Team Secret Whales as "Team Secret"; accepting *any* non-current
  name falsely de-duplicated a same-day game against a different opponent); OE's results for those
  games are taken out of the score before decoding the rest (a series split across the OE cutoff
  otherwise decoded to the opposite series result; OE results contradicting the score raise → no C);
  winners from the score for BO1/sweeps/deciding games, otherwise from final frames (towers →
  inhibitors → gold, blue side from the frame's `esportsTeamId`) decoded under the score constraint
  (47/47 on real final frames), and alternating order (leader first) for a series with any frame
  missing; engine time `t = startTime + (number − 1) s`, ordered by `(t, startTime, match_id,
  number)` — never by input position. Series that started at/after the current one or were still
  running are not replayed (measured on HOLDOUT: 17/450 games touched, mean |Δp| 0.0008, Brier Δ
  −0.00012, `t = +0.86`). Before the game has frames nobody knows which side is blue, and C's
  intercept (a ≈ +0.14) is the blue-side edge, so the no-frames number is the **side-neutral mean**
  `(p(A blue) + 1 − p(B blue)) / 2` (`side_neutral: true`; the player never uses it as an anchor).
  The board's C number feeds the too_early headline and the
  no-frames pre-game view (`source: "xregion"`, `_xregion_pregame`; Board.tsx must not call
  Stage 1 for these matches), and is the **blend anchor** for minutes 3–15 on the headline, curve,
  per-second timeline and live playback (adopted by "not significantly worse": series `t = +2.27`
  in the better direction, minute-3 jump 18.6 → 0.11 points). Stage 4 still gets no pre-game
  features. Logging follows the same rule as everywhere: only `match.predictable` boards, source
  `"xregion"` for the pre-game row. Domestic and same-home-league boards are byte-identical with
  and without C (verified against HEAD's oracle on 36 boards; `XregionBoard` pins it), and do not
  send any of C's upstream requests.
  **Every `exp` computed from the exported state uses the fdlibm port** (`xregion.portable_exp` /
  `web/xregion.ts` `exp`, set by `Engine.from_state`; `logistic` defaults to it), never
  `math.exp` / `Math.exp`: no two platforms' `exp` agree to the last bit (this machine's
  `math.exp` is MSVC's — 1147 of 200k inputs not correctly rounded, and 7% differ from V8; Safari
  uses the system libm), so with the platform `exp` the two implementations disagreed by an ulp on
  a sizeable share of boards and `test:diff` could not tell that from a porting bug. The gate and
  `build_state` (`Engine(P)` / `run_elo`, default `exp=math.exp`) are deliberately left on
  `math.exp`, so the gate reproduction stays bit-for-bit (re-verified: max|Δp| = 0, DEV and
  HOLDOUT, all seven models); the state is data both sides read from the same JSON.
- **Isotonic calibration is Stage 4 only. Do not generalise it to Stage 1/2.** Same change, opposite
  verdicts, because the calibration sets differ by 20×: Stage 4 has ~35k snapshots (Brier
  `t = +6.91`, ECE `t = +7.12`, 8/8 folds), Stage 1/2 have 1716 games (Brier `t = -4.74`, 6/6 folds
  *worse* — it overfits the calibration set exactly as README describes). Gates:
  `research/gate_calibration.py` and `research/gate_calibration_pre.py`.
- **The "probability is clipped" warning must key off the actual output, not the input.** It lives
  in `IngameModel.clip_warning()` and fires only when the calibrated probability reaches `p_min` /
  `p_max`. It used to be hardcoded in `make_row` as "minute ≥ 20 and |golddiff| ≥ 5000", which was
  true under Platt (capped at 0.94) and became a false claim under isotonic (an 8k lead returns
  0.959, nowhere near the 0.99 cap). A stale warning is worse than none — Stage 3's agents and the
  UI quote it as fact.

- **Never replay a finished game through `/esports/board` in a process that logs predictions.**
  `esports_board` calls `log_prediction`, and on a finished game that records a "prediction"
  made from the final frame (p = 0.01 / 0.99) into `predictions/log.jsonl` — the file used to
  score the model. A diff run against the live service wrote four such rows on 2026-09-12
  (removed; backup `log.jsonl.bak-before-cleanup-*`). `tools/board_oracle.py` replaces
  `log_prediction` with a no-op for exactly this reason. The unattended writer is
  `collect_live.log_predictions`, which requests `/esports/board/{id}?curves=false` **only for
  matches with an `in_game` game** — any new caller of the board needs the same guard.
- **`daily_update.run()` forces `PYTHONIOENCODING=utf-8` on its children.** On Chinese Windows a
  piped child writes GBK; read back as UTF-8, `fetch_data.py`'s "已更新" never matched, so every
  local run from 2026-09-04 reported "数据没有变化" and never retrained — and every run logged
  success. The same garbling makes the `games` / `个快照` parse fail, which silently skips the
  gate's minimum-count checks.
- **Live names must be mapped to OE names before any lookup.** lolesports sends
  `summonerName` with the team code glued on (`IGTheShy`; OE has `TheShy`) and champions as Data
  Dragon ids (`LeeSin`, `MonkeyKing`; OE has `Lee Sin`, `Wukong`). Looked up raw, every
  player-champion familiarity on the board was 0 and 13% of champions vanished from both the
  Stage 2 champion win rates and Stage 4's composition scaling index — no error, just a plausible
  number. `feature_store.champion_key` (+ `CHAMP_ID_ALIAS`) is applied inside the lookups, so
  training, which passes OE names, is unchanged; `esports_feed.oe_player_name` strips only the two
  match team codes. `web/names.ts` mirrors both. Measured on the held-out test split
  (`research/gate_live_names.py`, 1755 games): broken names made Stage 2 no better than Stage 1
  (59.5% vs 59.4%); aligned names give 61.7%, Brier `t = +3.16`, log loss `t = +2.85` over 5
  time blocks (accuracy `t = +1.78`), and reproduce the training matrix's draft features exactly.
- **The board blends Stage 2 into Stage 4 for minutes 3–15.** `api._blend_prior` /
  `web/ingame.ts blendPrior` (`BLEND_FROM`/`BLEND_TO`) move the headline from the post-draft
  probability (pre-game if there is none) to the in-game one, linearly in log-odds; switched on
  by `blend=True` only on the board paths (headline, curve, per-second timeline, live playback).
  `/predict/ingame` still returns the raw Stage 4 number, so `test:golden` is unaffected. It was
  adopted for continuity, not accuracy (`research/gate_anchor.py`: t = +0.83, never worse; the
  minute-3 jump fell from 14.3 to 0.1 points). Change it in both implementations — `test:diff`
  catches a mismatch.
- **Collected winners are provisional until Oracle's Elixir confirms them.** `collect_live.py`
  infers a game's winner from the series-score increment, which in LPL once credited one point to
  two games. `oe_correct()` overwrites collected labels with OE results every 6 h
  (`source="oe"`, the old value kept in `was`). Do not train on collected labels that skip it.
- **Do not smooth the win-probability curve over time.** A trailing average removes the
  per-second chatter (XGBoost is piecewise-constant in golddiff, plus slice switches at
  12.5/17.5/22.5 min) but lags exactly when the game is being decided: Brier `t = -13.25` over 150
  held-out games (`research/gate_smoothing.mjs`). Any de-chattering has to happen in feature space
  (e.g. averaging over a golddiff neighbourhood) and pass the same gate.
- **A failed start calibration is not a result.** `_game_start` (and `gameStart` in
  `frontend/src/web/feed.ts`) retries every `START_RETRY_SEC` until the calibration windows have
  settled; only then does it accept `frames[0]`. Caching the first-sight failure put every live
  minute and every collected snapshot label 8–20 s off for the whole game.

## Conventions

- Docstrings, comments, log output, and the UI are in **Chinese**; identifiers and field names in
  English. Module docstrings carry the *reasoning* — why a design exists and which failure it
  prevents — and are the fastest way into an unfamiliar file. Match that when adding code.
- Every path is anchored to the file's own location (`Path(__file__).parent`), so scripts run from
  any working directory. The repository is public: machine-specific values (server address, SSH
  key path) come from `LOL_SERVER` / `LOL_SSH_KEY` and never appear in the source.
- `LOL_ARTIFACTS` redirects training output; `daily_update.py` uses it to train into
  `artifacts_staging/` without touching what is live. Other env vars: `NVIDIA_API_KEY` (Stage 3),
  `TAVILY_API_KEY` / `BRAVE_API_KEY` / `SERPER_API_KEY` (retrieval), `DEBATE_KEY` (gates `/debate`
  behind an `X-Debate-Key` header), `LOL_STALE_DAYS`, `LOL_PRED_LOG`, `LOL_API` (where
  `collect_live.py` reaches the API, default `http://127.0.0.1:8000`). See
  `deploy/service.env.example`.
- `agent_config.json` is generated by `research/screen_agents.py` and its keys are the Chinese role
  names (`协调者` / `补充者` / `验证者` / `质疑者`). It overrides `_DEFAULTS` in `debate.py`, and the
  model IDs in it are the live truth — README's Stage 3 table lists an earlier selection.
- `data/`, `artifacts/`, `backfill/`, `predictions/`, `collected/` hold data, not code.
  `google_credentials.json` / `google_token.json` are real credentials, gitignored, never printed.
- `attic/` is superseded work kept for reference only. Do not import from it.

## Deployment

**As of 2026-09-13 there is no server.** The Oracle box that used to run everything under "If a
server comes back" can be reclaimed at any time, so it is no longer maintained or deployed to. It
may still be up, running code from before that date — do not treat its data or behaviour as
current, and do not propose server deploys. This changes only when a new server turns up or the
free A1 grab succeeds.

What actually runs:

- **The public site** — GitHub Pages, rebuilt by `.github/workflows/pages.yml` on every push to
  `main` that touches `frontend/` (see Commands). Visitors' browsers do all the live inference, so
  shipping a front-end change is just pushing it.
- **This Windows machine** — four scheduled tasks installed by `deploy/windows_install.ps1`:
  `LoL-Predict` (`run_api.py`, the API on 127.0.0.1:8000), `LoL-Collect`, `LoL-Update`
  (`daily_update.py --publish-web`: retrain — `train.py`, `ingame_train.py` into
  `artifacts_staging/`, plus `xregion.py --build` as a **side path** —, gate, swap, restart
  `LoL-Predict` via `schtasks /end` + `/run`, then push the refreshed model files to the site —
  `export_web_model.py` copies `artifacts/xregion.json` to `frontend/public/web/`) and
  `LoL-Backfill`. A failed xregion build or `sanity_check` (e.g. OE adds an unclassified event code)
  keeps yesterday's `xregion.json` **whole** — self-consistent, since its `events` de-dup list and its
  ratings come from one build; never mix files — and is recorded in `update_status.json` → `xregion`
  and `/health`; it no longer blocks Stage 1/2/4 or the site (it used to stop the whole update every
  day until someone edited the code). The batch jobs
  start through `run_task.py` (windowless `pythonw` with stdout redirected into `logs/`). Tasks run
  as the logged-in user with no stored credentials, so they need a session (a locked screen is
  fine). All four have `-StartWhenAvailable`, so a run missed while the machine was off happens as soon
  as it is back (for collection that only means "next sample now" — a missed live game cannot be
  sampled afterwards). The install
  script's header lists the differences from the systemd units that would otherwise fail silently
  (local time vs UTC, catch-up, the restart command).
- The Python side still has to stay correct with no server: it is the reference `test:golden` and
  `test:diff` hold the browser implementation to, and the local tasks run it.

### If a server comes back

Systemd units in `deploy/`; a server runs four of them —
`lol-predict` (the API), `lol-update` (twice-daily fetch + retrain via `daily_update.py`),
`lol-collect` (live snapshots every 2 min), `lol-backfill` (daily, after the update, since it needs
Oracle's Elixir results as labels). `deploy/install.sh` installs and verifies **only** `lol-predict`
(env file at mode 600, port-8000 cleanup, `/health` poll); the three timer units are installed by
hand. The unit files carry the reasoning for each setting in comments;
the load-bearing ones are single worker (the feature store is 2–4 GB resident per worker),
`TimeoutStartSec=300` (cold start loads five years of CSV), `KillSignal=SIGINT` (uvicorn ignores
SIGTERM for graceful shutdown), and `EnvironmentFile` (editing `.bashrc` does nothing for a systemd
service). Note that systemd unit files do not support trailing comments.

`lol-predict` binds **127.0.0.1**, not the public interface — `/esports/board` fans out to upstream
with the shared lolesports key, and getting that key rate-limited kills the data source. Reach it
over an SSH tunnel or an authenticated reverse proxy.

Neither machine's data is authoritative by default: the server's OAuth token once died for ten days
(2026-08-31 → 09-09) while the local copy kept updating. Check `/health` → `data_through` and the
`update` block on whichever machine you are reading from before trusting its numbers.
