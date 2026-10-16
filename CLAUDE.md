# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` documents the *methodology* (the walk-forward gate, what was accepted/rejected, the
data problems). Read it before touching the models or the research scripts. This file covers the
operational and structural things that are spread across several files. Module docstrings carry
the detailed reasoning; this file points at them.

## Commands

**In every block below, `python` means `.venv\Scripts\python.exe`.** A bare `python` (and bare
`uvicorn`) on this machine is a separate system Python 3.13 with a **different XGBoost** (3.3.0 vs
the venv's 3.4.1) — models trained or golden cases exported with it will not match what the tasks
produce. `requirements.txt` pins nothing exactly (`xgboost>=2.0`), so the existing `.venv` is the
only real pin; install into it with `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
The scheduled tasks start `.venv\Scripts\pythonw.exe` directly, and everything they spawn uses
`sys.executable`.

A fresh clone has no data or models: `data/`, `artifacts/`, `collected/`, `predictions/`,
`backfill/`, `frontend/scripts/golden/` and the built `static/index.html` + `static/assets/` are
gitignored (only `frontend/public/web/*.json` and `static/legacy.html` are tracked). Bootstrap order:
`fetch_data.py --auth` → `train.py` → `ingame_train.py` → `xregion.py --build`; `export_web_model.py`
additionally needs real frames in `collected/`.

```bash
python fetch_data.py --years 2025 2026     # refresh season CSVs from Oracle's Elixir (Drive); --all, --retry N
python fetch_data.py --list                # list remote files without downloading
python fetch_data.py --auth                # one-time OAuth, needed because the anon quota is shared
python fetch_data.py --auth --force        # re-auth; the token dies every 7 days while the Cloud app is in Testing

python train.py                            # Stage 1 + 2  -> artifacts/model_{pre,post}_draft.json
python ingame_train.py                     # Stage 4      -> artifacts/model_ingame{,_live}.json
python xregion.py --build                  # cross-region model C -> artifacts/xregion.json (sanity-checked; --asof)
python xregion.py --check artifacts/xregion.json

python run_api.py --port 8010 --no-log     # foreground dev API (UI at /, OpenAPI at /docs)
```

**The three training commands write straight into the live `artifacts/`** — what `LoL-Predict`
loads on its next restart and what `export_web_model.py` publishes, with no gate and no backup.
For experiments set `LOL_ARTIFACTS` to a scratch dir. Port 8000 is the always-on `LoL-Predict`
instance (the Vite dev proxy targets it); a second API instance writes to the same
`predictions/log.jsonl` whenever it serves an in-game board unless `LOL_PRED_LOG` points elsewhere.

Tests — stdlib `unittest`, no pytest, no extra dependency (deliberate). 206 tests, all in
`tests/test_regressions.py`, ~12s, no network or API keys (the debate tests stub the LLM). Without
`artifacts/` the suite stays green but silently skips the two board classes (`InternationalBoard`,
`XregionBoard`). Test method names are Chinese, so select by class:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -k SideOrdering -v
```

Front end (Node is only needed to build — `api.py` just serves the built files):

```bash
npm --prefix frontend install
npm --prefix frontend run dev        # :5173, proxies the API prefixes to 127.0.0.1:8000
npm --prefix frontend run build      # -> static/index.html + static/assets/ (gitignored); runs check-css-vars
npm --prefix frontend run typecheck
```

Static GitHub Pages build — no backend at all; see `frontend/src/web/` under Architecture:

```bash
python tools/export_web_model.py          # models + team table + xregion state -> frontend/public/web/, plus golden cases
npm --prefix frontend run test:golden     # browser implementation vs Python (needs a prior export on this machine)
npm --prefix frontend run test:diff       # browser board vs a fresh Python process, finished games (network + artifacts + data)
npm --prefix frontend run test:diff -- --days 7 --per 4
npm --prefix frontend run dev:pages       # :5174
npm --prefix frontend run build:pages     # -> dist-pages/
python tools/publish_web.py --no-push     # export + test:golden + commit frontend/public/web only (without --no-push: a public push)
```

The site is https://ra1nforest.github.io/LOL_Predictions/ (public repo `Ra1nForest/LOL_Predictions`);
`.github/workflows/pages.yml` rebuilds it on every push to `main` that touches `frontend/`. After a
successful model swap `LoL-Update` runs `tools/publish_web.py`, which commits **in this working copy**,
refuses unless `main` is checked out, commits only `frontend/public/web/`, and refuses to push past
anyone else's unpushed commit. Outcome: `update_status.json` → `web_publish` and `/health`.

CI runs only `build:pages` (the CSS-variable check + `tsc`). **`test:golden` and `test:diff` never
run in CI** — the training data is not in the repo, so CI cannot export the golden cases. A push
that changes `frontend/src/web/` deploys without either check, so run both locally first.

Research gates and ad-hoc tools are run from the project root, never from inside their directory.
Gates import `research/harness.py` (`walk_forward`, `paired_t`, `gate`; it has no CLI):

```bash
python research/gate_league_mix.py         # example gate
python tools/debate_dryrun.py --dump       # exercises the whole Stage 3 protocol with stubs, no key
node research/gate_smoothing.mjs           # gates for display-only changes import frontend/src/web/*.ts directly
```

Unattended pipeline (what the local Windows scheduled tasks run — see Deployment):

```bash
python daily_update.py --dry-run --no-fetch    # train into artifacts_staging/ + metric gate; no fetch, swap or restart
python daily_update.py --service LoL-Predict   # + fetch, gate, file-by-file swap (old set kept in artifacts_prev/,
                                               #   rolled back if /health fails), restart; --publish-web also updates the site
python collect_live.py                     # one collection round (LoL-Collect, every 2 min): collected/ + prediction logging via :8000
python collect_live.py --status            # counts only
python prediction_log.py --resolve --score # backfill real results for logged predictions, then score them
python run_task.py collect collect_live.py --stdout   # run a task in the foreground; normally output goes to logs/<name>.log
```

Plain `--dry-run` stops right after the fetch when no CSV changed; `--no-fetch` (dead OAuth token,
Drive quota) and `--force` make it train anyway. `--no-restart` swaps without restarting.

## Architecture

Four stages, three of which produce a number, plus the cross-region model C (`xregion.py`). The
Python runtime modules live in the project root. `api.py` imports nothing from `research/` or
`tools/`, but the scheduled tasks do: `research/backfill_late.py` (LoL-Backfill, plus
`collect_live.oe_correct` and `prediction_log.resolve`, which import its `oe_index` / `find_oe` /
`oe_league`) and `tools/publish_web.py` → `tools/export_web_model.py` (LoL-Update). Nothing imports
`attic/`.

- **`feature_store.py`** is the single definition of Stage 1/2 features, used by *both*
  `build_training_matrix()` (offline) and `FeatureStore.make_row()` (online). Any feature change
  belongs here and nowhere else. `select_features(mdf, stage)` decides which columns a stage sees;
  `pre_draft` is `post_draft` minus anything matching `DRAFT_PREFIXES`.
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
  must be added to `XP_DERIVED` (in `ingame_train.py`) or the live variant leaks XP back in.
  Stage 4 calibrates with isotonic regression (Stage 1/2 stay Platt, see Invariants); the
  `calibrator` field in `calib_ingame*.json` selects it and `IngameModel.calibrate()` falls back to
  Platt without it. **Stage 4 features have no single definition**: `ingame_model.build()` computes
  them offline, `IngameModel.make_row()` recomputes the same columns online without importing
  `ingame_model`, `web/ingame.ts` is a third copy and `research/merge_backfill.derive` a fourth.
  Only `make_row` ↔ `ingame.ts` is pinned (`test:golden`); nothing checks `build()` ↔ `make_row()`,
  so a Stage 4 feature change must touch all of them by hand.
- **`api.py`** is the whole HTTP surface (~1900 lines, all routes in one file). Models, the feature
  store, the session store, the esports feed and model C's state (`_load_xregion()` →
  `STATE["xregion"]`) are built once in the `lifespan` handler and live in the module-level `STATE`
  dict. If `artifacts/xregion.json` is missing or fails `sanity_check`, cross-region boards simply
  show no C — visible only in the startup log and `/health`. Stage 4 inference goes through
  `_ingame_core` (`/predict/ingame`, `/session/{sid}/tick`, the board). `/predict` builds its Stage
  1/2 response inline; `_predict_core` is only the board's Stage-2 float. `esports_board` (~440
  lines) holds the board logic. `/debate`'s `_brief_for` calls the models itself and must keep the
  same variant choice as `_ingame_core`.
- **`esports_feed.py`** wraps the unofficial lolesports endpoints and is the main place that knows
  their quirks (lagged `startingTime`, the schedule's stale `state`, broadcast-vs-game start,
  team-name aliases); `api.py`'s `_refresh_wins` / `_clinched` / `_xregion_series` handle score
  freshness and name casing on top (mirrored in `web/board.ts`). `LEAGUE_IDS` (four majors +
  international events, keyed by upstream `league.name.upper()`) decides what counts as live;
  `Match.league` keeps the upstream casing (`"Worlds"`), so the major test (`is_major()`) and the
  lolesports→OE league lookup (`research/backfill_late.oe_league()`) are the only places for those
  checks, and both are case-insensitive.
- **`xregion.py`** is the single runtime definition of model C ("分层 Elo"), the board's pre-game
  number for international matchups Stage 1/2 cannot serve: the all-league game table, the engine,
  the walk-forward coefficients, the exported state (`artifacts/xregion.json`, from `--build` and the
  daily update) and the board's in-event replay (`board_predict`). `research/xregion_data.py` and
  `research/gate_cross_region.py` import it; it must never import `research/`. `api._xregion_board`
  (twin: `board.ts xregionBoard`) only fetches and hands the data to xregion's pure functions. Trap:
  `EsportsFeed._get` caches by URL with the **first caller's TTL**, so never read a URL that live
  data also uses (getEventDetails) with a long TTL — use `match_ids` for stable ids.
- **`explain.py`** turns field names into sentences. Raw field names must never reach a response
  unless `raw_features: true`; a field with no template is dropped, not shown raw.
- **`debate.py`** (Stage 3) runs the 4-round protocol against NVIDIA NIM. `agents.py` is the older
  advisory layer, still reachable via `POST /predict` with `"agent_review": true` in the body.
  `websearch.py` provides retrieval (Tavily / Brave / Serper, probed in that order); with none
  configured the service still runs.
- **`frontend/`** is the UI source (Vite + React + TypeScript). `npm run build` emits
  `static/index.html` + `static/assets/` (gitignored) for `api.py` to mount at `/`; it also runs
  `check-css-vars.mjs` (a misspelt `var(--x)` fails silently — TypeScript cannot see it).
  `static/legacy.html` is the previous single-file front end, kept for comparison. Components
  (`src/components/`, `App.tsx`) have no domain logic: everything they render comes from the
  response (`reasons`, `warnings`, metrics), produced by `api.py` or, in the Pages build, by `web/`.
  **Response types are hand-written, so they only guarantee front-end-internal consistency.**
  Derive them from a real response, never from a guess: `/predict` nests its stages under
  `stages["1_pre_draft"]` / `["2_post_draft"]`, which is not what the obvious guess produces.
  The `/debate` client timeout must stay **above** `debate.py`'s own 300s budget, or the front
  end abandons a call that was still going to succeed.
- **`frontend/src/web/`** is a **second implementation of the serving path**, for the static
  GitHub Pages build (`vite --mode pages`, `VITE_STATIC=1`; `src/api/client.ts` lazy-loads
  `web/static-api.ts` when `IS_STATIC`, so the server bundle does not contain it). `feed.ts` ports
  `esports_feed.py`; `board.ts` ports `/esports/live`, `/esports/upcoming` and `/esports/board`;
  `stage1.ts` / `ingame.ts` / `xgb.ts` run Stage 1 and the `_live` Stage 4 variant in the browser;
  `stage2.ts` computes the board's post-draft line from two exported tallies (champion×league,
  player×champion); `xregion.ts` ports the board half of `xregion.py` over `public/web/xregion.json`
  (copied only if `sanity_check` passes); `teams.ts` stands in for the FeatureStore + `map_team`;
  `names.ts` and `pyfmt.ts` (Python's round-half-even formatting) keep strings identical;
  `explain.ts` ports `explain.py` (its text tables are *exported*, not copied). The static
  `/predict` is Stage 1 only and Stage 3 is absent. The browser calls lolesports directly — both
  hosts send `Access-Control-Allow-Origin: *` — so traffic is spread over visitors' own IPs.
  Being a second implementation, **it is only trustworthy while two checks pass**: `test:golden`
  (six groups: Stage 4 features/probabilities, `_ingame_core` and `/predict` responses, the board's
  Stage 2 line, the international board helpers, model C — features, model C and response fields
  bit-for-bit, tree probabilities within 1e-6 because the trees run in float32) and `test:diff` (the
  whole board, against `tools/board_oracle.py`). The oracle is a **fresh** Python process on
  purpose: the long-running service carries live-time cache state that is never recomputed, and
  diffing against it measures that state, not the code. Any change to features, models, `explain.py`
  or `esports_feed.py` must be mirrored in `web/` and re-verified, and **after every retrain
  `tools/export_web_model.py` must be re-run** — the files in `frontend/public/web/` are a snapshot,
  and a stale one serves the old model without error. **Pages-only features have no Python twin and
  no check**: live per-frame playback (`web/player.ts` via `useLivePlayback.ts`) and the per-second
  timeline (`board.ts fineTimeline` / `feed.ts fineRows`). Node 24 runs the `.ts` files directly
  (type stripping), which is why `erasableSyntaxOnly` is on and `web/` imports carry `.ts` extensions.
- **`frontend/src/i18n.ts`** switches the UI to English unless the browser's first preferred
  language (`navigator.languages[0]`) is Chinese (`?lang=en|zh` forces it). Hard-coded UI text goes
  through `T("中文原文")` (the Chinese string is the key). Generated sentences — reasons, warnings,
  notes, errors — are **not** translated at the source: `web/*.ts` and `api.py` must keep emitting
  the exact Chinese that `test:golden` compares against Python. Instead `src/api/client.ts` runs
  every response through `localize()` in English mode, which matches each sentence against Chinese
  templates (compiled to regexes) and fills the English one; anything it doesn't recognise stays
  Chinese (dev mode logs `[i18n] 没有英文`). Paths that bypass `client.ts` (`useLivePlayback.ts`) call
  `localize()` themselves. Changing a sentence in `explain.py`/`api.py`/`web/` therefore needs the
  matching template in `i18n.ts` updated too. Rune names are fetched in the UI locale; the default
  `zh_CN` matches Python for `test:diff`.

## Invariants that are easy to break silently

Nearly every bug this project has had produced a plausible wrong number rather than an exception.
`tests/test_regressions.py` exists to pin these down — each test names the trap it guards. If one
goes red, work out whether that trap is back before changing the test. Not every bullet below has a
test (the year list, the LTA copies, Stage 3's no-number rule and the board-on-finished-games rule
do not), so a refactor near them needs a manual check.

- **Training and serving must read the same data.** The five-year list is a literal in five places
  (`api.DATA`, `train.DATA`, `ingame_train.YEARS`, `xregion.YEARS`, `ingame_model`'s `__main__`);
  only `xregion` checks itself against `train.DATA`, and `FeatureStore.from_csv` /
  `build_training_matrix` skip a missing file silently. Loading a different span does not error; it
  just makes player-champion familiarity systematically low. Adding a year means editing all five.
- **`raw["league"].replace({"LTA N": "LCS", "LTA S": "CBLOL"})`** must survive any refactor of the
  loaders. Without it a full year of NA matches silently disappears. It has four runtime copies
  (`feature_store.py` ×2, `ingame_model.py`, `xregion.py`); only xregion's is tested.
- **Never feed OE's `blue_firstpick`, `atakhans` or `turretplates` into a feature** — present,
  non-null, and semantically dead or year-dependent (README's data-problems table). `EXCLUDED` in
  `feature_store.py` matches exact names only, so it would not catch a prefixed `diff_`/`b_`/`r_`
  column; what actually keeps them out is that `ROLL_STATS` does not list them.
- **`frames[-1].gameState` is the authoritative liveness signal**, not the schedule's `state`.
  Likewise `frame_sides()` takes blue/red from the frame's `gameMetadata`, not from the persisted
  API — the two disagree about 8% of the time (for the whole of Worlds 2024/2025 the persisted
  sides are inverted), and following the wrong one attributes one team's numbers to the other
  without any error.
- **No fuzzy team matching.** `TEAM_ALIASES` is explicit, and `map_team` returns `None` for any name
  not in `known` (an empty list included); only `known=None` skips validation. A `difflib` fallback
  once mapped `Beijing JDG Esports` onto `LNG Esports`. Alias keys match case-sensitively while
  `known` matches case-insensitively, so an all-caps getEventDetails name misses its alias (which is
  why `_xregion_series` takes names from the schedule). Every Stage 1/2 mapping (schedule, upcoming,
  live list, board) gets its list from `api._known_for(league)`: the league's own teams for a major,
  `store.known_teams(None)` otherwise (exported as `teams.json` → `known_all`). Model C maps against
  its state's team names (`STATE["xregion_known"]`), and `collect_live.oe_correct` against OE's team
  names for that league.
- **Stage 3 never changes the probability.** The number always comes from Stage 2 or Stage 4;
  the agents emit classified risk points only, and `external_fact` entries without a URL are
  downgraded.
- **Accept a modelling change only at `|t| > 2.5`** on the paired walk-forward gate
  (`research/harness.py`, `paired_t` / `gate`). Do not use `ratio = |Δ| / SD` — it is an effect
  size and never accumulates evidence. Single-holdout comparisons are noise at this data volume.
  The 2.5 rule governs accuracy claims; display-continuity changes (the board blend at t = +0.83,
  the C anchor at t = +2.27) were adopted as "not significantly worse" with the user's explicit
  sign-off — such a change needs that sign-off, not just a number.
- **New gate scripts get their CSV paths from `research/multiyear_gate.data_path()` / `load_year`**
  (`load_year` returns `None` for a missing file and keeps only the four majors). Nine older
  research scripts build their own path; when a year is added or a file goes missing, check them —
  that is how a year silently drops out of a comparison.
- Stage 4 trains on **all leagues, not just the four majors** (`TARGET = None` in
  `ingame_model.py`, gated at `t = +5.19`), while `/predict`, `/predict/ingame`, `/session/start`
  and `/debate` still only accept the four (the board also serves international events — next
  bullets). Narrowing the training pool back to the majors would undo a measured gain.
- **International boards: Stage 1/2 only when both teams share a home league; Stage 4 always, with
  the *event* league.** `api.pregame_league` (twin `web/board.ts`) decides and is exposed as
  `match.pregame_league`. A same-home-league match runs Stage 1/2 with the **home league**; when it
  returns `None` (cross-region, or a team with no major-league history) the board has no Stage 1/2
  anywhere and Stage 4 runs with `include_pre=False`, because Stage 1/2 have no skill cross-region
  (`research/gate_international.py`: 523 games, accuracy ~50%, calibration slope ~0). Passing the
  home league to Stage 4 (its `is_*` must stay all-zero, as international rows were in training),
  or letting it fall back to `_pre_context`, would not error. Boards with an unmapped team still get
  Stage 4 but are never logged (`prediction_log.resolve` could never match them). The first three
  minutes never reach `test:diff`; `frontend/scripts/golden/intl_cases.json` pins them. DCGI needs no
  `LEAGUE_TO_OE` entry: OE labels it `DCGI`, which `oe_league` passes through — never map it to
  `DCup`, OE's December LPL cup.
- **Cross-region boards use model C (`xregion.py`), and only in its in-event online form.** Only the
  version that replays the running event's finished games passed the gate
  (`research/gate_cross_region.py`: series t = +2.87, a thin pass; the OE-only next-day version
  fails, and OE lags about a day). If the event schedule can't be fetched, or the series score lags
  the game number (`withheld: "score_lag"`), show no C — never fall back to `artifacts/xregion.json`
  alone. There is no post-draft line (variant D failed). Before frames exist C shows the
  side-neutral mean, because its intercept is the blue-side edge. Domestic and same-home-league
  boards must stay byte-identical and send none of C's requests (`XregionBoard`). Every `exp` from
  the exported state goes through `xregion.portable_exp` / `web/xregion.ts exp`, never
  `math.exp`/`Math.exp`: libms differ by an ulp, and `test:diff` can't tell that from a porting bug;
  the gate and `build_state` deliberately stay on `math.exp`. Replay rules and numbers live in the
  `xregion.py` docstrings (pinned by `golden/xregion_cases.json` and `XregionReplay`).
- **Isotonic calibration is Stage 4 only.** It passed on ~35k snapshots (Brier `t = +6.91`) and
  failed on Stage 1/2's ~1.7k-game calibration set (`t = -4.74`, overfits exactly as README
  describes; `research/gate_calibration{,_pre}.py`).
- **The "probability is clipped" warning keys off the calibrated output** (`IngameModel.clip_warning`
  vs `p_min`/`p_max`), never a minute/golddiff rule. A stale warning is worse than none — Stage 3
  and the UI quote it as fact.
- **Never call `/esports/board` on a finished game in a process that logs predictions.**
  `esports_board` would write final-frame p ≈ 0.01/0.99 rows into `predictions/log.jsonl`, the
  scoring file. `tools/board_oracle.py` stubs `log_prediction`, and `collect_live.log_predictions`
  (the unattended writer) only requests matches with an `in_game` game; new callers need the same
  guard.
- **`daily_update.run()` forces `PYTHONIOENCODING=utf-8` on its children.** Otherwise children on
  Chinese Windows write GBK, the `已更新` / `games` / `个快照` parses fail, and every run "succeeds"
  without retraining or count-gating (this happened 2026-09-04 → 09-12). The count checks depend on
  those print formats: change `train.py`'s "N games" or `ingame_train.py`'s "→ N 个快照" line and the
  check is skipped silently.
- **Live names must be mapped to OE names before any lookup** (`feature_store.champion_key` inside
  the lookups, `esports_feed.oe_player_name`; twin `web/names.ts`). Raw lolesports names
  (`IGTheShy`, `LeeSin`) silently find nothing: familiarity goes to 0, 13% of player-champion slots
  lose their champion, and Stage 2 was no better than Stage 1 until this was fixed
  (`research/gate_live_names.py`: 59.5% → 61.7%, Brier `t = +3.16`).
- **The board blends its pre-game anchor into Stage 4 over minutes 3–15** (post-draft, else
  pre-game, else C; `api._blend_prior` / `web/ingame.ts blendPrior`), on board paths only —
  `/predict/ingame` stays raw. Adopted for continuity, not accuracy (`research/gate_anchor.py`).
  Change both twins; `test:diff` catches a mismatch.
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
  any working directory. The repository is public: machine-specific values (hosts, key paths, cache
  dirs) come from env vars and never appear in the source.
- Env vars are read from the process environment only — nothing loads an env file. On this machine
  they must be user environment variables (`setx`), which the scheduled tasks inherit;
  `deploy/service.env.example` is only the systemd `EnvironmentFile` template. `LOL_ARTIFACTS`
  redirects training output (`daily_update.py` uses it to train into `artifacts_staging/`). Others:
  `NVIDIA_API_KEY` (Stage 3), `TAVILY_API_KEY` / `BRAVE_API_KEY` / `SERPER_API_KEY` (retrieval),
  `DEBATE_KEY` (gates `/debate` behind an `X-Debate-Key` header), `LOL_STALE_DAYS`, `LOL_PRED_LOG`,
  `LOL_API` (where `collect_live.py` reaches the API, default `http://127.0.0.1:8000`),
  `LOL_XREGION_CACHE` (research table cache for `research/xregion_data.py`).
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
- **This Windows machine** — four scheduled tasks installed by `deploy/windows_install.ps1`, each
  running `.venv\Scripts\pythonw.exe` on `run_api.py` or `run_task.py <log> <script>` (output →
  `logs/<name>.log`): `LoL-Predict` (the API on 127.0.0.1:8000), `LoL-Collect` (every 2 min),
  `LoL-Update` (`daily_update.py --service LoL-Predict --publish-web`: fetch → retrain into
  `artifacts_staging/` → gate → swap → restart `LoL-Predict` via `schtasks /end` + `/run` → publish
  the site) and `LoL-Backfill` (`research/backfill_late.py`, after the update, since it needs OE
  results as labels). Triggers are fixed in UTC (Update 00:00 and 12:00, Backfill 13:30) and the
  installer converts them to local time; `update_status.json` and the logs use naive local time.
  Tasks run as the logged-in user with no stored credentials, so they need a session (a locked
  screen is fine), and all have `-StartWhenAvailable` (a missed live game cannot be sampled
  afterwards). Never wrap a task in `cmd.exe`: the window flashes, and `schtasks /end` leaves an
  orphaned old-model API holding :8000 that still passes `/health`. Change task definitions only in
  `windows_install.ps1` — re-running it runs the unittest suite first, then unregisters and recreates
  all four, so hand edits via `Set-ScheduledTask` are lost. Its header lists the differences from
  the systemd units that would otherwise fail silently.
- **The tasks run straight from this working tree.** Uncommitted edits to `train.py`,
  `feature_store.py`, `api.py`, … are what `LoL-Update` retrains and gates with, and `LoL-Predict`
  loads them on its next restart (twice a day). The site publish works in this checkout too: it fails
  unless `main` is checked out, discards hand edits in `frontend/public/web/` when the export or
  `test:golden` fails, and stalls while an unpushed local commit touches anything else — all visible
  only in `update_status.json` → `web_publish`.
- **What a run skips.** If no CSV changed, `daily_update` stops after the fetch with exit 0: no
  retrain, no xregion rebuild, no `prediction_log.resolve`, no site publish. An export or web-code
  change reaches the site only when the data changes, or by running `tools/publish_web.py` (or
  `daily_update.py --force`) by hand. If `--service` names no existing task, the restart is skipped
  and the run still records `ok` while the API serves the old model. `xregion.py --build` is a side
  path: when it or `sanity_check` fails, yesterday's `artifacts/xregion.json` stays whole (never mix
  old and new — ratings and the `events` de-dup list come from one build), the failure shows in
  `/health` → `update.xregion`, and Stage 1/2/4 and the site still update. `/health` →
  `xregion.oe_asof` shows how far behind the live state is.
- The Python side still has to stay correct with no server: it is the reference `test:golden` and
  `test:diff` hold the browser implementation to, and the local tasks run it.

Before trusting any number, check `/health` → `data_through` and `update` (`level` comes from
`last_fetch_ok`): `data_freshness` can say fresh while the update pipeline is dead
(`update_status.py`).

### If a server comes back

Systemd units in `deploy/`; a server runs four of them — `lol-predict` (the API), `lol-update`
(twice-daily `daily_update.py`), `lol-collect` (every 2 min), `lol-backfill` (daily, after the
update). The unit files explain each setting in comments (single worker — the feature store is
2–4 GB resident; `TimeoutStartSec=300` — cold start loads five years of CSV; `EnvironmentFile`;
systemd allows no trailing comments). The trap is `lol-update`: `NoNewPrivileges`,
`ProtectKernelTunables`, `ProtectControlGroups` and `RestrictSUIDSGID` each silently block its
`sudo systemctl restart`, which also needs `/etc/sudoers.d/50-lol-restart` (installed by hand).
`deploy/install.sh` is stale (it checks `venv/` where the units use `venv312/`, and only three model
files) and installs only `lol-predict`. `deploy/wsl_migrate.sh` is a dormant WSL2 path; the Windows
tasks exist because WSL2 broke on 2026-08-30.

`lol-predict` binds **127.0.0.1**, not the public interface — `/esports/board` fans out to upstream
with the shared lolesports key, and getting that key rate-limited kills the data source. Reach it
over an SSH tunnel or an authenticated reverse proxy.
