# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` documents the *methodology* (the walk-forward gate, what was accepted/rejected, the
data problems). Read it before touching the models or the research scripts. This file covers the
operational and structural things that are spread across several files.

## Commands

```bash
pip install -r requirements.txt

python fetch_data.py --years 2025 2026     # refresh season CSVs from Oracle's Elixir (Drive)
python fetch_data.py --list                # list remote files without downloading
python fetch_data.py --auth                # one-time OAuth, needed because the anon quota is shared

python train.py                            # Stage 1 + 2  -> artifacts/model_{pre,post}_draft.json
python ingame_train.py                     # Stage 4      -> artifacts/model_ingame{,_live}.json

uvicorn api:app --reload --port 8000       # UI at /, OpenAPI at /docs
```

Tests — stdlib `unittest`, no pytest, no extra dependency (deliberate; even the front end has no
build step). They run in ~8s and need neither network nor API keys; the debate tests stub the LLM.

```bash
python -m unittest discover -s tests -v
```

Front end (Node only needed locally — the server has none):

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

One class or one test (test method names are Chinese, so `-k` on the class is usually easier):

```bash
python -m unittest discover -s tests -k SideOrdering -v
```

Research gates and ad-hoc tools are run from the project root, never from inside their directory:

```bash
python research/gate_league_mix.py --folds 5 --seeds 3
python research/harness.py
python tools/debate_dryrun.py --dump       # exercises the whole Stage 3 protocol with stubs, no key
```

Unattended pipeline (what the server actually runs):

```bash
python daily_update.py --dry-run           # fetch + train into artifacts_staging/, no swap, no restart
python daily_update.py                     # + metric gate, atomic swap, restart, /health probe
python collect_live.py --status            # live late-game snapshot collector (systemd runs it every 2 min)
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
- **`api.py`** is the whole HTTP surface (~1300 lines, all routes in one file). Models, the feature
  store, the session store, and the esports feed are built once in the `lifespan` handler and live
  in the module-level `STATE` dict. Inference is `_predict_core` / `_ingame_core`; the routes are
  thin wrappers over those two.
- **`esports_feed.py`** wraps the unofficial lolesports endpoints and is the only place that knows
  about their quirks (lagged `startingTime`, the schedule's stale `state`, broadcast-vs-game start,
  team-name aliases). `/esports/board` in `api.py` is the big consumer.
- **`explain.py`** turns field names into sentences. Raw field names must never reach a response
  unless `raw_features: true`; a field with no template is dropped, not shown raw.
- **`debate.py`** (Stage 3) runs the 4-round protocol against NVIDIA NIM. `agents.py` is the older
  advisory layer, still reachable via `/predict?agent_review`. `websearch.py` provides retrieval
  (Tavily / Brave / Serper, probed in that order); with none configured the service still runs.
- **`frontend/`** is the UI source (Vite + React + TypeScript). `npm run build` emits into
  `static/`, which is what `api.py` mounts — **the server has no Node**, so builds happen locally
  and only the artefacts (`static/index.html` + `static/assets/*`) get deployed. `npm run dev`
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
  player×champion); Stage 3 is not in the static build.
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
  `ingame_model.py`, gated at `t = +5.19`), while inference in `api.py` still only accepts the four.
  Narrowing the training pool back to the majors would undo a measured gain.
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
  `log_prediction` with a no-op for exactly this reason.
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
  behind an `X-Debate-Key` header), `LOL_STALE_DAYS`, `LOL_PRED_LOG`. See
  `deploy/service.env.example`.
- `agent_config.json` is generated by `research/screen_agents.py` and its keys are the Chinese role
  names (`协调者` / `补充者` / `验证者` / `质疑者`). It overrides `_DEFAULTS` in `debate.py`, and the
  model IDs in it are the live truth — README's Stage 3 table lists an earlier selection.
- `data/`, `artifacts/`, `backfill/`, `predictions/`, `collected/` hold data, not code.
  `google_credentials.json` / `google_token.json` are real credentials, gitignored, never printed.
- `attic/` is superseded work kept for reference only. Do not import from it.

## Deployment

Systemd units in `deploy/`; the server runs four of them —
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
over an SSH tunnel or an authenticated reverse proxy. README's "Exposure" section describes the
earlier public-port setup and is stale on this point.

Server-side data is authoritative: it has the live-collected snapshots and the freshest CSVs, so a
local run may legitimately differ from what the deployed service reports.
