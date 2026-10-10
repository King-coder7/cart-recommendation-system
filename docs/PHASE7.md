# Phase 7 — Automating training and filtering

**Business question:** Can this system run without constant human intervention?

Yes, if inventory rules refresh every night **without** pulling a cart-history
report, and the model retrains on a **monthly** (or trigger) cadence after a
human approves the extract cost.

Phase 7 uses `data/cart_export_19_05.csv` (latest extract, through 2019-07) and
the four nightly files in `catalog/`.

## Pipeline

```mermaid
flowchart TD
  inv[Inventory / merchandising feeds] --> nightly[nightly_refresh.py]
  nightly --> json[catalog/*.json]
  json --> reload[POST /v1/admin/reload-rules]
  reload --> api[Running API - no restart]

  human[Human approves extract spend] --> wh[Warehouse cart report ~4h $1.21-$4.53]
  wh --> csv[cart_export_19_05.csv]
  csv --> monthly[monthly_retrain.py]
  monthly --> cand[recommender.candidate.joblib]
  cand --> gate{HitRate@10 at least 0.40?}
  gate -->|yes| bak[Copy live artifact to .bak]
  bak --> live[Promote recommender.joblib]
  live --> restart[Restart / redeploy container]
  gate -->|no| page[Page a human - keep previous model]
```

| Job | Script | Input | Output |
|---|---|---|---|
| Nightly filter refresh | `scripts/nightly_refresh.py` | latest **local** cart file + inventory semantics | `catalog/*.json`, optional API reload |
| Monthly / triggered retrain | `scripts/monthly_retrain.py` | `cart_export_19_05.csv` after approval | `artifacts/recommender.joblib` + `.bak` + `last_retrain_report.json` |
| Catalog generator | `models/generate_catalog_lists.py` | same cart file | discontinued, OOS, new, specials |

Safe deploy: train a **candidate** artifact, score a holdout, **backup** the live
joblib, then copy candidate → live only if the gate passes. A bad promote can
roll back to `artifacts/recommender.joblib.bak`.

## Nightly files (as of 2019-07 from `cart_export_19_05.csv`)

These stand in for feeds that merchandising/inventory would emit each night:

| File | Count | Meaning in this system |
|---|---:|---|
| `discontinued.json` | 97 | No sales in the last 12 months |
| `out_of_stock.json` | 42 | Sold last month, missing this month (temporary gap) |
| `new_items.json` | 12 | First sale in the last 12 months (371–382, 388, …) |
| `special_items.json` | 3 pins | New-arrival merchandising (388, 382, 381) |

```bash
python3 scripts/nightly_refresh.py --cart-data data/cart_export_19_05.csv --skip-reload
```

The nightly job **must not** request a warehouse cart extract. That report is
the expensive path.

## What runs when

### Nightly (no human)

1. Rebuild the four JSON files from the latest **already-downloaded** extract
   (or from a cheap inventory dump, if ops replaces the generator).
2. `POST /v1/admin/reload-rules` so discontinued/OOS/promos change **without**
   retraining or restarting ECS.
3. Alert if `generated_at` is missing or the job fails.

Cost: minutes of compute, not $1–$5.

### Weekly (light human)

- Glance at JSON sizes and `artifacts/last_retrain_report.json`.
- Page if `out_of_stock.json` is empty (feed broken) or huge (over-blocking).
- Page if `new_items.json` is empty after a known launch.

### Monthly (human approves spend, job is automatic after that)

1. Someone approves **one** cart-history report (~4 hours, **$1.21–$4.53**).
2. `python3 scripts/monthly_retrain.py --train-data data/cart_export_19_05.csv --request-extract`
3. Gate: HitRate@10 on holdout ≥ 0.40 (and not a collapse vs last month).
4. If pass: backup live joblib, promote candidate, **redeploy** the ECS task
   (new image or bind-mounted artifact). Reload rules is not enough for a new
   model; the process must load the new joblib.
5. If fail: keep the previous artifact and page.

Latest run on `cart_export_19_05.csv` vs November 2017 holdout:

| | |
|---|---:|
| Sessions | 32,510 |
| Products | 364 |
| HitRate@10 | **0.6042** (1206/1996) |
| Promoted | yes |

Treat that HitRate as a **smoke test**. November 2017 is not a 2019 cart. When
a newer answer file exists, swap `--holdout-data`.

### Triggered (human if a new extract is required)

Same as Phase 5: ≥15 new SKUs in a week, HitRate drop ≥5pp, big catalog purge,
or a launch whose SKUs are absent from the artifact.

## What requires a human

| Decision | Why it is not fully automatic |
|---|---|
| Buying a cart extract | Direct dollars and a 4-hour warehouse job |
| Promoting a candidate that **fails** the HitRate floor | Could ship a broken ranker to the website |
| Editing `special_items.json` campaigns | Merchandising judgment (holiday, new arrival) |
| Choosing ARM64 vs x86 and ECS redeploy | Infra; one-click after the artifact is good |
| Raising min/max ECS tasks | Cost vs latency tradeoff |

Everything else — hiding OOS, dropping discontinued, injecting one new-item
slot — runs unattended.

## Risks and failure modes

| Risk | What you would see | Mitigation |
|---|---|---|
| Stale JSON (>26h) | Dead PDPs return (OOS/discontinued leak) | Alert on `generated_at`; nightly pager |
| Nightly job requests a cart extract | Bill spikes; 4h delay | Nightly script has **no** extract flag |
| Empty `new_items.json` after a launch | Exploration off; popularity bias returns | Weekly check; merch can pin in specials |
| Pins fill all 10 slots | No model recommendations | Cap pins at 3 in the generator |
| Promote a corrupt joblib | `/health` 500; empty recs | `.bak` rollback; health check on `/health` |
| Holdout is the wrong year | Gate is noisy | Document 2017 holdout as smoke only |
| ECS task still on old image | New joblib never served | Redeploy after promote; `/health` is not enough to prove model version — check `POST /v1/recommend` verbose or logs |
| Security group / ALB health | 1/1 tasks but Safari fails | Target group port **8080**, listener **8000**, path `/health` |
| Lab credentials expire | CLI/ECR push fails | Learner Lab: copy a fresh AWS Details block |

## How this answers the business question

The website does not need an engineer every morning. Inventory and campaigns
move through JSON and a reload endpoint. The expensive signal — full cart
history — is pulled **monthly** (or on a churn alarm) after someone accepts
~$2 and four hours. Promotion is gated so automation cannot silently replace
a working model with a bad one. That is unattended enough for production, with
humans only on spend, failed gates, and merchandising intent.
