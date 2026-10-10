# cart-recommendation-system

Production-style cart recommender: temporal co-purchase scoring, cold-start
handling, business-rule filtering, and a FastAPI/Docker serving layer.

- Deployment notes: [docs/PRODUCTION.md](docs/PRODUCTION.md)
- Phase 5 (November 2017 / cold start): [docs/PHASE5.md](docs/PHASE5.md)
- Phase 6 (model vs business rules): [docs/PHASE6.md](docs/PHASE6.md)
- Phase 7 (automation): [docs/PHASE7.md](docs/PHASE7.md)
- Executive summary: [docs/Executive_Summary_Module01.pdf](docs/Executive_Summary_Module01.pdf)

## Phase 1: Baseline co-purchase

Conditional probability from historical carts. Strongest full-file baseline on
`cart_export_17_10.csv`:

- January 2016: HitRate@10 = 0.5267 (148/281)
- December 2016: HitRate@10 = 0.5642 (246/436)

Leak-free 2011–2015 train, January 2016 test: HitRate@10 = 0.3986 (112/281).

## Phase 2: Drift

Same model, January vs December 2016: no degradation (0.53 → 0.56). Catalog was
stable; no new products in those test sets.

## Phase 3a: Temporal weighting

On a leak-free January 2016 split (train only through 2015), a 180-day half-life
raises HitRate@10 from **0.3986 → 0.4733**.

## Phase 3c: LSTM

Same split: HitRate@10 = **0.4270**. Better than the unweighted baseline, weaker
than temporal weighting. The API serves the temporal model.

## Phase 4–7: Serving

| Piece | Location |
|---|---|
| HTTP API | `api/main.py` |
| Business rules | `api/engine.py` |
| Nightly catalog JSON | `catalog/*.json` |
| Serialized model | `artifacts/recommender.joblib` |
| Docker | `Dockerfile`, `docker-compose.yml` |
| Jobs | `scripts/nightly_refresh.py`, `scripts/monthly_retrain.py` |

Grader contract:

```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{ "cart": ["240", "200", "277", "78"], "top_n": 10 }'
```

```bash
source .venv/bin/activate
pip install -r requirements-api.txt
python3 models/generate_catalog_lists.py --cart-data data/cart_export_19_05.csv
python3 models/train_serving_model.py --train-data data/cart_export_19_05.csv
uvicorn api.main:app --host 0.0.0.0 --port 8080
```
