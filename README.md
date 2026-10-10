# cart-recommendation-system

Production-style cart recommender: temporal co-purchase scoring, cold-start
handling, business-rule filtering, and a FastAPI/Docker serving layer.

Full deployment, cold-start, rules, and automation notes live in
[docs/PRODUCTION.md](docs/PRODUCTION.md). Phase 5 (November 2017 retrain,
catalog churn, retraining schedule) is in [docs/PHASE5.md](docs/PHASE5.md).
Phase 6 (model vs business rules) is in [docs/PHASE6.md](docs/PHASE6.md).
Phase 7 (automation) is in [docs/PHASE7.md](docs/PHASE7.md).

## Phase 1: Baseline Co-Purchase Recommender

**Model:** Conditional probability based on product co-occurrence in historical carts.

**Training Data:** 2,175 unique shopping sessions from 2011-2015 containing 252 unique products.

**Evaluation:** Sequential prediction on 2016-01 dataset with 281 test cases.

**Results:** HitRate@10 = 0.3986 (112/281 correct predictions)

## Phase 3a: Temporal weighting

On a leak-free January 2016 split (train only through 2015), a 180-day half-life
raises HitRate@10 from **0.3986 → 0.4733**.

## Phase 3c: LSTM next-item model

Same split: LSTM HitRate@10 = **0.4270**. Better than the unweighted baseline,
weaker than temporal weighting. The API therefore serves the temporal model.

## Phase 4–7: Serving

| Piece | Location |
|---|---|
| HTTP API | `api/main.py` |
| Business rules | `api/engine.py` |
| Nightly catalog JSON | `catalog/*.json` |
| Serialized model | `artifacts/recommender.joblib` |
| Docker | `Dockerfile`, `docker-compose.yml` |
| Nightly / monthly jobs | `scripts/nightly_refresh.py`, `scripts/monthly_retrain.py` |

```bash
source .venv/bin/activate
pip install -r requirements-api.txt
python3 models/generate_catalog_lists.py --cart-data data/cart_export_19_05.csv
python3 models/train_serving_model.py --train-data data/cart_export_19_05.csv
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/v1/recommend \
  -H 'content-type: application/json' \
  -d '{"cart_items":[227,228],"top_k":10,"seed":42}'
```
