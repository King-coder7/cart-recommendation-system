# Phase 5 — More data and the cold-start problem

**Business question:** How do we recommend products that have never been sold before?

## Retrain and November 2017 evaluation

The production scorer is the Phase 3a temporal co-purchase model (180-day half-life). It was retrained on `data/cart_export_17_10.csv` and scored on `data/answer_2017-11.csv` using the same HitRate@10 metric as Phases 1–3.

```bash
python3 models/evaluate_phase5.py \
  --train-data data/cart_export_17_10.csv \
  --test-data data/answer_2017-11.csv
```

| Model | Train | Test | HitRate@10 |
|---|---|---|---:|
| Phase 1 unweighted co-purchase | `cart_export_17_10.csv` | Nov 2017 (1,996 cases) | 0.5772 (1152/1996) |
| **Phase 3a temporal, 180-day** | same | same | **0.5862 (1170/1996)** |
| Phase 1, 2011–2015 only | `cart_export_11-15` window | Jan 2016 | 0.3986 (112/281) |

Retraining on the later extract is not a small tweak. HitRate jumps from ~0.40 on the 2015-trained January 2016 split to **0.59** on November 2017 because the 2017 test carts actually contain 2017 SKUs.

## Catalog churn (what the extract shows)

The assignment brief cites 32 new products in 2017, 4 discontinued, +13.7% SKUs in 2016 and +7% in 2017. Counting distinct `cart_product_id` values in `cart_export_17_10.csv` / `cart_export_19_05.csv`:

| Through year | Distinct SKUs | New vs prior year | Growth |
|---|---:|---:|---:|
| 2015 | 253 | — | — |
| 2016 | 296 | 43 | **+17.0%** |
| 2017 | 327 | **31** | **+10.5%** |

- **31 products first appear in 2017:** 315–342, 344–346 (brief said 32; one SKU may live only in a merchandising list, not in carts).
- **18 SKUs sold in 2016 but not in 2017:** 52, 141, 182, 191, 193, 219, 221, 222, 243, 249, 252, 271, 290–294, 300 (brief said 4 discontinued; the cart extract is a stricter “no sales that year” cut).
- **733 / 1,996** November cases have a 2017-new SKU in the *answer*.
- After retraining on 2017 history, **1,386 / 1,996** top-10 lists include at least one 2017-new SKU, and HitRate on the 733 cold-answer cases is **0.6508**.

Direction matches the brief: the catalog is growing every year, and a frozen co-purchase model cannot name items it has never seen.

## Why co-purchase fails on true cold-start

The model only scores pairs that appeared together in a cart. If product 388 has **zero** sales:

1. It never enters the co-purchase graph.
2. Conditional probability for “bought with 240” is undefined.
3. Popularity fallback also ignores it (popularity is sales count).
4. The API would never return it unless the **serving layer** injects it.

That is popularity bias in this system: recommendations concentrate on SKUs with history. EquiRate-style balanced injection is the serving-layer analogue — force a small number of new/long-tail items into the list so they can earn their first co-purchase edges.

A model trained only through 2016 would give those 31 SKUs **no score at all**, even though 733 November answers include them. Retraining after they start selling is necessary, but it does not help the *first* week a SKU is on the site.

## Retraining schedule

Cart-history extracts take ~4 hours and cost **$1.21–$4.53** each. Nightly retraining is wasteful. Phase 2 also showed co-purchase relationships stayed stable for ~11 months on a fixed catalog. Catalog *growth* is the real reason to retrain, not slow drift of old pairs.

| Cadence | What runs | Retrain the model? |
|---|---|---|
| **Nightly** | Refresh discontinued / OOS / new / special JSON; reload rules | **No** |
| **Monthly** | One approved cart extract; retrain temporal-180d; holdout HitRate@10 | **Yes** |
| **Triggered (off-cycle)** | See signals below | **Yes**, after spend approval |

**Default: retrain monthly.** That is frequent enough to absorb ~30 new SKUs/year without paying for a 4-hour extract every night, and infrequent enough that Phase 2’s stability result still holds.

### Signals that trigger an extra retrain

Retrain before the next monthly window if **any** of these fire:

1. **`new_items.json` grows by ≥15 SKUs in a week** (2017 added 31 in a year; a launch week that size is a cold-start shock).
2. **HitRate@10 on a frozen holdout drops ≥5 percentage points** vs last promote.
3. **Discontinued or OOS count jumps** (large catalog purge; old neighbors become unsellable).
4. **Merchandising launches a campaign** whose SKUs are not in the current artifact.

Human approval is required to buy a new cart extract and to promote a candidate whose HitRate is below 0.40 or more than 5pp worse than last month.

## Cold-start strategies (implemented in the serving layer)

The model stays a co-purchase scorer. New and unsellable items are handled **after** scoring in `api/engine.py`, driven by nightly files in `catalog/`.

### 1. Promotional overrides (`special_items.json`)

Merchandising pins up to three SKUs (new arrivals, holiday features) at the top of the list if they are in stock and not discontinued. This is how a product with **no sales** still appears on day one: the business chooses it, not the model.

- **Business goal:** launch and campaign visibility; we can sell what we spent to promote.
- **Customer experience:** the first slots can be “featured,” not random junk. Pins are capped at 3 so the rest of the list still looks like “people also bought.”
- **Data collection:** every pin click/add-to-cart is a labeled co-purchase for the next monthly retrain.

### 2. Randomized exploration (`new_items.json`)

One slot in the top-10 is reserved for a uniform sample of in-stock new SKUs that are not already in the cart. This is balanced injection: we do not wait for popularity.

- **Business goal:** give the long tail a chance to sell; reduce concentration on bestsellers.
- **Customer experience:** only **one** explore slot, and only among sellable new items, so relevance does not collapse.
- **Data collection:** exploration is how a never-sold SKU gets its first co-purchase row. Without it, the monthly retrain still cannot see that item.

### 3. Monthly retrain (closes the loop)

Once exploration and promotions generate sales, those pairs enter `cart_export_*.csv` and become normal temporal co-purchase scores. Cold-start handling is temporary. The long-term system should recommend new items *because they now co-occur*, not forever because they are pinned.

## Justification (one paragraph)

The website cannot wait for a co-purchase graph to discover a product that has never sold, and it cannot retrain every night at $1–$5 and four hours per extract. Monthly retraining absorbs catalog growth (~10% in 2017) while Phase 2 showed old relationships stay useful for months. Between retrains, promotional pins protect revenue and launches, and a single randomized new-item slot fights popularity bias without turning the widget into a random catalog. Both mechanisms create the interaction data the next model needs. Discontinued and out-of-stock filters (Phase 6) sit in the same serving layer so we never “solve” cold start by recommending something we cannot sell.
