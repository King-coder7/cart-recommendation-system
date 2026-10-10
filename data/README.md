
Production-style cart-based recommendation system designed to forecast the products a customer is likely to add next based on historical purchase patterns, temporal context, and evaluation of recommendation quality.

The project uses shopping cart histories to build co-purchase relationships between products and evaluates recommendation performance using historical answer sets. A baseline recommender is implemented using conditional probabilities derived from repeated product co-occurrence within cart sessions. This provides a strong benchmark for understanding product affinity and temporal drift across different time windows.

The repository is structured to support experimentation with different training periods and evaluation dates, including validation against 2016 datasets. It is intended as a foundation for more advanced recommendation approaches, including temporal modeling, cold-start handling, and deployment as a production-ready API.

## Phase 1: Baseline Co-Purchase Recommender

A co-purchase frequency recommender was implemented to score products based on how often they appear together in historical shopping sessions. For a cart containing items {A, B, C}, the model recommends items most frequently bought with any of {A, B, C}, ranked by conditional probability.

### Data & Evaluation
- Training data: historical cart sessions from 2011-2017
- Test data: answer sets from January and December 2016
- Evaluation metric: HitRate@10

### Best Phase 1 Result
The strongest baseline result was achieved using training data from 2011-2017 October.

- January 2016: HitRate@10 = 0.5267 (148/281)
- December 2016: HitRate@10 = 0.5642 (246/436)

This confirms that a simple co-purchase model can achieve a strong baseline and serves as an effective benchmark for more advanced recommendation methods.

## Phase 2: Model Drift Analysis

### Research Question
How stable is a co-purchase recommender when deployed over time in a real-world e-commerce environment?

### Drift Theory
Customer behavior is not stationary. Products change, preferences shift, and recommendations can silently degrade when the underlying data distribution changes over time. This is known as model drift.

There are two primary types relevant to this project:
- Concept drift: the relationship between product patterns and customer behavior changes over time.
- Data drift: the input data distribution changes because of new or discontinued products, shifting purchasing behavior.

### Evaluation Setup
The best Phase 1 model was evaluated across an 11-month gap using the same recommender logic and the same evaluation metric.

- Training data: cart_export_17_10.csv
- January 2016 test set: answer_2016-01.csv
- December 2016 test set: answer_2016-12.csv
- Metric: HitRate@10

### Results
| Period | HitRate@10 | Hits | Test Cases |
|---|---:|---:|---:|
| January 2016 | 0.5267 | 148/281 | 281 |
| December 2016 | 0.5642 | 246/436 | 436 |
| Improvement | +0.0375 | +98 hits | +7.1% |

### Interpretation
The model did not degrade over the 11-month period; instead, it improved by 7.1%. This suggests that the co-purchase relationships in the data remained relatively stable and that seasonal demand patterns may have aligned better with the December evaluation period.

### Product Catalog Stability
The test data was almost entirely composed of known products from the training history.

- January 2016: 0 new products, 197 discontinued products
- December 2016: 0 new products, 179 discontinued products
- In both cases, the test sets used only previously known products; no cold-start new-product cases appeared.

### Why the Model Stayed Strong
The model remained effective because the underlying product relationships were still consistent over time. The strong December result likely reflects seasonal purchase behavior that aligns with recurring historical co-purchase patterns. The absence of new products also reduced the risk of input drift and prevented cold-start failures.

### Production Implications
This result suggests that a co-purchase recommender based on known product relationships can remain effective for an extended period without retraining, especially when the catalog is stable. However, the model should still be monitored periodically because real-world drift eventually appears through:
- product life cycles,
- changing customer preferences,
- new catalog additions,
- seasonal variation.

A reasonable monitoring strategy would be to retrain quarterly or semi-annually and to monitor HitRate@10 on a validation set. If performance drops materially over a sustained period, a fresh training window should be used.

## Future Work
The next steps for the project include:
- implementing temporal weighting to emphasize recent co-purchase behavior,
- comparing fixed windows of historical data to determine the most stable training period,
- adding more advanced models such as collaborative filtering, sequence models, or hybrid recommenders,
- deploying the final model through a production-ready API.

This project focuses on:
- cart/session-based product recommendation,
- co-purchase signal extraction from historical transaction data,
- evaluation using HitRate@K,
- temporal drift analysis across training windows,
- extensibility toward production deployment and API serving.