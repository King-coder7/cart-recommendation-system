# cart-recommendation-system
Production-style cart-based recommendation system with temporal modeling, cold-start handling, and API deployment
## Phase 1: Baseline Co-Purchase Recommender

**Model:** Conditional probability based on product co-occurrence in historical carts.

**Training Data:** 2,175 unique shopping sessions from 2011-2015 containing 252 unique products.

**Evaluation:** Sequential prediction on 2016-01 dataset with 281 test cases.

**Results:** HitRate@10 = 0.3986 (112/281 correct predictions)

**Interpretation:** The baseline model succeeds in recommending the next product ~40% of the time, 
providing a strong foundation for comparison with advanced models (collaborative filtering, 
neural networks, LSTM-based sequential models).