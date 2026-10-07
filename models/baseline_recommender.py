"""Baseline cart-based recommendation model.

This recommender uses co-purchase frequency: given items in a cart,
recommend items most often purchased together with those items.

For the answer file format, we use previous_products as context and
predict from answer_product_ids.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_product_list(value: str) -> List[int]:
    """Parse a string like '[227,228]' into [227, 228]."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1]
        try:
            return [int(x.strip()) for x in value.split(',') if x.strip()]
        except ValueError:
            return []
    return []


def load_cart_history(path: str | Path) -> Dict[int, List[int]]:
    """Load training data: group by cart_session -> list of product_ids."""
    df = pd.read_csv(path)
    
    # Group by cart_session (which represents a single purchase transaction)
    cart_items: Dict[int, List[int]] = defaultdict(list)
    for _, row in df.iterrows():
        session = int(row['cart_session'])
        product_id = int(row['cart_product_id'])
        cart_items[session].append(product_id)
    
    # Remove duplicates within each session, preserve order
    return {sid: list(dict.fromkeys(products)) for sid, products in cart_items.items()}


def build_co_purchase_model(train_carts: Dict[int, List[int]]) -> Dict[int, Counter]:
    """Build model: product_id -> {related_product: count}."""
    model: Dict[int, Counter] = defaultdict(Counter)
    
    for products in train_carts.values():
        if len(products) < 2:
            continue
        
        unique = list(dict.fromkeys(products))
        for product in unique:
            for other in unique:
                if product != other:
                    model[product][other] += 1
    
    return model


def recommend_items(
    model: Dict[int, Counter],
    cart_items: List[int],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Given cart items, recommend top-k related products by conditional probability."""
    scores: Dict[int, float] = defaultdict(float)
    seen = set(cart_items)
    
    for item in cart_items:
        neighbors = model.get(item, Counter())
        total = sum(neighbors.values())
        if total == 0:
            continue
        
        for candidate, count in neighbors.items():
            if candidate not in seen:
                scores[candidate] += count / total
    
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


def load_answer_file(path: str | Path) -> List[dict]:
    """Load answer file with previous_products and answer_product_ids."""
    df = pd.read_csv(path)
    
    results = []
    for _, row in df.iterrows():
        previous = parse_product_list(row['previous_products'])
        answer = parse_product_list(row['answer_product_ids'])
        
        if previous and answer:
            results.append({
                'previous_products': previous,
                'answer_product_ids': set(answer),
            })
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a cart-based recommendation model.")
    parser.add_argument("--train-data", type=str, required=True, help="CSV containing historical cart purchases")
    parser.add_argument("--test-data", type=str, required=True, help="CSV containing answer / validation carts")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to rank")
    args = parser.parse_args()

    print(f"\n=== Phase 1: Baseline Cart Recommender ===\n")
    
    print(f"Loading training data from {args.train_data}...")
    train_carts = load_cart_history(args.train_data)
    print(f"Loaded {len(train_carts)} unique sessions with {sum(len(p) for p in train_carts.values())} total product instances")
    
    print(f"Building co-purchase model...")
    model = build_co_purchase_model(train_carts)
    print(f"Model contains {len(model)} unique products")
    
    print(f"Loading answer file from {args.test_data}...")
    test_cases = load_answer_file(args.test_data)
    print(f"Loaded {len(test_cases)} test cases")
    
    hits = 0
    for test_case in test_cases:
        previous = test_case['previous_products']
        answer = test_case['answer_product_ids']
        
        recs = recommend_items(model, previous, top_k=args.top_k)
        rec_items = {product for product, _ in recs}
        
        # Check if any recommended product is in the answer set
        if rec_items & answer:
            hits += 1
    
    hitrate = hits / len(test_cases) if test_cases else 0.0
    print(f"\n=== Results ===")
    print(f"HitRate@{args.top_k}: {hitrate:.4f} ({hits}/{len(test_cases)})\n")


if __name__ == "__main__":
    main()
