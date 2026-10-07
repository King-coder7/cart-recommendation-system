"""Baseline cart-based recommendation model."""

from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


CART_COLUMNS = [
    "cart_id",
    "basket_id",
    "order_id",
    "transaction_id",
    "session_id",
    "cart",
    "cart_session",
    "sale_id",
]

ITEM_COLUMNS = [
    "cart_product_id",
    "item_id",
    "product_id",
    "product",
    "sku",
    "item",
    "sku_id",
]


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def parse_items(value) -> List[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        # Handle list-like strings: [1, 2, 3]
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1]

        # Try common separators
        separators = [";", ",", "|", "/"]
        for sep in separators:
            if sep in cleaned:
                parts = [part.strip() for part in cleaned.split(sep)]
                return [part for part in parts if part]
        return [cleaned]

    return [str(value).strip()]


def load_cart_history(path: str | Path) -> Dict[str, List[str]]:
    path = Path(path)
    df = pd.read_csv(path)

    cart_col = find_column(df, CART_COLUMNS)
    item_col = find_column(df, ITEM_COLUMNS)

    if cart_col is None or item_col is None:
        raise ValueError(f"Could not infer cart and item columns from {path}. Available columns: {list(df.columns)}")

    cart_items: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        cart_id = str(row[cart_col]).strip()
        item_id = str(row[item_col]).strip()
        if cart_id and item_id:
            cart_items[cart_id].append(item_id)

    return {cart_id: list(dict.fromkeys(items)) for cart_id, items in cart_items.items()}


def build_co_purchase_model(train_carts: Dict[str, List[str]]) -> Dict[str, Counter]:
    model: Dict[str, Counter] = defaultdict(Counter)

    for cart in train_carts.values():
        unique_items = list(dict.fromkeys(str(item).strip() for item in cart if str(item).strip()))
        if len(unique_items) < 2:
            continue

        for item in unique_items:
            for other in unique_items:
                if item == other:
                    continue
                model[item][other] += 1

    return model


def recommend_items(
    model: Dict[str, Counter],
    cart_items: List[str],
    top_k: int = 10,
    exclude_current: bool = True,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = defaultdict(float)
    seen = set(str(item).strip() for item in cart_items if str(item).strip())

    for item in cart_items:
        item = str(item).strip()
        if not item:
            continue
        neighbors = model.get(item, Counter())
        total = sum(neighbors.values())
        if total == 0:
            continue

        for candidate, count in neighbors.items():
            if exclude_current and candidate in seen:
                continue
            scores[candidate] += count / total

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


def load_ground_truth(path: str | Path) -> Dict[str, set]:
    """Load test answer data.
    
    Handles answer files with columns like:
    - sale_id, answer_product_ids, previous_products
    - cart_id, item, answer_item
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Try to find sale/cart ID column
    cart_col = find_column(df, CART_COLUMNS)
    
    # Special case: answer files often use sale_id and answer_product_ids
    if cart_col is None and "sale_id" in df.columns:
        cart_col = "sale_id"
    
    if cart_col is None:
        raise ValueError(f"Could not find a cart column in {path}. Columns: {list(df.columns)}")

    # Look for answer product columns
    answer_col = find_column(df, ["answer_product_ids", "answer_item", "true_item", "answer_item", "label", "item"])
    
    if answer_col is None:
        raise ValueError(f"Could not find answer product column in {path}. Columns: {list(df.columns)}")

    truth: Dict[str, set] = {}
    for _, row in df.iterrows():
        cart_id = str(row[cart_col]).strip()
        answer_value = row[answer_col]
        
        items = parse_items(answer_value)
        if cart_id and items:
            truth[cart_id] = set(items)

    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a cart-based recommendation model.")
    parser.add_argument("--train-data", type=str, required=True, help="CSV containing historical cart purchases")
    parser.add_argument("--test-data", type=str, required=True, help="CSV containing answer / validation carts")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to rank")
    args = parser.parse_args()

    print(f"\n=== Phase 1: Baseline Cart Recommender ===\n")
    print(f"Loading training data: {args.train_data}")
    train_carts = load_cart_history(args.train_data)
    print(f"Loaded {len(train_carts)} unique carts")

    print(f"Building co-purchase model...")
    model = build_co_purchase_model(train_carts)
    print(f"Model contains {len(model)} unique items")

    print(f"Loading test data: {args.test_data}")
    test_truth = load_ground_truth(args.test_data)
    print(f"Loaded {len(test_truth)} test cases")

    baseline_hits = 0
    evaluated = 0
    skipped = 0

    for sale_id, true_items in test_truth.items():
        # In the training data, cart_id == sale_id
        # But we also need the context items. For now, use empty context if not in training.
        context = train_carts.get(sale_id, [])
        
        if not context:
            # If sale_id not in training, try using the true items as context (fallback)
            context = list(true_items)
        
        if not context:
            skipped += 1
            continue
        
        recs = recommend_items(model, context, top_k=args.top_k)
        rec_items = {item for item, _ in recs}
        
        if true_items & rec_items:
            baseline_hits += 1
        evaluated += 1

    hitrate = baseline_hits / evaluated if evaluated else 0.0
    print(f"\n=== Results ===")
    print(f"HitRate@{args.top_k}: {hitrate:.4f} ({baseline_hits}/{evaluated})")
    print(f"Skipped: {skipped}\n")


if __name__ == "__main__":
    main()
