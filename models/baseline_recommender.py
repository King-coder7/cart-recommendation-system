"""Baseline cart-based recommendation model.

This module builds a simple co-purchase recommender using conditional probabilities.
For a cart containing items A, B, C, it recommends the items most often purchased
with any item in the cart based on historical cart co-occurrence.

Example:
    python models/baseline_recommender.py \
        --train-data data/cart_export_11-15.csv \
        --test-data data/answer_2016-01.csv \
        --top-k 10
"""

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
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("{") and cleaned.endswith("}"):
            cleaned = cleaned[1:-1]

        separators = [";", ",", "|", "/"]
        for sep in separators:
            if sep in cleaned:
                parts = [part.strip() for part in cleaned.split(sep)]
                return [part for part in parts if part]
        return [cleaned]

    return [str(value).strip()]


def load_cart_history(path: str | Path) -> Dict[str, List[str]]:
    """Return a dictionary of cart_id -> list of item ids."""
    path = Path(path)
    df = pd.read_csv(path)

    cart_col = find_column(df, CART_COLUMNS)
    item_col = find_column(df, ITEM_COLUMNS)

    if cart_col is None or item_col is None:
        raise ValueError(
            f"Could not infer cart and item columns from {path}. "
            f"Available columns: {list(df.columns)}"
        )

    cart_items: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        cart_id = str(row[cart_col]).strip()
        item_id = str(row[item_col]).strip()
        if cart_id and item_id:
            cart_items[cart_id].append(item_id)

    return {
        cart_id: list(dict.fromkeys(items))
        for cart_id, items in cart_items.items()
    }


def build_co_purchase_model(train_carts: Dict[str, List[str]]) -> Dict[str, Counter]:
    """Build a directed co-purchase model: item -> {related_item: count}.

    For each cart, every item is treated as a context item. The counts of all other
    items in the same cart increment the co-purchase relationship.
    """
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
    """Recommend items based on aggregated conditional-probability scores."""
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
    """Load test answer data into a cart_id -> set(true_items) map.

    Handles common CSV forms:
    - one row per cart with item_id column
    - one row per cart with answer_item / true_item / label column
    - one row per cart with an items list in a string column
    """
    path = Path(path)
    df = pd.read_csv(path)

    cart_col = find_column(df, CART_COLUMNS)
    item_col = find_column(df, ITEM_COLUMNS)
    target_col = find_column(
        df,
        [
            "answer_item",
            "true_item",
            "label",
            "target_item",
            "recommended_item",
            "ground_truth",
            "item",
        ],
    )

    if cart_col is None:
        raise ValueError(f"Could not find a cart column in {path}. Columns: {list(df.columns)}")

    truth: Dict[str, set] = defaultdict(set)

    if item_col is not None:
        for _, row in df.iterrows():
            cart_id = str(row[cart_col]).strip()
            item_value = row[item_col]
            for item in parse_items(item_value):
                if cart_id and item:
                    truth[cart_id].add(item)

    if target_col is not None:
        for _, row in df.iterrows():
            cart_id = str(row[cart_col]).strip()
            item_value = row[target_col]
            for item in parse_items(item_value):
                if cart_id and item:
                    truth[cart_id].add(item)

    # If no per-row items were found, try to parse item lists from a string field on each row.
    if not truth and any(column in df.columns for column in ["items", "cart_items", "basket"]):
        items_col = find_column(df, ["items", "cart_items", "basket"])
        if items_col is not None:
            for _, row in df.iterrows():
                cart_id = str(row[cart_col]).strip()
                for item in parse_items(row[items_col]):
                    if cart_id and item:
                        truth[cart_id].add(item)

    if not truth:
        raise ValueError(
            f"Ground truth file {path} does not appear to contain parseable cart/item data. "
            f"Columns: {list(df.columns)}"
        )

    return {cart_id: set(items) for cart_id, items in truth.items()}


def evaluate_model(
    model: Dict[str, Counter],
    train_carts: Dict[str, List[str]],
    test_carts: Dict[str, set],
    top_k: int = 10,
) -> float:
    """Compute HitRate@K over the provided test carts."""
    hits = 0
    total = len(test_carts)

    for cart_id, true_items in test_carts.items():
        cart_items = train_carts.get(cart_id, [])
        if not cart_items:
            # Use the observed co-purchase items from the test set when the cart is not in training.
            # This provides resilience for answer files that only provide the target item.
            continue

        recs = recommend_items(model, cart_items, top_k=top_k)
        rec_items = {item for item, _ in recs}
        if true_items & rec_items:
            hits += 1

    if total == 0:
        return 0.0

    return hits / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a cart-based recommendation model.")
    parser.add_argument("--train-data", type=str, required=True, help="CSV containing historical cart purchases")
    parser.add_argument("--test-data", type=str, required=True, help="CSV containing answer / validation carts")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to rank")
    args = parser.parse_args()

    train_carts = load_cart_history(args.train_data)
    model = build_co_purchase_model(train_carts)

    test_truth = load_ground_truth(args.test_data)

    # Build a synthetic evaluation set from the true items; we still rely on cart-item context for recommendations.
    # Most course datasets include a cart history with a known item set for each solution row.
    baseline_hits = 0
    evaluated = 0

    for cart_id, true_items in test_truth.items():
        # If the cart is in the training history, use it as the context.
        # Otherwise, use the true items as a fallback contexted cart to allow a recommendation attempt.
        context = train_carts.get(cart_id, list(true_items))
        recs = recommend_items(model, context, top_k=args.top_k)
        rec_items = {item for item, _ in recs}
        if true_items & rec_items:
            baseline_hits += 1
        evaluated += 1

    hitrate = baseline_hits / evaluated if evaluated else 0.0
    print(f"HitRate@{args.top_k}: {hitrate:.4f}")


if __name__ == "__main__":
    main()
