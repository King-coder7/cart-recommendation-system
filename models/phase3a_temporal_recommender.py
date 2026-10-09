"""Phase 3a: Temporal-weighted co-purchase recommender.

Same conditional-probability ranking as Phase 1, but co-purchase counts
are weighted by recency using exponential decay:

    weight = 0.5 ** (age_days / half_life_days)

Recent carts contribute more than older ones. An unweighted baseline
is evaluated in the same run for a direct comparison.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_product_list(value: str) -> List[int]:
    """Parse a string like '[227,228]' into [227, 228]."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        try:
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            return []
    return []


def load_cart_history_with_dates(
    path: str | Path,
) -> Tuple[Dict[str, List[int]], Dict[str, datetime]]:
    """Load carts grouped by session, plus a timestamp per session.

    Training files only have year/month, so each session is dated to
    the latest year-month observed for that cart (day = 1).
    """
    df = pd.read_csv(path)

    cart_items: Dict[str, List[int]] = defaultdict(list)
    cart_dates: Dict[str, datetime] = {}

    for _, row in df.iterrows():
        session = str(row["cart_session"]).strip()
        product_id = int(row["cart_product_id"])
        cart_items[session].append(product_id)

        year = int(row["year"])
        month = int(row["month"])
        dt = datetime(year, month, 1)
        if session not in cart_dates or dt > cart_dates[session]:
            cart_dates[session] = dt

    carts = {sid: list(dict.fromkeys(products)) for sid, products in cart_items.items()}
    return carts, cart_dates


def parse_iso_date(value: str, flag_name: str) -> datetime:
    """Parse YYYY-MM-DD, ignoring trailing punctuation from copy-paste."""
    cleaned = value.strip().rstrip(".,;")
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(
            f"Invalid {flag_name} '{value}'. Use YYYY-MM-DD, e.g. 2016-01-01"
        ) from exc


def recency_weight(age_days: float, half_life_days: float) -> float:
    """Exponential decay so that weight halves every half_life_days."""
    if half_life_days <= 0:
        return 1.0
    if age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)


def build_co_purchase_model(
    train_carts: Dict[str, List[int]],
    cart_dates: Dict[str, datetime] | None = None,
    reference_date: datetime | None = None,
    half_life_days: float | None = None,
) -> Dict[int, Counter]:
    """Build product -> {related_product: weighted_count}.

    If half_life_days is None, every co-purchase counts as 1 (Phase 1).
    """
    model: Dict[int, Counter] = defaultdict(Counter)

    for session_id, products in train_carts.items():
        if len(products) < 2:
            continue

        weight = 1.0
        if half_life_days is not None and cart_dates is not None and reference_date is not None:
            age_days = (reference_date - cart_dates[session_id]).days
            weight = recency_weight(age_days, half_life_days)

        unique = list(dict.fromkeys(products))
        for product in unique:
            for other in unique:
                if product != other:
                    model[product][other] += weight

    return model


def recommend_items(
    model: Dict[int, Counter],
    cart_items: List[int],
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Recommend top-k items by summed conditional probability."""
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
        previous = parse_product_list(row["previous_products"])
        answer = parse_product_list(row["answer_product_ids"])

        if previous and answer:
            results.append(
                {
                    "previous_products": previous,
                    "answer_product_ids": set(answer),
                }
            )

    return results


def evaluate(model: Dict[int, Counter], test_cases: List[dict], top_k: int) -> Tuple[int, float]:
    hits = 0
    for test_case in test_cases:
        recs = recommend_items(model, test_case["previous_products"], top_k=top_k)
        rec_items = {product for product, _ in recs}
        if rec_items & test_case["answer_product_ids"]:
            hits += 1

    hitrate = hits / len(test_cases) if test_cases else 0.0
    return hits, hitrate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3a: train and evaluate a temporally weighted cart recommender."
    )
    parser.add_argument("--train-data", type=str, required=True, help="CSV of historical cart purchases")
    parser.add_argument("--test-data", type=str, required=True, help="CSV of answer / validation carts")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to rank")
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=365,
        help="Days until a co-purchase is weighted at 50% of a current one",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="YYYY-MM-DD date treated as 'now' for recency (default: latest training month)",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="YYYY-MM-DD; drop training sessions on or after this date to avoid leakage",
    )
    args = parser.parse_args()

    print("\n=== Phase 3a: Temporal-Weighted Cart Recommender ===\n")

    print(f"Loading training data from {args.train_data}...")
    train_carts, cart_dates = load_cart_history_with_dates(args.train_data)
    n_products = sum(len(p) for p in train_carts.values())
    print(f"Loaded {len(train_carts)} unique sessions with {n_products} total product instances")

    if args.cutoff_date:
        cutoff = parse_iso_date(args.cutoff_date, "--cutoff-date")
        keep = [sid for sid, dt in cart_dates.items() if dt < cutoff]
        dropped = len(train_carts) - len(keep)
        train_carts = {sid: train_carts[sid] for sid in keep}
        cart_dates = {sid: cart_dates[sid] for sid in keep}
        n_products = sum(len(p) for p in train_carts.values())
        print(
            f"Applied cutoff {cutoff.date()}: dropped {dropped} future sessions, "
            f"{len(train_carts)} remain ({n_products} product instances)"
        )

    if args.reference_date:
        reference_date = parse_iso_date(args.reference_date, "--reference-date")
    else:
        reference_date = max(cart_dates.values()) if cart_dates else datetime.now()
    min_date = min(cart_dates.values()) if cart_dates else reference_date
    span_days = (reference_date - min_date).days
    print(f"Training window: {min_date.date()} → {reference_date.date()} ({span_days} days)")
    print(f"Decay half-life: {args.half_life_days:.0f} days")

    print("Building unweighted co-purchase model (Phase 1 baseline)...")
    baseline_model = build_co_purchase_model(train_carts)
    print(f"Baseline model contains {len(baseline_model)} unique products")

    print("Building temporally weighted co-purchase model...")
    temporal_model = build_co_purchase_model(
        train_carts,
        cart_dates=cart_dates,
        reference_date=reference_date,
        half_life_days=args.half_life_days,
    )
    print(f"Temporal model contains {len(temporal_model)} unique products")

    print(f"Loading answer file from {args.test_data}...")
    test_cases = load_answer_file(args.test_data)
    print(f"Loaded {len(test_cases)} test cases")

    baseline_hits, baseline_hr = evaluate(baseline_model, test_cases, args.top_k)
    temporal_hits, temporal_hr = evaluate(temporal_model, test_cases, args.top_k)
    delta = temporal_hr - baseline_hr

    print(f"\n=== Results ===")
    print(f"Baseline  HitRate@{args.top_k}: {baseline_hr:.4f} ({baseline_hits}/{len(test_cases)})")
    print(
        f"Temporal  HitRate@{args.top_k}: {temporal_hr:.4f} ({temporal_hits}/{len(test_cases)})  "
        f"[half-life={args.half_life_days:.0f}d]"
    )
    print(f"Delta vs baseline: {delta:+.4f} ({delta * 100:+.2f} pp)\n")


if __name__ == "__main__":
    main()
