"""Phase 3A: Temporal Weighting Recommender.

A stronger baseline than the static co-purchase model described in Phase 1.
The idea is to keep the same recommendation logic but weight product associations
by recency so that newer co-purchase patterns matter more than outdated ones.

The script is intentionally robust:
- If the input CSV contains a date/timestamp column, it uses actual recency.
- If not, it still supports a configurable decay and can infer a reference month
  from filenames such as cart_export_17_10.csv.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def infer_reference_date_from_filename(path: str | Path) -> datetime | None:
    """Infer a likely reference month from filenames like cart_export_17_10.csv."""
    name = str(Path(path).name)
    patterns = [
        r"(\d{4})[-_](\d{1,2})",
        r"(\d{2})[-_](\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            year_part, month_part = match.groups()
            year = int(year_part)
            if len(str(year)) == 2:
                year = 2000 + year if year < 50 else 1900 + year
            month = int(month_part)
            if 1 <= month <= 12:
                return datetime(year, month, 1)
    return None


def pick_column(df: pd.DataFrame, candidate_names: Iterable[str]) -> str | None:
    normalized = {str(col).lower(): col for col in df.columns}
    for name in candidate_names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def parse_datetime(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
    ]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def load_cart_history(path: str | Path, reference_date: datetime | None = None) -> tuple[Dict[str, List[int]], Dict[str, datetime]]:
    """Load sessions and their optional dates.

    Returns a mapping of cart_session -> ordered list of product_ids and a mapping
    of cart_session -> last-seen datetime for recency weighting.
    """
    df = pd.read_csv(path)

    session_col = pick_column(df, ["cart_session", "session_id", "cart_id", "session"])
    product_col = pick_column(df, ["cart_product_id", "product_id", "item_id", "product"])
    date_col = pick_column(df, ["timestamp", "date", "event_date", "purchase_date", "cart_date", "created_at"])

    if session_col is None or product_col is None:
        raise ValueError(f"Expected CSV columns like 'cart_session' and 'cart_product_id' in {path}")

    carts: Dict[str, List[int]] = defaultdict(list)
    session_dates: Dict[str, datetime] = {}

    for _, row in df.iterrows():
        session = str(row[session_col]).strip()
        product = int(row[product_col])
        carts[session].append(product)

        if date_col is not None:
            parsed = parse_datetime(row[date_col])
            if parsed is not None:
                current = session_dates.get(session)
                if current is None or parsed > current:
                    session_dates[session] = parsed

    cleaned = {sid: list(dict.fromkeys(products)) for sid, products in carts.items()}
    return cleaned, session_dates


def build_temporal_model(
    train_carts: Dict[str, List[int]],
    session_dates: Dict[str, datetime],
    reference_date: datetime | None = None,
    half_life_days: int = 365,
) -> Dict[int, Counter]:
    """Build a recency-weighted co-purchase model.

    If session dates are unavailable, all weights fall back to 1.0 so the model
    behaves like the Phase 1 baseline while still supporting temporal weighting
    when data includes dates.
    """
    model: Dict[int, Counter] = defaultdict(Counter)

    if reference_date is None:
        # Fall back to the latest observed date, if available; otherwise, use a neutral reference.
        reference_date = max(session_dates.values()) if session_dates else None

    for session_id, products in train_carts.items():
        if len(products) < 2:
            continue

        unique = list(dict.fromkeys(products))
        session_date = session_dates.get(session_id)
        weight = 1.0
        if session_date is not None and reference_date is not None:
            delta_days = max((reference_date - session_date).days, 0)
            weight = 2 ** (-delta_days / half_life_days)

        for product in unique:
            for other in unique:
                if product != other:
                    model[product][other] += weight

    return model


def recommend_items(model: Dict[int, Counter], cart_items: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
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
    parser = argparse.ArgumentParser(description="Phase 3A: Temporal weighting co-purchase recommender.")
    parser.add_argument("--train-data", type=str, required=True, help="Historical cart transactions CSV")
    parser.add_argument("--test-data", type=str, required=True, help="Answer/validation CSV")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--half-life-days", type=int, default=365, help="Recency half-life in days")
    parser.add_argument("--reference-date", type=str, default=None, help="Optional date to anchor recency weighting; format YYYY-MM-DD")
    args = parser.parse_args()

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)

    inferred_ref = infer_reference_date_from_filename(train_path)
    ref_date = None
    if args.reference_date:
        ref_date = datetime.strptime(args.reference_date, "%Y-%m-%d")
    elif inferred_ref is not None:
        ref_date = inferred_ref

    print(f"\n=== Phase 3A: Temporal Weighting Recommender ===\n")
    print(f"Loading training data from {train_path}...")
    train_carts, session_dates = load_cart_history(train_path, reference_date=ref_date)
    print(f"Loaded {len(train_carts)} unique sessions with {sum(len(p) for p in train_carts.values())} total product instances")

    print(f"Building temporal co-purchase model (half-life = {args.half_life_days} days)...")
    model = build_temporal_model(train_carts, session_dates, reference_date=ref_date, half_life_days=args.half_life_days)
    print(f"Model contains {len(model)} unique products")

    print(f"Loading answer file from {test_path}...")
    test_cases = load_answer_file(test_path)
    print(f"Loaded {len(test_cases)} test cases")

    hits = 0
    for test_case in test_cases:
        previous = test_case['previous_products']
        answer = test_case['answer_product_ids']
        recs = recommend_items(model, previous, top_k=args.top_k)
        rec_items = {product for product, _ in recs}
        if rec_items & answer:
            hits += 1

    hitrate = hits / len(test_cases) if test_cases else 0.0
    print(f"\n=== Results ===")
    print(f"HitRate@{args.top_k}: {hitrate:.4f} ({hits}/{len(test_cases)})")
    print(f"Reference date: {ref_date.strftime('%Y-%m-%d') if ref_date else 'not available'}")
    print(f"Half-life: {args.half_life_days} days")
    print("\nTemporal weighting keeps the same co-purchase logic, but boosts newer product affinities.")


if __name__ == "__main__":
    main()
