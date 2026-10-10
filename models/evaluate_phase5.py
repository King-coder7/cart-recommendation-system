"""Phase 5: retrain on cart_export_17_10 and evaluate November 2017 cold-start impact."""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from baseline_recommender import load_answer_file
from phase3a_temporal_recommender import (
    build_co_purchase_model,
    evaluate,
    load_cart_history_with_dates,
    recommend_items,
)


def catalog_by_year(path: str) -> dict:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["cart_product_id"] = df["cart_product_id"].astype(int)
    by_year = defaultdict(set)
    for year, group in df.groupby("year"):
        by_year[int(year)].update(int(p) for p in group["cart_product_id"])
    return by_year


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/cart_export_17_10.csv")
    parser.add_argument("--test-data", default="data/answer_2017-11.csv")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--half-life-days", type=float, default=180)
    args = parser.parse_args()

    carts, dates = load_cart_history_with_dates(args.train_data)
    reference = max(dates.values())
    model = build_co_purchase_model(
        carts, cart_dates=dates, reference_date=reference, half_life_days=args.half_life_days
    )
    test_cases = load_answer_file(args.test_data)
    hits, hitrate = evaluate(model, test_cases, args.top_k)

    by_year = catalog_by_year(args.train_data)
    before_2017 = set()
    for year, products in by_year.items():
        if year < 2017:
            before_2017 |= products
    new_2017 = by_year.get(2017, set()) - before_2017

    cold_answer_cases = 0
    cold_hits = 0
    new_in_recs = 0
    for case in test_cases:
        recs = {pid for pid, _ in recommend_items(model, case["previous_products"], args.top_k)}
        answer_has_new = bool(case["answer_product_ids"] & new_2017)
        if answer_has_new:
            cold_answer_cases += 1
            if recs & case["answer_product_ids"]:
                cold_hits += 1
        if recs & new_2017:
            new_in_recs += 1

    print("\n=== Phase 5: 2017 catalog churn ===")
    print(f"Train file: {args.train_data}")
    print(f"Test file:  {args.test_data} ({len(test_cases)} cases)")
    print(f"Temporal HitRate@{args.top_k}: {hitrate:.4f} ({hits}/{len(test_cases)})")
    print(f"New 2017 SKUs vs pre-2017 catalog: {len(new_2017)}")
    print(f"Test cases whose answer includes a 2017-new SKU: {cold_answer_cases}/{len(test_cases)}")
    print(
        "HitRate on those cold-answer cases: "
        f"{(cold_hits / cold_answer_cases) if cold_answer_cases else 0:.4f} ({cold_hits}/{cold_answer_cases})"
    )
    print(f"Test cases where top-{args.top_k} includes a 2017-new SKU: {new_in_recs}/{len(test_cases)}")
    print("New 2017 product ids:", sorted(new_2017))


if __name__ == "__main__":
    main()
