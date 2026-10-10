"""Train the production temporal recommender and serialize it with joblib."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib

from phase3a_temporal_recommender import (
    build_co_purchase_model,
    load_cart_history_with_dates,
    parse_iso_date,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and serialize the serving model.")
    parser.add_argument("--train-data", type=str, default="data/cart_export_19_05.csv")
    parser.add_argument("--half-life-days", type=float, default=180)
    parser.add_argument("--artifact", type=str, default="artifacts/recommender.joblib")
    parser.add_argument("--cutoff-date", type=str, default=None)
    args = parser.parse_args()

    carts, dates = load_cart_history_with_dates(args.train_data)
    if args.cutoff_date:
        cutoff = parse_iso_date(args.cutoff_date, "--cutoff-date")
        keep = [sid for sid, dt in dates.items() if dt < cutoff]
        carts = {sid: carts[sid] for sid in keep}
        dates = {sid: dates[sid] for sid in keep}

    reference = max(dates.values()) if dates else datetime.utcnow()
    model = build_co_purchase_model(
        carts,
        cart_dates=dates,
        reference_date=reference,
        half_life_days=args.half_life_days,
    )
    popularity: Counter = Counter()
    for products in carts.values():
        popularity.update(products)

    serializable_model = {
        int(src): {int(dst): float(score) for dst, score in neighbors.items()}
        for src, neighbors in model.items()
    }
    artifact = {
        "version": f"temporal-{int(args.half_life_days)}d",
        "algorithm": "temporal_weighted_co_purchase",
        "half_life_days": args.half_life_days,
        "train_data": Path(args.train_data).name,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "reference_date": reference.date().isoformat(),
        "n_sessions": len(carts),
        "n_products": len(popularity),
        "model": serializable_model,
        "popularity": {int(k): int(v) for k, v in popularity.items()},
    }

    out = Path(args.artifact)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)
    print(f"Wrote {out} | sessions={len(carts)} products={len(popularity)} version={artifact['version']}")


if __name__ == "__main__":
    main()
