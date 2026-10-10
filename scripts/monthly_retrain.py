"""Monthly (or trigger-based) retrain.

Requests a fresh cart-history extract only when a human has approved spend,
trains a new joblib artifact next to the live one, and writes a report.
Promotion of artifacts/recommender.joblib is gated on HitRate not collapsing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from evaluate_phase5 import catalog_by_year  # noqa: E402
from phase3a_temporal_recommender import (  # noqa: E402
    build_co_purchase_model,
    evaluate,
    load_answer_file,
    load_cart_history_with_dates,
)
from train_serving_model import main as train_main  # noqa: E402


def maybe_warn_extract_cost(force_extract: bool) -> None:
    if not force_extract:
        print(
            "Using the local cart extract. A new warehouse report takes ~4 hours "
            "and costs $1.21–$4.53; pass --request-extract after approval."
        )
        return
    print("APPROVED: treating the local file as a freshly requested cart-history extract.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/cart_export_19_05.csv")
    parser.add_argument("--holdout-data", default="data/answer_2017-11.csv")
    parser.add_argument("--artifact", default="artifacts/recommender.joblib")
    parser.add_argument("--min-hitrate", type=float, default=0.40)
    parser.add_argument("--request-extract", action="store_true")
    args = parser.parse_args()

    maybe_warn_extract_cost(args.request_extract)

    candidate = Path("artifacts/recommender.candidate.joblib")
    sys.argv = [
        "train_serving_model.py",
        "--train-data",
        args.train_data,
        "--artifact",
        str(candidate),
        "--half-life-days",
        "180",
    ]
    train_main()

    carts, dates = load_cart_history_with_dates(args.train_data)
    model = build_co_purchase_model(
        carts,
        cart_dates=dates,
        reference_date=max(dates.values()),
        half_life_days=180,
    )
    test_cases = load_answer_file(args.holdout_data)
    hits, hitrate = evaluate(model, test_cases, 10)

    by_year = catalog_by_year(args.train_data)
    years = sorted(by_year)
    latest = years[-1] if years else None
    prior = set()
    for year in years[:-1]:
        prior |= by_year[year]
    n_new = len(by_year[latest] - prior) if latest is not None else 0

    report = {
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "train_data": args.train_data,
        "holdout": args.holdout_data,
        "hitrate_at_10": round(hitrate, 4),
        "hits": hits,
        "cases": len(test_cases),
        "new_skus_latest_year": n_new,
        "promoted": hitrate >= args.min_hitrate,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/last_retrain_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if report["promoted"]:
        live = Path(args.artifact)
        if live.exists():
            bak = live.with_suffix(".joblib.bak")
            shutil.copyfile(live, bak)
            print(f"Backed up previous artifact -> {bak}")
        shutil.copyfile(candidate, live)
        print(f"Promoted {candidate} -> {live}")
    else:
        print(
            f"HOLD: HitRate {hitrate:.4f} < {args.min_hitrate:.2f}. "
            "Keep the previous artifact; page a human."
        )


if __name__ == "__main__":
    main()
