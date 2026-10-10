"""Build nightly catalog JSON files from the latest cart export.

These files stand in for inventory and merchandising feeds:
- discontinued.json: SKUs with no sales in the last 12 months
- out_of_stock.json: SKUs sold earlier in the current year but not in the latest month
- new_items.json: SKUs whose first sale is in the last 12 months
- special_items.json: merchandising pins (new arrivals + seasonal bestsellers)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path} ({len(payload.get('product_ids', payload.get('items', [])))} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nightly catalog constraint files.")
    parser.add_argument("--cart-data", type=str, default="data/cart_export_19_05.csv")
    parser.add_argument("--out-dir", type=str, default="catalog")
    args = parser.parse_args()

    df = pd.read_csv(args.cart_data)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["cart_product_id"] = df["cart_product_id"].astype(int)

    max_year = int(df["year"].max())
    max_month = int(df.loc[df["year"] == max_year, "month"].max())
    generated_at = datetime(max_year, max_month, 1).isoformat()

    recent_mask = (df["year"] > max_year - 1) | (
        (df["year"] == max_year - 1) & (df["month"] >= max_month)
    )
    recent = set(df.loc[recent_mask, "cart_product_id"])
    all_products = set(df["cart_product_id"])
    discontinued = sorted(all_products - recent)

    latest_month = set(
        df.loc[(df["year"] == max_year) & (df["month"] == max_month), "cart_product_id"]
    )
    prev_year, prev_month = (max_year, max_month - 1) if max_month > 1 else (max_year - 1, 12)
    previous_month = set(
        df.loc[(df["year"] == prev_year) & (df["month"] == prev_month), "cart_product_id"]
    )
    # Proxy for nightly inventory: sold last month, missing this month, still in catalog.
    out_of_stock = sorted(p for p in (previous_month - latest_month) if p not in discontinued)

    first_seen = df.groupby("cart_product_id")[["year", "month"]].min()
    new_items = sorted(
        int(pid)
        for pid, row in first_seen.iterrows()
        if (int(row["year"]) > max_year - 1)
        or (int(row["year"]) == max_year - 1 and int(row["month"]) >= max_month)
    )

    december = df[(df["month"] == 12)]
    seasonal = (
        december.groupby("cart_product_id").size().sort_values(ascending=False).head(3).index.tolist()
        if not december.empty
        else []
    )
    newest = list(reversed(new_items[-3:]))
    specials = []
    pin = 0
    used = set()
    for pid, campaign in [(p, "new_arrival") for p in newest] + [(p, "holiday_feature") for p in seasonal]:
        pid = int(pid)
        if pid in used or pid in discontinued or pid in out_of_stock:
            continue
        specials.append(
            {
                "product_id": pid,
                "campaign": campaign,
                "boost": 1000,
                "pin_slot": pin,
            }
        )
        used.add(pid)
        pin += 1
        if pin >= 3:
            break

    out_dir = Path(args.out_dir)
    meta = {
        "generated_at": generated_at,
        "source": Path(args.cart_data).name,
        "as_of": f"{max_year}-{max_month:02d}",
    }
    write_json(out_dir / "discontinued.json", {**meta, "product_ids": discontinued})
    write_json(out_dir / "out_of_stock.json", {**meta, "product_ids": out_of_stock})
    write_json(out_dir / "new_items.json", {**meta, "product_ids": new_items})
    write_json(out_dir / "special_items.json", {**meta, "items": specials})


if __name__ == "__main__":
    main()
