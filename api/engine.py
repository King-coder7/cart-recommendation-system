"""Model scoring plus the business-rules serving layer.

The trained co-purchase model only ranks historically observed items.
Catalog files generated nightly decide what can actually be sold, what
is being promoted, and which cold-start SKUs should get exploration.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import joblib

from models.phase3a_temporal_recommender import recommend_items

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "recommender.joblib"
CATALOG_DIR = ROOT / "catalog"


def _as_int_set(values: Iterable[Any]) -> Set[int]:
    return {int(v) for v in values}


@dataclass
class CatalogRules:
    generated_at: str
    discontinued: Set[int]
    out_of_stock: Set[int]
    new_items: Set[int]
    specials: List[dict]
    source_files: Dict[str, str] = field(default_factory=dict)

    def unsellable(self) -> Set[int]:
        return self.discontinued | self.out_of_stock

    def sellable(self, product_id: int) -> bool:
        return product_id not in self.unsellable()


def load_catalog_rules(catalog_dir: Path = CATALOG_DIR) -> CatalogRules:
    def read(name: str) -> dict:
        path = catalog_dir / name
        if not path.exists():
            return {}
        with path.open() as f:
            return json.load(f)

    discontinued = read("discontinued.json")
    out_of_stock = read("out_of_stock.json")
    new_items = read("new_items.json")
    special_items = read("special_items.json")

    generated = (
        special_items.get("generated_at")
        or new_items.get("generated_at")
        or discontinued.get("generated_at")
        or datetime.utcnow().isoformat()
    )
    return CatalogRules(
        generated_at=str(generated),
        discontinued=_as_int_set(discontinued.get("product_ids", [])),
        out_of_stock=_as_int_set(out_of_stock.get("product_ids", [])),
        new_items=_as_int_set(new_items.get("product_ids", [])),
        specials=list(special_items.get("items", [])),
        source_files={
            "discontinued": str(catalog_dir / "discontinued.json"),
            "out_of_stock": str(catalog_dir / "out_of_stock.json"),
            "new_items": str(catalog_dir / "new_items.json"),
            "special_items": str(catalog_dir / "special_items.json"),
        },
    )


class RecommenderEngine:
    def __init__(
        self,
        artifact_path: Path = ARTIFACT_PATH,
        catalog_dir: Path = CATALOG_DIR,
        candidate_pool: int = 50,
        explore_slots: int = 1,
    ) -> None:
        self.artifact_path = artifact_path
        self.catalog_dir = catalog_dir
        self.candidate_pool = candidate_pool
        self.explore_slots = explore_slots
        self.artifact: dict = {}
        self.model: dict = {}
        self.popularity: Dict[int, int] = {}
        self.rules = CatalogRules("", set(), set(), set(), [])
        self.load_model()
        self.reload_rules()

    def load_model(self) -> None:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Missing model artifact at {self.artifact_path}. "
                "Run: python3 models/train_serving_model.py"
            )
        payload = joblib.load(self.artifact_path)
        self.artifact = payload
        raw_model = payload["model"]
        self.model = {
            int(src): {int(dst): float(score) for dst, score in neighbors.items()}
            for src, neighbors in raw_model.items()
        }
        self.popularity = {int(k): int(v) for k, v in payload.get("popularity", {}).items()}

    def reload_rules(self) -> CatalogRules:
        self.rules = load_catalog_rules(self.catalog_dir)
        return self.rules

    def recommend(
        self,
        cart_items: List[int],
        top_k: int = 10,
        explore: bool = True,
        seed: Optional[int] = None,
    ) -> dict:
        started = time.perf_counter()
        parsed_cart = [int(x) for x in cart_items]
        known = set(self.model) | set(self.popularity)
        seen = set(parsed_cart)
        known_cart = [pid for pid in parsed_cart if pid in known]
        rules = self.rules
        blocked = rules.unsellable()
        rng = random.Random(seed)

        raw = recommend_items(self.model, known_cart, top_k=self.candidate_pool)
        model_ranked = [(pid, score) for pid, score in raw if pid not in seen and pid not in blocked]

        if len(model_ranked) < self.candidate_pool:
            extras = sorted(
                self.popularity.items(),
                key=lambda item: (-item[1], item[0]),
            )
            already = {pid for pid, _ in model_ranked} | seen | blocked
            for pid, pop in extras:
                if pid in already:
                    continue
                model_ranked.append((pid, pop / (1.0 + extras[0][1] if extras else 1.0) * 1e-6))
                already.add(pid)
                if len(model_ranked) >= self.candidate_pool:
                    break

        applied: List[str] = ["drop_discontinued", "drop_out_of_stock"]
        selected: List[dict] = []
        selected_ids: Set[int] = set()

        for item in sorted(rules.specials, key=lambda row: row.get("pin_slot", 10_000)):
            pid = int(item["product_id"])
            if pid in seen or pid in blocked or pid in selected_ids:
                continue
            selected.append(
                {
                    "product_id": pid,
                    "score": 1.0 + float(item.get("boost", 0)),
                    "source": "promotion",
                    "campaign": item.get("campaign", "special"),
                }
            )
            selected_ids.add(pid)
            applied.append("promotion_override")
            if len(selected) >= top_k:
                break

        if explore and self.explore_slots > 0 and len(selected) < top_k:
            eligible_new = [
                pid
                for pid in rules.new_items
                if pid not in seen and pid not in blocked and pid not in selected_ids
            ]
            if eligible_new:
                injected = rng.sample(eligible_new, k=min(self.explore_slots, len(eligible_new)))
                for pid in injected:
                    selected.append(
                        {
                            "product_id": pid,
                            "score": 0.0,
                            "source": "cold_start_explore",
                            "campaign": "new_item_exploration",
                        }
                    )
                    selected_ids.add(pid)
                applied.append("cold_start_explore")

        for pid, score in model_ranked:
            if len(selected) >= top_k:
                break
            if pid in selected_ids or pid in seen or pid in blocked:
                continue
            source = "cold_start_model" if pid in rules.new_items else "model"
            selected.append(
                {
                    "product_id": pid,
                    "score": round(float(score), 6),
                    "source": source,
                    "campaign": None,
                }
            )
            selected_ids.add(pid)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "recommendations": selected[:top_k],
            "latency_ms": round(latency_ms, 3),
            "model_version": self.artifact.get("version"),
            "train_data": self.artifact.get("train_data"),
            "rules_generated_at": rules.generated_at,
            "filters_applied": sorted(set(applied)),
            "cart_items": parsed_cart,
        }


def example_payload() -> dict:
    return {
        "cart_items": [227, 228],
        "top_k": 10,
    }
