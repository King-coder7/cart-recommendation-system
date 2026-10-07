"""Phase 2: Drift Analysis - Evaluate temporal model degradation and product catalog changes.

Measures:
- Performance on January 2016 vs December 2016 test sets
- Impact of new products (not in training data)
- Impact of discontinued products
- Concept drift (changing co-purchase relationships over time)
- Input drift (new product introductions)
"""

from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

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


def load_cart_history(path: str | Path) -> Dict[str, List[int]]:
    """Load training data: group by cart_session -> list of product_ids."""
    df = pd.read_csv(path)
    
    cart_items: Dict[str, List[int]] = defaultdict(list)
    for _, row in df.iterrows():
        session = str(row['cart_session']).strip()
        product_id = int(row['cart_product_id'])
        cart_items[session].append(product_id)
    
    return {sid: list(dict.fromkeys(products)) for sid, products in cart_items.items()}


def build_co_purchase_model(train_carts: Dict[str, List[int]]) -> Dict[int, Counter]:
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
    """Given cart items, recommend top-k related products."""
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


def get_product_ids(test_cases: List[dict]) -> Set[int]:
    """Extract all product IDs from test cases."""
    all_products = set()
    for test_case in test_cases:
        all_products.update(test_case['previous_products'])
        all_products.update(test_case['answer_product_ids'])
    return all_products


def evaluate_and_categorize(
    model: Dict[int, Counter],
    test_cases: List[dict],
    train_products: Set[int],
    test_products: Set[int],
    top_k: int = 10,
) -> dict:
    """Evaluate model and categorize hits by product novelty."""
    
    new_products = test_products - train_products
    discontinued_products = train_products - test_products
    known_products = test_products & train_products
    
    hits_total = 0
    hits_with_known = 0
    hits_with_new = 0
    misses_due_to_unknown = 0
    
    cases_using_new = 0
    cases_using_discontinued = 0
    cases_all_known = 0
    
    for test_case in test_cases:
        previous = test_case['previous_products']
        answer = test_case['answer_product_ids']
        
        # Categorize the test case
        prev_has_new = any(p in new_products for p in previous)
        prev_has_discontinued = any(p in discontinued_products for p in previous)
        ans_has_new = any(p in new_products for p in answer)
        
        if prev_has_new:
            cases_using_new += 1
        if prev_has_discontinued:
            cases_using_discontinued += 1
        if not prev_has_new and not prev_has_discontinued:
            cases_all_known += 1
        
        recs = recommend_items(model, previous, top_k=top_k)
        rec_items = {product for product, _ in recs}
        
        # Check hit
        hit = bool(rec_items & answer)
        
        if hit:
            hits_total += 1
            # Categorize the hit
            if ans_has_new:
                hits_with_new += 1
            else:
                hits_with_known += 1
        else:
            # If we missed, did the answer contain new products?
            if ans_has_new:
                misses_due_to_unknown += 1
    
    hitrate_total = hits_total / len(test_cases) if test_cases else 0.0
    hitrate_known = hits_with_known / (len(test_cases) - (hits_with_new + misses_due_to_unknown)) if (len(test_cases) - (hits_with_new + misses_due_to_unknown)) > 0 else 0.0
    
    return {
        'hitrate_total': hitrate_total,
        'hits_total': hits_total,
        'total_cases': len(test_cases),
        'new_products_count': len(new_products),
        'discontinued_products_count': len(discontinued_products),
        'known_products_count': len(known_products),
        'hits_with_known': hits_with_known,
        'hits_with_new': hits_with_new,
        'misses_due_to_unknown': misses_due_to_unknown,
        'cases_using_new': cases_using_new,
        'cases_using_discontinued': cases_using_discontinued,
        'cases_all_known': cases_all_known,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Drift Analysis - evaluate model on multiple time periods.")
    parser.add_argument("--train-data", type=str, required=True, help="CSV containing historical cart purchases")
    parser.add_argument("--test-jan", type=str, required=True, help="CSV containing January 2016 test cases")
    parser.add_argument("--test-dec", type=str, required=True, help="CSV containing December 2016 test cases")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to rank")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("PHASE 2: MODEL DRIFT ANALYSIS")
    print("="*70)
    
    # Load training data
    print(f"\n[1/5] Loading training data from {args.train_data}...")
    train_carts = load_cart_history(args.train_data)
    train_products = set()
    for products in train_carts.values():
        train_products.update(products)
    print(f"      {len(train_carts)} sessions | {len(train_products)} unique products")
    
    # Build model
    print(f"\n[2/5] Building co-purchase model...")
    model = build_co_purchase_model(train_carts)
    print(f"      Model contains {len(model)} products with co-purchase relationships")
    
    # Load and evaluate January 2016
    print(f"\n[3/5] Evaluating on January 2016 (answer_2016-01.csv)...")
    test_jan = load_answer_file(args.test_jan)
    test_jan_products = get_product_ids(test_jan)
    results_jan = evaluate_and_categorize(model, test_jan, train_products, test_jan_products, top_k=args.top_k)
    
    print(f"      Test Set: {results_jan['total_cases']} cases")
    print(f"      Products: {results_jan['known_products_count']} known | {results_jan['new_products_count']} new | {results_jan['discontinued_products_count']} discontinued")
    print(f"      HitRate@{args.top_k}: {results_jan['hitrate_total']:.4f} ({results_jan['hits_total']}/{results_jan['total_cases']})")
    
    # Load and evaluate December 2016
    print(f"\n[4/5] Evaluating on December 2016 (answer_2016-12.csv)...")
    test_dec = load_answer_file(args.test_dec)
    test_dec_products = get_product_ids(test_dec)
    results_dec = evaluate_and_categorize(model, test_dec, train_products, test_dec_products, top_k=args.top_k)
    
    print(f"      Test Set: {results_dec['total_cases']} cases")
    print(f"      Products: {results_dec['known_products_count']} known | {results_dec['new_products_count']} new | {results_dec['discontinued_products_count']} discontinued")
    print(f"      HitRate@{args.top_k}: {results_dec['hitrate_total']:.4f} ({results_dec['hits_total']}/{results_dec['total_cases']})")
    
    # Analysis and comparison
    print("\n" + "="*70)
    print("DRIFT ANALYSIS RESULTS")
    print("="*70)
    
    hitrate_change = results_dec['hitrate_total'] - results_jan['hitrate_total']
    hitrate_pct_change = (hitrate_change / results_jan['hitrate_total']) * 100 if results_jan['hitrate_total'] > 0 else 0
    
    print(f"\n📊 PERFORMANCE CHANGE (Jan → Dec 2016):")
    print(f"   January  HitRate@{args.top_k}: {results_jan['hitrate_total']:.4f}")
    print(f"   December HitRate@{args.top_k}: {results_dec['hitrate_total']:.4f}")
    print(f"   Change: {hitrate_change:+.4f} ({hitrate_pct_change:+.1f}%)")
    
    print(f"\n🆕 PRODUCT CATALOG CHANGES:")
    print(f"   January:   {results_jan['new_products_count']} new products | {results_jan['discontinued_products_count']} discontinued")
    print(f"   December:  {results_dec['new_products_count']} new products | {results_dec['discontinued_products_count']} discontinued")
    
    print(f"\n📈 NEW PRODUCT IMPACT:")
    print(f"   January cases with new products:   {results_jan['cases_using_new']}/{results_jan['total_cases']} ({100*results_jan['cases_using_new']/results_jan['total_cases']:.1f}%)")
    print(f"   December cases with new products:  {results_dec['cases_using_new']}/{results_dec['total_cases']} ({100*results_dec['cases_using_new']/results_dec['total_cases']:.1f}%)")
    print(f"   Hits on new products (Jan):        {results_jan['hits_with_new']}")
    print(f"   Hits on new products (Dec):        {results_dec['hits_with_new']}")
    
    print(f"\n❌ KNOWN PRODUCTS (No Input Drift):")
    print(f"   January cases with known-only:     {results_jan['cases_all_known']}/{results_jan['total_cases']} ({100*results_jan['cases_all_known']/results_jan['total_cases']:.1f}%)")
    print(f"   December cases with known-only:    {results_dec['cases_all_known']}/{results_dec['total_cases']} ({100*results_dec['cases_all_known']/results_dec['total_cases']:.1f}%)")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if hitrate_change < 0:
        print(f"\n⚠️  MODEL DEGRADATION DETECTED: Performance declined by {abs(hitrate_pct_change):.1f}%")
    elif hitrate_change > 0:
        print(f"\n✅ MODEL STABILITY: Performance improved by {hitrate_pct_change:.1f}% (seasonal boost or better distribution match)")
    else:
        print(f"\n➡️  MODEL STABLE: No significant performance change")
    
    new_product_impact = ((results_dec['cases_using_new'] / results_dec['total_cases']) - 
                          (results_jan['cases_using_new'] / results_jan['total_cases'])) * 100
    
    print(f"\n📦 CATALOG EXPANSION: New products appearing in {new_product_impact:+.1f}% more test cases (Dec vs Jan)")
    print(f"   → Model has no learned co-purchase relationships for these {results_dec['new_products_count']} new items")
    print(f"   → Recommending new products requires explicit retraining")
    
    print(f"\n💡 RECOMMENDATION:")
    if results_dec['cases_using_new'] > results_jan['cases_using_new']:
        print(f"   • Retrain model quarterly to capture new products and changing relationships")
        print(f"   • Current ~11-month-old model shows {'degradation' if hitrate_change < 0 else 'stability'} on fresh data")
    else:
        print(f"   • Model remains effective on known products")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
