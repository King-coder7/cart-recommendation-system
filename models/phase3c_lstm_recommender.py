"""Phase 3c: LSTM next-item cart recommender.

Trains a session LSTM on ordered cart prefixes: given items [a, b, c],
predict the next product. At evaluation time the model ranks the full
catalog (excluding items already in the cart) and HitRate@K is scored
the same way as Phase 1/3a.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for Phase 3c. Install with:\n"
        "  pip install torch\n"
        "or: pip install -r requirements.txt"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_recommender import (  # noqa: E402
    build_co_purchase_model,
    load_answer_file,
    recommend_items as baseline_recommend,
)


PAD_IDX = 0


def parse_iso_date(value: str, flag_name: str) -> datetime:
    cleaned = value.strip().rstrip(".,;")
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(
            f"Invalid {flag_name} '{value}'. Use YYYY-MM-DD, e.g. 2016-01-01"
        ) from exc


def load_cart_sequences(path: str | Path) -> Tuple[Dict[str, List[int]], Dict[str, datetime]]:
    """Load ordered product sequences per cart_session.

    Rows are sorted by cart_id so the sequence matches add-to-cart order.
    """
    df = pd.read_csv(path)
    df["cart_session"] = df["cart_session"].astype(str).str.strip()
    df["cart_product_id"] = df["cart_product_id"].astype(int)
    df["cart_id"] = df["cart_id"].astype(int)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df = df.sort_values(["cart_session", "cart_id"])

    cart_items: Dict[str, List[int]] = defaultdict(list)
    cart_dates: Dict[str, datetime] = {}

    for row in df.itertuples(index=False):
        session = str(row.cart_session)
        cart_items[session].append(int(row.cart_product_id))
        dt = datetime(int(row.year), int(row.month), 1)
        if session not in cart_dates or dt > cart_dates[session]:
            cart_dates[session] = dt

    carts = {sid: list(dict.fromkeys(products)) for sid, products in cart_items.items()}
    return carts, cart_dates


def apply_cutoff(
    carts: Dict[str, List[int]],
    cart_dates: Dict[str, datetime],
    cutoff: datetime,
) -> Tuple[Dict[str, List[int]], Dict[str, datetime]]:
    keep = [sid for sid, dt in cart_dates.items() if dt < cutoff]
    return {sid: carts[sid] for sid in keep}, {sid: cart_dates[sid] for sid in keep}


def evaluate_hitrate(
    recommend_fn,
    test_cases: List[dict],
    top_k: int,
) -> Tuple[int, float]:
    hits = 0
    for test_case in test_cases:
        recs = recommend_fn(test_case["previous_products"], top_k)
        rec_items = {product for product, _ in recs}
        if rec_items & test_case["answer_product_ids"]:
            hits += 1
    hitrate = hits / len(test_cases) if test_cases else 0.0
    return hits, hitrate


class PrefixDataset(Dataset):
    def __init__(self, prefixes: List[List[int]], targets: List[int]) -> None:
        self.prefixes = prefixes
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.prefixes[idx], self.targets[idx]


def collate_prefixes(batch: List[Tuple[List[int], int]]):
    prefixes, targets = zip(*batch)
    lengths = torch.tensor([len(p) for p in prefixes], dtype=torch.long)
    seqs = [torch.tensor(p, dtype=torch.long) for p in prefixes]
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    return padded, lengths, torch.tensor(targets, dtype=torch.long)


class SessionLSTM(nn.Module):
    """Embed products, run an LSTM over the cart prefix, score every SKU."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(sequences))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        padded_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(sequences.size(0), device=sequences.device)
        last_hidden = padded_out[batch_idx, last_idx]
        return self.output(self.dropout(last_hidden))


def build_examples(
    carts: Sequence[List[int]],
    item_to_idx: Dict[int, int],
) -> Tuple[List[List[int]], List[int]]:
    prefixes: List[List[int]] = []
    targets: List[int] = []
    for products in carts:
        mapped = [item_to_idx[p] for p in products if p in item_to_idx]
        if len(mapped) < 2:
            continue
        for i in range(1, len(mapped)):
            prefixes.append(mapped[:i])
            targets.append(mapped[i])
    return prefixes, targets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_lstm(
    model: SessionLSTM,
    train_loader: DataLoader,
    val_prefixes: List[List[int]],
    val_targets: List[int],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> SessionLSTM:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for sequences, lengths, targets in train_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(sequences, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_hr = next_item_hitrate(model, val_prefixes, val_targets, device, top_k=10)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"      epoch {epoch:02d}/{epochs}  loss={avg_loss:.4f}  val HitRate@10={val_hr:.4f}")

        if val_hr > best_val + 1e-4:
            best_val = val_hr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"      early stop at epoch {epoch} (best val HitRate@10={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def next_item_hitrate(
    model: SessionLSTM,
    prefixes: List[List[int]],
    targets: List[int],
    device: torch.device,
    top_k: int,
) -> float:
    if not prefixes:
        return 0.0
    model.eval()
    hits = 0
    batch_size = 256
    for start in range(0, len(prefixes), batch_size):
        batch_p = prefixes[start : start + batch_size]
        batch_t = targets[start : start + batch_size]
        lengths = torch.tensor([len(p) for p in batch_p], dtype=torch.long, device=device)
        seqs = [torch.tensor(p, dtype=torch.long) for p in batch_p]
        padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX).to(device)
        logits = model(padded, lengths)
        topk = torch.topk(logits, k=min(top_k, logits.size(1)), dim=1).indices.cpu().tolist()
        for recs, target in zip(topk, batch_t):
            if target in recs:
                hits += 1
    return hits / len(prefixes)


class LSTMRecommender:
    def __init__(
        self,
        model: SessionLSTM,
        item_to_idx: Dict[int, int],
        idx_to_item: Dict[int, int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def recommend(self, cart_items: List[int], top_k: int) -> List[Tuple[int, float]]:
        mapped = [self.item_to_idx[p] for p in cart_items if p in self.item_to_idx]
        if not mapped:
            return []

        sequences = torch.tensor([mapped], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(mapped)], dtype=torch.long, device=self.device)
        logits = self.model(sequences, lengths)[0]

        seen = set(mapped)
        logits[PAD_IDX] = -1e9
        for idx in seen:
            logits[idx] = -1e9

        k = min(top_k, logits.numel() - 1)
        values, indices = torch.topk(logits, k=k)
        recs: List[Tuple[int, float]] = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            product = self.idx_to_item.get(idx)
            if product is not None:
                recs.append((product, float(score)))
        return recs


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3c: LSTM next-item cart recommender.")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cutoff-date", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()

    print("\n=== Phase 3c: LSTM Next-Item Cart Recommender ===\n")
    print(f"Device: {device}")

    print(f"Loading training data from {args.train_data}...")
    train_carts, cart_dates = load_cart_sequences(args.train_data)
    print(f"Loaded {len(train_carts)} unique sessions")

    if args.cutoff_date:
        cutoff = parse_iso_date(args.cutoff_date, "--cutoff-date")
        before = len(train_carts)
        train_carts, cart_dates = apply_cutoff(train_carts, cart_dates, cutoff)
        print(
            f"Applied cutoff {cutoff.date()}: dropped {before - len(train_carts)} future sessions, "
            f"{len(train_carts)} remain"
        )

    session_ids = list(train_carts.keys())
    random.shuffle(session_ids)
    n_val = max(1, int(len(session_ids) * args.val_frac))
    val_ids = set(session_ids[:n_val])
    train_ids = [sid for sid in session_ids if sid not in val_ids]

    train_seqs = [train_carts[sid] for sid in train_ids]
    val_seqs = [train_carts[sid] for sid in val_ids]

    catalog = sorted({p for products in train_carts.values() for p in products})
    item_to_idx = {product: i + 1 for i, product in enumerate(catalog)}
    idx_to_item = {i: product for product, i in item_to_idx.items()}
    vocab_size = len(item_to_idx) + 1
    print(f"Catalog: {len(catalog)} products | vocab_size={vocab_size} (incl. PAD)")

    train_prefixes, train_targets = build_examples(train_seqs, item_to_idx)
    val_prefixes, val_targets = build_examples(val_seqs, item_to_idx)
    print(f"Next-item examples: {len(train_prefixes)} train | {len(val_prefixes)} val")

    if not train_prefixes:
        raise SystemExit("No training prefixes found (need carts with at least 2 products).")

    train_loader = DataLoader(
        PrefixDataset(train_prefixes, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_prefixes,
    )

    print("Training LSTM...")
    model = SessionLSTM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    model = train_lstm(
        model,
        train_loader,
        val_prefixes,
        val_targets,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    recommender = LSTMRecommender(model, item_to_idx, idx_to_item, device)

    print("Building Phase 1 co-purchase baseline on the same training window...")
    baseline_model = build_co_purchase_model(train_carts)

    print(f"Loading answer file from {args.test_data}...")
    test_cases = load_answer_file(args.test_data)
    print(f"Loaded {len(test_cases)} test cases")

    baseline_hits, baseline_hr = evaluate_hitrate(
        lambda cart, k: baseline_recommend(baseline_model, cart, k),
        test_cases,
        args.top_k,
    )
    lstm_hits, lstm_hr = evaluate_hitrate(recommender.recommend, test_cases, args.top_k)
    delta = lstm_hr - baseline_hr
    relative = (delta / baseline_hr * 100.0) if baseline_hr else 0.0

    print("\n=== Results ===")
    print(f"Baseline HitRate@{args.top_k}: {baseline_hr:.4f} ({baseline_hits}/{len(test_cases)})")
    print(f"LSTM     HitRate@{args.top_k}: {lstm_hr:.4f} ({lstm_hits}/{len(test_cases)})")
    print(f"Delta vs baseline: {delta:+.4f} ({relative:+.1f}% relative)\n")


if __name__ == "__main__":
    main()
