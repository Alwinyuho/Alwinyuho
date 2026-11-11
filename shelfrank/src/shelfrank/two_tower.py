"""Two-tower retrieval model with a text path for cold-start items.

Item tower: projected text embedding + a learned ID embedding. The ID embedding
is only used for items seen in training. During training it is randomly
dropped, so the text path alone must also place items well. That is what lets
the model recommend a brand-new item it has never seen interactions for.

User tower: the user's recent items, combined as a recency-weighted mean plus
the last item, then projected.

Loss: in-batch sampled softmax with logQ correction, which stops popular items
from being over-penalised for appearing often as negatives.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import config


class TwoTower(nn.Module):
    def __init__(
        self,
        text_embeddings: np.ndarray,
        seen_in_train: np.ndarray,
        dim: int = 128,
        id_dropout: float = 0.3,
        temperature: float = 0.05,
    ):
        super().__init__()
        n_items = len(text_embeddings)
        self.pad = n_items
        self.id_dropout = id_dropout
        self.temperature = temperature
        text = np.vstack([text_embeddings, np.zeros((1, text_embeddings.shape[1]))])
        self.register_buffer("text", torch.tensor(text, dtype=torch.float32))
        self.register_buffer(
            "seen", torch.tensor(np.append(seen_in_train, False), dtype=torch.float32)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text.shape[1], dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.ids = nn.Embedding(n_items + 1, dim, padding_idx=self.pad)
        nn.init.normal_(self.ids.weight, std=0.02)
        self.user_proj = nn.Linear(dim * 2, dim)

    def item_vectors(self, items: torch.Tensor, drop_ids: bool = False) -> torch.Tensor:
        keep = self.seen[items]
        if drop_ids:
            keep = keep * (torch.rand_like(keep) > self.id_dropout).float()
        vectors = self.text_proj(self.text[items]) + self.ids(items) * keep.unsqueeze(-1)
        return F.normalize(vectors, dim=-1)

    def user_vectors(self, histories: torch.Tensor) -> torch.Tensor:
        """``histories`` is (batch, HISTORY_LEN), left-padded, most recent item last."""
        mask = (histories != self.pad).float()
        vectors = self.item_vectors(histories) * mask.unsqueeze(-1)
        positions = torch.arange(histories.shape[1], device=histories.device, dtype=torch.float32)
        recency = torch.exp((positions - histories.shape[1] + 1) / 5.0) * mask
        mean = (vectors * recency.unsqueeze(-1)).sum(1) / recency.sum(1, keepdim=True).clamp(
            min=1e-6
        )
        last = vectors[:, -1]
        return F.normalize(self.user_proj(torch.cat([mean, last], dim=-1)), dim=-1)

    def loss(self, histories, targets, log_q):
        users = self.user_vectors(histories)
        items = self.item_vectors(targets, drop_ids=True)
        logits = users @ items.T / self.temperature - log_q[targets].unsqueeze(0)
        same = targets.unsqueeze(0) == targets.unsqueeze(1)
        logits = logits.masked_fill(
            same & ~torch.eye(len(targets), dtype=torch.bool, device=targets.device), float("-inf")
        )
        return F.cross_entropy(logits, torch.arange(len(targets), device=targets.device))


def pad_histories(histories: list[list[int]], pad: int, length: int = config.HISTORY_LEN):
    out = np.full((len(histories), length), pad, dtype=np.int64)
    for row, history in enumerate(histories):
        recent = history[-length:]
        if recent:
            out[row, -len(recent) :] = recent
    return out


def train_two_tower(
    model: TwoTower,
    histories: list[list[int]],
    targets: np.ndarray,
    item_counts: np.ndarray,
    epochs: int = 6,
    batch_size: int = 1024,
    lr: float = 2e-3,
    device: str = "cpu",
    log=print,
) -> TwoTower:
    torch.manual_seed(config.SEED)
    model.to(device)
    padded = torch.tensor(pad_histories(histories, model.pad), device=device)
    target_tensor = torch.tensor(targets, device=device)
    probabilities = (item_counts + 1) / (item_counts + 1).sum()
    log_q = torch.tensor(np.log(np.append(probabilities, 1.0)), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(config.SEED)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(target_tensor), generator=generator).to(device)
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            loss = model.loss(padded[batch], target_tensor[batch], log_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        log(f"epoch {epoch + 1}: loss {total / batches:.4f}")
    model.eval()
    return model


@torch.no_grad()
def score_all(
    model: TwoTower, histories: list[list[int]], device: str = "cpu", batch_size: int = 2048
) -> np.ndarray:
    """Return (queries, items) similarity scores."""
    model.eval()
    items = model.item_vectors(torch.arange(model.pad, device=device))
    out = []
    for start in range(0, len(histories), batch_size):
        padded = torch.tensor(
            pad_histories(histories[start : start + batch_size], model.pad), device=device
        )
        out.append((model.user_vectors(padded) @ items.T).cpu().numpy())
    return np.vstack(out)
