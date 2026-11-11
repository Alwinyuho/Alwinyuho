import numpy as np
import pandas as pd
import torch

from shelfrank.baselines import ItemKNN, popularity_scores
from shelfrank.ranker import rank_positions, top_k
from shelfrank.two_tower import TwoTower, pad_histories, score_all

DAY = 86_400_000


def interactions():
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "b", "c", "c"],
            "item": [0, 1, 0, 1, 2, 3],
            "timestamp": [0, DAY, 0, DAY, 100 * DAY, 100 * DAY],
        }
    )


def test_recent_interactions_count_more():
    scores = popularity_scores(interactions(), 4, half_life_days=10)
    assert scores[2] > scores[0]  # one recent event outweighs two events from 100 days ago
    assert scores[3] == scores[2]


def test_item_knn_links_co_reviewed_items():
    knn = ItemKNN(interactions(), 4)
    scores = knn.score([[0]])[0]
    assert scores[1] > 0
    assert scores[2] == 0 and scores[3] == 0


def test_pad_histories_keeps_most_recent_items_at_the_end():
    padded = pad_histories([[1, 2, 3], []], pad=9, length=2)
    assert padded.tolist() == [[2, 3], [9, 9]]


def test_unseen_item_has_no_id_contribution():
    text = np.eye(4, dtype=np.float32)
    seen = np.array([True, True, True, False])
    model = TwoTower(text, seen, dim=8)
    with torch.no_grad():
        model.ids.weight[3] = 100.0  # would dominate if it were used
        vector = model.item_vectors(torch.tensor([3]))
        text_only = torch.nn.functional.normalize(model.text_proj(model.text[[3]]), dim=-1)
    assert torch.allclose(vector, text_only, atol=1e-6)


def test_two_tower_loss_trains_and_scores():
    text = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    model = TwoTower(text, np.ones(6, dtype=bool), dim=8)
    histories = torch.tensor(pad_histories([[0], [1], [2]], model.pad, length=3))
    loss = model.loss(histories, torch.tensor([3, 4, 5]), torch.zeros(7))
    loss.backward()
    assert torch.isfinite(loss)
    assert score_all(model, [[0], [1, 2]]).shape == (2, 6)


def test_top_k_and_rank_positions():
    scores = np.array([[0.2, 0.9, 0.5]])
    assert top_k(scores, 2).tolist() == [[1, 2]]
    assert rank_positions(scores).tolist() == [[3, 1, 2]]
