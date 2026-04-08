"""
ALS Collaborative Filtering on Mix × Sound Co-occurrence Matrix

Steps:
1. Load listening data and filter to known sounds_ids
2. Build mix × sound sparse matrix weighted by play_count
3. Apply BM25 weighting and train ALS model
4. Save item_factors and user_factors as .npy for downstream use
5. Define ALSRecommender PyTorch module for inference

Usage:
    python models/als_co_occurence_matrix.py
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LISTENING_CSV = "/Users/emulie/Downloads/bq-results-20260325-195823-1774468728937.csv"
SOUNDS_IDS_JSON_PATH = "data/sounds_ids.json"

ITEM_FACTORS_PATH = "data/als_item_factors.npy"
USER_FACTORS_PATH = "data/als_user_factors.npy"
SOUND_LABELS_PATH = "data/als_sound_labels.json"
MIX_LABELS_PATH = "data/als_mix_labels.json"

# ---------------------------------------------------------------------------
# ALS hyperparameters
# ---------------------------------------------------------------------------
ALS_FACTORS = 64
ALS_REGULARIZATION = 0.1
ALS_ALPHA = 0.05
BM25_K1 = 100
BM25_B = 0.8
MIN_MIX_LENGTH = 2  # drop mixes with <= this many sounds after filtering


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_listening_df(csv_path: str, sounds_ids: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["sounds"] = df["sounds"].apply(json.loads)

    all_sounds = set(sounds_ids)
    df["sounds"] = df["sounds"].apply(lambda x: [s for s in x if s in all_sounds])
    df = df[df["sounds"].apply(len) > MIN_MIX_LENGTH].copy()
    return df


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def build_mix_sound_matrix(df: pd.DataFrame, sounds_ids: list):
    df = df.copy()
    df["mix_id"] = df["sounds"].apply(lambda s: "|".join(sorted(s)))

    mix_counts = df.groupby("mix_id").size().reset_index(name="play_count")

    df_mix = (
        df[["mix_id", "sounds"]].merge(mix_counts, on="mix_id").explode("sounds", ignore_index=True)
    )

    df_mix["mix_idx"] = df_mix["mix_id"].astype("category").cat.codes
    mix_labels = df_mix["mix_id"].astype("category").cat.categories.tolist()

    sound_cat = pd.CategoricalDtype(categories=sounds_ids, ordered=False)
    df_mix["sound_idx"] = df_mix["sounds"].astype(sound_cat).cat.codes

    n_mixes = len(mix_labels)
    n_sounds = len(sounds_ids)
    print(f"Matrix shape: {n_mixes:,} mixes × {n_sounds} sounds")

    mix_item = csr_matrix(
        (df_mix["play_count"].astype("float32"), (df_mix["mix_idx"], df_mix["sound_idx"])),
        shape=(n_mixes, n_sounds),
    )
    return mix_item, mix_labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_als(mix_item: csr_matrix):
    # implicit expects (item × user) → transpose to (sound × mix)
    mix_item_weighted = bm25_weight(mix_item.T, K1=BM25_K1, B=BM25_B).T.tocsr()

    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        alpha=ALS_ALPHA,
    )
    model.fit(mix_item_weighted)
    return model, mix_item_weighted


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------
class ALSRecommender(nn.Module):
    """
    PyTorch inference wrapper for a trained ALS model.

    Replicates implicit's fold-in exactly:
        mix_emb = (Y.T @ diag(c) @ Y + λI)^-1 @ Y.T @ c
    where c = interaction_vector (BM25-weighted confidence values).

    This is a per-query solve — unlike a global precomputed foldin_matrix —
    because the confidence matrix diag(c) differs for every mix.

    score = mix_emb @ sound_emb.T + mix_emb @ (YtY_reg)^-1 @ sound_emb.T
    Both terms share the same mix_emb, so the second term is computed via
    the same fold-in path; the full expression reduces to:
        scores = item_factors @ mix_emb
    """

    def __init__(self, item_factors: np.ndarray, regularization: float):
        super().__init__()
        self.regularization = regularization
        self.register_buffer("item_factors", torch.tensor(item_factors, dtype=torch.float32))

    def _fold_in(self, interaction_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute mix embedding via confidence-weighted ALS fold-in.
        Args:
            interaction_vector: [n_sounds] BM25-weighted confidence values
        Returns:
            mix_emb: [D]
        """
        Y = self.item_factors  # [n_sounds, D]
        c = interaction_vector  # [n_sounds]

        # Y.T @ diag(c) @ Y  =  (Y * c[:, None]).T @ Y
        YtCY = (Y * c.unsqueeze(1)).T @ Y  # [D, D]
        reg = self.regularization * torch.eye(Y.shape[1], dtype=Y.dtype, device=Y.device)
        rhs = Y.T @ c  # [D]

        mix_emb = torch.linalg.solve(YtCY + reg, rhs)  # [D]
        return mix_emb

    def forward(self, interaction_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            interaction_vector: [n_sounds] BM25-weighted float tensor
                (same values that were in mix_item_weighted at training time)
        Returns:
            scores: [n_sounds]
        """
        mix_emb = self._fold_in(interaction_vector)  # [D]
        scores = self.item_factors @ mix_emb  # [n_sounds]
        return scores

    def recommend(
        self,
        interaction_vector: torch.Tensor,
        N: int = 10,
        filter_already_liked: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns top-N (indices, scores), optionally masking sounds already in mix.
        interaction_vector: [n_sounds] — BM25-weighted row from mix_item_weighted.
        """
        with torch.no_grad():
            scores = self.forward(interaction_vector)  # [n_sounds]

        if filter_already_liked:
            scores = scores.clone()
            scores[interaction_vector > 0] = -torch.inf

        top_scores, top_ids = torch.topk(scores, k=N)
        return top_ids, top_scores


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_weights(model, sound_labels: list, mix_labels: list) -> None:
    np.save(ITEM_FACTORS_PATH, model.item_factors)
    np.save(USER_FACTORS_PATH, model.user_factors)
    with open(SOUND_LABELS_PATH, "w") as f:
        json.dump(sound_labels, f)
    with open(MIX_LABELS_PATH, "w") as f:
        json.dump(mix_labels, f)
    print(f"Saved item_factors  → {ITEM_FACTORS_PATH}  {model.item_factors.shape}")
    print(f"Saved user_factors  → {USER_FACTORS_PATH}  {model.user_factors.shape}")
    print(f"Saved sound_labels  → {SOUND_LABELS_PATH}")
    print(f"Saved mix_labels    → {MIX_LABELS_PATH}")


def build_als_recommender(model) -> ALSRecommender:
    return ALSRecommender(model.item_factors, model.regularization)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open(SOUNDS_IDS_JSON_PATH) as f:
        sounds_ids = json.load(f)

    print("Loading listening data...")
    df_listening = load_listening_df(LISTENING_CSV, sounds_ids)
    print(f"Rows after filtering: {len(df_listening):,}")

    print("\nBuilding mix × sound matrix...")
    mix_item, mix_labels = build_mix_sound_matrix(df_listening, sounds_ids)

    print("\nTraining ALS model...")
    model, mix_item_weighted = train_als(mix_item)
    print(f"item_factors: {model.item_factors.shape}")
    print(f"user_factors: {model.user_factors.shape}")

    print("\nSaving weights...")
    save_weights(model, sounds_ids, mix_labels)

    print("\nBuilding ALSRecommender...")
    als_recommender = build_als_recommender(model)
    als_recommender.eval()

    # Sanity check: compare PyTorch output to implicit's model.recommend
    mix_idx = 150
    N = 10
    imp_ids, imp_scores = model.recommend(
        mix_idx, mix_item_weighted[mix_idx], N=N, filter_already_liked_items=True
    )

    row = np.asarray(mix_item_weighted[mix_idx].todense(), dtype=np.float32).squeeze()
    pt_ids, pt_scores = als_recommender.recommend(torch.tensor(row), N=N, filter_already_liked=True)
    pt_ids = pt_ids.numpy()

    print(f"\nSanity check (mix_idx={mix_idx}):")
    print(f"  implicit top-{N}: {imp_ids}")
    print(f"  pytorch  top-{N}: {pt_ids}")
    match = set(imp_ids) == set(pt_ids)
    print(f"  Top-{N} sets match: {match}")
