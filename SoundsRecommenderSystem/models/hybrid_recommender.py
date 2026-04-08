"""
Hybrid Sound Recommender: ALS (collaborative) + Tags (content-based)

Both models are aligned to the canonical sounds_ids order so the input vector
index i always refers to sounds_ids[i].

Input:  [n_sounds] float32 volume vector (0–1)
Output: top-N (indices, scores)

Two implementations:
  - HybridRecommenderPyTorch  — exact confidence-weighted ALS fold-in
  - build_hybrid_tflite       — TFLite-compatible (Dense-only, precomputed fold-in)

Build:
    hybrid = build_hybrid_recommender(als_model, tags_model, sounds_ids)
    hybrid.eval()
    ids, scores = hybrid.recommend(volume_tensor, N=10)

    tflite_bytes = build_hybrid_tflite(als_model, tags_model, sounds_ids)
    with open("hybrid_recommender.tflite", "wb") as f:
        f.write(tflite_bytes)
"""

import json
import numpy as np
import torch
import torch.nn as nn

from models.als_co_occurence_matrix import ALSRecommender, ALS_FACTORS, ALS_REGULARIZATION
from models.sounds_similarity_tags_model import SoundsSimilarityTagsModel


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------


def _align_tag_factors(tags_model: SoundsSimilarityTagsModel, sounds_ids: list) -> np.ndarray:
    """
    Build a [n_sounds, D_tags] matrix in sounds_ids order.
    Sounds missing from the tags model get a zero vector.
    """
    D_tags = tags_model.X.shape[1]
    tag_factors = np.zeros((len(sounds_ids), D_tags), dtype=np.float32)
    for i, sid in enumerate(sounds_ids):
        if sid in tags_model.sound_id_to_idx:
            j = tags_model.sound_id_to_idx[sid]
            tag_factors[i] = tags_model.X[j]
    return tag_factors


# ---------------------------------------------------------------------------
# PyTorch version
# ---------------------------------------------------------------------------


class HybridRecommenderPyTorch(nn.Module):
    """
    Combines ALS collaborative filtering and tag-based content similarity.

    ALS path (exact):
        mix_als_emb = (Y.T @ diag(c) @ Y + λI)^-1 @ Y.T @ c   [D_als]
        als_scores  = item_factors @ mix_als_emb                 [n_sounds]

    Tags path:
        mix_tag_emb = (tag_factors.T @ volume) / (||volume||_1 + ε)   [D_tags]
        tag_scores  = tag_factors @ mix_tag_emb                         [n_sounds]

    Combined:
        scores = als_weight * als_scores + tag_weight * tag_scores
    """

    def __init__(
        self,
        item_factors: np.ndarray,  # [n_sounds, D_als]
        tag_factors: np.ndarray,  # [n_sounds, D_tags]  — aligned to sounds_ids
        regularization: float,
        als_weight: float = 0.7,
        tag_weight: float = 0.3,
    ):
        super().__init__()
        self.regularization = regularization
        self.als_weight = als_weight
        self.tag_weight = tag_weight

        self.register_buffer("item_factors", torch.tensor(item_factors, dtype=torch.float32))
        self.register_buffer("tag_factors", torch.tensor(tag_factors, dtype=torch.float32))

    # -- ALS fold-in (identical to ALSRecommender._fold_in) ------------------

    def _als_fold_in(self, volume: torch.Tensor) -> torch.Tensor:
        Y = self.item_factors  # [n_sounds, D_als]
        c = volume  # [n_sounds]
        YtCY = (Y * c.unsqueeze(1)).T @ Y  # [D_als, D_als]
        reg = self.regularization * torch.eye(Y.shape[1], dtype=Y.dtype, device=Y.device)
        rhs = Y.T @ c  # [D_als]
        return torch.linalg.solve(YtCY + reg, rhs)  # [D_als]

    # -- Tags mix embedding --------------------------------------------------

    def _tag_mix_emb(self, volume: torch.Tensor) -> torch.Tensor:
        # volume-weighted mean of active sound tag embeddings
        weight_sum = volume.sum() + 1e-8
        return (self.tag_factors.T @ volume) / weight_sum  # [D_tags]

    # -- Forward -------------------------------------------------------------

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: [n_sounds] float32, values in [0, 1]
        Returns:
            scores: [n_sounds] float32
        """
        als_emb = self._als_fold_in(volume)  # [D_als]
        als_scores = self.item_factors @ als_emb  # [n_sounds]

        tag_emb = self._tag_mix_emb(volume)  # [D_tags]
        tag_scores = self.tag_factors @ tag_emb  # [n_sounds]

        return self.als_weight * als_scores + self.tag_weight * tag_scores

    def recommend(
        self,
        volume: torch.Tensor,
        N: int = 10,
        filter_already_liked: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            volume: [n_sounds] float32 volume vector (0–1)
            N: number of recommendations
            filter_already_liked: mask sounds already in the mix
        Returns:
            (indices, scores): both [N]
        """
        with torch.no_grad():
            scores = self.forward(volume)

        if filter_already_liked:
            scores = scores.clone()
            scores[volume > 0] = -torch.inf

        top_scores, top_ids = torch.topk(scores, k=N)
        return top_ids, top_scores


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_hybrid_recommender(
    als_model,
    tags_model: SoundsSimilarityTagsModel,
    sounds_ids: list,
    als_weight: float = 0.7,
    tag_weight: float = 0.3,
) -> HybridRecommenderPyTorch:
    """
    Build and return a HybridRecommenderPyTorch from trained models.

    als_model  : fitted implicit.als.AlternatingLeastSquares
    tags_model : fitted SoundsSimilarityTagsModel
    sounds_ids : canonical ordering list (from sounds_ids.json)
    """
    tag_factors = _align_tag_factors(tags_model, sounds_ids)
    return HybridRecommenderPyTorch(
        item_factors=als_model.item_factors,
        tag_factors=tag_factors,
        regularization=als_model.regularization,
        als_weight=als_weight,
        tag_weight=tag_weight,
    )


# ---------------------------------------------------------------------------
# TFLite version
# ---------------------------------------------------------------------------


def build_hybrid_tflite(
    als_model,
    tags_model: SoundsSimilarityTagsModel,
    sounds_ids: list,
    N: int = 10,
    als_weight: float = 0.7,
    tag_weight: float = 0.3,
    filter_already_liked: bool = True,
    output_path: str = "data/hybrid_recommender.tflite",
) -> bytes:
    """
    Build a TFLite model whose forward pass IS recommend():
        input : [1, n_sounds] float32 volume vector (0–1), L1-normalised
        output: indices [1, N] int32, scores [1, N] float32

    TFLite cannot run linalg.solve, so the ALS fold-in uses a precomputed
    foldin_matrix = (YtY + λI)^-1 @ Y.T  (drops per-query confidence weighting).

    Graph:
        ALS:  volume → Dense(D_als) → Dense(n_sounds) → * als_weight ─┐
        Tags: volume → Dense(D_tags) → Dense(n_sounds) → * tag_weight ─┤ Add
                                                                        ↓
                               tf.where(volume > 0, -1e9, scores)  [mask liked]
                                                                        ↓
                                             tf.math.top_k(masked, k=N)
                                                                        ↓
                                                        (indices, scores)
    """
    import tensorflow as tf

    n_sounds = len(sounds_ids)
    Y = als_model.item_factors.astype(np.float32)  # [n_sounds, D_als]
    D_als = Y.shape[1]

    # Precomputed ALS fold-in matrix
    YtY = Y.T @ Y
    reg = np.eye(D_als, dtype=np.float32) * float(als_model.regularization)
    foldin_matrix = np.linalg.solve(YtY + reg, Y.T)  # [D_als, n_sounds]

    tag_factors = _align_tag_factors(tags_model, sounds_ids)  # [n_sounds, D_tags]
    D_tags = tag_factors.shape[1]

    # Build Keras functional model
    inp = tf.keras.Input(shape=(n_sounds,), name="volume")  # [1, n_sounds]

    # ALS path
    als_emb = tf.keras.layers.Dense(
        D_als,
        use_bias=False,
        name="als_foldin",
        kernel_initializer=tf.constant_initializer(foldin_matrix.T),  # [n_sounds, D_als]
    )(inp)
    als_out = tf.keras.layers.Dense(
        n_sounds,
        use_bias=False,
        name="als_scores",
        kernel_initializer=tf.constant_initializer(Y),  # [D_als, n_sounds]
    )(als_emb)
    als_out = tf.keras.layers.Lambda(lambda x: x * als_weight, name="als_weighted")(als_out)

    # Tags path
    tag_emb = tf.keras.layers.Dense(
        D_tags,
        use_bias=False,
        name="tag_foldin",
        kernel_initializer=tf.constant_initializer(tag_factors),  # [n_sounds, D_tags]
    )(inp)
    tag_out = tf.keras.layers.Dense(
        n_sounds,
        use_bias=False,
        name="tag_scores",
        kernel_initializer=tf.constant_initializer(tag_factors.T),  # [D_tags, n_sounds]
    )(tag_emb)
    tag_out = tf.keras.layers.Lambda(lambda x: x * tag_weight, name="tag_weighted")(tag_out)

    # Combine scores
    scores = tf.keras.layers.Add(name="combined_scores")([als_out, tag_out])  # [1, n_sounds]

    # Mask already-liked sounds (volume > 0 → -1e9)
    if filter_already_liked:
        scores = tf.keras.layers.Lambda(
            lambda x: tf.where(x[0] > 0.0, tf.fill(tf.shape(x[1]), -1e9), x[1]),
            name="mask_liked",
        )([inp, scores])

    # Top-N as the final output
    top_scores, top_indices = tf.keras.layers.Lambda(
        lambda x: tf.math.top_k(x, k=N),
        name="top_k",
    )(scores)

    keras_model = tf.keras.Model(
        inputs=inp,
        outputs={"indices": top_indices, "scores": top_scores},
        name="hybrid_recommender",
    )
    keras_model.trainable = False

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_bytes)

    print(f"Saved TFLite model → {output_path}  ({len(tflite_bytes) / 1024:.1f} KB)")
    return tflite_bytes


# ---------------------------------------------------------------------------
# Main — build both from saved weights
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from models.als_co_occurence_matrix import build_als_recommender

    SOUNDS_IDS_JSON_PATH = "data/sounds_ids.json"
    ALS_ITEM_FACTORS_PATH = "data/als_item_factors.npy"
    SOUNDS_TAG_PATH = "~/Data/ContentAirTable/content_database_sounds.csv"
    MUSIC_TAG_PATH = "~/Data/ContentAirTable/content_database_music.csv"
    CONTENT_TO_SOUNDS_IDS_PATH = "data/content_to_sound_id.json"

    with open(SOUNDS_IDS_JSON_PATH) as f:
        sounds_ids = json.load(f)

    # -- Reconstruct a lightweight als_model namespace from saved weights ----
    class _ALSProxy:
        """Minimal proxy so we can pass saved weights to the factory functions."""

        def __init__(self, item_factors_path, regularization):
            self.item_factors = np.load(item_factors_path)
            self.regularization = regularization

    als_proxy = _ALSProxy(ALS_ITEM_FACTORS_PATH, ALS_REGULARIZATION)

    # -- Load tags model -----------------------------------------------------
    tags_model = SoundsSimilarityTagsModel(
        path_music_csv=MUSIC_TAG_PATH,
        path_sounds_csv=SOUNDS_TAG_PATH,
        path_sounds_ids_json=SOUNDS_IDS_JSON_PATH,
        path_content_to_sounds_ids=CONTENT_TO_SOUNDS_IDS_PATH,
        co_occ_vectorization_strategy="ppmi",
        co_occ_densification_strategy="svd",
    )

    # -- PyTorch hybrid ------------------------------------------------------
    hybrid = build_hybrid_recommender(als_proxy, tags_model, sounds_ids)
    hybrid.eval()
    print(f"\nHybridRecommenderPyTorch ready")
    print(f"  item_factors : {hybrid.item_factors.shape}")
    print(f"  tag_factors  : {hybrid.tag_factors.shape}")

    # Quick inference test
    n_sounds = len(sounds_ids)
    test_volume = torch.zeros(n_sounds)
    for sid in ["ambience.river", "ambience.rain", "ambience.waterfall"]:
        if sid in tags_model.sound_id_to_idx:
            idx = sounds_ids.index(sid) if sid in sounds_ids else -1
            if idx >= 0:
                test_volume[idx] = 0.8

    ids, scores = hybrid.recommend(test_volume, N=10)
    print(f"\nTop-10 recommendations for test mix:")
    for i, s in zip(ids.numpy(), scores.numpy()):
        print(f"  [{i:4d}] {sounds_ids[i]:<50s} {s:.4f}")

    # -- TFLite hybrid -------------------------------------------------------
    try:
        build_hybrid_tflite(als_proxy, tags_model, sounds_ids)
    except ImportError:
        print("\nTensorFlow not installed — skipping TFLite export.")
