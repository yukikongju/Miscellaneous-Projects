"""

**Scoring**

score = (
    α * normalize(content_sim)
  + β * normalize(co_occ)
  + γ * normalize(pop_penalty)
  + δ * normalize(-diversity_penalty)
)

α = 0.2   # content
β = 0.5   # co-occurrence (strongest)
γ = 0.1   # popularity
δ = 0.3   # diversity

- Similarity Matrix: X_dense [n_sounds x 20 dims]
  1. sounds x tags from metadata (binary sparse)
  2. PPMI (normalize for popular tags)
  3. Truncated SVD (20 dims, dense, normalized)
- Co-occurence Matrix: ALS
    1. mix x sounds from listening events (binary sparse)

input: mix => [24, 5, 6] OR [0, ..., 0.8, ...]
X_dense: (n_sounds, n_tags) [sound_similarity matrix]
sounds_volume: (n_sounds, ) [sounds_volume] 0-1 float
co_occ: (n_sounds, n_sounds) => score



"""

#  from typing import Dict
import torch.nn as nn
import numpy as np


class DummyMatrixFactorizationModel(nn.Module):
    NUM_SOUNDS = 495

    def __init__(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
