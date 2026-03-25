"""

Transformations
- sounds remap: removes sounds that don't start with prefix
- Remove single-sounds mixes
- Popularity Normalization: log-popularity dampening, TF-IDF
- Per-Mix Volume Normalization
- Confidence Weighting (ALS-style)
- Binary
- Deduplication of Near-Identical Mixes (to remove power users who listens the same mix)
- Minimum Support Filtering (Remove Sounds that appear in fewer than N mixes)


Target Shape: Mix x Sounds (session-level matrix)
X : [num_mix, num_sounds]

     rain  ocean  birds  binaural  deepsleep  campfire  ...  (150 sounds)
m1 [ 0.28,  1.00,  0.00,    0.00,      0.00,     0.00, ... ]
m2 [ 0.00,  0.00,  0.50,    0.00,      0.00,     0.00, ... ]
m3 [ 0.55,  0.00,  0.00,    0.50,      0.50,     0.00, ... ]

"""

import json
import numpy as np
import pandas as pd

# import scipy.sparse as sp
from typing import List, Dict, Tuple


def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sounds"] = df["sounds"].apply(json.loads)
    df["sounds_volume"] = df["sounds_volume"].apply(lambda x: list(map(float, json.loads(x))))
    return df


def _filter_sounds_without_allowed_prefix(
    df: pd.DataFrame,
    prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Remove individual sounds from each mix that don't start with an allowed prefix."""
    if prefixes is None:
        prefixes = [
            "ambience",
            "asmr",
            "binaural",
            "isochronic",
            "music",
            "solfeggio",
            "soundjourney",
        ]

    allowed = tuple(p + "." for p in prefixes)

    def _filter_row(row):
        pairs = [
            (s, v) for s, v in zip(row["sounds"], row["sounds_volume"]) if s.startswith(allowed)
        ]
        sounds, volumes = zip(*pairs) if pairs else ([], [])
        return pd.Series({"sounds": list(sounds), "sounds_volume": list(volumes)})

    df = df.copy()
    df[["sounds", "sounds_volume"]] = df.apply(_filter_row, axis=1)
    return df


def _filter_single_sound_mixes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove mixes with fewer than 2 sounds."""
    return df[df["sounds"].apply(len) >= 2].reset_index(drop=True)


def _filter_minimum_support(df: pd.DataFrame, min_support: int = 5) -> pd.DataFrame:
    """
    Remove sounds that appear in fewer than min_support mixes.
    Re-applies single-sound filter after removal.
    """
    from collections import Counter

    sound_counts = Counter(s for sounds in df["sounds"] for s in sounds)
    allowed = {s for s, count in sound_counts.items() if count >= min_support}

    df = df.copy()
    df["sounds_volume"] = df.apply(
        lambda row: [v for s, v in zip(row["sounds"], row["sounds_volume"]) if s in allowed],
        axis=1,
    )
    df["sounds"] = df["sounds"].apply(lambda sounds: [s for s in sounds if s in allowed])

    return _filter_single_sound_mixes(df)


def _deduplicate_near_identical_mixes(
    df: pd.DataFrame,
    similarity_threshold: float = 0.9,
) -> pd.DataFrame:
    """
    Per user, remove mixes whose sound-set Jaccard similarity to any already-kept
    mix exceeds the threshold. Keeps the first occurrence of each near-identical group.
    """

    def _jaccard(a: list, b: list) -> float:
        sa, sb = set(a), set(b)
        union = len(sa | sb)
        return len(sa & sb) / union if union else 0.0

    keep_indices = []

    for _, user_df in df.groupby("user_id"):
        indices = user_df.index.tolist()
        sounds_list = user_df["sounds"].tolist()
        kept: list[int] = []  # positions within this user's list

        for i in range(len(indices)):
            is_dup = any(
                _jaccard(sounds_list[i], sounds_list[j]) >= similarity_threshold for j in kept
            )
            if not is_dup:
                kept.append(i)

        keep_indices.extend(indices[i] for i in kept)

    return df.loc[keep_indices].reset_index(drop=True)


def _load_sound_vocab(
    sounds_ids_path: str = "data/sounds_ids.json",
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Load the fixed sound vocabulary from disk.

    Returns
    -------
    sound_ids   : ordered list of sound names
    sound_to_idx: sound name -> column index
    idx_to_sound: column index -> sound name
    """
    with open(sounds_ids_path) as f:
        sound_ids: list[str] = json.load(f)
    sound_to_idx = {s: i for i, s in enumerate(sound_ids)}
    idx_to_sound = {i: s for i, s in enumerate(sound_ids)}
    return sound_ids, sound_to_idx, idx_to_sound


def _build_mix_sound_matrix(
    df: pd.DataFrame,
    sound_ids: List[str],
    sound_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    Build a dense Mix x Sounds matrix using a fixed sound vocabulary.

    Columns are ordered by sound_ids, giving a stable shape regardless
    of which sounds appear in the current data slice. Sounds in the data that
    are not in the vocabulary are silently dropped.

    Returns
    -------
    X       : ndarray of shape (num_mixes, num_sounds)
    """
    X = np.zeros((len(df), len(sound_ids)), dtype=np.float32)
    for mix_pos, (_, row) in enumerate(df.iterrows()):
        for sound, vol in zip(row["sounds"], row["sounds_volume"]):
            if sound in sound_to_idx:
                X[mix_pos, sound_to_idx[sound]] = vol

    return X


def _normalize_per_mix_volume(X: np.ndarray) -> np.ndarray:
    """L1-normalize each row so that volumes within a mix sum to 1."""
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return X / row_sums


def _apply_log_dampening(X: np.ndarray) -> np.ndarray:
    """Apply log1p element-wise: x -> log(1 + x)."""
    return np.log1p(X)


def _apply_tfidf(X: np.ndarray) -> np.ndarray:
    """
    Weight each entry by IDF so that sounds appearing in many mixes are down-weighted.
    TF is taken to be the raw volume (or whatever is already in the matrix).
    IDF(sound) = log(N / document_frequency(sound))
    """
    N = X.shape[0]
    doc_freq = (X > 0).sum(axis=0).astype(float)
    doc_freq[doc_freq == 0] = 1.0
    idf = np.log(N / doc_freq)
    return X * idf


def _apply_confidence_weighting(X: np.ndarray, alpha: float = 40.0) -> np.ndarray:
    """
    ALS-style confidence: C_ui = 1 + alpha * r_ui for observed entries.
    Unobserved entries remain 0 (confidence 1 is handled implicitly by ALS).
    """
    return np.where(X > 0, 1.0 + alpha * X, 0.0).astype(np.float32)


def _binarize(X: np.ndarray) -> np.ndarray:
    """Set every non-zero entry to 1."""
    return (X > 0).astype(np.float32)


def build_mix_sound_pipeline(listening_file_path: str, sound_id_file_path: str):
    df = pd.read_csv(listening_file_path)
    sound_ids, sound_to_idx, idx_to_sound = _load_sound_vocab(sounds_ids_path=sound_id_file_path)

    df = _convert_data_types(df)
    df = _filter_sounds_without_allowed_prefix(df)
    df = _filter_single_sound_mixes(df)
    df = _filter_minimum_support(df, min_support=5)
    df = _deduplicate_near_identical_mixes(df, similarity_threshold=0.9)
    X = _build_mix_sound_matrix(df, sound_ids, sound_to_idx)

    n_mixes, n_sounds = X.shape
    print(f"Matrix shape : {n_mixes} mixes x {n_sounds} sounds")

    X_norm = _normalize_per_mix_volume(X)  # row-sum = 1
    X_norm = _apply_tfidf(X_norm)  # popularity normalization
    X_conf = _apply_confidence_weighting(X_norm)  # ALS confidence scores
    # X_bin  = binarize(X)                      # binary presence/absence
    return X_conf, sound_to_idx, idx_to_sound


if __name__ == "__main__":
    LISTENING_EVENT_FILE_PATH = (
        "/Users/emulie/Downloads/script_job_67b7f56b753852fd2a3f35baed75edbc_0.csv"
    )
    SOUNDS_LISTENING_IDS_PATH = "data/sounds_ids.json"
    X, sound_to_idx, idx_to_sound = build_mix_sound_pipeline(
        listening_file_path=LISTENING_EVENT_FILE_PATH, sound_id_file_path=SOUNDS_LISTENING_IDS_PATH
    )
    print()
