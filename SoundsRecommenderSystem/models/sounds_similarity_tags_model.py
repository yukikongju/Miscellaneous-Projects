"""
Sounds Similarity Embedding using Tags

Steps:
1. Build df_sounds_music with tags
2. Create Tags Occurence Matrix
3. Vectorize co-occurence matrix: PPMI (current), Jaccard, TF-IDF,
4. Make co-occurence matrix dense: SVD w/ normalization

Usage

Later:
- add sound embedding: CLAP, wav2vec (speech focused)
- add automatic sounds tag pipeline for new sounds


"""

import json
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import List


class SoundsSimilarityTagsModel:
    TAG_COL_LABELS = {
        "Content ID": "content_id",
        "ℹ️ Themes": "themes",
        "ℹ️ User Needs": "user_needs",
        "ℹ️ Environment": "environment",
        "ℹ️ Sound Features": "sound_features",
    }

    def __init__(
        self,
        path_music_csv: str,
        path_sounds_csv: str,
        path_sounds_ids_json: str,
        # path_sounds_to_content_ids: str,
        path_content_to_sounds_ids: str,
        co_occ_vectorization_strategy: str,
        co_occ_densification_strategy: str,
    ) -> None:
        # initialize dense tag matrix
        self.df_sounds_music = self._init_df_sound_music(path_music_csv, path_sounds_csv)
        self.sounds_id = self._read_json(path_sounds_ids_json)
        # self.sound_to_content_id = self._read_json(path_sounds_to_content_ids)
        self.content_to_sound_id = self._read_json(path_content_to_sounds_ids)
        self.missing_sounds = pd.read_csv("data/missing_sounds.csv")
        self.missing_sounds = self.missing_sounds.set_index("sound_id")

        self.all_tags = self._get_all_tags(self.df_sounds_music)
        self.df_tags = self._build_tag_matrix(
            df_sounds_music=self.df_sounds_music, tags=self.all_tags
        )

        # !WARNING - print missing ids
        print(
            f"In sounds_id, not in content_to_sound {set(self.sounds_id) - set(self.content_to_sound_id.values())}"
        )
        print(
            f"In df, not in content_to_sound map: {set(self.df_sounds_music['content_id']) - set(self.content_to_sound_id.keys())}"
        )
        # set(model.sounds_id) - set(model.df_tags_cleaned.index)
        # set(model.df_tags_cleaned.index) - set(model.sounds_id)

        # put in order + add missing tags
        self.df_tags_cleaned = self.df_tags.rename(self.content_to_sound_id)
        self.df_tags_cleaned = self.df_tags_cleaned.groupby(level=0).sum()
        # missing_sounds = set(self.sounds_id) - set(self.df_tags_cleaned['sounds'].unique())
        # print(missing_sounds)
        self.df_tags_cleaned = pd.concat([self.df_tags_cleaned, self.missing_sounds])
        self.df_tags_cleaned = self.df_tags_cleaned[self.df_tags_cleaned.index.isin(self.sounds_id)]

        self.df_tags_cleaned = self.df_tags_cleaned.sort_index(axis=0)
        self.sound_id_to_idx = {sid: i for i, sid in enumerate(self.df_tags_cleaned.index)}
        self.idx_to_sound_id = {i: sid for i, sid in enumerate(self.df_tags_cleaned.index)}
        self.sound_ids = set(self.sound_id_to_idx.keys())

        # transform co-occurence matrix
        self.X = self._vectorize_tag_matrix(
            df_tags=self.df_tags_cleaned, strategy=co_occ_vectorization_strategy
        )
        self.X = self._densify_tag_matrix(X=self.X, strategy=co_occ_densification_strategy)

    def _read_json(self, json_file: str):
        with open(json_file) as f:
            content = json.load(f)
        return content

    def _init_df_sound_music(self, path_music_csv: str, path_sounds_csv: str):
        df_music: pd.DataFrame = pd.read_csv(path_music_csv)
        df_sounds: pd.DataFrame = pd.read_csv(path_sounds_csv)
        df_sounds_cleaned = df_sounds[self.TAG_COL_LABELS.keys()].rename(
            columns=self.TAG_COL_LABELS
        )
        df_musics_cleaned = df_music[self.TAG_COL_LABELS.keys()].rename(columns=self.TAG_COL_LABELS)
        df_sounds_music = pd.concat([df_sounds_cleaned, df_musics_cleaned])
        return df_sounds_music

    def _get_all_tags(self, df_sounds_music: pd.DataFrame):
        all_tags = []
        for col in self.TAG_COL_LABELS.values():
            tags = df_sounds_music[col].dropna().str.split(",").explode().str.strip()
            all_tags += tags[tags != ""].unique().tolist()
        return sorted(set(all_tags))

    def _build_tag_matrix(self, df_sounds_music: pd.DataFrame, tags: List[str]):
        tag_cols = self.TAG_COL_LABELS.values()
        df = df_sounds_music.copy()

        def parse_tags(val):
            if pd.isna(val) or val == "NaN" or val == "":
                return []
            return [t.strip() for t in str(val).split(",")]

        # Build binary matrix
        rows = []
        for _, row in df.iterrows():
            tag_set = set()
            for col in tag_cols:
                tag_set.update(parse_tags(row[col]))
            rows.append({tag: int(tag in tag_set) for tag in tags})

        df_tags = pd.DataFrame(rows, index=df["content_id"])
        df_tags.index.name = "content_id"
        return df_tags

    def _vectorize_tag_matrix(self, df_tags: pd.DataFrame, strategy: str = "ppmi"):
        if strategy == "ppmi":
            X = df_tags.values.astype(float)  # (n_sounds, n_tags)
            C = X @ X.T  # (n_sounds, n_sounds)

            # PPMI
            total = C.sum()
            marginals = C.sum(axis=1) / total
            P_ij = C / total
            P_i_P_j = np.outer(marginals, marginals)

            with np.errstate(divide="ignore", invalid="ignore"):
                pmi = np.where(P_ij > 0, np.log2(P_ij / (P_i_P_j + 1e-10)), 0)

            ppmi = np.clip(pmi, 0, None)  # Positive PMI: drop negative values
            return ppmi
        else:
            raise NotImplementedError(
                f"{strategy=} not implemented. Please select one of the following: 'ppmi', "
            )

    def _densify_tag_matrix(self, X: np.ndarray, strategy: str = "svd"):
        if strategy == "svd":
            svd = TruncatedSVD(n_components=30, random_state=42)
            X_dense = normalize(
                svd.fit_transform(X), norm="l2"
            )  # note: normalize with cosine similarity: faiss, annoy, hnswlib

            print(f"Explained variance: {svd.explained_variance_ratio_.sum():.1%}")
            return X_dense
        else:
            raise NotImplementedError(
                f"{strategy=} not implemented. Please select one of the following: 'svd', "
            )

    def get_mix_embedding(self, mix: List[str]):
        # method 1
        # vectors = []
        # for sound_id in mix:
        #     if sound_id in self.sound_ids:
        #         idx = self.sound_id_to_idx[sound_id]
        #         v = self.X[idx]
        #         vectors.append(v)
        # mix_embedding1 = np.mean(vectors, axis=0)
        # method 2:
        indexes = [self.sound_id_to_idx[sound_id] for sound_id in mix]
        mix_embedding = np.mean(self.X[indexes], axis=0)
        #! WARNING - WHAT TO DO WHEN vectors is empty? mix embedding is null vector
        return mix_embedding

    def get_similar_sound_candidates(self, mix: List[str]):  # ? FIXME: return scores (?)
        mix_vector = self.get_mix_embedding(mix)
        similarity_scores = self.X @ mix_vector
        ranked_indexes = np.argsort(similarity_scores)
        return ranked_indexes

    def get_candidate_redundancy_score(self, mix: List[str]):
        """
        Max cosine similarity between candidate and any sound in the mix
        """
        mix_idx = [self.sound_id_to_idx[sound_id] for sound_id in mix]
        mix_mat = self.X[mix_idx]
        redundancy_scores = (self.X @ mix_mat.T).max(axis=1)
        return redundancy_scores


if __name__ == "__main__":
    SOUNDS_TAG_PATH = "~/Data/ContentAirTable/content_database_sounds.csv"
    MUSIC_TAG_PATH = "~/Data/ContentAirTable/content_database_music.csv"
    SOUNDS_IDS_JSON_PATH = "data/sounds_ids.json"
    # SOUNDS_IDS_TO_CONTENT_IDS_JSON_PATH = "../data/sounds_id_to_content_id_cleaned_sorted.json"
    CONTENT_TO_SOUNDS_IDS_JSON_PATH = "data/content_to_sound_id.json"
    # LISTENING_CSV = "/Users/emulie/Downloads/script_job_67b7f56b753852fd2a3f35baed75edbc_0.csv"
    model = SoundsSimilarityTagsModel(
        path_music_csv=MUSIC_TAG_PATH,
        path_sounds_csv=SOUNDS_TAG_PATH,
        path_sounds_ids_json=SOUNDS_IDS_JSON_PATH,
        # path_sounds_to_content_ids=SOUNDS_IDS_TO_CONTENT_IDS_JSON_PATH,
        path_content_to_sounds_ids=CONTENT_TO_SOUNDS_IDS_JSON_PATH,
        co_occ_vectorization_strategy="ppmi",
        co_occ_densification_strategy="svd",
    )
    mix = ["ambience.river", "ambience.heavyrain", "ambience.waterfall", "ambience.rain"]
    x = model.get_mix_embedding(mix)
    ranked_idx = model.get_similar_sound_candidates(mix)
    sounds_id_ranked = [model.idx_to_sound_id[i] for i in ranked_idx]
    print()
