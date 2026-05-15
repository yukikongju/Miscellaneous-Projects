"""
Raw data ingestion and member-level stratified splitting.

Usage:
    from data.loader import load_raw, split_by_member, drop_identifiers
    df = load_raw()
    train, val, test = split_by_member(df)
    X_train = drop_identifiers(train)

Caveats:
    - split_by_member performs member-level (not row-level) splits to prevent
      leakage; a member's sessions appear in exactly one fold.
    - drop_identifiers removes SESSION_KEY, SESSION_DATE, and MEMBER_KEY;
      keep the originals if you need them for allocation output.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import DATA_PATH, RANDOM_SEED, TEST_SIZE, VAL_SIZE

ID_COLS = ["SESSION_KEY", "SESSION_DATE", "MEMBER_KEY"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["SESSION_DATE"])
    return df


def split_by_member(df: pd.DataFrame):
    """
    Member-level split: no MEMBER_KEY appears in more than one fold.
    Stratified by whether a member has single vs. multiple sessions.
    Returns train_df, val_df, test_df (all identifier columns still present).
    """
    member_sessions = df.groupby("MEMBER_KEY")["SESSION_KEY"].count().reset_index()
    member_sessions.columns = ["MEMBER_KEY", "n_sessions"]
    member_sessions["stratum"] = (member_sessions["n_sessions"] > 1).astype(int)

    members = member_sessions["MEMBER_KEY"].values
    strata = member_sessions["stratum"].values

    # First cut: train vs temp (val + test)
    train_members, temp_members = train_test_split(
        members,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=strata,
        random_state=RANDOM_SEED,
    )

    # Second cut: val vs test from temp
    temp_strata = member_sessions.set_index("MEMBER_KEY").loc[temp_members, "stratum"]
    val_members, test_members = train_test_split(
        temp_members,
        test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        stratify=temp_strata,
        random_state=RANDOM_SEED,
    )

    train_df = df[df["MEMBER_KEY"].isin(train_members)].copy()
    val_df = df[df["MEMBER_KEY"].isin(val_members)].copy()
    test_df = df[df["MEMBER_KEY"].isin(test_members)].copy()

    assert len(set(train_members) & set(test_members)) == 0
    assert len(set(train_members) & set(val_members)) == 0

    return train_df, val_df, test_df


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop SESSION_KEY, SESSION_DATE, MEMBER_KEY — not features."""
    return df.drop(columns=[c for c in ID_COLS if c in df.columns])
