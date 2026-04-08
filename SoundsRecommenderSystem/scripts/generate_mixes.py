"""
Sample the most popular mixes of length 2–3 from df_listening and save to
data/mixes.txt, one mix per line (sounds comma-separated).

Strategy:
  - Build mix play_count from listening data (filtered to sounds_ids)
  - For each target length, take the top-K most played mixes
  - Total ~2–3K mixes balanced across lengths

Usage:
    python scripts/generate_mixes.py
    python scripts/generate_mixes.py --top_per_len 1000 --output data/generated_mixes.txt
"""

import argparse
import json

import pandas as pd


LISTENING_CSV = "/Users/emulie/Downloads/bq-results-20260325-195823-1774468728937.csv"
SOUNDS_IDS_PATH = "data/sounds_ids.json"
DEFAULT_OUTPUT = "data/generated_mixes.txt"
DEFAULT_TOP = 1000  # per length → 2K total for lengths 2–3


def load_mix_counts(csv_path: str, sounds_ids: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["sounds"] = df["sounds"].apply(json.loads)

    known = set(sounds_ids)
    df["sounds"] = df["sounds"].apply(lambda x: [s for s in x if s in known])
    df = df[df["sounds"].apply(len) > 0].copy()

    df["mix_id"] = df["sounds"].apply(lambda s: "|".join(sorted(s)))
    df["mix_len"] = df["sounds"].apply(len)

    mix_counts = df.groupby(["mix_id", "mix_len"]).size().reset_index(name="play_count")
    return mix_counts


def sample_top_mixes(mix_counts: pd.DataFrame, lengths: list, top_per_len: int) -> list:
    mixes = []
    for length in lengths:
        top = mix_counts[mix_counts["mix_len"] == length].nlargest(top_per_len, "play_count")[
            "mix_id"
        ]
        mixes.extend(top.tolist())
        print(
            f"  Length {length}: {len(top):,} mixes (top {top_per_len} of "
            f"{(mix_counts['mix_len'] == length).sum():,} unique)"
        )
    return mixes


def save_mixes(mixes: list, output_path: str) -> None:
    with open(output_path, "w") as f:
        for mix_id in mixes:
            f.write(",".join(mix_id.split("|")) + "\n")
    print(f"\n{len(mixes):,} mixes saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listening_csv", default=LISTENING_CSV)
    parser.add_argument("--sounds_ids", default=SOUNDS_IDS_PATH)
    parser.add_argument(
        "--top_per_len", type=int, default=DEFAULT_TOP, help="Top N mixes to keep per length"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.sounds_ids) as f:
        sounds_ids = json.load(f)

    print("Loading listening data...")
    mix_counts = load_mix_counts(args.listening_csv, sounds_ids)

    print(f"\nSampling top {args.top_per_len} mixes per length:")
    mixes = sample_top_mixes(mix_counts, lengths=[2, 3], top_per_len=args.top_per_len)

    save_mixes(mixes, args.output)
